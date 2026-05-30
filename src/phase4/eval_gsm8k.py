import os
import sys
import re
import json
import argparse
from tqdm import tqdm

# Add local path to sys.path to allow imports from current dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Workaround for HuggingFace Transformers bug in environments with incomplete torch.distributed
try:
    from types import ModuleType
    import torch.distributed.tensor.device_mesh
except ModuleNotFoundError:
    try:
        tensor_mod = ModuleType("torch.distributed.tensor")
        device_mesh_mod = ModuleType("torch.distributed.tensor.device_mesh")
        class DummyDeviceMesh:
            pass
        device_mesh_mod.DeviceMesh = DummyDeviceMesh
        tensor_mod.device_mesh = device_mesh_mod
        sys.modules["torch.distributed.tensor"] = tensor_mod
        sys.modules["torch.distributed.tensor.device_mesh"] = device_mesh_mod
    except Exception:
        pass

# --- Dynamic PyTorch Patching for TorchAO / Unsloth Compatibility ---
try:
    import torch
    class _StubDtype:
        """Hashable stand-in for torch.intN / torch.uintN (N=1..7)."""
        __slots__ = ("_name",)
        def __init__(self, name):
            self._name = name
        def __repr__(self):
            return self._name
        def __hash__(self):
            return hash(self._name)
        def __eq__(self, other):
            return isinstance(other, _StubDtype) and self._name == other._name
        def __ne__(self, other):
            return not self.__eq__(other)
        is_floating_point = False

    for _prefix in ["int", "uint"]:
        for _i in range(1, 8):
            _attr = f"{_prefix}{_i}"
            if not hasattr(torch, _attr):
                setattr(torch, _attr, _StubDtype(f"torch.{_attr}"))
                
    try:
        import torch.utils._pytree as _pt
        if not hasattr(_pt, "register_constant"):
            _pt.register_constant = lambda x: None
    except Exception:
        pass
except ImportError:
    pass
# --------------------------------------------------------------------

try:
    import torch
    
    # Dynamic hotpatch to bypass PEFT import check crash for old torchao versions
    try:
        import peft.import_utils as _piu
        _piu.is_torchao_available = lambda: False
    except Exception:
        pass
        
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from rewards import extract_xml_answer, is_mathematically_equivalent
except ImportError as e:
    print(f"[-] Missing library: {e}. Please ensure torch, unsloth, datasets, and rewards are available.")
    sys.exit(1)

# Constants
SYSTEM_PROMPT = """A conversation between User and Assistant. The Assistant is a precise mathematical reasoner.

When solving a math problem, the Assistant MUST:

1. Use <think>...</think> tags to reason step by step BEFORE giving the final answer
2. Inside <think>, identify ALL quantities in the problem before calculating anything
3. Inside <think>, write out EVERY multiplication and subtraction explicitly — never skip steps
4. Pay special attention to problems involving ratios, rates, and "X times faster/more/less" — these always require two operations, not one
5. After </think>, state the final answer as a plain number with no units, symbols, or extra text

The final answer must appear on its own line in this exact format:
#### <number>

Bad example (incomplete think, skipped step):
<think>Multiply sprints by distance.</think>
180
#### 180

Good example (full think, all steps explicit):
<think>
James runs 3 sprints per session.
He runs 3 sessions per week.
Each sprint is 60 meters.
Total = 3 × 3 × 60 = 540 meters.
</think>
James runs 540 meters per week.
#### 540"""


GSM8K_FEW_SHOT = """Question: There are 15 clients in a shop. If 5 clients leave and 3 new clients enter, how many clients are in the shop now?
Answer: There were 15 clients. 5 left, so 15 - 5 = 10 clients. 3 new clients entered, so 10 + 3 = 13 clients. The answer is 13.

Question: Weng earns $12 an hour for baby-sitting. Yesterday, she baby-sat for 5 hours. How much money did she earn?
Answer: Weng earns $12 an hour. She baby-sat for 5 hours, so she earned 12 * 5 = 60 dollars. The answer is 60.

Question: Betty is saving money to buy a wallet that costs $100. Betty has 20% of the money. How much more money does she need?
Answer: Betty has 20% of the money, which is 0.20 * 100 = 20 dollars. She needs 100 - 20 = 80 more dollars. The answer is 80.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia bought 5 bagels for $3 each, so they cost 5 * 3 = 15 dollars. She started with 23 dollars, so she has 23 - 15 = 8 dollars left. The answer is 8.

"""

def extract_ground_truth(answer_text):
    match = re.search(r"####\s*(-?\d+)", answer_text)
    if match:
        return match.group(1).strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned reasoning model on GSM8K.")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2.5-1.5B-Instruct", help="Path or HF ID of base model.")
    parser.add_argument("--adapter", type=str, default="kridaydave/Qwen-1.5B-LFGRPO-OPTIM", help="Path or HF ID of LoRA adapter.")
    parser.add_argument("--output", type=str, default="data/eval_results_step100.json", help="Path to save evaluation results.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation samples (default: evaluate all 1319 test examples).")
    parser.add_argument("--few_shot", action="store_true", help="Use standard few-shot chain of thought prompt instead of ChatML zero-shot.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--load_in_4bit", action="store_false", dest="load_in_4bit", help="Disable 4-bit quantization.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Evaluation environment: {device.upper()}")
    print(f"[*] Base model: {args.base_model}")
    print(f"[*] Adapter path: {args.adapter}")
    print(f"[*] Output results file: {args.output}")

    # Load model and tokenizer
    print("[*] Loading base model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        device_map="auto"
    )

    if args.adapter:
        print(f"[*] Loading adapter weights from: {args.adapter}")
        model.load_adapter(args.adapter, adapter_name="default")
    
    # Enable Fast Inference
    FastLanguageModel.for_inference(model)

    # Clean up and force ChatML template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading GSM8K test split...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    eval_data = list(dataset)
    if args.limit is not None:
        eval_data = eval_data[:args.limit]
        print(f"[+] Sub-sampling evaluation to {args.limit} examples.")
    else:
        print(f"[+] Evaluating all {len(eval_data)} test examples.")

    correct = 0
    outputs = []

    print("[*] Starting evaluation loop...")
    for idx, item in enumerate(tqdm(eval_data, desc="GSM8K Eval")):
        question = item["question"]
        gold_answer = extract_ground_truth(item["answer"])

        if args.few_shot:
            prompt = GSM8K_FEW_SHOT + f"Question: {question}\nAnswer:"
            max_new = 150
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not prompt:
                prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            # Pre-fill assistant thinking monologue starter
            prompt += "<think>\n"
            max_new = 512

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        completion = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False).strip()
        # Clean up assistant tags and end tokens
        completion = completion.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        if not args.few_shot:
            completion = "<think>\n" + completion

        predicted_answer = extract_xml_answer(completion)
        
        is_correct = predicted_answer is not None and gold_answer is not None and is_mathematically_equivalent(predicted_answer, gold_answer)
        if is_correct:
            correct += 1

        outputs.append({
            "idx": idx,
            "question": question,
            "completion": completion,
            "predicted": predicted_answer,
            "ground_truth": gold_answer,
            "correct": is_correct
        })

    accuracy = correct / len(outputs) if outputs else 0.0
    print("\n==================================================")
    print("GSM8K EVALUATION COMPLETE")
    print("==================================================")
    print(f"Total Evaluated: {len(outputs)}")
    print(f"Correct Answers: {correct}")
    print(f"Pass@1 Accuracy: {accuracy:.2%}")
    print("==================================================")

    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_data = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(outputs),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "few_shot": args.few_shot,
        "results": outputs
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Saved evaluation summary to: {args.output}")

if __name__ == "__main__":
    main()
