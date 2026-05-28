"""
GRPO eval + tag diversity analysis. One cell end-to-end.
Usage in Colab:
  Set MODE = "grpo"     -> loads LoRA at /content/grpo_cot_output
  Set MODE = "baseline" -> loads base instruct model, zero-shot eval
"""

import os
import re
import sys
import json
import subprocess
from collections import Counter

# --- Dynamic PyTorch Patching for TorchAO / Unsloth Compatibility ---
try:
    import torch
    # 1. Patch missing low-precision integer dtypes (introduced in PyTorch 2.6)
    for prefix in ["int", "uint"]:
        for i in range(1, 8):
            attr = f"{prefix}{i}"
            if not hasattr(torch, attr):
                class DummyDtype:
                    def __init__(self, name):
                        self.name = name
                    def __repr__(self):
                        return self.name
                    def __hash__(self):
                        return hash(self.name)
                    def __eq__(self, other):
                        return isinstance(other, DummyDtype) and self.name == other.name
                setattr(torch, attr, DummyDtype(f"torch.{attr}"))
                
    # 2. Patch missing torch.utils._pytree.register_constant (introduced in PyTorch 2.5/2.6)
    import torch.utils._pytree
    if not hasattr(torch.utils._pytree, "register_constant"):
        torch.utils._pytree.register_constant = lambda x: None
except ImportError:
    pass
# --------------------------------------------------------------------

# --- Core Dependencies Check and Auto-installer ---
try:
    import unsloth
    import datasets
    import transformers
    import peft
    import bitsandbytes
    import accelerate
    import tqdm
    print("[+] Core machine learning environments detected.")
except ImportError:
    print("[*] Core libraries missing or incomplete. Starting automated container setup...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-q", "uv"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "datasets", "accelerate", "peft", "bitsandbytes", "transformers", "tqdm"], check=True)
    print("[+] Container patched and configured successfully.")

# --- Imports (Unsloth must be imported first to patch system libraries) ---
from unsloth import FastLanguageModel
import torch
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# ===== CONFIG =====
MODE = "grpo"  # "grpo" or "baseline"
LIMIT = 50
ZERO_SHOT = True  # True to use system prompt / ChatML (required for LF-GRPO reasoning), False for few-shot CoT
# ==================

# Set model path
if MODE == "grpo":
    MODEL_PATH = "kridaydave/Qwen-1.5B-LFGRPO-OPTIM"
    print(f"[+] Using GRPO LoRA at {MODEL_PATH}")
else:
    MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[+] Using base model: {MODEL_PATH}")

# Constants
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant must think step-by-step "
    "inside <think>...</think> tags to solve the mathematical problem, and then provide "
    "the final numeric answer outside the tags."
)

GSM8K_FEW_SHOT = """Question: There are 15 clients in a shop. If 5 clients leave and 3 new clients enter, how many clients are in the shop now?
Answer: There were 15 clients. 5 left, so 15 - 5 = 10 clients. 3 new clients entered, so 10 + 3 = 13 clients. The answer is 13.

Question: Weng earns $12 an hour for baby-sitting. Yesterday, she baby-sat for 5 hours. How much money did she earn?
Answer: Weng earns $12 an hour. She baby-sat for 5 hours, so she earned 12 * 5 = 60 dollars. The answer is 60.

Question: Betty is saving money to buy a wallet that costs $100. Betty has 20% of the money. How much more money does she need?
Answer: Betty has 20% of the money, which is 0.20 * 100 = 20 dollars. She needs 100 - 20 = 80 more dollars. The answer is 80.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia bought 5 bagels for $3 each, so they cost 5 * 3 = 15 dollars. She started with 23 dollars, so she has 23 - 15 = 8 dollars left. The answer is 8.

"""

TRANSITION_TOKENS = ["wait", "hmm", "but", "thinking", "actually", "let me check"]

def extract_ground_truth(answer_text):
    m = re.search(r"####\s*(-?\d+)", answer_text)
    return m.group(1) if m else None

def extract_predicted_number(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.replace(",", "")
    boxed = re.search(r"\\boxed\{([0-9.-]+)\}", text)
    if boxed:
        return boxed.group(1)
    match = re.findall(r"[T|t]he answer is\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match[-1]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None

def get_all_tags(text):
    return re.findall(r"</?(\w+)>", text)

# Load model
print(f"[*] Loading model from: {MODEL_PATH}")

if MODE == "grpo":
    local_config_path = os.path.join(MODEL_PATH, "adapter_config.json")
    if os.path.exists(local_config_path):
        print(f"[+] Found local config at {local_config_path}")
        with open(local_config_path) as f:
            base_name = json.load(f).get("base_model_name_or_path")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_name,
            max_seq_length=1024,
            load_in_4bit=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, MODEL_PATH)
    else:
        print(f"[+] Local config not found. Loading model directly via FastLanguageModel: {MODEL_PATH}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=1024,
            load_in_4bit=True,
            device_map="auto"
        )
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
        device_map="auto"
    )

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

FastLanguageModel.for_inference(model)

# Load GSM8K test set
dataset = load_dataset("openai/gsm8k", "main", split="test")
eval_data = list(dataset)[:LIMIT]
print(f"[*] Evaluating {len(eval_data)} questions...")

# Run eval
device = "cuda"
outputs = []
correct = 0
zero_shot = ZERO_SHOT

for idx, item in enumerate(tqdm(eval_data)):
    question = item["question"]
    gold_answer = extract_ground_truth(item["answer"])

    if zero_shot:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Fallback guard to prevent empty prompt tokenization crash
        if not prompt:
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        # Pre-fill assistant response with '<think>\n' to lock reasoning model into thinking monologues
        prompt += "<think>\n"
        max_new = 512
    else:
        prompt = GSM8K_FEW_SHOT + f"Question: {question}\nAnswer:"
        max_new = 150

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id, do_sample=False
        )

    completion = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()
    # Clean up standard assistant end-of-generation special tokens to keep outputs tidy
    completion = completion.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if zero_shot:
        completion = "<think>\n" + completion
    pred = extract_predicted_number(completion)
    is_correct = (pred == gold_answer)
    if is_correct:
        correct += 1

    outputs.append({
        "question": question,
        "completion": completion,
        "predicted": pred,
        "ground_truth": gold_answer,
        "correct": is_correct
    })

acc = correct / len(outputs)
print(f"\nAccuracy: {correct}/{len(outputs)} = {acc:.2%}")

# Save
os.makedirs("data", exist_ok=True)
save_name = f"data/{MODE}_eval_outputs.jsonl"
with open(save_name, "w") as f:
    for o in outputs:
        f.write(json.dumps(o) + "\n")
print(f"[+] Saved to {save_name}")

# ===== TAG DIVERSITY ANALYSIS =====
print("\n" + "=" * 60)
print("TAG DIVERSITY ANALYSIS")
print("=" * 60)

multi_block = single_block = no_think = invented_count = 0
all_invented_tags = []

for o in outputs:
    text = o["completion"]
    tags = get_all_tags(text)
    unique = set(tags)
    nb = text.count("<think>")

    if nb > 1:
        multi_block += 1
    elif nb == 1:
        single_block += 1
    else:
        no_think += 1

    invented = [t for t in unique if t.lower() not in ["think", "/think"]]
    if invented:
        invented_count += 1
        all_invented_tags.extend(invented)

T = len(outputs)
print(f"Total completions: {T}")
print(f"  Single <think> block: {single_block} ({100*single_block/T:.1f}%)")
print(f"  Multi-block (hack):  {multi_block} ({100*multi_block/T:.1f}%)")
print(f"  No think tag:        {no_think} ({100*no_think/T:.1f}%)")

if all_invented_tags:
    inv_counts = Counter(all_invented_tags)
    print(f"\nInvented tags: {invented_count}/{T} completions")
    print(f"Unique invented tags: {sorted(set(all_invented_tags))}")
    print(f"Occurrences: {inv_counts.most_common()}")
    if len(set(all_invented_tags)) >= 2:
        print(">>> MULTIPLE invented tags -> Schema Generalization (Interp A)")
    else:
        print(">>> SINGLE invented tag type -> Format Hallucination (Interp B)")
else:
    print("\nNo invented tags detected.")

hack_rate = multi_block / T
print(f"\nReward hack rate: {100*hack_rate:.1f}% ", end="")
if hack_rate > 0.2:
    print("-- conciseness gain is substantially confounded")
else:
    print("-- edge case, not systematic")

# Show multi-block examples
if multi_block > 0:
    print(f"\n--- Multi-block examples (up to 5) ---")
    shown = 0
    for o in outputs:
        if o["completion"].count("<think>") > 1 and shown < 5:
            print(f"  Q: {o['question'][:60]}...")
            print(f"  Tags: {get_all_tags(o['completion'])}")
            print(f"  Preview: {o['completion'][:150].replace(chr(10), ' | ')}")
            print()
            shown += 1

print("\nDone.")
