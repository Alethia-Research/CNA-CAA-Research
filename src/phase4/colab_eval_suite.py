"""
LF-GRPO Checkpoint Evaluation & Tag Diversity Suite
Alethia Research — Periphery Alignment & Central Logic
Optimized for Google Colab Tesla T4 (Single-GPU, low memory)

This script sequentially evaluates:
1. Baseline Model (Qwen2.5-1.5B-Instruct)
2. Checkpoint at Step 100 (saved adapter)
3. Checkpoint at Step 200 (saved adapter)

It measures:
- Math Accuracy on 50 GSM-8K test questions (Goal: beat 26/50 = 52%)
- Thinking Block Stats (Single-block, Multi-block hacks, No think)
- Invented Tag Detection (capturing Schema Generalization)
"""

import os
import re
import sys
import json
import shutil
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
    print("[*] Core libraries missing. Starting automated setup using uv...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-q", "uv"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "datasets", "accelerate", "peft", "bitsandbytes", "transformers", "tqdm"], check=True)
    print("[+] Setup complete. Running imports...")

# --- Imports (Unsloth must be imported first to patch system libraries) ---
from unsloth import FastLanguageModel
import torch
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# ===== CONFIGURATION =====
LIMIT = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

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

# =========================

def extract_ground_truth(answer_text):
    m = re.search(r"####\s*(-?\d+)", answer_text)
    return m.group(1) if m else None

def extract_predicted_number(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.replace(",", "")
    
    # 1. LaTeX \boxed{number}
    boxed = re.search(r"\\boxed\{([0-9.-]+)\}", text)
    if boxed:
        return boxed.group(1)
        
    # 2. "The answer is X"
    match = re.findall(r"[T|t]he answer is\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match[-1]
        
    # 3. Fallback: Last number
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None

def is_mathematically_equivalent(s1: str, s2: str) -> bool:
    if not s1 or not s2:
        return False
    s1_clean = s1.replace(",", "").strip()
    s2_clean = s2.replace(",", "").strip()
    try:
        return float(s1_clean) == float(s2_clean)
    except ValueError:
        return s1_clean.lower() == s2_clean.lower()

def get_all_tags(text):
    return re.findall(r"</?(\w+)>", text)

def unzip_checkpoint(path, dest_dir):
    """Safely extracts a zipped checkpoint from Drive or local."""
    if os.path.exists(dest_dir):
        print(f"[+] Found unzipped checkpoint directory at '{dest_dir}'")
        return dest_dir
        
    if not os.path.exists(path):
        # Check standard Drive location
        drive_path = f"/content/drive/MyDrive/grpo_checkpoints/{os.path.basename(path)}"
        if os.path.exists(drive_path):
            path = drive_path
        else:
            print(f"[-] Checkpoint file not found at '{path}' or '{drive_path}'")
            return None

    print(f"[*] Unzipping checkpoint '{path}' to '{dest_dir}'...")
    os.makedirs(dest_dir, exist_ok=True)
    shutil.unpack_archive(path, dest_dir, "zip")
    print(f"[+] Extraction complete: {dest_dir}")
    return dest_dir

def evaluate_model(model, tokenizer, eval_data, label="Model"):
    print(f"\n==== Starting Evaluation: {label} ====")
    correct = 0
    outputs = []
    
    for idx, item in enumerate(tqdm(eval_data, desc=f"Evaluating {label}")):
        question = item["question"]
        gold_answer = extract_ground_truth(item["answer"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Pre-fill assistant response to force cognitive monologue
        prompt += "<think>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,  # Prevent early truncation of planning
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False      # Greedy decoding for stable benchmarking
            )

        completion = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()
        completion = completion.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        completion = "<think>\n" + completion
        
        pred = extract_predicted_number(completion)
        is_correct = pred is not None and gold_answer is not None and is_mathematically_equivalent(pred, gold_answer)
        
        if is_correct:
            correct += 1

        outputs.append({
            "question": question,
            "completion": completion,
            "predicted": pred,
            "ground_truth": gold_answer,
            "correct": is_correct
        })

    # Summary and analysis
    acc = correct / len(outputs)
    print(f"\n[+] {label} accuracy: {correct}/{len(outputs)} = {acc:.2%}")
    
    # Tag analysis
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
            
        invented = [t for t in unique if t.lower() not in ["think", "/think", "boxed", "/boxed"]]
        if invented:
            invented_count += 1
            all_invented_tags.extend(invented)
            
    print(f"  Single-block <think>: {single_block}/{len(outputs)}")
    print(f"  Multi-block <think> (hack): {multi_block}/{len(outputs)}")
    print(f"  No <think> tag: {no_think}/{len(outputs)}")
    if all_invented_tags:
        print(f"  Invented tags detected: {sorted(set(all_invented_tags))} (Count: {invented_count})")
        
    return {
        "label": label,
        "accuracy": acc,
        "correct_count": correct,
        "multi_block": multi_block,
        "invented_count": invented_count,
        "invented_tags": list(set(all_invented_tags)),
        "outputs": outputs
    }

def run_suite(step100_path, step200_path):
    # 1. Load GSM8K Test Dataset
    print("[*] Loading GSM8K test split...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    eval_data = list(dataset)[:LIMIT]
    print(f"[+] Loaded {len(eval_data)} evaluation samples.")

    # 2. Check and unzip if necessary
    step100_dir = "./step_100_unzipped"
    step200_dir = "./step_200_unzipped"
    
    if step100_path.endswith(".zip"):
        step100_path = unzip_checkpoint(step100_path, step100_dir)
    if step200_path.endswith(".zip"):
        step200_path = unzip_checkpoint(step200_path, step200_dir)

    results = []

    # ================= BASELINE MODEL EVAL =================
    print("\n" + "=" * 50)
    print("1. EVALUATING BASELINE (Qwen2.5-1.5B-Instruct)")
    print("=" * 50)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
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
    baseline_res = evaluate_model(model, tokenizer, eval_data, label="Baseline Model")
    results.append(baseline_res)
    
    # Unload baseline to save VRAM
    del model
    torch.cuda.empty_cache()

    # ================= STEP 100 CHECKPOINT EVAL =================
    if step100_path and os.path.exists(step100_path):
        print("\n" + "=" * 50)
        print("2. EVALUATING STEP 100 CHECKPOINT")
        print("=" * 50)
        
        # Load base model again
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
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
        # Load adapter
        print(f"[*] Attaching Step 100 LoRA Adapter from '{step100_path}'...")
        model = PeftModel.from_pretrained(model, step100_path)
        FastLanguageModel.for_inference(model)
        
        step100_res = evaluate_model(model, tokenizer, eval_data, label="Step 100 Model")
        results.append(step100_res)
        
        # Unload
        del model
        torch.cuda.empty_cache()
    else:
        print(f"\n[!] Skipping Step 100 (Path '{step100_path}' not found).")

    # ================= STEP 200 CHECKPOINT EVAL =================
    if step200_path and os.path.exists(step200_path):
        print("\n" + "=" * 50)
        print("3. EVALUATING STEP 200 CHECKPOINT (FINAL)")
        print("=" * 50)
        
        # Load base model again
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
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
        # Load adapter
        print(f"[*] Attaching Step 200 LoRA Adapter from '{step200_path}'...")
        model = PeftModel.from_pretrained(model, step200_path)
        FastLanguageModel.for_inference(model)
        
        step200_res = evaluate_model(model, tokenizer, eval_data, label="Step 200 Model")
        results.append(step200_res)
        
        # Unload
        del model
        torch.cuda.empty_cache()
    else:
        print(f"\n[!] Skipping Step 200 (Path '{step200_path}' not found).")

    # ================= FINAL COMPARISON REPORT =================
    print("\n" + "=" * 60)
    print("FINAL LF-GRPO CHECKPOINT COMPARISON REPORT")
    print("=" * 60)
    
    print("\n| Model | Math Accuracy (GSM-8K) | Correct Count | Multi-Block Hacks | Invented Tag Count | Unique Invented Tags |")
    print("|---|---|---|---|---|---|")
    for r in results:
        tags_str = ", ".join(r["invented_tags"]) if r["invented_tags"] else "None"
        print(f"| {r['label']} | {r['accuracy']:.2%} | {r['correct_count']}/{LIMIT} | {r['multi_block']} | {r['invented_count']} | {tags_str} |")
        
    print("\n=======================================================")
    print("Goal to beat: 26/50 (52.00%)")
    for r in results:
        status = "PASSED! 🎉" if r["correct_count"] >= 26 else "FAILED ❌"
        print(f"  - {r['label']}: {r['correct_count']}/50 -> {status}")
    print("=======================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LF-GRPO checkpoints sequentially.")
    parser.add_argument("--step100", type=str, default="/content/grpo_cot_output/checkpoint-100", 
                        help="Path to Step 100 model directory or zip file")
    parser.add_argument("--step200", type=str, default="/content/grpo_cot_output/checkpoint-200", 
                        help="Path to Step 200 model directory or zip file")
    
    args, _ = parser.parse_known_args()
    
    run_suite(args.step100, args.step200)
