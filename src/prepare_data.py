import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

def prepare_data():
    print("[*] Preparing data for LFSFT retraining...")
    os.makedirs("data", exist_ok=True)
    
    # 1. Load Alpaca dataset (Helpful subset)
    print("[*] Loading Alpaca dataset...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    
    # 2. Load PKU-SafeRLHF dataset (Harmless subset)
    print("[*] Loading PKU-SafeRLHF dataset...")
    safe_rlhf = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    
    formatted_data = []
    
    # Format helpful prompts (Alpaca)
    # We want 5,000 helpful prompts
    print("[*] Formatting Helpful prompts...")
    alpaca_indices = list(range(len(alpaca)))
    random.seed(42)
    random.shuffle(alpaca_indices)
    
    count_helpful = 0
    for idx in alpaca_indices:
        if count_helpful >= 5000:
            break
        item = alpaca[idx]
        instruction = item["instruction"]
        user_input = item["input"]
        response = item["output"]
        
        if user_input:
            prompt = f"{instruction}\n\n{user_input}"
        else:
            prompt = instruction
            
        chatml_text = (
            "<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
        
        formatted_data.append({
            "text": chatml_text,
            "category": "helpful"
        })
        count_helpful += 1

    # Format harmless prompts (PKU-SafeRLHF)
    # We filter for safe responses to unsafe prompts
    print("[*] Formatting Harmless prompts...")
    safe_rlhf_indices = list(range(len(safe_rlhf)))
    random.shuffle(safe_rlhf_indices)
    
    count_harmless = 0
    for idx in safe_rlhf_indices:
        if count_harmless >= 5000:
            break
        item = safe_rlhf[idx]
        prompt = item["prompt"]
        
        # We look for a refusal / safe completion. 
        # Ideally, we choose the response that is safe when the other is unsafe.
        r0 = item["response_0"]
        r1 = item["response_1"]
        s0 = item["is_response_0_safe"]
        s1 = item["is_response_1_safe"]
        
        safe_response = None
        if s0 and not s1:
            safe_response = r0
        elif s1 and not s0:
            safe_response = r1
        elif s0 and s1:
            # Both are safe, pick one randomly
            safe_response = r0 if random.random() > 0.5 else r1
            
        if safe_response is None:
            continue # Skip if both are unsafe
            
        chatml_text = (
            "<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{safe_response}<|im_end|>"
        )
        
        formatted_data.append({
            "text": chatml_text,
            "category": "harmless"
        })
        count_harmless += 1
        
    print(f"[+] Formatted {count_helpful} helpful and {count_harmless} harmless examples.")
    
    # Shuffle the final combined dataset
    random.shuffle(formatted_data)
    
    output_path = "data/lfsft_dataset.jsonl"
    print(f"[*] Saving prepared dataset to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
            
    print("[+] Dataset preparation complete!")

if __name__ == "__main__":
    prepare_data()
