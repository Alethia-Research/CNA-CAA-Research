import os
import argparse
import json
from datasets import load_dataset

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant must think step-by-step "
    "inside <think>...</think> tags to solve the mathematical problem, and then provide "
    "the final numeric answer outside the tags."
)

def prepare_sft_cot_data(limit_samples=1000, output_dir="data"):
    print("[*] Loading GSM8K dataset from Hugging Face...")
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(f"[-] Error loading GSM8K: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "gsm8k_sft_cot.jsonl")
    
    samples = list(dataset)
    if limit_samples is not None:
        samples = samples[:min(limit_samples, len(samples))]
        
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in samples:
            raw_answer = item["answer"]
            if "####" in raw_answer:
                parts = raw_answer.split("####")
                reasoning = parts[0].strip()
                final_answer = parts[-1].strip().replace(",", "")
            else:
                reasoning = raw_answer.strip()
                final_answer = ""

            # Format strictly with ChatML structure
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{item['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>{reasoning}</think>{final_answer}<|im_end|>\n"
            )
            
            json_line = {"text": text}
            f.write(json.dumps(json_line) + "\n")
            count += 1
            
    print(f"[+] Successfully saved {count} SFT CoT formatted samples to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GSM8K SFT-CoT formatting data")
    parser.add_argument("--limit", type=int, default=1000, help="Number of samples (default: 1000)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    prepare_sft_cot_data(limit_samples=args.limit, output_dir=args.output_dir)
