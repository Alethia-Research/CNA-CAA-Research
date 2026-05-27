import os
import argparse
import json
from datasets import load_dataset

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant must think step-by-step "
    "inside <think>...</think> tags to solve the mathematical problem, and then provide "
    "the final numeric answer outside the tags."
)

def extract_target_answer(answer_text):
    """
    Extracts the clean numeric answer after '####' in GSM8K answers.
    """
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip().replace(",", "")

def prepare_data(limit_train=None, limit_test=None, output_dir="data"):
    print("[*] Loading GSM8K dataset from Hugging Face...")
    try:
        dataset = load_dataset("openai/gsm8k", "main")
    except Exception as e:
        print(f"[-] Error loading GSM8K: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for split, limit, filename in [
        ("train", limit_train, "gsm8k_train_grpo.jsonl"),
        ("test", limit_test, "gsm8k_test_grpo.jsonl")
    ]:
        print(f"[*] Processing {split} split...")
        data_split = dataset[split]
        if limit is not None:
            data_split = data_split.select(range(min(limit, len(data_split))))
            
        out_path = os.path.join(output_dir, filename)
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data_split:
                prompt_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]}
                ]
                target_answer = extract_target_answer(item["answer"])
                
                # Write formatted sample to JSONL
                json_line = {
                    "prompt": prompt_messages,
                    "target_answer": target_answer,
                    "original_answer": item["answer"]
                }
                f.write(json.dumps(json_line) + "\n")
                count += 1
                
        print(f"[+] Saved {count} samples to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset for GRPO training")
    parser.add_argument("--limit_train", type=int, default=None, help="Limit number of train samples")
    parser.add_argument("--limit_test", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for JSONL files")
    args, _ = parser.parse_known_args()
    
    prepare_data(limit_train=args.limit_train, limit_test=args.limit_test, output_dir=args.output_dir)
