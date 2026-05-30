import os
import argparse
import json
from datasets import load_dataset

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
