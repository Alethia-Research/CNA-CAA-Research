import os
import re
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Standard GSM8K few-shot examples with Chain-of-Thought reasoning
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
    # GSM8K ground truth answers end with "#### <number>"
    match = re.search(r"####\s*(-?\d+)", answer_text)
    if match:
        return match.group(1)
    return None

def extract_predicted_number(completion_text):
    # Look for "The answer is X" pattern
    match = re.findall(r"[T|t]he answer is\s*(-?\d+)", completion_text)
    if match:
        return match[-1]
    # Fallback: Extract the last number in the completion
    numbers = re.findall(r"-?\d+", completion_text)
    if numbers:
        return numbers[-1]
    return None

def evaluate_gsm8k(model_path, device="cuda", limit_samples=50):
    print(f"[*] Loading model and tokenizer from: {model_path}")
    if not os.path.exists(model_path):
        print(f"[-] Model path {model_path} does not exist. Exiting.")
        return

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print("[*] LoRA adapter checkpoint detected. Loading base model first...")
        with open(adapter_config_path, "r") as f:
            config = json.load(f)
        base_model_name = config.get("base_model_name_or_path")
        print(f"[*] Base model from adapter config: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()

    print("[*] Loading GSM8K test split...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    eval_data = list(dataset)
    if limit_samples is not None:
        eval_data = eval_data[:limit_samples]
        print(f"[+] Sub-sampling evaluation to {limit_samples} questions.")

    correct = 0
    total = 0

    print("[*] Running math reasoning evaluation...")
    for item in tqdm(eval_data, desc="GSM8K Evaluation"):
        question = item["question"]
        gold_answer = extract_ground_truth(item["answer"])
        
        # Build prompt with few-shot chain of thought prefix
        prompt = GSM8K_FEW_SHOT + f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract just the newly generated answer part
        generated_answer = full_text[len(prompt):].strip()
        
        pred_val = extract_predicted_number(generated_answer)
        
        is_correct = (pred_val == gold_answer)
        if is_correct:
            correct += 1
        total += 1
        
        # Print first 2 questions for debugging/visibility
        if total <= 2:
            print(f"\n--- Example {total} ---")
            print(f"Question: {question}")
            print(f"Generated CoT: {generated_answer}")
            print(f"Predicted Value: {pred_val} | Ground Truth: {gold_answer}")
            print(f"Match: {is_correct}")
            print("-" * 40)

    acc = correct / total if total > 0 else 0.0
    print("\n==================================================")
    print("GSM8K EVALUATION RESULTS SUMMARY")
    print("==================================================")
    print(f"Total Evaluated: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Math Accuracy  : {acc:.2%}")
    print("==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GSM8K on fine-tuned models")
    parser.add_argument("--model_path", type=str, default="data/results/qwen-1.5b-control", help="Path to the model checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions (default: 50)")
    args = parser.parse_args()
    
    evaluate_gsm8k(args.model_path, device=args.device, limit_samples=args.limit)
