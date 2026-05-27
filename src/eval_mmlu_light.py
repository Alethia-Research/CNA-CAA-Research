import os
import sys
from types import ModuleType

# Workaround for HuggingFace Transformers bug in environments with incomplete torch.distributed
try:
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

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Standard subjects representing math, science, humanities, and social sciences
TEST_SUBJECTS = [
    "college_computer_science",
    "elementary_mathematics",
    "clinical_knowledge",
    "professional_law",
    "philosophy",
    "econometrics"
]

def format_subject(subject_name):
    return subject_name.replace("_", " ")

def format_example(question, choices, answer_idx=None):
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    if answer_idx is not None:
        prompt += f" {chr(65+answer_idx)}\n\n"
    return prompt

def evaluate_mmlu(model_path, device="cuda", limit_per_subject=50):
    print(f"[*] Loading model and tokenizer from: {model_path}")
    is_local = os.path.exists(model_path)
    if not is_local:
        print(f"[*] Model path '{model_path}' not found locally. Attempting to load directly from Hugging Face Hub...")

    # Self-healing tokenizer loader
    tokenizer = None
    tokenizer_paths = [model_path, "Qwen/Qwen2.5-1.5B-Instruct"]
    for path in tokenizer_paths:
        if not path:
            continue
        for use_fast in [True, False]:
            try:
                print(f"[*] Trying to load tokenizer from '{path}' (use_fast={use_fast})...")
                temp_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=use_fast, trust_remote_code=True)
                test_ids = temp_tokenizer("test").input_ids
                if len(test_ids) > 0:
                    tokenizer = temp_tokenizer
                    print(f"[+] Successfully loaded functional tokenizer from '{path}' (use_fast={use_fast})")
                    break
            except Exception as e:
                print(f"[-] Failed loading from '{path}' (use_fast={use_fast}): {e}")
        if tokenizer is not None:
            break

    if tokenizer is None:
        print("[-] Critical: Failed to load a functional tokenizer from any source. Exiting.")
        return

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Get token IDs for A, B, C, D choice tokens
    choices = ["A", "B", "C", "D"]
    choice_ids = []
    for c in choices:
        # Get the ID of the choice character
        tokens = tokenizer.encode(c, add_special_tokens=False)
        # Handle cases where tokenizers prepend spaces or add special chars
        choice_ids.append(tokens[-1])
    
    print(f"[+] Choice token IDs map: {dict(zip(choices, choice_ids))}")

    overall_correct = 0
    overall_total = 0
    subject_accuracies = {}

    for subject in TEST_SUBJECTS:
        print(f"\n[*] Evaluating subject: {format_subject(subject)}...")
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"[-] Error loading subject {subject}: {e}. Skipping.")
            continue

        dev_dataset = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)
        
        # Build 5-shot prompt prefix using dev split examples
        few_shot_prefix = "The following are multiple choice questions (with answers).\n\n"
        for i in range(min(5, len(dev_dataset))):
            ex = dev_dataset[i]
            few_shot_prefix += format_example(ex["question"], ex["choices"], ex["answer"])

        correct = 0
        total = 0
        
        # Limit evaluation examples for speed
        eval_examples = list(dataset)
        if limit_per_subject is not None:
            eval_examples = eval_examples[:limit_per_subject]

        for ex in tqdm(eval_examples, desc=f"MMLU: {format_subject(subject)}"):
            prompt = few_shot_prefix + format_example(ex["question"], ex["choices"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1] # Get logits of the last generated token
                
                # Compare logits of A, B, C, D
                choice_logits = logits[choice_ids]
                prediction_idx = torch.argmax(choice_logits).item()
                
            if prediction_idx == ex["answer"]:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        subject_accuracies[subject] = acc
        print(f"-> Accuracy for {format_subject(subject)}: {acc:.2%} ({correct}/{total})")
        
        overall_correct += correct
        overall_total += total

    print("\n==================================================")
    print("MMLU EVALUATION RESULTS SUMMARY")
    print("==================================================")
    for subject, acc in subject_accuracies.items():
        print(f"- {format_subject(subject):<30}: {acc:.2%}")
    
    macro_acc = np.mean(list(subject_accuracies.values()))
    micro_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    print("-" * 50)
    print(f"Macro Average Accuracy: {macro_acc:.2%}")
    print(f"Micro Average Accuracy (Total): {micro_acc:.2%}")
    print("==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MMLU on fine-tuned models")
    parser.add_argument("--model_path", type=str, default="data/results/qwen-1.5b-control", help="Path to the model checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions per subject (default: 50)")
    args = parser.parse_args()
    
    evaluate_mmlu(args.model_path, device=args.device, limit_per_subject=args.limit)
