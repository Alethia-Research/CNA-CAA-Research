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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
def clean_code(prompt, completion):
    # Remove markdown code block markers if the model wrapped its output in them
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
        
    # Standard cleaning: if the model repeated the prompt signature, remove it
    prompt_lines = [line.strip() for line in prompt.split("\n") if line.strip()]
    comp_lines = [line.strip() for line in completion.split("\n") if line.strip()]
    
    # Simple heuristic: if completion starts with the same signature, slice it out
    if comp_lines and prompt_lines and comp_lines[0].startswith(prompt_lines[0][:15]):
        # Model repeated signature; let's try to extract from the indentation onward
        lines = completion.split("\n")
        signature_found = False
        for idx, line in enumerate(lines):
            if "def " in line and any(p in line for p in prompt_lines[0][:10]):
                completion = "\n".join(lines[idx + 1:])
                break
                
    return prompt + "\n" + completion

def evaluate_humaneval(model_path, device="cuda", limit_samples=40):
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

    # Ensure pad token is set for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print("[*] Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    
    problems = list(dataset)
    if limit_samples is not None:
        problems = problems[:limit_samples]
        print(f"[+] Sub-sampling evaluation to {limit_samples} coding tasks.")

    correct = 0
    total = 0

    print("[*] Running coding execution evaluation...")
    for item in tqdm(problems, desc="HumanEval Evaluation"):
        prompt = item["prompt"]
        test = item["test"]
        entry_point = item["entry_point"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract generated part
        completion = full_text[len(prompt):]
        
        # Combine prompt signature and generated completion body
        full_code = clean_code(prompt, completion)
        
        # Build execution string
        exec_code = f"{full_code}\n\n{test}\ncheck({entry_point})"
        
        passed = False
        try:
            # Execute in a sandboxed dictionary namespace
            global_namespace = {}
            exec(exec_code, global_namespace)
            passed = True
        except AssertionError:
            passed = False
        except Exception as e:
            # Code failed to compile or run
            passed = False
            
        if passed:
            correct += 1
        total += 1
        
        # Print first code sample for visibility/debugging
        if total <= 1:
            print(f"\n--- Example {total} (Function: {entry_point}) ---")
            print("--- Generated Code ---")
            print(full_code)
            print("----------------------")
            print(f"Execution Test Passed: {passed}")
            print("-" * 50)

    acc = correct / total if total > 0 else 0.0
    print("\n==================================================")
    print("HUMANEVAL CODING EVALUATION RESULTS SUMMARY")
    print("==================================================")
    print(f"Total Evaluated: {total}")
    print(f"Passed Tests   : {correct}")
    print(f"Coding Accuracy: {acc:.2%}")
    print("==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate HumanEval on fine-tuned models")
    parser.add_argument("--model_path", type=str, default="data/results/qwen-1.5b-control", help="Path to the model checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--limit", type=int, default=40, help="Number of questions (default: 40)")
    args = parser.parse_args()
    
    evaluate_humaneval(args.model_path, device=args.device, limit_samples=args.limit)
