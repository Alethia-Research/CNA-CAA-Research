# All-in-One GRPO Cognitive Training Script

Copy the code block below and paste it into a **single Colab cell** to run the complete setup, formatting, and training.

```python
# =====================================================================
# ALL-IN-ONE GRPO COGNITIVE MONOLOGUE TRAINING PIPELINE
# Designed for Google Colab and Single-GPU Notebook environments
# =====================================================================

import os
import re
import sys
import json
import subprocess

# 1. Automatic Dependency Installer (runs when executed in Google Colab)
try:
    import unsloth
    import trl
    import datasets
    import vllm
    import accelerate
    import peft
    import bitsandbytes
    print("[+] All core libraries detected.")
except ImportError:
    print("[*] Core libraries missing or incomplete. Starting automatic package installations...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-q", "uv"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "datasets", "trl", "transformers", "accelerate", "peft", "bitsandbytes", "vllm"], check=True)
    print("[+] Libraries installed successfully! Please restart the runtime if import warnings persist.")

# 2. Imports (unsloth must be imported first to patch system libraries)
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# =====================================================================
# 3. REWARD FUNCTIONS & LOGIT-DIFF MATRICES
# =====================================================================

def get_completion_text(comp) -> str:
    """
    Safely extracts the generated text string from a completion.
    Handles ChatML formatted lists of message dicts.
    """
    if isinstance(comp, list):
        for msg in comp:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        if len(comp) > 0 and isinstance(comp[-1], dict):
            return comp[-1].get("content", "")
    elif isinstance(comp, str):
        return comp
    return ""

def extract_xml_answer(text: str) -> str:
    """
    Extracts the final answer after the </think> tag if present.
    Looks for LaTeX \boxed{...} first, then the last number.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text_clean = text.replace(",", "")
    
    # Match \boxed{number}
    boxed_match = re.search(r"\\boxed\{([0-9.-]+)\}", text_clean)
    if boxed_match:
        return boxed_match.group(1)
        
    # Fallback to the last number in text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text_clean)
    if numbers:
        return numbers[-1]
    return ""

def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    """
    Validates that the model uses <think>...</think> tags correctly.
    """
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        
        has_start = "<think>" in comp_text
        has_end = "</think>" in comp_text
        if has_start and has_end:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            if start_idx < end_idx and len(comp_text[end_idx + 8:].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif has_start or has_end:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards

def math_correctness_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Standard mathematical answer matching.
    """
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        if extracted and extracted == target_clean:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def p_grpo_format_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Posterior-GRPO Formatting: zeroes format rewards on failed answers.
    """
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = (extracted and extracted == target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        has_start = "<think>" in comp_text
        has_end = "</think>" in comp_text
        if has_start and has_end:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            if start_idx < end_idx and len(comp_text[end_idx + 8:].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        else:
            rewards.append(0.2)
    return rewards

def step_grpo_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Step-GRPO Decaying Reward: scales correct answers by gamma^steps.
    """
    rewards = []
    gamma = 0.99
    transition_tokens = ["wait", "hmm", "but", "thinking", "actually", "let me check"]
    
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = (extracted and extracted == target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        think_content = ""
        if "<think>" in comp_text and "</think>" in comp_text:
            think_content = comp_text.split("<think>")[-1].split("</think>")[0].lower()
        else:
            think_content = comp_text.lower()
            
        steps = 0
        for token in transition_tokens:
            steps += think_content.count(token)
            
        rewards.append(float(gamma ** steps))
    return rewards

# Global step tracker for stage shifting
current_step = 0

class StepTrackerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        global current_step
        current_step = state.global_step

def get_combined_reward_fn(mode="step-grpo", stage_steps=50):
    """
    Combines formatting and correctness rewards and dynamically transitions
    the weights from format-priming (Stage 1) to correctness (Stage 2).
    """
    def reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
        global current_step
        
        if mode == "p-grpo":
            f_rewards = p_grpo_format_reward_fn(prompts, completions, target_answer)
        else:
            f_rewards = format_reward_fn(prompts, completions)
            
        if mode == "step-grpo":
            c_rewards = step_grpo_reward_fn(prompts, completions, target_answer)
        else:
            c_rewards = math_correctness_reward_fn(prompts, completions, target_answer)
            
        # Two-stage weights
        if current_step <= stage_steps:
            w_format, w_correct = 1.0, 0.1
        else:
            w_format, w_correct = 0.2, 1.0
            
        combined = []
        for f, c in zip(f_rewards, c_rewards):
            combined.append(w_format * f + w_correct * c)
        return combined
    return reward_fn

# =====================================================================
# 4. TRAINING LAUNCH ORCHESTRATION
# =====================================================================

def run_pipeline(model_name="unsloth/Qwen2.5-3B-Instruct", mode="step-grpo", max_steps=150, stage_steps=50, limit_train=1000, no_vllm=False, num_generations=4, max_completion_length=384):
    print(f"[*] Starting All-in-One Pipeline in mode: {mode.upper()}")
    
    # Format math questions (GSM8K)
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The Assistant must think step-by-step "
        "inside <think>...</think> tags to solve the mathematical problem, and then provide "
        "the final numeric answer outside the tags."
    )
    
    print("[*] Pre-processing GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_subset = dataset["train"]
    if limit_train is not None:
         train_subset = train_subset.select(range(min(limit_train, len(train_subset))))
         
    os.makedirs("data", exist_ok=True)
    dataset_path = "data/gsm8k_train_grpo.jsonl"
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in train_subset:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]}
            ]
            target_answer = item["answer"].split("####")[-1].strip().replace(",", "")
            json_line = {
                "prompt": prompt_messages,
                "target_answer": target_answer
            }
            f.write(json.dumps(json_line) + "\n")
            
    print(f"[+] Data prepared at {dataset_path}")
    
    # Configure Torch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect if vLLM can actually be imported (prevents CUDA version mismatch crashes)
    vllm_available = False
    if device == "cuda":
        try:
            import vllm
            vllm_available = True
        except Exception as e:
            print(f"[!] Warning: vLLM is installed but cannot be imported (likely due to CUDA version mismatch). Falling back to standard HF generation. Error: {e}")
            
    use_vllm = not no_vllm and device == "cuda" and vllm_available
    
    # Load Model
    print(f"[*] Loading {model_name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512 + max_completion_length,
        load_in_4bit=True,
        fast_inference=use_vllm,
        gpu_memory_utilization=0.95
    )
    
    # Configure LoRA
    print("[*] Setting up LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth"
    )
    
    # Load formatted dataset
    train_dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Configure GRPOTrainer
    training_args = GRPOConfig(
        output_dir="./grpo_cot_output",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=512,
        max_completion_length=max_completion_length,
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
        logging_steps=5,
        save_steps=50,
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        report_to="none"
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[get_combined_reward_fn(mode=mode, stage_steps=stage_steps)],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[StepTrackerCallback()]
    )
    
    print("[*] Launching training loop...")
    trainer.train()
    
    final_save_path = "./grpo_cot_output/final_lora"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"[+] Pipeline complete! LoRA adapters saved at {final_save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run complete GRPO training pipeline.")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B-Instruct", help="Base model")
    parser.add_argument("--mode", type=str, choices=["standard", "p-grpo", "step-grpo"], default="step-grpo", help="GRPO Mode")
    parser.add_argument("--max_steps", type=int, default=150, help="Max steps")
    parser.add_argument("--stage_steps", type=int, default=50, help="Stage 1 steps")
    parser.add_argument("--limit_train", type=int, default=1000, help="Train set limit")
    parser.add_argument("--no_vllm", action="store_true", help="Disable vLLM colocate mode")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt")
    parser.add_argument("--max_completion_length", type=int, default=384, help="Max length of generated sequence")
    
    args, _ = parser.parse_known_args()
    
    run_pipeline(
        model_name=args.model_name,
        mode=args.mode,
        max_steps=args.max_steps,
        stage_steps=args.stage_steps,
        limit_train=args.limit_train,
        no_vllm=args.no_vllm,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length
    )
```
