# =====================================================================
# ALL-IN-ONE GRPO COGNITIVE MONOLOGUE TRAINING PIPELINE
# Designed for Google Colab and Single-GPU Notebook environments
# =====================================================================

import os
import re
import sys
import json
import subprocess

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
    Standard formatting reward. Checks if the model output follows the <think>...</think> structure.
    Returns:
        1.0 for perfect single-block structure.
        0.3 for multi-block formats (exploit prevention).
        0.2 for unclosed blocks.
        0.5 for malformed/empty trailing.
        0.0 for no structure.
    """
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")
        
        # Strict single-block format check
        if num_start == 1 and num_end == 1:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            # Ensure correct ordering and some content after </think>
            if start_idx < end_idx and len(comp_text[end_idx + 8:].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif num_start > 1 or num_end > 1:
            # Multi-block formatting penalty
            rewards.append(0.3)
        elif num_start == 1 and num_end == 0:
            # Unclosed block penalty
            rewards.append(0.2)
        elif num_start == 0 and num_end == 1:
            # Missing start tag but has end tag
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
    Posterior-GRPO Formatting Reward.
    Zeroes out the formatting (process) reward if the final math answer is incorrect.
    This prevents the model from being reinforced for 'looking smart' while getting the math wrong.
    """
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        
        # Check correctness first
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = (extracted and extracted == target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        # If correct, evaluate format strictly
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")
        
        if num_start == 1 and num_end == 1:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            if start_idx < end_idx and len(comp_text[end_idx + 8:].strip()) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif num_start > 1 or num_end > 1:
            rewards.append(0.3)
        elif num_start == 1 and num_end == 0:
            rewards.append(0.2)
        elif num_start == 0 and num_end == 1:
            rewards.append(0.2)
        else:
            rewards.append(0.2)
            
    return rewards

def step_grpo_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    """
    Step-GRPO Decaying Reward.
    If the answer is correct, applies an exponential decay based on the number
    of cognitive transition steps (tokens like 'Wait', 'Hmm') inside the monologue.
    Applies multi-block penalty and handles unclosed tag fallback.
    R_j = max(0.0, gamma^steps - block_penalty) * 𝟙[correct]
    """
    rewards = []
    gamma = 0.99
    # Transition tokens indicating a new step/reasoning transition
    transition_tokens = ["wait", "hmm", "but", "thinking", "actually", "let me check"]
    
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = (extracted and extracted == target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        # 1. Capture all completed think blocks globally
        think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
        think_content_list = [block.lower() for block in think_blocks]
        
        # 2. Fallback for unclosed think tag: extract everything after the last <think>
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")
        if num_start > 0 and num_end == 0:
            unclosed_monologue = comp_text.split("<think>")[-1].lower()
            think_content_list.append(unclosed_monologue)
            
        # If no think tags were present at all, check the whole response
        if num_start == 0:
            think_content = comp_text.lower()
        else:
            think_content = " ".join(think_content_list)
            
        # 3. Count occurrence of cognitive transition tokens globally
        steps = 0
        for token in transition_tokens:
            steps += think_content.count(token)
            
        # 4. Multi-block penalty calculation: n - 1 extra blocks penalize 0.1 each
        block_penalty = 0.0
        if num_start > 1:
            block_penalty = 0.1 * (num_start - 1)
            
        # Calculate decayed reward and apply block penalty
        decayed_reward = max(0.0, float(gamma ** steps) - block_penalty)
        rewards.append(decayed_reward)
        
    return rewards

# Global step tracker for stage shifting
current_step = 0

class StepTrackerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        global current_step
        current_step = state.global_step

    def on_substep_end(self, args, state, control, **kwargs):
        # Verification of gradient flow - runs at the end of the very first backward pass
        model = kwargs.get("model")
        if model is not None and state.global_step == 0:
            print("\n" + "="*50)
            print("GRADIENT INSULATION VERIFICATION REPORT")
            print("="*50)
            
            frozen_grads = []
            active_grads = []
            
            for name, param in model.named_parameters():
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                if param.requires_grad:
                    active_grads.append((name, grad_norm))
                else:
                    frozen_grads.append((name, grad_norm))
                    
            print(f"[+] Total trainable parameters: {len(active_grads):,}")
            print("[*] Sample active parameter gradient norms:")
            for name, val in active_grads[:5]:
                print(f"  - {name}: grad_norm = {val:.6f}")
                
            print(f"[+] Total frozen parameters: {len(frozen_grads):,}")
            print("[*] Sample frozen parameter gradient norms:")
            for name, val in frozen_grads[:5]:
                print(f"  - {name}: grad_norm = {val:.6f}")
                
            # Perform strict validation: frozen parameters MUST have 0.0 gradient norm
            non_zero_frozen = [name for name, val in frozen_grads if val > 1e-9]
            if len(non_zero_frozen) > 0:
                print(f"[-] CRITICAL FAILURE: {len(non_zero_frozen)} frozen parameters received gradients!")
                print(f"[-] Sample leaked gradient: {non_zero_frozen[0]}")
                raise RuntimeError("Gradient insulation check failed: Frozen layers received gradients.")
            else:
                print("[+] VERIFICATION SUCCESS: 100% gradient insulation confirmed! All frozen layers have 0.0 gradient norm.")
            print("="*50 + "\n")

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

def run_pipeline(model_name="unsloth/Qwen2.5-3B-Instruct", mode="step-grpo", max_steps=150, stage_steps=50, limit_train=1000, no_vllm=False, num_generations=4, max_completion_length=384, layers_to_transform_str=None):
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
    dataset_path = "./data/gsm8k_train_grpo.jsonl"
    if not os.path.exists(dataset_path):
        print("[*] Training dataset missing. Running prepare_grpo_data.py...")
        from prepare_grpo_data import prepare_data
        prepare_data(limit_train=limit_train, output_dir="./data")
        
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
    
    # Resolve layers to transform dynamically
    layers_to_transform = None
    if layers_to_transform_str:
        preset = layers_to_transform_str.strip().lower()
        if preset in ["last_4", "last_8", "last_2"]:
            num_layers = model.config.num_hidden_layers
            if preset == "last_4":
                layers_to_transform = list(range(num_layers - 4, num_layers))
            elif preset == "last_8":
                layers_to_transform = list(range(num_layers - 8, num_layers))
            elif preset == "last_2":
                layers_to_transform = list(range(num_layers - 2, num_layers))
            print(f"[+] Dynamic layer configuration resolved preset '{preset}' to layers: {layers_to_transform}")
        else:
            try:
                layers_to_transform = [int(x.strip()) for x in layers_to_transform_str.split(",") if x.strip()]
                print(f"[+] Custom layer configuration resolved to layers: {layers_to_transform}")
            except ValueError:
                print(f"[-] Invalid format for --layers_to_transform '{layers_to_transform_str}'. Adapting all layers.")
                layers_to_transform = None

    # Configure LoRA (SFT Warm-Start Aware)
    if hasattr(model, "peft_config"):
        print("[+] Existing PEFT LoRA adapter detected. Warm-start active. Continuing training on active adapter...")
    else:
        print("[*] Setting up LoRA...")
        peft_kwargs = {
            "model": model,
            "r": 32,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "use_gradient_checkpointing": "unsloth"
        }
        if layers_to_transform is not None:
            peft_kwargs["layers_to_transform"] = layers_to_transform
            
        model = FastLanguageModel.get_peft_model(**peft_kwargs)
    
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
        max_steps=max_steps,
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
    parser.add_argument("--layers_to_transform", type=str, default=None, help="Preset (last_4, last_8, last_2) or comma-separated layer indices to adapt via LoRA (rest are frozen)")
    
    args, _ = parser.parse_known_args()
    
    run_pipeline(
        model_name=args.model_name,
        mode=args.mode,
        max_steps=args.max_steps,
        stage_steps=args.stage_steps,
        limit_train=args.limit_train,
        no_vllm=args.no_vllm,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        layers_to_transform_str=args.layers_to_transform
    )
