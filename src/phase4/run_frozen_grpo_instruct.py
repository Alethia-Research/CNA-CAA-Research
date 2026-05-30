# =====================================================================
# SELF-CONTAINED FROZEN-LAYER GRPO ON INSTRUCT RUNNER
# Zero-dependency pipeline: includes dynamic patching, data prep,
# custom step-decay reward functions, autograd validation, and trainer.
# Optimized for Google Colab and Single Tesla T4 GPU (16GB VRAM)
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

# --- Core Dependencies Check and Auto-installer ---
try:
    import unsloth
    import trl
    import datasets
    import transformers
    print("[+] Core machine learning environments detected.")
except ImportError:
    print("[*] Core libraries missing. Starting automated container setup...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-q", "uv"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "datasets", "trl", "transformers", "accelerate", "peft", "bitsandbytes", "vllm"], check=True)
    print("[+] Container patched and configured successfully. Restarting execution cell.")

from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# =====================================================================
# 1. INLINE OPENAI DATASET PREPARATION
# =====================================================================
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

def prepare_gsm8k_dataset(limit=1000, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "gsm8k_train_grpo.jsonl")
    
    if os.path.exists(out_path):
        print(f"[+] Verified active dataset cache at {out_path}")
        return out_path
        
    print("[*] Loading OpenAI GSM8K from Hugging Face...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    # Restrict volume to prevent container timeouts
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
        
    print(f"[*] Compiling and formatting {len(dataset)} samples for GRPO...")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            # Extract final answer target
            answer_text = item["answer"]
            if "####" in answer_text:
                target = answer_text.split("####")[-1].strip().replace(",", "")
            else:
                target = answer_text.strip().replace(",", "")
                
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]}
            ]
            
            f.write(json.dumps({
                "prompt": prompt_messages,
                "target_answer": target
            }) + "\n")
            
    print(f"[+] Dataset compiled successfully at {out_path}")
    return out_path

# =====================================================================
# 2. INLINE CUSTOM REWARD FUNCTIONS
# =====================================================================
def get_completion_text(comp) -> str:
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
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text_clean = text.replace(",", "")
    
    boxed_match = re.search(r"\\boxed\{([0-9.-]+)\}", text_clean)
    if boxed_match:
        return boxed_match.group(1)
        
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text_clean)
    if numbers:
        return numbers[-1]
    return ""

def is_mathematically_equivalent(s1: str, s2: str) -> bool:
    if not s1 or not s2:
        return False
    try:
        return float(s1) == float(s2)
    except ValueError:
        return s1.strip().lower() == s2.strip().lower()

def format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")
        
        if num_start == 1 and num_end == 1:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            if start_idx < end_idx and len(comp_text[end_idx + 8 :].strip()) > 0:
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
            rewards.append(0.0)
    return rewards

def math_correctness_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        if extracted and is_mathematically_equivalent(extracted, target_clean):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def p_grpo_format_reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
    rewards = []
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")
        if num_start == 1 and num_end == 1:
            start_idx = comp_text.find("<think>")
            end_idx = comp_text.find("</think>")
            if start_idx < end_idx and len(comp_text[end_idx + 8 :].strip()) > 0:
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
    rewards = []
    gamma = 0.99
    transition_tokens = ["wait", "hmm", "but", "thinking", "actually", "let me check"]
    
    for comp, target in zip(completions, target_answer):
        comp_text = get_completion_text(comp)
        extracted = extract_xml_answer(comp_text)
        target_clean = target.strip().replace(",", "")
        is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)
        
        if not is_correct:
            rewards.append(0.0)
            continue
            
        think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
        think_content_list = [block.lower() for block in think_blocks]
        
        num_start = comp_text.count("<think>")
        num_end = comp_text.count("</think>")
        if num_start > 0 and num_end == 0:
            unclosed_monologue = comp_text.split("<think>")[-1].lower()
            think_content_list.append(unclosed_monologue)
            
        global_content = comp_text.lower()
        steps = 0
        for token in transition_tokens:
            steps += global_content.count(token)
            
        block_penalty = 0.0
        if num_start > 1:
            block_penalty = 0.5 * (num_start - 1)
            
        decayed_reward = max(0.0, float(gamma**steps) - block_penalty)
        rewards.append(decayed_reward)
    return rewards

# =====================================================================
# 3. INTERACTIVE STEP TRACKER AND AUTOGRAD INSULATION CALLBACK
# =====================================================================
current_step = 0

class StepTrackerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        global current_step
        current_step = state.global_step

    def on_substep_end(self, args, state, control, **kwargs):
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
                
            non_zero_frozen = [name for name, val in frozen_grads if val > 1e-9]
            if len(non_zero_frozen) > 0:
                print(f"[-] CRITICAL FAILURE: {len(non_zero_frozen)} frozen parameters received gradients!")
                raise RuntimeError("Gradient insulation check failed: Frozen layers received gradients.")
            else:
                print("[+] VERIFICATION SUCCESS: 100% gradient insulation confirmed! All frozen layers have 0.0 gradient norm.")
            print("="*50 + "\n")

def get_combined_reward_fn(mode="step-grpo", stage_steps=50):
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
            
        # Two-stage dynamic weight scaling
        if current_step <= stage_steps:
            w_format = 1.0
            w_correct = 0.1
        else:
            w_format = 0.2
            w_correct = 1.0
            
        return [w_format * f + w_correct * c for f, c in zip(f_rewards, c_rewards)]
    return reward_fn

# =====================================================================
# 4. TRAINING INTERRUPT CONTROL
# =====================================================================
def execute_frozen_grpo(model_name="unsloth/Qwen2.5-1.5B-Instruct"):
    print("\n" + "="*70)
    print("LAUNCHING SELF-CONTAINED FROZEN-LAYER GRPO PIPELINE")
    print("="*70)
    print(f"Model Target:       {model_name}")
    print("Active LoRA Layers: L24-L27 (last_4 periphery parameters only)")
    print("Frozen Layers:      L0-L23 (central logic fully insulated)")
    print("VRAM constraints:   Optimized for Tesla T4 (16GB VRAM) Colocate Mode")
    print("="*70 + "\n")

    # A. Dataset compilation
    data_path = prepare_gsm8k_dataset(limit=1000, output_dir="data")
    train_dataset = load_dataset("json", data_files=data_path, split="train")

    # B. Load Model & Tokenizer
    # Dynamic verification of vLLM usability to prevent CUDA-mismatch import crashes
    has_vllm = False
    try:
        import vllm
        has_vllm = True
        print("[+] vLLM library successfully importable.")
    except ImportError:
        print("[!] Warning: vLLM import failed (likely due to CUDA mismatch). Falling back to Hugging Face generation.")

    print("[*] Initializing FastLanguageModel in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=896,
        load_in_4bit=True,
        fast_inference=has_vllm,
        gpu_memory_utilization=0.95
    )

    # C. Target Modules and Layer Freezing
    num_layers = model.config.num_hidden_layers
    layers_to_transform = list(range(num_layers - 4, num_layers))
    print(f"[+] Confined training to layers: {layers_to_transform}")

    print("[*] Wrapping PEFT adapters with Unsloth configuration...")
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        layers_to_transform=layers_to_transform
    )

    # D. GRPO Configurations
    training_args = GRPOConfig(
        output_dir="./grpo_cot_output",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=384,
        max_steps=150,
        use_vllm=has_vllm,
        vllm_mode="colocate" if has_vllm else None,
        logging_steps=5,
        save_steps=50,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none"
    )

    # E. Trainer Launch
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[get_combined_reward_fn(mode="step-grpo", stage_steps=50)],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[StepTrackerCallback()]
    )

    print("[*] Commencing reinforcement learning loop...")
    trainer.train()

    # F. Save adapters
    final_save_path = "./grpo_cot_output/final_lora"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"[+] Pipeline complete! LoRA adapters saved at {final_save_path}")

if __name__ == "__main__":
    execute_frozen_grpo()
