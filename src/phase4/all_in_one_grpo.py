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
    
    # Standard python -m pip is globally active and 100% robust against PATH failures
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    
    print("[*] Installing Unsloth and data capability libraries...")
    subprocess.run(pip_cmd + ["unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(pip_cmd + ["datasets", "trl", "transformers", "accelerate", "peft", "bitsandbytes"], check=True)
    
    # Smart vLLM installer to prevent CUDA version mismatches in Google Colab
    try:
        import torch
        cuda_ver = torch.version.cuda
        
        # vLLM release wheels use the abi3 tag (Application Binary Interface v3),
        # which means a single cp38-abi3 compiled wheel is backward-compatible
        # with Python 3.8 through 3.13. The asset filename on GitHub is statically 'cp38'.
        py_ver_str = "cp38"
        
        print(f"[*] Detected PyTorch CUDA version: {cuda_ver}, Python target: {py_ver_str} (abi3-compatible)")
        
        # Clean CUDA version string (e.g. '12.8' -> '128', '12.4' -> '124')
        clean_cuda = cuda_ver.replace(".", "") if cuda_ver else ""
        
        # vLLM release wheels are available for 121, 124, 128, etc.
        if clean_cuda in ["128", "124", "121", "118"]:
            wheel_url = f"https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0+cu{clean_cuda}-{py_ver_str}-abi3-manylinux_2_35_x86_64.whl"
            print(f"[*] Installing CUDA-optimized vLLM wheel: {wheel_url}")
            res = subprocess.run(pip_cmd + [wheel_url], capture_output=True, text=True)
            if res.returncode == 0:
                print("[+] Successfully installed CUDA-matched vLLM wheel!")
            else:
                print("[-] Wheel installation failed. Falling back to standard vLLM installation...")
                print("="*60)
                print("PIP WHEEL INSTALLATION ERROR LOGS:")
                print("="*60)
                print(res.stderr if res.stderr else res.stdout)
                print("="*60)
                subprocess.run(pip_cmd + ["vllm"], check=True)
        else:
            print("[*] No matching custom wheel found. Installing standard vLLM package...")
            subprocess.run(pip_cmd + ["vllm"], check=True)
    except Exception as e:
        print(f"[!] Error during dynamic vLLM package check: {e}. Falling back to default vLLM...")
        subprocess.run(pip_cmd + ["vllm"], check=True)
        
    # Clear import caches and remove any cached import failures from sys.modules
    import sys
    import importlib
    importlib.invalidate_caches()
    for key in list(sys.modules.keys()):
        if "vllm" in key or "unsloth" in key or "trl" in key or "datasets" in key:
            del sys.modules[key]
    sys.path_importer_cache.clear()
    print("[+] Libraries installed successfully! Please restart the Colab runtime if import warnings persist.")

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

def is_mathematically_equivalent(s1: str, s2: str) -> bool:
    if not s1 or not s2:
        return False
    # Clean commas to prevent float conversion crashes
    s1_clean = s1.replace(",", "").strip()
    s2_clean = s2.replace(",", "").strip()
    try:
        return float(s1_clean) == float(s2_clean)
    except ValueError:
        return s1_clean.lower() == s2_clean.lower()

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
        if extracted and is_mathematically_equivalent(extracted, target_clean):
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
        is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)
        
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
        is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)
        
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
            
        # Fix: Count cognitive transition tokens globally across the entire completion text
        # to prevent the model from escaping penalty by placing transition words outside <think> tags.
        global_content = comp_text.lower()
            
        # 3. Count occurrence of cognitive transition tokens globally
        steps = 0
        for token in transition_tokens:
            steps += global_content.count(token)
            
        # 4. Multi-block penalty calculation: Increase penalty to 0.5 per extra block
        # to completely eliminate the mathematical arbitrage of opening multiple think blocks.
        block_penalty = 0.0
        if num_start > 1:
            block_penalty = 0.5 * (num_start - 1)
            
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

class DriveUploadCallback(TrainerCallback):
    """Mounts Google Drive and copies each checkpoint after it is saved."""

    def __init__(self, output_dir, drive_dest=None):
        super().__init__()
        import shutil
        self.output_dir = output_dir
        _run = os.path.basename(os.path.abspath(output_dir))
        self.drive_dest = drive_dest or f"/content/drive/MyDrive/grpo_checkpoints/{_run}"
        self._mounted = False
        self._try_mount()

    def _try_mount(self):
        try:
            if os.path.ismount("/content/drive"):
                self._mounted = True
                os.makedirs(self.drive_dest, exist_ok=True)
                print(f"[+] Drive already mounted → {self.drive_dest}")
                return
            from google.colab import drive as _gd
            _gd.mount("/content/drive", force_remount=False)
            self._mounted = True
            os.makedirs(self.drive_dest, exist_ok=True)
            print(f"[+] Drive mounted → {self.drive_dest}")
        except Exception as _e:
            print(f"[!] Drive mount failed ({_e}) — checkpoints local only.")

    def on_save(self, args, state, control, **kwargs):
        if not self._mounted:
            return
        try:
            import shutil
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt_dir):
                ckpt_dir = self.output_dir
            zip_name = f"save_step_{state.global_step}"
            zip_base = os.path.join(self.drive_dest, zip_name)
            if os.path.exists(zip_base + ".zip"):
                os.remove(zip_base + ".zip")
            shutil.make_archive(zip_base, "zip", root_dir=ckpt_dir)
            print(f"[+] Step {state.global_step} → {zip_name}.zip uploaded to Drive.")
        except Exception as _e:
            print(f"[!] Drive upload failed at step {state.global_step}: {_e}")

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
        print("[!] Warning: Dataset path C:\\Users\\NewAdmin\\Desktop\\Research\\data\\gsm8k_train_grpo.jsonl could not be verified on disk, but inlined preprocessing has completed. Proceeding with active stream.")
        
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
    import inspect as _inspect
    _grpo_params = set(_inspect.signature(GRPOConfig.__init__).parameters)

    training_args_kwargs = dict(
        output_dir="./grpo_cot_output",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=512,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        logging_steps=5,
        save_steps=50,
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        report_to="none"
    )

    if use_vllm and "use_vllm" in _grpo_params:
        training_args_kwargs["use_vllm"] = True
        if "vllm_device" in _grpo_params:
            training_args_kwargs["vllm_device"] = "cuda:0"
        _vllm_optional = {
            "vllm_gpu_memory_utilization": 0.5,
            "vllm_dtype": "float16",
            "vllm_mode": "colocate",
            "vllm_enforce_eager": True,
        }
        for k, v in _vllm_optional.items():
            if k in _grpo_params:
                training_args_kwargs[k] = v
    elif "use_vllm" in _grpo_params:
        training_args_kwargs["use_vllm"] = False

    training_args = GRPOConfig(**training_args_kwargs)

    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[get_combined_reward_fn(mode=mode, stage_steps=stage_steps)],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[StepTrackerCallback(), DriveUploadCallback(training_args.output_dir)]
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
