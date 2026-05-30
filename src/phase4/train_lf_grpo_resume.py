# =====================================================================
# LF-GRPO LOOPHOLE-FREE RETRAINING PIPELINE
# Designed for Google Colab Tesla T4 & Single-GPU Environments
# Alethia Research — Periphery Alignment & Central Logic
# =====================================================================

import os
import re
import sys
import json
import shutil
import subprocess
from types import ModuleType
from importlib.machinery import ModuleSpec

# ── Patch: sitecustomize.py global subprocess patch ──────────────────────────
_SITECUSTOMIZE_CONTENT = '''\
try:
    import torch as _torch
    class _StubDtype:
        __slots__ = ("_name",)
        def __init__(self, name):
            self._name = name
        def __repr__(self):
            return self._name
        def __hash__(self):
            return hash(self._name)
        def __eq__(self, other):
            return isinstance(other, _StubDtype) and self._name == other._name
        def __ne__(self, other):
            return not self.__eq__(other)
        is_floating_point = False
    for _pfx in ("int", "uint"):
        for _i in range(1, 8):
            _a = f"{_pfx}{_i}"
            if not hasattr(_torch, _a):
                setattr(_torch, _a, _StubDtype(f"torch.{_a}"))
except Exception:
    pass
'''
try:
    import site
    for d in site.getsitepackages():
        try:
            with open(os.path.join(d, "sitecustomize.py"), "w", encoding="utf-8") as f:
                f.write(_SITECUSTOMIZE_CONTENT)
            break
        except Exception:
            continue
except Exception:
    pass

import torch

# ── Dynamic torchvision and system mocks ─────────────────────────────────────
from unittest.mock import MagicMock
class TorchvisionMockFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("torchvision"):
            return ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        mock = MagicMock()
        mock.__name__ = spec.name
        mock.__path__ = []
        return mock
    def exec_module(self, module): pass
sys.meta_path.insert(0, TorchvisionMockFinder())

# --- Core Dependencies Check and Auto-installer ---
try:
    import unsloth
    import datasets
    import transformers
    import peft
    import bitsandbytes
    import accelerate
    import trl
    print("[+] Core machine learning environments detected.")
except ImportError:
    print("[*] Core libraries missing. Starting automated setup using uv...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-q", "uv"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "datasets", "accelerate", "peft", "bitsandbytes", "transformers", "tqdm", "trl"], check=True)
    print("[+] Setup complete. Running imports...")

# Load Unsloth
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

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

# =====================================================================
# 1. CORE REWARD FUNCTIONS WITH LOOPHOLE FIXES
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
    s1_clean = s1.replace(",", "").strip()
    s2_clean = s2.replace(",", "").strip()
    try:
        return float(s1_clean) == float(s2_clean)
    except ValueError:
        return s1_clean.lower() == s2_clean.lower()

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
        elif num_start == 0 or num_end == 0:
            if num_start == 0 and num_end == 0:
                rewards.append(0.0)
            else:
                rewards.append(0.2)
        else:
            rewards.append(0.3)
    return rewards

def think_depth_reward_fn(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
        if not think_blocks:
            rewards.append(0.0)
            continue
        lines = [l.strip() for l in think_blocks[0].split('\n') if l.strip()]
        rewards.append(min(1.0, len(lines) / 6.0))
    return rewards

def quantity_inventory_reward_fn(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
        if not think_blocks:
            rewards.append(0.0)
            continue
        think_text = think_blocks[0]
        numbers_found = re.findall(r'\d+', think_text)
        has_explicit_steps = bool(re.search(
            r'(total|sum|remaining|result|equals|=)\s*[:=]?\s*\d',
            think_text.lower()
        ))
        has_multiple_quantities = len(set(numbers_found)) >= 4
        score = 0.0
        if has_multiple_quantities: score += 0.5
        if has_explicit_steps: score += 0.5
        rewards.append(score)
    return rewards

def final_format_reward_fn(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        if re.search(r'####\s*-?\d+(?:\.\d+)?', comp_text):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# ── Tag-Spam Penalty (Electrified Fence) ───────────────────────────────────
_TAG_WHITELIST = frozenset(["think", "boxed"])

def tag_spam_penalty_fn(prompts, completions, **kwargs) -> list[float]:
    penalties = []
    tag_pattern = re.compile(r"</?([a-zA-Z][a-zA-Z0-9_]*)>")
    digit_tag_pattern = re.compile(r"</?\d+>")

    for comp in completions:
        comp_text = get_completion_text(comp)
        bad_tag_types = set()

        for match in tag_pattern.finditer(comp_text):
            tag_name = match.group(1).lower()
            if tag_name not in _TAG_WHITELIST:
                bad_tag_types.add(tag_name)

        if digit_tag_pattern.search(comp_text):
            bad_tag_types.add("__digit__")

        if bad_tag_types:
            penalties.append(max(-1.5, -0.3 * len(bad_tag_types)))
        else:
            penalties.append(0.0)
    return penalties

# =====================================================================
# 2. UNIFIED LOOPHOLE-FREE REWARD GATING & LENGTH DECAY
# =====================================================================

current_step = 0

def get_combined_reward_fn():
    """
    Combines formatting and correctness rewards under a gated alignment logic:
    - If the mathematical answer is INCORRECT: returns exactly 0.0 (plus tag penalties).
    - If CORRECT: rewards formatting, correctness with Monologue Word-Count Decay
      (decay = 0.995 ** words), and auxiliary rewards.
    - Prevents 'looking smart while being wrong'.
    - Prevents vocabulary shifting to bypass conciseness.
    """
    def reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
        global current_step
        
        # 1. Pre-calculate standard metrics
        f_rewards = format_reward_fn(prompts, completions)
        d_rewards = think_depth_reward_fn(prompts, completions)
        i_rewards = quantity_inventory_reward_fn(prompts, completions)
        ff_rewards = final_format_reward_fn(prompts, completions)
        spam_penalties = tag_spam_penalty_fn(prompts, completions)
        
        combined = []
        for comp, target, f, d, i, ff, sp in zip(completions, target_answer, f_rewards, d_rewards, i_rewards, ff_rewards, spam_penalties):
            comp_text = get_completion_text(comp)
            extracted = extract_xml_answer(comp_text)
            target_clean = target.strip().replace(",", "")
            is_correct = extracted and is_mathematically_equivalent(extracted, target_clean)
            
            # --- GATE 1: Answer correctness check ---
            if not is_correct:
                # 0.0 total reward if incorrect (plus any tag spam penalties)
                combined.append(0.0 + sp)
                continue
                
            # --- GATE 2: Length-based Monologue Conciseness ---
            # Extract content within first <think> block
            think_blocks = re.findall(r"<think>(.*?)</think>", comp_text, re.DOTALL)
            num_start = comp_text.count("<think>")
            
            monologue_content = ""
            if think_blocks:
                monologue_content = think_blocks[0]
            elif num_start > 0:
                # Unclosed block fallback
                monologue_content = comp_text.split("<think>")[-1]
                
            # Count actual words inside thinking monologue to prevent vocabulary hacks
            words = len(monologue_content.split())
            
            # Grace window of 100 words to allow thorough reasoning on multi-step problems,
            # followed by a mild decay of 0.996 per word to suppress extreme verbosity.
            if words <= 100:
                conciseness_decay = 1.0
            else:
                conciseness_decay = 0.996 ** (words - 100)
            
            # Multi-block penalty
            block_penalty = 0.0
            if num_start > 1:
                block_penalty = 0.5 * (num_start - 1)
                
            c_reward = max(0.0, conciseness_decay - block_penalty)
            
            # Stage weights (Stage 3 correctness dominant, but since we are gated,
            # we reward compliance strictly only when the math is 100% correct)
            w_format = 0.1
            w_correct = 1.0
            w_depth = 0.2
            w_inventory = 0.1
            w_final = 0.2
            
            total_reward = (
                w_format * f +
                w_correct * c_reward +
                w_depth * d +
                w_inventory * i +
                w_final * ff +
                sp
            )
            combined.append(total_reward)
            
        return combined
    return reward_fn

# =====================================================================
# 3. TRAINING & DRIVE UPLOAD CALLBACKS
# =====================================================================

class StepTrackerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        global current_step
        current_step = state.global_step

class DriveUploadCallback(TrainerCallback):
    def __init__(self, output_dir, drive_dest=None):
        super().__init__()
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
                print(f"[+] Drive already mounted -> {self.drive_dest}")
                return
            from google.colab import drive as _gd
            _gd.mount("/content/drive", force_remount=False)
            self._mounted = True
            os.makedirs(self.drive_dest, exist_ok=True)
            print(f"[+] Drive mounted -> {self.drive_dest}")
        except Exception as _e:
            print(f"[!] Drive mount skipped ({_e}). Checkpoints local only.")

    def on_save(self, args, state, control, **kwargs):
        if not self._mounted:
            return
        try:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt_dir):
                ckpt_dir = self.output_dir
            zip_name = f"save_step_{state.global_step}"
            zip_base = os.path.join(self.drive_dest, zip_name)
            if os.path.exists(zip_base + ".zip"):
                os.remove(zip_base + ".zip")
            shutil.make_archive(zip_base, "zip", root_dir=ckpt_dir)
            print(f"[+] Checkpoint step {state.global_step} zipped & uploaded to Drive.")
        except Exception as _e:
            print(f"[!] Drive upload skipped at step {state.global_step}: {_e}")

# =====================================================================
# 4. RETRAINING EXECUTION PIPELINE
# =====================================================================

def run_retraining(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    resume_from_adapter=None,
    max_steps=100,
    learning_rate=1.5e-5,
    output_dir="./grpo_cot_resumed",
    save_steps=50,
):
    print(f"[*] Starting Loophole-Free LF-GRPO Resumed Training...")
    
    # 1. Dataset Preparation
    print("[*] Preparing GSM8K Dataset...")
    from datasets import load_dataset as hf_load_dataset
    raw_gsm = hf_load_dataset("openai/gsm8k", "main", split="train")
    
    os.makedirs("data", exist_ok=True)
    dataset_path = "data/gsm8k_train_grpo.jsonl"
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in raw_gsm:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]}
            ]
            ans_text = item["answer"]
            target_answer = ans_text.split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"prompt": prompt_messages, "target_answer": target_answer}) + "\n")
            
    print(f"[+] Loaded {len(raw_gsm)} training samples.")
    
    # Load formatted dataset
    from datasets import load_dataset
    train_dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 2. Base Model & Tokenizer Loading (with 4-bit Quantization)
    print(f"[*] Loading model {model_name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512 + 384,
        load_in_4bit=True,
        device_map="auto"
    )
    
    # 3. LoRA Setup (Selective late-layer adaptation - Frozen-Layer GRPO)
    # LFSFT confirms freezing early-mid layers L0-L23 protects central math engine.
    num_layers = model.config.num_hidden_layers
    resolved_layers = list(range(num_layers - 4, num_layers)) # Adapts L24-L27
    
    print(f"[+] Applying Frozen-Layer GRPO: Freezing L0-L{resolved_layers[0]-1}. Adapting late-layers: {resolved_layers}")
    
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=64,
        lora_alpha=96,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=resolved_layers,
        use_gradient_checkpointing="unsloth"
    )
    
    # 4. Resume Adapter weights if provided
    if resume_from_adapter:
        # Check if zipped
        if resume_from_adapter.endswith(".zip"):
            unzipped_dir = "./resume_adapter_unzipped"
            if os.path.exists(unzipped_dir):
                shutil.rmtree(unzipped_dir)
            shutil.unpack_archive(resume_from_adapter, unzipped_dir, "zip")
            resume_from_adapter = unzipped_dir
            
        print(f"[*] Attaching previous adapter weights from '{resume_from_adapter}'...")
        model.load_adapter(resume_from_adapter, adapter_name="default")
        print("[+] Checkpoint adapter attached successfully!")

    # 5. Configure GRPOConfig
    training_args = GRPOConfig(
        output_dir=output_dir,
        beta=0.0,  # Skip reference model deepcopy
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=448,
        max_completion_length=448,
        max_steps=max_steps,
        logging_steps=5,
        save_steps=save_steps,
        optim="paged_adamw_8bit",
        report_to="none",
        use_vllm=False  # Disabled for T4 VRAM safety during resumed run
    )

    # Hotfix model warnings expected by TRL
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if not hasattr(model, "_keep_in_fp32_modules"):
        model._keep_in_fp32_modules = []

    # 6. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[get_combined_reward_fn()],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[StepTrackerCallback(), DriveUploadCallback(output_dir)]
    )

    # 7. Run Resumed Training Loop
    print(f"[*] Resuming retraining from step 100 for {max_steps} steps...")
    trainer.train()
    
    # Save final loophole-free adapter
    final_save_path = os.path.join(output_dir, "final_loophole_free_lora")
    print(f"[*] Saving final loophole-free LoRA adapters to {final_save_path}...")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("[+] LF-GRPO Retraining Execution Completed Successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resume LF-GRPO training using loophole-free rewards.")
    parser.add_argument("--resume_from", type=str, default="/content/grpo_cot_output/checkpoint-100",
                        help="Path to previous checkpoint directory or zip file to resume from")
    parser.add_argument("--max_steps", type=int, default=100, help="Number of steps to train")
    parser.add_argument("--output_dir", type=str, default="./grpo_cot_resumed", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint save interval")
    
    args, _ = parser.parse_known_args()
    
    run_retraining(
        resume_from_adapter=args.resume_from,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        save_steps=args.save_steps
    )
