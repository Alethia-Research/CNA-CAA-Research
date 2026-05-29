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

from unsloth import FastLanguageModel # Must be imported first!
import os
import argparse
from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# Import our custom reward functions
from rewards import (
    format_reward_fn,
    math_correctness_reward_fn,
    p_grpo_format_reward_fn,
    step_grpo_reward_fn
)

# Global tracker for training step
current_step = 0

class StepTrackerCallback(TrainerCallback):
    """
    HF Trainer Callback to track the active global step count,
    enabling dynamic stage weight scaling, and verify gradient insulation.
    """
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

def get_combined_reward_fn(mode="standard", stage_steps=100, format_weight_stage1=1.0, format_weight_stage2=0.2, correct_weight_stage1=0.1, correct_weight_stage2=1.0):
    """
    Combines formatting and correctness rewards into a single reward function.
    Implements two-stage reward scaling based on the training step tracker.
    Monitors and reports emergent syntactic tag mutations in real-time during training.
    """
    def reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
        global current_step
        import re
        from collections import Counter
        from rewards import get_completion_text
        
        # Monitor emergent tag mutations
        invented_tags = []
        for comp in completions:
            comp_text = get_completion_text(comp)
            tags = re.findall(r"</?(\w+)>", comp_text)
            for t in tags:
                if t.lower() not in ["think", "boxed"]:
                    invented_tags.append(t)
                    
        if invented_tags:
            print(f"\n" + "="*50)
            print(f"[!] EMERGENT TAG MUTATION DETECTED (Step {current_step})")
            print(f"    Unique invented tags: {sorted(set(invented_tags))}")
            print(f"    Occurrences: {Counter(invented_tags).most_common()}")
            print("="*50 + "\n")
        
        # 1. Calculate formatting reward
        if mode == "p-grpo":
            # P-GRPO zeroes out formatting rewards on incorrect trajectories
            f_rewards = p_grpo_format_reward_fn(prompts, completions, target_answer)
        else:
            f_rewards = format_reward_fn(prompts, completions)
            
        # 2. Calculate correctness reward
        if mode == "step-grpo":
            # Step-GRPO applies exponential decay to correct trajectories
            c_rewards = step_grpo_reward_fn(prompts, completions, target_answer)
        else:
            c_rewards = math_correctness_reward_fn(prompts, completions, target_answer)
            
        # 3. Two-stage scaling weights
        if current_step <= stage_steps:
            w_format = format_weight_stage1
            w_correct = correct_weight_stage1
        else:
            w_format = format_weight_stage2
            w_correct = correct_weight_stage2
            
        combined = []
        for f, c in zip(f_rewards, c_rewards):
            combined.append(w_format * f + w_correct * c)
            
        return combined
    return reward_fn

def main():
    parser = argparse.ArgumentParser(description="Unsloth GRPO Reasoning Training Script")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B-Instruct", help="Base model name or path")
    parser.add_argument("--dataset_path", type=str, default="data/gsm8k_train_grpo.jsonl", help="Path to prepared training JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./grpo_cot_output", help="Output directory for checkpoints")
    parser.add_argument("--mode", type=str, choices=["standard", "p-grpo", "step-grpo"], default="standard", help="GRPO optimization variant")
    
    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of training steps")
    parser.add_argument("--stage_steps", type=int, default=50, help="Number of steps for Stage 1 (format-priming)")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt (Group size)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_prompt_len", type=int, default=512, help="Max prompt token length")
    parser.add_argument("--max_completion_len", type=int, default=384, help="Max completion token length")
    
    # Hardware/Optimizer settings
    parser.add_argument("--no_vllm", action="store_true", help="Disable vLLM colocate mode (required if running on CPU or unsupported hardware)")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--gpu_utilization", type=float, default=0.95, help="vLLM GPU memory utilization fraction")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint saving frequency in steps")
    parser.add_argument("--layers_to_transform", type=str, default=None, help="Preset (last_4, last_8, last_2) or comma-separated layer indices to adapt via LoRA (rest are frozen)")
    
    args, _ = parser.parse_known_args()
    
    print(f"[*] Starting GRPO training in mode: {args.mode.upper()}")
    print(f"[*] Model target: {args.model_name}")
    print(f"[*] Dataset: {args.dataset_path}")
    print(f"[*] Output directory: {args.output_dir}")
    
    # 1. Set environment properties
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_vllm = not args.no_vllm and device == "cuda"
    print(f"[+] Device detected: {device.upper()}. Using vLLM: {use_vllm}")
    
    # 2. Load Model & Tokenizer using Unsloth FastLanguageModel
    print("[*] Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_prompt_len + args.max_completion_len,
        load_in_4bit=True,
        fast_inference=use_vllm,
        gpu_memory_utilization=args.gpu_utilization
    )
    
    # Resolve layers to transform dynamically
    layers_to_transform = None
    if args.layers_to_transform:
        preset = args.layers_to_transform.strip().lower()
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
                layers_to_transform = [int(x.strip()) for x in args.layers_to_transform.split(",") if x.strip()]
                print(f"[+] Custom layer configuration resolved to layers: {layers_to_transform}")
            except ValueError:
                print(f"[-] Invalid format for --layers_to_transform '{args.layers_to_transform}'. Adapting all layers.")
                layers_to_transform = None

    # Apply PEFT LoRA adapters
    print("[*] Applying PEFT LoRA adapters...")
    peft_kwargs = {
        "model": model,
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "use_gradient_checkpointing": "unsloth"
    }
    if layers_to_transform is not None:
        peft_kwargs["layers_to_transform"] = layers_to_transform
        
    model = FastLanguageModel.get_peft_model(**peft_kwargs)
    
    # 3. Load dataset
    print(f"[*] Loading training dataset from {args.dataset_path}...")
    try:
        # Load local JSONL dataset
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    except Exception as e:
        print(f"[-] Error loading dataset: {e}")
        return
        
    # 4. Prepare reward function
    combined_reward = get_combined_reward_fn(
        mode=args.mode,
        stage_steps=args.stage_steps
    )
    
    # 5. Configure GRPOTrainer arguments
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        max_steps=args.max_steps,
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
        logging_steps=5,
        save_steps=args.save_steps,
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        report_to="none"
    )
    
    # 6. Instantiate GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[combined_reward],
        args=training_args,
        train_dataset=dataset,
        callbacks=[StepTrackerCallback()]
    )
    
    # 7. Start Training
    print("[*] Launching training loop...")
    trainer.train()
    
    # 8. Save final LoRA adapters
    final_save_path = os.path.join(args.output_dir, "final_lora")
    print(f"[*] Saving final LoRA adapters to {final_save_path}...")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("[+] Training pipeline execution finished successfully!")

if __name__ == "__main__":
    main()
