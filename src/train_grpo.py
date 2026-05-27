from unsloth import FastLanguageModel # Must be imported first!
import os
import argparse
import torch
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
    enabling dynamic stage weight scaling.
    """
    def on_step_end(self, args, state, control, **kwargs):
        global current_step
        current_step = state.global_step

def get_combined_reward_fn(mode="standard", stage_steps=100, format_weight_stage1=1.0, format_weight_stage2=0.2, correct_weight_stage1=0.1, correct_weight_stage2=1.0):
    """
    Combines formatting and correctness rewards into a single reward function.
    Implements two-stage reward scaling based on the training step tracker.
    """
    def reward_fn(prompts, completions, target_answer, **kwargs) -> list[float]:
        global current_step
        
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
    
    # Apply PEFT LoRA adapters
    print("[*] Applying PEFT LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth"
    )
    
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
