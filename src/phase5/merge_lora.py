#!/usr/bin/env python3
"""
Merges a trained LoRA adapter with its base model to produce a full-parameter model.
Required because mergekit operates on full-parameter tensors rather than adapter weights.
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA Adapter with Base Model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen2.5-1.5B-Instruct",
        help="Path or HF ID of the base model."
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the directory containing the LoRA adapter (adapter_model.safetensors)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grpo_cot_output/final_merged_model",
        help="Path to save the merged full-parameter model."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.lora_path):
        print(f"[!] Error: LoRA path '{args.lora_path}' does not exist.")
        sys.exit(1)
        
    print(f"[*] Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"[*] Loading LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    
    print("[*] Merging weights in-place...")
    model = model.merge_and_unload()
    
    print(f"[*] Saving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[+] Successfully merged LoRA adapter and saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
