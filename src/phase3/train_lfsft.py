import os
import argparse
import json
import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)

# Custom Dataset that loads ChatML jsonl
class ChatMLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.examples = []
        print(f"[*] Reading and pre-tokenizing dataset from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item["text"]
                tokenized = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=False
                )
                self.examples.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"]
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Custom Completion-Only Data Collator for ChatML
class CompletionOnlyDataCollator:
    def __init__(self, tokenizer, response_template="<|im_start|>assistant\n"):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        
    def __call__(self, features):
        batch = {}
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        labels = padded_input_ids.clone()
        
        for i, tokens in enumerate(padded_input_ids):
            temp_len = len(self.response_template_ids)
            found = False
            for idx in range(len(tokens) - temp_len + 1):
                if tokens[idx : idx + temp_len].tolist() == self.response_template_ids:
                    # Mask everything before and including the assistant tag
                    labels[i, : idx + temp_len] = -100
                    found = True
                    break
            if not found:
                labels[i, :] = -100
                
        batch["input_ids"] = padded_input_ids
        batch["attention_mask"] = padded_attention_mask
        batch["labels"] = labels
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="LFSFT training script")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model ID")
    parser.add_argument("--dataset_path", type=str, default="data/lfsft_dataset.jsonl", help="Dataset file path")
    parser.add_argument("--output_dir", type=str, default="data/results/qwen-1.5b-lfsft", help="Output directory")
    parser.add_argument("--control", action="store_true", help="Run standard SFT instead of LFSFT")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (defaults: LFSFT=5e-5, Control=2e-5)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (if > 0, overrides epochs)")
    args, _ = parser.parse_known_args()
    return args

def train():
    args = parse_args()
    
    # Set default LR based on mode
    if args.lr is None:
        args.lr = 2e-5 if args.control else 5e-5
        
    print(f"[*] Mode: {'CONTROL (Full SFT)' if args.control else 'LFSFT (Layer-Frozen)'}")
    print(f"[*] Model ID: {args.model_id}")
    print(f"[*] Learning rate: {args.lr}")
    print(f"[*] Effective Batch size: {args.batch_size * args.grad_accum}")
    
    # 1. Load Tokenizer & Model
    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("[*] Loading model...")
    # Load model in float16 for control run to optimize VRAM on T4 GPU
    model_dtype = torch.float16 if args.control else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Enable gradient checkpointing for VRAM efficiency
    model.gradient_checkpointing_enable()
    
    # 2. Configure Parameter Freezing
    if not args.control:
        print("[*] Freezing layers L0-L23, embed_tokens, norm, and lm_head...")
        for name, param in model.named_parameters():
            # Check if layer is part of L24-L27 (indices 24, 25, 26, 27)
            is_trainable = False
            for layer_idx in range(24, 28):
                if f"model.layers.{layer_idx}." in name:
                    is_trainable = True
                    break
            
            if is_trainable:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        print("[*] Control run: enabling gradients for all layers...")
        for param in model.parameters():
            param.requires_grad = True

    # 3. Print parameter training stats
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"[+] Total parameters: {all_params:,}")
    print(f"[+] Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # 4. Load dataset
    dataset = ChatMLDataset(args.dataset_path, tokenizer)
    collator = CompletionOnlyDataCollator(tokenizer)
    
    # 5. Create filtered Optimizer to avoid optimizer state allocation for frozen params
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    
    # Use Paged 8-bit AdamW for Control/SFT training to optimize VRAM on T4 GPU in Google Colab
    if args.control:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(trainable_params_list, lr=args.lr, weight_decay=0.01)
            print("[+] Using bitsandbytes PagedAdamW8bit optimizer for memory efficiency...")
        except ImportError:
            optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=0.01)
            print("[!] bitsandbytes not found, falling back to torch.optim.AdamW...")
    else:
        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=0.01)
    
    # Schedulers
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = (len(dataset) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * 0.03)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"[+] Total training steps: {total_steps} (warmup: {warmup_steps})")

    # 6. Trainer Setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs if args.max_steps <= 0 else -1.0,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="no",
        save_strategy="epoch" if args.max_steps <= 0 else "no",
        logging_steps=5,
        fp16=not args.control, # Enable mixed-precision only for LFSFT (Control SFT uses pure FP16 loaded model)
        dataloader_num_workers=0, # Set to 0 to prevent CPU RAM thrashing / swapping in Colab
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        optimizers=(optimizer, scheduler)
    )

    print("[*] Starting training...")
    trainer.train()
    
    print(f"[*] Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[+] Training complete!")

if __name__ == "__main__":
    train()
