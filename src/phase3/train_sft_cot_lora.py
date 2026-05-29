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

import os
from unsloth import FastLanguageModel
import argparse
import inspect
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def train_sft_cot(model_name="unsloth/Qwen2.5-1.5B-Instruct", dataset_path="data/gsm8k_sft_cot.jsonl", epochs=1, batch_size=2, lr=2e-4):
    print(f"[*] Starting SFT CoT Format Warm-Start")
    print(f"[*] Base Model: {model_name}")
    print(f"[*] Dataset: {dataset_path}")
    
    # 1. Load Model and Tokenizer in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512 + 384,
        load_in_4bit=True,
    )
    
    # Resolve layers to transform (freeze L0-L23, train L24-L27)
    num_layers = model.config.num_hidden_layers
    layers_to_transform = list(range(num_layers - 4, num_layers))
    print(f"[+] Restricting LoRA adapters to late-layer periphery: {layers_to_transform}")
    
    # 2. Apply PEFT LoRA only to the targeted late-layer periphery
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        layers_to_transform=layers_to_transform,
    )
    
    # 3. Load dataset
    if not os.path.exists(dataset_path):
        print(f"[*] SFT dataset missing. Generating it now...")
        from prepare_sft_cot_data import prepare_sft_cot_data
        prepare_sft_cot_data(limit_samples=1000)
        
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 4. SFT Training Arguments
    training_args = TrainingArguments(
        output_dir="./sft_cot_output",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none"
    )
    
    # 5. Initialize SFTTrainer (Self-Healing version-agnostic parameters)
    sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "dataset_text_field": "text",
        "max_seq_length": 512 + 384,
        "dataset_num_proc": 2,
        "packing": False, # Verify token lengths remain discrete
        "args": training_args,
    }
    
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    
    print("[*] Launching SFT format training...")
    trainer.train()
    
    final_save_path = "./sft_cot_output/final_lora"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"[+] SFT warm-start complete! LoRA adapters saved at {final_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-Frozen SFT-CoT Warm-Start")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-1.5B-Instruct", help="Base model ID")
    parser.add_argument("--dataset_path", type=str, default="data/gsm8k_sft_cot.jsonl", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_sft_cot(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
