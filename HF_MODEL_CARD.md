---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- text-generation
- peft
- lora
- trl
- unsloth
- grpo
- reinforcement-learning
- alignment-tax
- math-reasoning
- chain-of-thought
---

# Qwen-1.5B-LFGRPO-OPTIM

This repository hosts the LoRA adapter weights for **Qwen-1.5B-LFGRPO-OPTIM**, a low-compute, alignment-optimized reasoning model. The model is trained using **Layer-Frozen Group Relative Policy Optimization (LF-GRPO)**, a novel alignment paradigm designed to mitigate the "alignment tax" in small language models.

* **Developed by:** Kriday Dave (Alethia Research Group)
* **Model type:** Causal Language Model with PEFT/LoRA Adapter
* **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Quantized in 4-bit)
* **Language(s):** English
* **License:** Apache-2.0
* **Repository:** [Alethia-Research GitHub](https://github.com/Alethia-Research)
* **Paper:** *Making Small Models Reason on a Colab Budget: Layer-Frozen Group Relative Policy Optimization*

---

## Model Description

Traditional reinforcement learning alignment (like standard GRPO) backpropagates formatting and correctness gradients across all layers of a language model. In smaller models (1.5B to 3B parameters), this triggers **Central Engine Disruption**—the destructive corruption of core mathematical and logical representations in early and middle layers ($L0$--$L23$).

**LF-GRPO** solves this by strictly freezing the model's central logic core ($L0$--$L23$) and confining parameter updates to the late-layer behavioral periphery ($L24$--$L27$). This allows the model to learn complex reasoning layout boundaries (such as step-by-step `<think>` tag monologues) without corrupting its underlying arithmetic capability.

### Functional Behaviors:
* **Structured Thinking:** The model breaks down word problems step-by-step using logical numbering arrays.
* **Conciseness Penalization:** Through step-decay relative rewards, the model maintains a short, high-density reasoning path, preventing verbosity drift.
* **Intact Core Arithmetic:** Avoids the standard post-alignment reasoning decay, preserving raw calculation precision.

---

## How to Get Started with the Model

You can load this adapter on top of the base Qwen-1.5B model using `peft` and `transformers`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the LF-GRPO adapter
model = PeftModel.from_pretrained(base_model, "kridaydave/Qwen-1.5B-LFGRPO-OPTIM")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

model.eval()

# Prompt format (Zero-Shot CoT with system guidance)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant must think step-by-step "
    "inside <think>...</think> tags to solve the mathematical problem, and then provide "
    "the final numeric answer outside the tags."
)

prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nJanet has 16 eggs. She eats 3 for breakfast and bakes muffins with 4. She sells the rest for $2 each. How much does she make?<|im_end|>\n<|im_start|>assistant\n<think>\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:]))
```

---

## Training Details

### Training Data
The model was trained on a 1,000-sample subset of the **OpenAI GSM8K** dataset, optimized specifically for step-by-step math logic.

### Training Procedure
* **Regime:** Two-stage optimization. Stage 1 (steps 0-100) focuses on format-priming and monologue tag alignment. Stage 2 (steps 101-300) optimizes for final math correctness and conciseness.
* **Group Relative Search:** Group size ($N=4$) is used to compute advantages relative to the group mean and standard deviation, bypassing the memory-heavy critic model.
* **Autograd Periphery Insulation:** Hard gradient masking applied at layer 24. 100% of parameters in layers 0-23 were kept frozen.

### Training Hyperparameters
* **LoRA Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **LoRA Rank / Alpha:** 32 / 32
* **Targeted Layers:** `[24, 25, 26, 27]`
* **Trainable Parameters:** 5,275,648 (0.34% of base model)
* **Optimizer:** `paged_adamw_8bit` (with CUDA page offloading)
* **Learning Rate:** 1.5e-5
* **Batch Configuration:** Batch=1, Accumulation=4 (effective batch size = 4)
* **Sequence Limits:** Prompt=512, Completion=384

---

## Evaluation Results

Evaluated on the OpenAI GSM8K test split (held-out prompts) under a zero-shot ChatML reasoning format:

* **Qwen2.5-1.5B-Instruct (Base Baseline):** ~42.0% - 50.0%
* **Standard GRPO (Full-Layer LoRA):** ~42.0% (degraded due to alignment tax / engine disruption)
* **LF-GRPO (This Work - Step 100):** ~50.0%
* **LF-GRPO (This Work - Step 200/300):** ~58.0% - 65.0% OOD accuracy (highly structured, concise CoT)

---

## Environmental Impact

* **Hardware Type:** 1 x Tesla T4 GPU (16GB VRAM)
* **Hours used:** ~2.0 hours
* **Cloud Provider:** Google Colab
* **Compute Region:** `us-central1`

---

## Technical Specifications

### Model Architecture
The underlying architecture is based on **Qwen2.5** (RoPE embeddings, SwiGLU gating, and RMSNorm layers) using a 28-layer parameter layout.

### Software
* **TRL** (Transformer Reinforcement Learning)
* **Unsloth** (Fast language model training & Triton kernels)
* **vLLM** (Fast CUDA graph decoders for advantage rollouts)

---

## Citation

```bibtex
@article{dave2026lfgrpo,
  title={Making Small Models Reason on a Colab Budget: Layer-Frozen Group Relative Policy Optimization},
  author={Dave, Kriday},
  journal={Alethia Research Group Technical Manuscript},
  year={2026}
}
```
