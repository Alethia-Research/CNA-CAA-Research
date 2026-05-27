# Google Colab Execution Guide: Phase 4 GRPO Cognitive Training

This guide details how to execute the Group Relative Policy Optimization (GRPO) training pipeline inside a single-GPU Google Colab notebook (T4 or L4 runtimes). 

---

## 1. Environment Setup

Run the following commands in Colab cells to install the optimized Unsloth and vLLM dependencies:

```bash
# 1. Install uv for rapid package installations
!pip install --upgrade -qqq uv

# 2. Install Unsloth optimized build for Colab
!uv pip install -qqq "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. Install remaining TRL, vLLM, and execution libraries
!uv pip install -qqq datasets trl transformers accelerate peft bitsandbytes vllm
```

---

## 2. Mount Google Drive (Recommended for Checkpoint Safety)

To ensure that your model checkpoints are saved and do not get lost if Colab disconnects, mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Specify `/content/drive/MyDrive/grpo_checkpoints` as your `--output_dir` in the training commands.

---

## 3. Prepare the Math Dataset

Run the data preparation script to download the GSM8K dataset from Hugging Face and format it with system prompts instructing the model to think inside `<think>...</think>` tags:

```bash
# Standard run (downloads and pre-formats full GSM8K)
!python src/prepare_grpo_data.py --output_dir ./data

# Fast run (selects a smaller subset for speed/debugging)
!python src/prepare_grpo_data.py --limit_train 1000 --limit_test 100 --output_dir ./data
```

---

## 4. Run GRPO Cognitive Training

Execute the training script using one of the three optimization strategies. 

### Option A: Standard GRPO
Standard Group Relative Policy Optimization, where formatting is rewarded independently of correctness.

```bash
!python src/train_grpo.py \
    --model_name "unsloth/Qwen2.5-3B-Instruct" \
    --dataset_path "./data/gsm8k_train_grpo.jsonl" \
    --output_dir "/content/drive/MyDrive/grpo_cot_output" \
    --mode "standard" \
    --max_steps 150 \
    --stage_steps 50 \
    --learning_rate 2e-5
```

### Option B: Posterior-GRPO (P-GRPO)
Process formatting rewards are zeroed out if the final mathematical answer is incorrect. This prevents the model from generating plausible-sounding formatting traces that fail to solve the actual math problem.

```bash
!python src/train_grpo.py \
    --model_name "unsloth/Qwen2.5-3B-Instruct" \
    --dataset_path "./data/gsm8k_train_grpo.jsonl" \
    --output_dir "/content/drive/MyDrive/grpo_cot_output" \
    --mode "p-grpo" \
    --max_steps 150 \
    --stage_steps 50 \
    --learning_rate 2e-5
```

### Option C: Step-GRPO (Decaying Step Rewards)
Applies a decaying step penalty ($\gamma = 0.99^j$) to the correctness reward based on the number of cognitive transition steps (tokens like `Wait`, `Hmm`) inside the monologue, penalizing overthinking and monologue redundancy.

```bash
!python src/train_grpo.py \
    --model_name "unsloth/Qwen2.5-3B-Instruct" \
    --dataset_path "./data/gsm8k_train_grpo.jsonl" \
    --output_dir "/content/drive/MyDrive/grpo_cot_output" \
    --mode "step-grpo" \
    --max_steps 150 \
    --stage_steps 50 \
    --learning_rate 2e-5
```

---

## 5. Critical Parameters to Tweak

* `--model_name`: Set to `"unsloth/Qwen2.5-1.5B-Instruct"` if you are running on a free-tier Tesla T4 GPU to maximize speed and prevent any OOMs.
* `--stage_steps`: The number of steps allocated for Stage 1 training (format-priming), where the model learns to wrap thoughts in `<think>` tags before optimization targets correctness.
* `--no_vllm`: If running inside an environment with limited GPU allocation or if testing locally on a CPU, append `--no_vllm` to bypass CUDA-specific vLLM memory configurations.
