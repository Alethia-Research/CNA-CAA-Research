# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Alethia Research** — Contrastive Neuron Attribution (CNA) for sparse behavioral circuit steering in LLMs without fine-tuning. Two tracks:
1. **CNA/CAA steering** — discover and steer safety/sycophancy/factual circuits via MLP activation scaling
2. **GRPO reasoning training** — train small models (1.5B–3B) with custom reward shaping on GSM-8K

All compute runs on **Google Colab Free/Pro (T4)**. No local GPU assumed.

## Commands

### Tests

```bash
# Run reward function unit tests (no dependencies beyond src/rewards.py)
python Tests/test_rewards.py

# Run with pytest if installed
pytest Tests/test_rewards.py -v
```

Tests import from `src/` via `sys.path` injection — no install step needed.

### CNA Steering Experiments

```bash
python src/cna_steering_experiment.py --help
python src/advanced_steering_suite.py --help
python src/calibrate_blacklist.py --help
python src/cross_model_blacklist_test.py --help
```

### GRPO Training

**All-in-one pipeline** (Colab-optimized, unsloth backend):
```bash
python src/all_in_one_grpo.py \
  --model_name unsloth/Qwen2.5-3B-Instruct \
  --mode step-grpo \
  --max_steps 150 \
  --stage_steps 50 \
  --layers_to_transform last_4   # or last_8, last_2, or "0,1,2,3"
```

**Modular trainer** (separate reward module):
```bash
python src/train_grpo.py \
  --model_name unsloth/Qwen2.5-1.5B-Instruct \
  --mode p-grpo \
  --layers_to_transform last_8
```

**GRPO modes:** `standard` | `p-grpo` | `step-grpo`  
**`--layers_to_transform`:** presets `last_2/last_4/last_8` or comma-separated layer indices; remaining layers are frozen and gradient-insulation-verified at step 0.

## Architecture

### CNA Track (`src/`)

CNA contrasts MLP activations between 5 positive and 5 negative prompts, selects the top ~200 contrastive neurons as a circuit, then scales those activations at inference by multiplier `m`:
- `m=0.0` — ablate (bypass safety, truth-serum sycophancy)
- `m=1.0` — baseline
- `m=2.0` — amplify

**Key files:**
- `src/cna_steering_experiment.py` — main cross-model sweep; outputs paper-ready comparison tables
- `src/calibrate_blacklist.py` — variance-based blacklist calibration to filter polyfunctional neurons
- `data/blacklists/` — JSON neuron index blacklists per model scale (transfer across Qwen/Llama)
- `data/circuits/` — discovered circuit indices (e.g., `safety_1.5b.json`)

Blacklists from 1.5B transfer predictably to 7B. Safety circuits consistently peak at **96–97% model depth** across architectures.

### GRPO Track (`src/rewards.py`, `src/train_grpo.py`, `src/all_in_one_grpo.py`)

**Reward system** (`src/rewards.py`) — all reward functions share the same signature `(prompts, completions, **kwargs) -> list[float]`:

| Function | Description |
|---|---|
| `format_reward_fn` | Validates `<think>...</think>` structure. Single-block = 1.0, multi-block = 0.3 (exploit prevention), unclosed = 0.2, none = 0.0 |
| `math_correctness_reward_fn` | Exact match on extracted number vs target |
| `p_grpo_format_reward_fn` | Posterior-GRPO: zeroes format reward if answer is wrong (prevents rewarding "looking smart") |
| `step_grpo_reward_fn` | Decays reward by `0.99^steps` where steps = count of transition tokens (`wait`, `hmm`, `actually`, etc.) globally; applies 0.5/extra-block penalty |

**Two-stage training** (`get_combined_reward_fn`): Stage 1 (steps 0–`stage_steps`) weights format heavily, Stage 2 shifts weight to correctness. Stage boundary controlled by `StepTrackerCallback`.

**`StepTrackerCallback.on_substep_end`** — fires after the first backward pass at step 0 to verify gradient insulation: raises `RuntimeError` if any frozen parameter receives non-zero gradients.

**`all_in_one_grpo.py` vs `train_grpo.py`:** `all_in_one_grpo.py` is the self-contained Colab pipeline (includes data prep, vLLM colocate, dynamic layer resolution inline). `train_grpo.py` is the modular version that imports from `src/rewards.py`. `src/phase4/` mirrors this structure for Phase 4 experiments.

### Data Flow

```
prepare_grpo_data.py → data/gsm8k_train_grpo.jsonl → train_grpo.py / all_in_one_grpo.py
```

Each JSONL line: `{"prompt": [...messages...], "target_answer": "42"}`.  
`extract_xml_answer()` parses model output: strips after `</think>`, prefers `\boxed{N}`, falls back to last number. Comma-stripped before comparison.

## Key Constraints

- **No `requirements.txt`** — dependencies (`unsloth`, `trl`, `transformers`, `datasets`) must be installed manually in Colab
- `unsloth` must be imported before other HF libs in training scripts
- `scratch/` is git-ignored — ad-hoc analysis scripts live there
- Paper artifacts (`findings/paper_draft.*`) are LaTeX — edit `findings/paper_draft.tex`, recompile for PDF
