# AGENTS.md — Alethia Research

Quick-start for agents. Shortcuts, gotchas, architecture.

## Two Tracks

| Track | Dir | Key Files |
|-------|-----|-----------|
| **CNA/CAA steering** | `src/` | `cna_steering_experiment.py`, `advanced_steering_suite.py`, `calibrate_blacklist.py`, `generalization_7b.py` |
| **GRPO reasoning** | `src/` | `all_in_one_grpo.py` (self-contained Colab), `train_grpo.py` (modular, imports `rewards.py`), `prepare_grpo_data.py` |
| **LFSFT** | `src/` | `train_lfsft.py`, `eval_lfsft.py`, `colab_eval_lfsft.py` |
| **Phase 4** | `src/phase4/` | Mirrors GRPO structure — `all_in_one_grpo.py`, `train_grpo.py`, `rewards.py`, `prepare_grpo_data.py` |

## Commands

```bash
# Tests (sys.path injection, no install)
python Tests/test_rewards.py
pytest Tests/test_rewards.py -v

# CNA steering (all --help)
python src/cna_steering_experiment.py --model qwen-1.5b --multiplier 0.0
python src/advanced_steering_suite.py
python src/calibrate_blacklist.py

# GRPO (Colab-optimized)
python src/all_in_one_grpo.py --model_name unsloth/Qwen2.5-3B-Instruct --mode step-grpo --max_steps 150 --stage_steps 50 --layers_to_transform last_4
python src/train_grpo.py --model_name unsloth/Qwen2.5-1.5B-Instruct --mode p-grpo --layers_to_transform last_8
```

GRPO modes: `standard` | `p-grpo` | `step-grpo`
layers_to_transform: `last_2`/`last_4`/`last_8` or comma-separated indices.

## Critical Gotchas

- **No `requirements.txt`** — deps (`unsloth`, `trl`, `transformers`, `datasets`, `vllm`, `bitsandbytes`) installed manually in Colab. `all_in_one_grpo.py` has auto-installer.
- **`unsloth` must be imported before** any other HF/transformers lib.
- **`NousResearch/neural-steering`** cloned at runtime by most steering scripts (`neuron_steer` import path). Must run from a dir with git/write access.
- **Blacklists/circuits** in `data/` as JSON neuron index lists. Transfer across Qwen/Llama scales.
- **Gradient insulation verification** fires in `StepTrackerCallback.on_substep_end` at step 0 — raises `RuntimeError` if frozen params get gradients.

## Data Flow

```
prepare_grpo_data.py → data/gsm8k_train_grpo.jsonl → all_in_one_grpo.py / train_grpo.py
```

Each JSONL: `{"prompt": [...messages...], "target_answer": "42"}`.

Answer parsing (`extract_xml_answer`): strips after `</think>`, prefers `\boxed{N}`, falls back to last number. Commas stripped.

## Reward Functions (src/rewards.py)

All share signature `(prompts, completions, **kwargs) -> list[float]`.

| Function | Key Behavior |
|----------|-------------|
| `format_reward_fn` | single `<think>...</think>` = 1.0, multi-block = 0.3, unclosed = 0.2, none = 0.0 |
| `math_correctness_reward_fn` | exact match extracted number vs target |
| `p_grpo_format_reward_fn` | zeroes format reward if answer wrong |
| `step_grpo_reward_fn` | decay `0.99^steps`, penalty for extra blocks |

Two-stage training: Stage 1 (steps `0–stage_steps`) weights format heavily, Stage 2 shifts to correctness.

## Conventions

- `scratch/` gitignored — ad-hoc analysis there
- All compute designed for **Google Colab T4 16GB** — no local GPU assumed
- Paper artifacts in `findings/` — LaTeX (`paper_draft.tex`), FINDINGS.md
- Results in `data/results/` as JSON
- Safety circuits peak at **96–97% model depth** across all architectures
- CNA maintains coherence where CAA degenerates to garbage
