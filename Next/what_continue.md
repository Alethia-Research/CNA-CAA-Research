# Resume Training Command

```bash
python train_lf_grpo_1000.py \
  --mode step-grpo \
  --max_steps 300 \
  --stage_steps -1 \
  --num_generations 2 \
  --save_steps 100 \
  --resume_adapter_from kridaydave/Qwen-1.5B-LFGRPO-OPTIM
```

`--stage_steps -1` = skip format-prime entirely, correctness weights from step 1.
Checkpoints auto-upload to Drive as `save_step_100.zip`, `save_step_200.zip`, `save_step_300.zip`.

---

# While Training (~1.5–2h) — Priority Order

## 1. Build Eval Harness (MOST URGENT — ~45 min)
Need this to actually report results. Without it the run is unverifiable.

- Load saved adapter: `FastLanguageModel.from_pretrained` base + `model.load_adapter("kridaydave/Qwen-1.5B-LFGRPO-OPTIM")`
- Run on GSM8K **test** split (1319 examples)
- Extract answer with `extract_xml_answer()` from `src/rewards.py`
- Report pass@1 accuracy
- Save results to `data/eval_results_step100.json`

Script to write: `src/phase4/eval_gsm8k.py`

## 2. Plot Reward Curves (~20 min)
Core paper figure. Data already exists in checkpoint dirs.

- Parse `grpo_cot_output/checkpoint-*/trainer_state.json` → `log_history`
- Plot `rewards/reward_fn` and `reward_std` vs global_step
- Mark stage transition at step 100 (vertical dashed line)
- Save to `findings/figures/reward_curve_step_grpo.png`

## 3. Queue Baseline Run in Second Colab Tab (~10 min setup)
Need comparison for paper Table 1: step-GRPO vs standard vs p-GRPO.

- Open second Colab session
- Run same command but `--mode standard --max_steps 300 --stage_steps -1`
- No adapter resume — fresh baseline from pretrained

## 4. Draft Paper §4 Experiments (~30 min)
Write the shell while numbers are missing, fill in later.

- Table structure: model | mode | pass@1 | avg_completion_len | reward@300
- §4.1: Dataset & Setup
- §4.2: Reward curve analysis (placeholder: "Figure X shows...")
- §4.3: GSM8K accuracy comparison (placeholder: "Table X shows...")
- File: `findings/paper_draft.tex` section 4
