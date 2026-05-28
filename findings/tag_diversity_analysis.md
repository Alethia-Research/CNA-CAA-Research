# Tag Diversity Analysis: GRPO Eval Outputs
**Alethia Research Group**
**Date: 2026-05-28**
**Status: Partial — requires Colab eval run to complete**

---

## Problem

The GRPO model (Qwen2.5-1.5B-Instruct + Step-GRPO LoRA, 150 steps) generated 50 outputs in a prior Colab eval session. Two documented examples show anomalous tag behavior:

1. **Multi-block `<think>`**: 4 sequential `<think>` blocks — reward hacking the step-decay penalty
2. **`<nowalkthrough>` invention**: Novel tag not present in training data

The key question: are these edge cases (format hallucination) or evidence of schema generalization?

## Framework

**Interpretation A — Schema Generalization**: Multiple distinct invented tags across multiple questions would prove GRPO trained abstract XML schema, not specific token strings.

**Interpretation B — Format Hallucination**: Single invented tag type or single occurrence = unstable grounding under distribution shift.

**Decision rule**: Count unique invented tag types across all 50 outputs.

- 0 invented tags: schema generalization not confirmed (A is false)
- 1 invented tag type (only `<nowalkthrough>`): supports B (format hallucination)
- 2+ invented tag types: supports A (schema generalization)

## Currently Known

From the 2 documented examples (out of 50):

| Example | Question | Think blocks | Invented tags |
|---------|----------|-------------|---------------|
| 1 (Janet's ducks) | Egg earnings | 1 (`<think>`) + `<nowalkthrough>` block | `<nowalkthrough>` |
| 2 (Robe bolts) | Fiber bolts | 4 `<think>` blocks | None |

**Unknown**: The remaining 48 outputs. Need to count:
- Total multi-block completions (reward hack prevalence)
- Total unique invented tag types (schema vs hallucination)
- Transition token distribution (are concise completions genuinely efficient or hacking?)

## Tooling Delivered

| File | Purpose |
|------|---------|
| `src/colab_grpo_eval_analysis.py` | Copy-paste Colab cell: loads model, runs 50-question eval, saves all outputs to JSONL, prints tag diversity stats inline |
| `src/analyze_grpo_outputs.py` | Offline analysis script: takes saved JSONL, produces full tag diversity report |

Both scripts fix the original tooling gap where `eval_gsm8k_light.py` only printed 2 examples and discarded the rest.

## Next Step

Run `colab_grpo_eval_analysis.py` in Colab with:
- `MODEL_PATH = "./grpo_cot_output/final_lora"` (GRPO model) — tags analysis
- Then `MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"` with `ZERO_SHOT = True` — baseline

Both JSONL outputs drop into `src/analyze_grpo_outputs.py` for full report.

## Predicted Outcome

Based on the LF-GRPO training log which implemented the multi-block penalty fix, the standard GRPO (without the fix) likely has 20-40% reward hack rate. The `<nowalkthrough>` tag is likely a single occurrence — format hallucination, not schema generalization. But this is speculation until we run.
