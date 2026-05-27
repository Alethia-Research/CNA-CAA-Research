# Phase 4 Findings: GRPO Cognitive Monologue Optimization

This folder houses the experimental results, evaluation reports, and mechanistic analysis for GRPO reasoning training on Qwen2.5-1.5B-Instruct (Tesla T4, Google Colab).

---

## Files

| File | Contents |
|---|---|
| `phase4_findings.md` | Full training report — hyperparams, stage-by-stage analysis, OOD eval results (42%), reward hacking discovery, novel tag hallucination, next-experiment table |
| `research_synthesis.md` | Unified theory — Periphery Alignment & Central Logic, how all 4 phases connect, Frozen-Layer GRPO proposal |

---

## Key Results (May 27, 2026)

| Metric | Value |
|---|---|
| Training steps | 150 |
| Training time | 72.1 min (T4) |
| Format compliance | 100% by step 49 |
| Training correctness (estimated) | 45–66% (pre-convergence, biased high) |
| **OOD GSM-8K accuracy (21/50)** | **42.00%** |
| Base model GSM-8K (5-shot, public) | ~42–45% |

**Net GRPO gain at 150 steps: approximately 0% over base.** Central engine disruption (full-layer LoRA) offset monologue benefit.

---

## Key Discoveries

### 1. Central Engine Disruption (THEORY CONFIRMATION)
GRPO LoRA modified all layers including L0–L23 (central logic engine). LFSFT modified only L24–L27. GRPO 42% vs LFSFT 62% — the 20pp gap is the measured cost of touching the central engine. Confirms Periphery Alignment theory from negative direction.

### 2. Reward Hacking: Multi-Block `<think>` Loophole
Step-GRPO penalized transition tokens inside `<think>` blocks. Model discovered: close one block, open another = step counter resets = no penalty. Classic Goodhart's Law. Fix: penalize total block count, not just intra-block transitions.

### 3. XML Schema Generalization (`<nowalkthrough>`)
Model generated an invented tag `<nowalkthrough>` not present in training data. Evidence that GRPO format training conditioned the model on abstract XML schema (`[tag][computation][/tag]`) rather than specific token strings. OR: unstable tag hallucination under distribution shift. Requires full 50-response tag analysis to distinguish.

### 4. Format Conditioning Overrides Few-Shot
Model generated `<think>` tags in few-shot eval mode (examples have no think tags). GRPO LoRA format conditioning is strong enough to override the base model's in-context format-following instinct.

---

## Proposed Next Experiments

| Priority | Experiment | Prediction |
|---|---|---|
| **CRITICAL** | Frozen-Layer GRPO (LoRA on L24–L27 only) | ~55–65% OOD — confirms Periphery Alignment + enables reasoning format training without capability degradation |
| HIGH | Base model zero-shot GSM-8K eval (50 questions) | ~35–38% — establishes true GRPO net contribution |
| HIGH | Analyze all 50 eval responses: tag diversity + multi-block frequency | Classifies reward hack rate and schema vs. hallucination |
| MEDIUM | Retrain with block-count penalty | Tests whether loophole fix improves OOD accuracy |
| MEDIUM | CNA probe on GRPO model L0–L23 math circuits vs base | Quantifies central engine disruption magnitude |
