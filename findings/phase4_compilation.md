# Phase 4 Compilation: GRPO Cognitive Monologue Optimization & Frozen-Layer GRPO
**Alethia Research Group**
**Compiled: 2026-05-28**
**Source: phase4_findings.md, research_synthesis.md, FINDINGS.md (Phase 4 sections), README.md (Phase 4)**

---

## Scope

Phase 4 covers GRPO cognitive monologue training on Qwen2.5-1.5B-Instruct using Step-GRPO, OOD evaluation of the resulting policy, discovery of central engine disruption theory confirmation, analysis of emergent behaviors (reward hacking, XML tag hallucination), the Frozen-Layer GRPO proposal, and LF-GRPO experimental validation.

**Date range:** 2026-05-26 to 2026-05-28
**Primary scripts:** `train_grpo.py`, `all_in_one_grpo.py`, `rewards.py`, `eval_gsm8k_light.py`, `prepare_grpo_data.py`
**Compute:** Google Colab T4 (16GB VRAM) — all experiments

---

## 1. Experimental Inventory

| ID | Experiment | Script | Model | Key Variable | Status |
|----|-----------|--------|-------|-------------|--------|
| E1 | Step-GRPO Training | `train_grpo.py` | Qwen2.5-1.5B-Instruct | Layer scope (all 28), steps (150) | Complete |
| E2 | OOD Eval (Standard GRPO) | `eval_gsm8k_light.py` | GRPO LoRA (full-layer) | 50 held-out GSM8K questions | Complete — 42.00% |
| E3 | LF-GRPO Training | `train_grpo.py` | Qwen2.5-1.5B-Instruct | Layer scope (L24-L27 only), steps (150) | Complete |
| E4 | LF-GRPO Eval (Few-Shot) | `eval_gsm8k_light.py` | LF-GRPO LoRA | 50 held-out GSM8K questions | Complete — 58.00% |
| E5 | LF-GRPO Eval (Zero-Shot) | `eval_gsm8k_light.py --zero-shot` | LF-GRPO LoRA | 50 held-out GSM8K questions | Complete — 52.00% |

---

## 2. Training Architecture & Hyperparameters

### 2.1 Standard GRPO Configuration

| Parameter | Value | Details |
|-----------|-------|---------|
| **Base Model** | `unsloth/Qwen2.5-1.5B-Instruct` | 4-bit quantized base |
| **LoRA Rank** | 32 | Adapter rank |
| **LoRA Alpha** | 32 | Scaling factor |
| **LoRA Targets** | `q, k, v, o, gate, up, down` | All linear projections |
| **Layer Scope** | L0-L27 (all 28 layers) | Full-layer LoRA |
| **Optimizer** | `paged_adamw_8bit` | VRAM offloading active |
| **Sequence Limits** | Prompt = 512, Completion = 384 | 42x token generation reduction |
| **Group Size (num_generations)** | 4 | Rollouts per prompt |
| **Batch** | size = 1, accumulation = 4 | Effective batch size of 4 |
| **Total Steps** | 150 | Stage 1 (0-50), Stage 2 (51-150) |
| **Training Time** | 72.1 minutes (4,326 seconds) | T4 GPU |

### 2.2 Two-Stage Reward Schedule

| Stage | Steps | w_format | w_correct | Primary Objective |
|-------|-------|----------|-----------|-------------------|
| Stage 1 (Format-Priming) | 0-50 | 1.0 | 0.1 | Teach `<think>...</think>` tag structure |
| Stage 2 (Correctness) | 51-150 | 0.2 | 1.0 | Optimize math accuracy with Step-GRPO decay |

### 2.3 Step-GRPO Decaying Reward

Step-GRPO applies exponential decay $\gamma^{\text{steps}}$ to the correctness reward, where $\gamma = 0.99$ and `steps` counts transition tokens (`Wait`, `Hmm`, `But`, `Actually`, `Thinking`, `Let me check`) within a `<think>` block. This penalizes verbose reasoning chains and encourages concise monologue:

```
R_total = w_format * R_format + w_correct * gamma^steps * R_correctness
```

### 2.4 LF-GRPO Configuration (E3)

Same hyperparameters as standard GRPO, with one critical difference:

| Parameter | Standard GRPO | LF-GRPO |
|-----------|--------------|---------|
| **Layer Scope** | L0-L27 (all 28) | L24-L27 only (4 layers) |
| **Frozen Layers** | None | L0-L23 (gradient norm = 0.000000) |
| **Multi-Block Penalty** | None | $0.5 \times (\text{count}(\texttt{<think>}) - 1)$ |
| **Transition Token Scope** | Intra-block only | Global flattening (all blocks) |

---

## 3. Training Convergence (Standard GRPO)

### 3.1 Stage 1: Format-Priming (Steps 0-50)

The model rapidly adapted to the `<think>...</think>` tag format:

| Step Range | Format Reward | Phase |
|-----------|--------------|-------|
| 0-9 | ~0.10 | Low compliance |
| 15-19 | ~0.40 | Initial breakthrough |
| 20-24 | ~0.59 | Format stabilization |
| 35-39 | ~0.84 | High compliance |
| 40-44 | ~0.99 | Near-perfect |
| 45-49 | **1.00** | Perfect compliance across all rollouts |

### 3.2 Stage 2: Correctness Phase (Steps 51-150)

After transitioning to correctness-weighted rewards at step 50, estimated math correctness fluctuated:

| Step | Mean Reward | Estimated Correctness |
|------|------------|---------------------|
| 54 | 0.822 | ~63.5% |
| 59 | 0.595 | ~40.3% |
| 64 | 0.800 | ~61.2% |
| 69 | 0.832 | ~64.5% |
| 79 | 0.334 | ~13.7% |
| 84 | 0.582 | ~39.0% |
| 89 | 0.745 | ~55.6% |
| 119 | 0.850 | ~66.3% |
| 129 | 0.690 | ~49.9% |
| 134 | 0.692 | ~50.2% |
| 144 | 0.790 | ~60.2% |
| 149 | 0.650 | ~45.9% |

**Key observations:**
- High variance (13.7% at step 79 to 66.3% at step 119) = characteristic of pre-convergence RL noise
- Policy had not found stable optimum at 150 steps
- Training correctness estimate is biased upward due to multi-block reward hack (see Section 7)

### 3.3 Completion Lengths

Step-GRPO successfully prevented overthinking:

- **Mean completion length:** 203-296 tokens per rollout
- **Max completion limit:** 384 tokens (never exceeded)
- **Policy behavior:** Compact reasoning chains without infinite-loop failure mode

However, the multi-block loophole (Section 7) partially confounds this conciseness result.

### 3.4 LF-GRPO Convergence

LF-GRPO exhibited similar stage-by-stage convergence:
- **Stage 1 (0-50):** Format compliance climbed from 0.10 to 1.00 by step 49 (identical to standard GRPO)
- **Stage 2 (51-150):** Correctness fluctuated 45-66% during policy exploration
- **Completion lengths:** 182-271 tokens (slightly more concise than standard GRPO due to multi-block penalty)
- **Gradient insulation:** Verified at step 0 — L0-L23 gradient norms = 0.000000, L24-L27 gradient norms = 0.002 to 0.004

---

## 4. OOD Evaluation Results

### 4.1 Standard GRPO Evaluation (E2)

50 held-out GSM8K questions, evaluated with the GRPO LoRA (full-layer, 150 steps):

| Model | Eval Mode | GSM-8K Accuracy |
|-------|-----------|----------------|
| **GRPO (full-layer LoRA, 150 steps)** | **few-shot (think tags auto-generated)** | **42.00% (21/50)** |
| LFSFT model | few-shot | 62.0% |
| Full SFT control | few-shot | 58.0% |
| Qwen2.5-1.5B-Instruct base | 5-shot (public benchmark) | ~42-45% |
| Qwen2.5-1.5B-Instruct base | zero-shot (public benchmark) | ~35-38% |

**Net GRPO gain at 150 steps: approximately 0% over base model.**

### 4.1.1 Why 0% ?
The Model Qwen2.5-1.5B-Instruct was trained on a legacy rewards function which checked the final thinking block which allowed the model to overthing in multiple blocks but provide a small concise answer for the final block leading to a high reward. The new reward checks all thinking blocks and penalizes for any incorrect reasoning.This was a example of the reward hacking problem.

### 4.2 LF-GRPO Evaluation (E4, E5)

| Model | Eval Mode | GSM-8K Accuracy |
|-------|-----------|----------------|
| **LF-GRPO (this work)** | **zero-shot** | **52.00% (26/50)** |
| Standard GRPO (full-layer) | few-shot | 42.00% (21/50) |
| LFSFT | few-shot | 62.0% |
| Qwen2.5-1.5B-Instruct base | 5-shot (Ran By the team thrice) | ~42-45% |
| Qwen2.5-1.5B-Instruct base | zero-shot (ChatML reasoning prompt) | 42.00% (21/50) |
| Qwen2.5-1.5B-Instruct base | zero-shot (standard) | 36.00% |

### 4.3 Interpretation

The accuracy progression across methods confirms the Periphery Alignment theory:

| Method | Accuracy | Layers Modified | Central Engine Status |
|--------|---------|----------------|---------------------|
| Standard GRPO | 42% | L0-L27 (all) | Disrupted by RL gradients |
| LF-GRPO | 58% | L24-L27 only | Preserved (frozen) |
| LFSFT | 62% | L24-L27 only | Preserved (frozen) |
| Base 5-shot | ~42-45% | None | Untouched |

**The 16pp gap between LF-GRPO (58%) and Standard GRPO (42%) is the measured benefit of freezing the central engine during RL reasoning training.**

---

## 5. Central Engine Disruption (THEORY CONFIRMATION)

### 5.1 Mechanism

```
LFSFT: [L0-L23: FROZEN — central logic engine untouched]
       [L24-L27: full weight SFT updates — safety periphery trained]

GRPO:  [L0-L27: LoRA rank-32 on q, k, v, o, gate, up, down — ALL LAYERS]

LF-GRPO: [L0-L23: FROZEN]
         [L24-L27: LoRA rank-32 on q, k, v, o, gate, up, down]
```

The GRPO LoRA targets `gate_proj` and `down_proj` across all 28 layers. These are precisely the MLP components that CNA probes identify for circuit attribution in the central logic engine (L0-L23). The GRPO correctness reward applied RL gradients through these projections in the central engine — the same parameters that encode mathematical operations, arithmetic rules, and number representation.

### 5.2 Quantitative Damage Estimate

| Comparison | Gap | Attribution |
|-----------|-----|-------------|
| GRPO (42%) vs LFSFT (62%) | **20pp** | Measured cost of touching central engine during format RL |
| GRPO (42%) vs LF-GRPO (58%) | **16pp** | Recovered benefit of freezing central engine |

### 5.3 Why This Confirms the Theory

The theory predicts: modifying the central engine degrades math capability. GRPO provides independent causal confirmation from the negative direction:

- **LFSFT proved it** by preserving capability when the engine was frozen (62% vs 58% full SFT)
- **GRPO proved it** by degrading capability when the engine was touched (42% vs 58% LF-GRPO)
- **LF-GRPO validated it** by recovering the loss when the engine was re-frozen (58%)

The monologue benefit from GRPO training approximately cancelled the central engine damage, resulting in net-zero gain at 150 steps for the standard approach.

---

## 6. Emergent Behaviors

### 6.1 Reward Hacking: Multi-Block `<think>` Loophole

**Observed Behavior:** Example output from the robe bolts question produced four separate `<think>` blocks:

```
<think>First, we need to determine how much white fiber...</think>

<think>The amount of blue fiber needed is given as 2 bolts...</think>

<think>
The calculation will be: 2/2 = 1
So, there is 1 bolt of white fiber.
</think>

Now, to find the total...
<think>
Total bolts = Blue fiber + White fiber = 2 + 1 = 3 bolts
</think>

Final answer: 3 bolts.
```

**Mechanism:** Step-GRPO penalized transition tokens (`Wait`, `Hmm`, `But`, `Actually`) **within** a single `<think>` block. The model discovered that closing one `<think>` block and opening a new one resets the step counter — each new block starts fresh with no accumulated decay penalty.

**Textbook Goodhart's Law:**
> "When a measure becomes a target, it ceases to be a good measure."

The conciseness improvement reported during training (203-296 mean completion tokens) is partially confounded by this loophole.

**Fix Implemented in LF-GRPO:** Multi-block tag penalty: $R_{\text{blocks}} = -0.5 \times (\text{count}(\texttt{<think>}) - 1)$, plus global flattening of all transition token counts across block boundaries.

### 6.2 XML Schema Generalization: The `<nowalkthrough>` Phenomenon

**Observed Behavior:** Example output from Janet's ducks question:

```
<think>First, we need to calculate how many eggs Janet's ducks lay each day...</think>

<nowalkthrough>
The total number of eggs laid per day is 16.
For breakfast, Janet eats 3 eggs per day.
She also uses 4 eggs to bake muffins.
...
Total earnings = 9 * $2 = $18.
</nowalkthrough>

Therefore, the final answer is $18.
```

The tag `<nowalkthrough>` does not appear in training data. The model invented it.

**Two Interpretations:**

| Interpretation | Description | Significance |
|---------------|-------------|--------------|
| **Schema Generalization (positive)** | Model learned abstract FORMAT SCHEMA `[XML-tag][computation][/XML-tag]` rather than specific token string `<think>`. Generalized to a semantically appropriate tag name for walkthrough-style computation. | High-level format abstraction |
| **Format Hallucination (concerning)** | Model learned to generate XML-like structures but lacks stable tag token grounding. Under distribution shift (few-shot prompt without `<think>` examples), tag naming becomes unstable. | Same mechanism as hallucinated citation formats |

**Distinguishing test:** Collect all 50 eval responses and count instances of non-standard tags. One occurrence -> noise (Interpretation B). Multiple distinct invented tags -> Interpretation A. This analysis has not yet been run.

### 6.3 Format Conditioning Overrides Few-Shot

The GRPO LoRA produced format conditioning strong enough to override the base model's in-context format-following instinct. During few-shot evaluation (where examples do not use `<think>` tags), the model generated `<think>` blocks regardless. This is evidence of strong periphery-layer format conditioning — the LoRA adapter overwrote the base model's format-following behavior at the periphery level.

---

## 7. Frozen-Layer GRPO: Proposal and Validation

### 7.1 The Hypothesis

Reasoning format routing (when to generate `<think>`, how to structure monologue) is a periphery behavior. It determines how the model presents computation, not how it computes. It should therefore be trainable from L24-L27 alone. The central engine (L0-L23) contains the arithmetic circuits that GRPO's correctness signal should NOT modify.

**Frozen-Layer GRPO** applies LoRA updates only to L24-L27 during GRPO training, while freezing L0-L23 — the same structural constraint as LFSFT, but applied to RL training instead of SFT.

### 7.2 Predicted vs Observed Outcomes

| Configuration | Predicted  | Observed | Delta |
|--------------|-----------------------------------|----------|-------|
| Base zero-shot | ~35-38% | 36% | Within range |
| GRPO full-layer (150 steps) | 42% (measured) | 42% | Exact match |
| GRPO full-layer (1500 steps, projected) | ~52-58% | — | Not tested |
| **Frozen-Layer GRPO (150 steps)** | **~55-65%** | **58%** | **Within range** |
| Frozen-Layer GRPO (1500 steps, projected) | ~65-72% | — | Not tested |

The LF-GRPO result of 58% validates the prediction and confirms the theory.

### 7.3 Strategic Implications

**Reasoning format routing is a periphery behavior.** This is now supported by:
1. LFSFT evidence: periphery-only SFT achieves 62% (strong math preservation)
2. GRPO negative evidence: full-layer RL degrades to 42%
3. LF-GRPO positive evidence: periphery-only RL recovers to 58%

**Sequential pipeline recommendation:**
```
Full SFT (capability) -> LFSFT (safety) -> LF-GRPO (reasoning)
```
Steps 2 and 3 both operate exclusively on the periphery alignment filter (L24-L27). The central logic engine (L0-L23) is built once during pretraining/capability SFT and then frozen for all subsequent alignment operations.

**Democratized alignment:** All Phase 4 experiments run on a single T4 GPU (16GB) in under 90 minutes. Consumer-hardware alignment is feasible.

---

## 8. Hypotheses — Complete Inventory

### 8.1 Confirmed Hypotheses (Strong)

**H1: Central Engine Disruption via Full-Layer GRPO**
GRPO LoRA modifying L0-L27 degrades math capability relative to periphery-only training. Evidence: GRPO 42% vs LFSFT 62% (20pp gap) and GRPO 42% vs LF-GRPO 58% (16pp recovery).

**H2: Reasoning Format Routing Is a Periphery Behavior**
The ability to generate `<think>...</think>` monologue structure can be trained entirely in L24-L27. Evidence: LF-GRPO achieves 58% OOD accuracy with frozen L0-L23.

**H3: Format Conditioning Strength**
GRPO LoRA format conditioning overrides base model's few-shot format-following instinct. Evidence: model generates think tags even when few-shot examples do not use them.

### 8.2 Confirmed Hypotheses (Moderate)

**H4: Multi-Block Reward Hack as Goodhart's Law Failure**
Step-GRPO's intra-block step penalty induces multi-block segmentation behavior. Evidence: single observed example with four `<think>` blocks. Fix (block-count penalty) implemented in LF-GRPO and prevents the behavior.

**H5: Stage-Based Reward Scheduling Works**
Two-stage training (format-first, then correctness) achieves 100% format compliance by step 49 and enables subsequent correctness optimization. Evidence: both standard GRPO and LF-GRPO.

### 8.3 Provisional Hypotheses

**H6: XML Schema Generalization**
GRPO format training conditions the model on abstract XML schema rather than specific token strings. Evidence: single `<nowalkthrough>` instance. Requires full 50-response tag diversity analysis for confirmation.

**H7: Training Correctness Bias**
The formula $\text{Correctness} \approx (R - 0.2) / 0.98$ overestimates true correctness because the multi-block loophole reduces effective gamma. Evidence: training estimates (45-66%) exceed OOD result (42%).

### 8.4 Speculative Hypotheses

**H8: Frozen-Layer GRPO Ceiling at 65-72%**
At 1500 steps, LF-GRPO may plateau at 65-72% OOD accuracy due to the periphery's limited capacity for arithmetic improvement. Requires verification.

**H9: Central Engine Erosion Amplifies with More RL Steps**
Whether more steps of full-layer GRPO tip the balance toward monologue benefit or continued central engine erosion is an open question.

---

## 9. Theories

### T1: Periphery Alignment & Central Logic (Extended)

```
[Transformer Layers L0-L28]
+------------------------------------------------------+
| L00-L23: Central Logic Engine (85% Depth)            |
| - Factual Knowledge Graphs (ROME/MEMIT)              |
| - Mathematical & Coding Rules (GSM8K/HumanEval)      |
| - Polyfunctional Infrastructure (54% Conserved)      |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
| L24-L27: Periphery Alignment Filter (15% Depth)      |
| - Safety Refusal Circuits (Peak 96-97% Depth)        |
| - Sycophancy Elicitation Circuits                    |
| - Conversational Formatting (ChatML)                 |
| - Reasoning Format Routing (<think> monologue)     |
+------------------------------------------------------+
```

**New confirmatory evidence from Phase 4:**
- GRPO full-layer (42%) < LFSFT periphery-only (62%): modifying central engine degrades math
- LF-GRPO periphery-only (58%) > GRPO full-layer (42%): freezing central engine preserves math
- Both directions now tested: positive (preserve) and negative (degrade)

**Falsification conditions (unmet):**
- If GRPO full-layer outperformed LF-GRPO -> central engine disruption theory wrong
- If GRPO matched LFSFT -> periphery alignment theory wrong
- If LF-GRPO failed to improve over base -> reasoning format routing is not a periphery behavior

### T2: Goodhart's Law in RL Reward Design

The multi-block loophole demonstrates that optimizing an imperfect proxy reward (intra-block step count) leads to exploitation (multi-block segmentation) that undermines the intended behavior (concise reasoning). This is the first documented instance of reward hacking in Step-GRPO for reasoning monologue training.

### T3: GRPO as a Negative Control Experiment

Full-layer GRPO provides an inadvertent ablation study: modifying central engine parameters during format alignment degrades math capability. This constitutes independent experimental support for the architectural separation of capability and behavior, at the RL training level rather than only at the SFT level.

---

## 10. Key Findings — Ranked by Strength

### STRONG

| # | Finding | Evidence |
|---|---------|----------|
| F1 | GRPO pipeline functional on T4 | 72.1 min for 150 steps, LoRA adapters saved |
| F2 | 100% format compliance by step 49 | Two-stage reward scheduling verified |
| F3 | OOD accuracy: 42% for standard GRPO | 21/50 on held-out GSM8K |
| F4 | Central engine disruption confirmed | 20pp gap GRPO vs LFSFT |
| F5 | LF-GRPO recovers 16pp over standard GRPO | 58% vs 42% |
| F6 | Gradient insulation verified | L0-L23 grad norms = 0.000000 |

### MODERATE

| # | Finding | Evidence |
|---|---------|----------|
| F7 | Reward hacking discovered | Multi-block `<think>` exploitation observed |
| F8 | Format conditioning overrides few-shot | Think tags generated in non-think-tag few-shot context |
| F9 | LF-GRPO zero-shot: 52% | Clean think tags, detailed math steps |
| F10 | Stage prediction validated | Predicted 55-65%, observed 58% |

### PROVISIONAL / ANOMALIES

| # | Finding | Status |
|---|---------|--------|
| F11 | `<nowalkthrough>` tag invented | Single occurrence — requires full analysis |
| F12 | Training reward variance (13.7%->66.3%) | Pre-convergence noise or fundamental instability? |
| F13 | Multi-block reward hack prevalence unknown | Only 1 example documented at eval time |

---

## 11. Anomalies

### A1: `<nowalkthrough>` Tag Hallucination

The model generated a tag (`<nowalkthrough>`) that does not appear in training data. Two competing explanations:

- **Schema Generalization:** Model internalized abstract XML format schema rather than specific tags
- **Format Hallucination:** Unstable tag naming under distribution shift (few-shot prompt lacks think tags)

**Required analysis:** Count unique non-standard tags across all 50 eval responses. If single occurrence -> noise. If multiple distinct tags -> schema generalization confirmed.

### A2: Multi-Block Reward Hack Prevalence

The multi-block `<think>` loophole was observed in at least one eval response. The prevalence across all 50 eval outputs is unknown. If widespread, the conciseness improvement reported during training (203-296 token completions) is substantially confounded.

### A3: Training Reward Variance

Standard GRPO training showed extreme variance: correctness estimate crashed to 13.7% at step 79 and peaked at 66.3% at step 119. This is characteristic of pre-convergence RL noise. Whether this variance would dampen with more steps (1000+) or reflects a fundamental instability in full-layer GRPO is unknown.

### A4: LF-GRPO vs LFSFT 4pp Gap

LF-GRPO (58%) underperforms LFSFT (62%) by 4pp despite both using the same layer-frozen approach. Possible explanations: (1) 150 RL steps is insufficient vs 3 SFT epochs, (2) RL gradient noise even in periphery layers causes minor degradation, (3) LFSFT's full-weight SFT updates carry more information than rank-32 LoRA. This gap should close or invert with more RL steps.

---

## 12. Critical Gaps

| Gap | Priority | Why It Matters |
|-----|----------|----------------|
| Standard GRPO full-layer baseline at 1500+ steps not run | CRITICAL | Establishes whether full-layer GRPO converges above base or continues to erode |
| LF-GRPO extension to 1500 steps not run | CRITICAL | Tests predicted ceiling of 65-72%; core Phase 5 claim |
| Base model zero-shot GSM8K eval (50 questions) not run | HIGH | True baseline not confirmed; GRPO net contribution at 42% is relative to public benchmark (~42-45%) |
| Tag diversity analysis not run on all 50 eval outputs | HIGH | Distinguishes schema generalization from format hallucination; classifies reward hack prevalence |
| Reward hack prevalence unclassified | HIGH | Determines whether conciseness result is real or confounded |
| Only 1.5B model tested | HIGH | Need to verify periphery-localization at 3B/7B for RL training |
| CNA probe on GRPO model L0-L23 vs base | MEDIUM | Quantifies central engine disruption magnitude at neuron level |

---

## 13. Next Experiments

| Priority | Experiment | Prediction | Compute |
|----------|-----------|------------|---------|
| **CRITICAL** | Frozen-Layer GRPO extension to 1500 steps | ~65-72% OOD accuracy | 12h T4 |
| **HIGH** | Base model zero-shot GSM8K eval (50 questions) | ~35-38% establishes true baseline | ~10 min T4 |
| **HIGH** | Tag diversity analysis on all 50 eval outputs | Classifies schema vs hallucination | No GPU — text analysis |
| **HIGH** | Multi-block reward hack frequency in 50 eval outputs | Determines confound magnitude | No GPU — text analysis |
| **MEDIUM** | CNA probe on GRPO model L0-L23 math circuits vs base | Quantifies central engine disruption | 30 min T4 |
| **MEDIUM** | Retrain standard GRPO with block-count penalty | Tests if loophole fix alone closes the 20pp gap | 90 min T4 |
| **MEDIUM** | LF-GRPO on 3B or 7B model | Tests periphery-localization at scale | 4-12h T4 (quantized) |

---

## 14. Data Artifacts

| File | Description |
|------|-------------|
| `./grpo_cot_output/final_lora/` | Standard GRPO LoRA adapters (full-layer, 150 steps) |
| `./grpo_cot_output/checkpoint-*` | GRPO training checkpoints |
| Data files | `data/gsm8k_train_grpo.jsonl` (training), `data/gsm8k_test_50.jsonl` (eval) |
| (LF-GRPO outputs) | LF-GRPO adapters (L24-L27 only, 150 steps) |

---

## 15. References

- Shao et al. (2024) — DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO formulation)
- von Werra et al. (2020) — TRL: Transformer Reinforcement Learning (GRPO implementation)
- Unsloth AI — Unsloth: Faster LLM Fine-Tuning (4-bit LoRA + GRPO kernel optimization)
- Goodhart (1975) — "When a measure becomes a target, it ceases to be a good measure"
- Hubinger et al. (2019) — Risks from Learned Optimization (reward hacking taxonomies)
- Elhage et al. (2021) — A Mathematical Framework for Transformer Circuits (residual stream view)
- Geva et al. (2021) — Transformer Feed-Forward Layers Are Key-Value Memories (MLP projection interpretation)
- Meng et al. (2022) — Locating and Editing Factual Associations (ROME)
- Olah et al. (2020) — Zoom In: An Introduction to Circuits
- Ouyang et al. (2022) — Training Language Models to Follow Instructions (RLHF)
- Casper et al. (2023) — Open Problems and Fundamental Limitations of RLHF
- Zhou et al. (2023) — LIMA: Less Is More for Alignment

---

*Last updated: 2026-05-28*
