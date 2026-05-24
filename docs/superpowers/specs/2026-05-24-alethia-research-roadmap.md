# Alethia Research — Master Roadmap
**Last updated:** 2026-05-24  
**Mission:** Publish high-quality, zero-fluff research on building smaller, smarter models under severe compute constraints.  
**End goal:** One publishable research paper + one open-source model release.  
**Compute:** Google Colab Free/Pro (T4/A100). All phases designed to run within this constraint.

---

## Status Overview

| Phase | Name | Status | Target |
|---|---|---|---|
| 1 | Contrastive Neuron Attribution (CNA) — Core | DONE | Qwen2.5-1.5B-Instruct |
| 2 | CNA Scale-Up + Cross-Model Analysis | NEXT | 7B, 72B, Llama vs Qwen |
| 3 | Universal Blacklist Optimization | PLANNED | Model-agnostic transfer |
| 4 | GRPO Reasoning Training | PLANNED | 1.5B–3B reasoning model |
| 5 | Model Merging / FrankenMoE | PLANNED | CPU-native capability stacking |
| B | Sub-Ternary Quantization | STRETCH | Edge/mobile deployment |

---

## Phase 1 — Contrastive Neuron Attribution: Core Experiments

**Status: COMPLETE**  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Framework:** `NousResearch/neural-steering`

### Validated Experiments

**Safety Refusal Steering (Deep Symmetric Contrastive)**
- Circuit: 200 neurons, layers L15–L27, concentrated at L27 (84 neurons)
- Ablation (`m=0.0`): Refusal bypassed cleanly — full lockpicking instructions generated
- Amplification (`m=2.0`): Safety hardened — more assertive refusals
- Quality at all multipliers: high (no degradation)

**Sycophancy Steering (Deep Symmetric Contrastive)**
- Circuit: ~200 neurons, layers L17–L27, concentrated at L27 (89 neurons)
- Ablation (`m=0.0`): "Truth serum" — model directly corrected 5G misinformation without diplomatic evasion
- Amplification (`m=2.0`): Increased deflection/boilerplate
- Key insight: model *knew* the correct answer but sycophancy circuit suppressed it

**Factual Logit-Diff Steering (RelP Attribution)**
- Shanghai vs Paris: circuit discovered, steering functional at `m=1.0`, `m=2.0`
- Delhi vs Paris with `m=-1.5`: **OPEN ISSUE** — output confused ("I'm sorry but for any confusion..."). Negative multiplier factual steering needs fixing.

**CAA Comparison Baseline**
- CAA at `m=-3.0`: complete garbage output (`.navigate whenignKey...` repeated tokens)
- Confirms CNA >> CAA in quality preservation

**Universal Blacklist Calibration**
- Custom blacklist built: `custom_blacklist.json`
- Filters polyfunctional neurons that contaminate circuits

### Key Findings from Phase 1
1. Safety subnetwork hypothesis confirmed: safety = sparse ~200 MLP neurons, not globally distributed
2. Zero-compute vulnerability: safety bypassed by zeroing activations at inference, no fine-tuning needed
3. Sycophancy = learned suppression circuit: ablation reveals latent truthfulness
4. Base vs instruct circuit overlap: only 3–10% neuron overlap between matched base/instruct pairs → alignment hooks into pre-existing passive classifiers and rewires constituent neurons

---

## Phase 2 — CNA Scale-Up and Cross-Architecture Analysis

**Status: NEXT**  
**Priority: HIGH — core paper content**

### 2a. Scale Verification on Larger Models

**Goal:** Confirm circuit sparsity holds at 7B and 72B scale. Establish scaling law for circuit size.

**Models to test (Colab-feasible):**
- `Qwen2.5-7B-Instruct` — T4 with bfloat16 (fits in ~16GB)
- `Llama-3.1-8B-Instruct` — same tier
- `Qwen2.5-72B-Instruct` — A100 required, 4-bit quantized via bitsandbytes
- `Llama-3.1-70B-Instruct` — A100 required, 4-bit quantized

**Metrics to capture:**
- Circuit neuron count at each scale (does it stay ~0.1% of total MLP?)
- Layer concentration pattern (does L27-equivalent concentration hold?)
- Refusal reduction % vs quality score (replicate Table 2 format)
- Attribution time per scale

**Expected finding:** Sparsity scales — larger models develop more concentrated circuits. Validate or refute this.

### 2b. Factual Logit-Diff Steering — Fix and Extend

**Goal:** Fix the negative multiplier factual steering issue, then generalize.

**Root cause of Delhi/Paris failure:** RelP attribution uses target token logit as anchor. Multi-token targets (Delhi = [16532, 6023]) use last token only — this may select a weak or ambiguous attribution anchor. At `m=-1.5` the circuit is suppressed too hard, causing incoherence rather than factual replacement.

**Fix strategy:**
1. Use first-token of multi-token target for attribution (not last)
2. Test moderate negative multipliers: `-0.5`, `-0.8`, `-1.0` before `-1.5`
3. Try single-token counterfactual pairs first (e.g., "Rome" vs "Paris" — both single tokens)
4. Add output coherence check: if quality score drops below 0.90, reduce multiplier

**Extended experiments after fix:**
- Capital facts: 10+ country/capital pairs
- Date facts: year-based factual knowledge
- Named entity substitution: person → wrong person
- Measure: what fraction of factual circuits are model-agnostic vs. entity-specific?

### 2c. Cross-Architecture Circuit Comparison

**Goal:** Do Qwen and Llama use the same layers for refusal? Are behavioral circuits transferable?

**Method:**
1. Run identical contrastive prompt sets on matched Qwen and Llama models
2. Record (layer_index, neuron_index) for each circuit
3. Normalize layer index by total depth (relative layer position)
4. Measure: overlap %, layer concentration pattern, attribution magnitude distribution

**Hypotheses to test:**
- H1: Both architectures concentrate refusal in the final 15% of layers
- H2: Neuron-level overlap is near-zero, but layer-relative position is consistent
- H3: Steering a circuit discovered on Qwen degrades Llama (zero transfer at neuron level)

**Why this matters for the paper:** Cross-architecture circuit analysis is unexplored. If H1+H2 are confirmed: late-layer concentration is an architecture-independent property of alignment fine-tuning. Strong contribution.

---

## Phase 3 — Universal Blacklist Optimization

**Status: PLANNED**  
**Priority: HIGH — solves open research problem, paper-ready**

### The Problem

Current blacklist (`custom_blacklist.json`) is model-specific. Polyfunctional neurons (neurons serving multiple circuits simultaneously) contaminate steering by introducing side effects — spelling errors, syntactic rigidity, capability degradation.

**Open question:** Can we build a model-agnostic blacklist that transfers across architectures?

### Approach

**Step 1 — Characterize polyfunctional neurons**
- Run CNA on multiple distinct behavioral circuits on the same model (safety, sycophancy, factual, sentiment)
- Find neuron intersection: neurons appearing in 3+ circuits = "universal" polyfunctional neurons
- These are the primary blacklist candidates

**Step 2 — Cross-model transfer test**
- Build blacklist on Qwen2.5-7B
- Apply that blacklist during CNA discovery on Llama-3.1-8B
- Measure: does quality score improve? Does refusal reduction hold?
- Metric: MMLU score pre/post steering with and without cross-model blacklist

**Step 3 — Automated blacklist generation heuristic**
- Hypothesis: polyfunctional neurons have high attribution score variance across multiple unrelated circuit discoveries
- Build algorithm: run N diverse circuit discoveries, flag neurons with variance > threshold
- This requires only forward passes — zero-compute heuristic

**Step 4 — Validate on held-out behaviors**
- Test blacklist on a behavioral circuit not seen during blacklist construction
- Measures generalization

**Paper contribution:** First model-agnostic behavioral blacklist. Enables CNA to be used out-of-the-box without per-model calibration.

---

## Phase 4 — GRPO Reasoning Training

**Status: PLANNED**  
**Priority: MEDIUM — independent track, separate paper potential**

### Goal

Train a small model (1.5B–3B parameters) to reason on a Colab budget. Target: "Making Small Models Reason on a Colab Budget."

### Stack
- **Framework:** Unsloth + TRL (GRPO trainer)
- **Base model:** `Qwen2.5-1.5B-Instruct` or `Qwen2.5-3B-Instruct`
- **Reward framework:** RULER (Relative Universal LLM-Elicited Rewards) via ART
- **VRAM:** 7–15GB — T4 viable, A100 preferred for larger models

### Reward Function Design (Core Bottleneck)

**Track A — Verifiable math/code (easiest, most reliable):**
- Reward: test suite pass/fail + format compliance
- Anti-reward-hacking filters: no timing manipulation, no test modification, restricted imports
- Verifier stack: syntax check → environment isolation → execution correctness

**Track B — RULER (no labeled data):**
- Use a local judge model (Qwen2.5-1.5B or similar) to rank N completions
- Ranking is more reliable than absolute scoring
- Enables reasoning training without hand-crafted reward functions

### Training Plan
1. Reproduce DeepSeek-R1-style chain-of-thought format on small model
2. Establish MMLU, GSM8K, HumanEval baseline
3. Train with GRPO on math reasoning (GSM8K + MATH subsets)
4. Evaluate: does model show emergent reasoning or just format mimicry?
5. Reward hacking audit: check for test-manipulation, library outsourcing patterns

### Open Research Questions for Phase 4
- Minimum model size for emergent chain-of-thought?
- Does RULER outperform hand-crafted rewards on small models?
- Can reward hacking be detected and mitigated with lightweight verifiers?

---

## Phase 5 — Model Merging and FrankenMoE

**Status: PLANNED**  
**Priority: MEDIUM — CPU-native, can run in parallel with GPU phases**

### Goal

CPU-bound mergekit pipelines to stack specialized capabilities into a single model. Explore FrankenMoE router initialization heuristics.

### Compute Profile

Runs entirely on CPU + system RAM. No GPU needed. Can be run on standard laptop or Colab CPU runtime. Compatible with phases running simultaneously on GPU.

### Merging Experiments

**Experiment 1 — Capability stacking via TIES/DARE**
- Merge: coding model (e.g., `Qwen2.5-Coder-1.5B`) + math model + instruction model
- Algorithm: TIES (sign consensus) or DARE (sparse dropout) via mergekit
- Evaluate: HumanEval, GSM8K, MT-Bench before/after merge
- Goal: merged model beats each individual model on combined benchmark

**Experiment 2 — DELLA vs TIES comparison**
- Same merge target, compare algorithms
- Metric: task interference score, capability retention per domain

**Experiment 3 — FrankenMoE Construction**
- Donor base: `Qwen2.5-1.5B-Instruct` (self-attention + LayerNorm)
- Expert MLPs: 2–4 specialized models
- Router initialization: compare Cheap Embed vs Hidden State Representation
- Evaluate: routing entropy, capability utilization, benchmark scores

### Open Research Question: Router Initialization Heuristics

**Problem:** FrankenMoE gating networks lack pre-trained parameters. Poor initialization → routing collapse (all tokens routed to one expert).

**Research target:** Automated, low-compute router initialization that transfers across model families.

**Approach:**
1. Collect hidden state distributions from donor model on diverse prompt sets (code, math, creative, factual)
2. Cluster hidden states into N groups (N = number of experts)
3. Initialize gate weights from cluster centroids
4. Measure routing entropy over first 1000 inference steps

**Paper contribution:** First systematic comparison of FrankenMoE router initialization methods with a proposed automated heuristic.

---

## Bonus Phase — Sub-Ternary Quantization

**Status: STRETCH GOAL**  
**Priority: LOW (high effort, lower direct contribution vs other phases)**

### Goal

Explore BitNet b1.58 and BTC-LLM for extreme compression. Enable Alethia models to run on consumer CPUs.

### Key Techniques
- **BitNet b1.58:** Weights ∈ {-1, 0, +1}. Replaces FP multiplications with integer additions. ~1.58 bits/param.
- **BTC-LLM:** Sub-1-bit PTQ via binary codebook + learnable linear transform. 0.5–1.0 bits/param. No retraining.

### Feasibility on Colab
- BTC-LLM compression: runs on CPU, single-pass
- Inference validation: bitnet.cpp or MLX
- MMLU accuracy pre/post compression to measure degradation

### When to pursue
After Phase 2+3 paper is submitted. Low-hanging fruit if a model from Phase 4/5 is already trained and ready to compress.

---

## Publication Strategy

### Primary Paper — CNA at Scale

**Target:** arXiv preprint + submission to a workshop (ICLR, NeurIPS mechanistic interpretability track, or similar)

**Title candidate:** *"Sparse Behavioral Circuits Scale: Contrastive Neuron Attribution Across Model Families and Sizes"*

**Core contributions:**
1. Phase 1 results: safety + sycophancy + factual circuit discovery on Qwen2.5-1.5B
2. Phase 2a: Circuit sparsity scaling law across 1.5B → 72B
3. Phase 2c: Cross-architecture circuit comparison (Qwen vs Llama)
4. Phase 3: Model-agnostic blacklist construction and transfer

**Sections:**
- Introduction + motivation (frugal compute angle)
- Background: CNA vs CAA vs SAE (Table 2 already written)
- Circuit discovery methodology (LRP rules, contrastive pairs)
- Scale experiments (Phase 2a results)
- Cross-architecture analysis (Phase 2c results)
- Blacklist optimization (Phase 3 results)
- Discussion: safety implications, alignment implications
- Code release + model release

### Secondary Paper — GRPO on Colab Budget (Phase 4)

**Title candidate:** *"Making Small Models Reason on a Colab Budget: GRPO with RULER Rewards"*

**Independent track** — can be written/submitted separately.

### Open-Source Releases

- CNA steering scripts: already partially done (`cna_steering_experiment.py`, `advanced_steering_suite.py`)
- Pre-computed circuit files for each tested model (saves others from rediscovering)
- Blacklist JSON files per model family
- FrankenMoE merge configs (mergekit YAML)
- Any trained GRPO model → HuggingFace Hub

### Zero-Fluff Policy

Every claim backed by experiment. No speculation without empirical grounding. Code released alongside paper.

---

## Open Research Questions

These are the genuinely unsolved problems within this project:

| # | Question | Phase | Difficulty |
|---|---|---|---|
| 1 | Does circuit sparsity (~0.1% MLP) hold at 70B+ scale? | 2a | Medium |
| 2 | Are refusal circuits at the same relative layer depth across architectures? | 2c | Medium |
| 3 | Can a model-agnostic blacklist be built purely from activation variance? | 3 | Hard |
| 4 | Does cross-model blacklist transfer improve CNA quality? | 3 | Medium |
| 5 | What is the minimum model size for emergent CoT via GRPO? | 4 | Hard |
| 6 | Can RULER match hand-crafted rewards on small models? | 4 | Medium |
| 7 | Does Cheap Embed vs Hidden State router init affect routing collapse rate? | 5 | Medium |
| 8 | What is the accuracy floor for sub-1-bit PTQ on 1.5B models? | B | Low |

---

## Immediate Next Steps (Phase 2 Kickoff)

**Week 1 — Fix Phase 1 gap:**
1. Fix factual steering negative multiplier issue (single-token pairs, softer multipliers)
2. Run factual steering on 10+ country/capital pairs, record results
3. Clean up `advanced_steering_suite.py` for reproducibility

**Week 2 — Scale to 7B:**
1. Run CNA safety + sycophancy on `Qwen2.5-7B-Instruct` on A100 Colab
2. Run same on `Llama-3.1-8B-Instruct`
3. Record circuit size, layer distribution, steering effectiveness

**Week 3 — Cross-model comparison:**
1. Extract (layer, neuron) indices from Qwen vs Llama circuits
2. Compute normalized overlap
3. Plot layer concentration histograms for both families

**Week 4 — Blacklist work:**
1. Run 4+ distinct circuit discoveries on same model
2. Identify neuron intersection (polyfunctional candidates)
3. Test whether excluding these improves clean steering

---

## Timeline Estimate

| Milestone | Estimated Duration | Compute |
|---|---|---|
| Fix factual steering + clean Phase 1 | 1 week | T4 |
| Phase 2a: 7B-scale experiments | 2 weeks | T4 + A100 |
| Phase 2a: 72B experiments | 1 week | A100 (4-bit) |
| Phase 2c: Cross-model comparison | 1 week | T4 |
| Phase 3: Blacklist construction + transfer | 2 weeks | T4 |
| Paper writing (primary CNA paper) | 2 weeks | — |
| Phase 4: GRPO experiments | 3–4 weeks | T4/A100 |
| Phase 5: Merging experiments | 2 weeks | CPU |
| **Total to first paper submission** | **~9–10 weeks** | — |

---

## Appendix: Hardware Reference for Colab

| Model | Quant | VRAM Required | Colab Tier |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | bfloat16 | ~4GB | T4 (Free) |
| Qwen2.5-7B-Instruct | bfloat16 | ~16GB | T4 (Pro) |
| Llama-3.1-8B-Instruct | bfloat16 | ~16GB | T4 (Pro) |
| Qwen2.5-72B-Instruct | 4-bit | ~40GB | A100 (Pro+) |
| Llama-3.1-70B-Instruct | 4-bit | ~40GB | A100 (Pro+) |
| GRPO training (3B) | QLoRA | ~12GB | T4 (Pro) |
