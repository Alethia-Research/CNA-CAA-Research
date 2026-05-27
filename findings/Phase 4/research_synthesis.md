# The Periphery Alignment Paradigm: A Unified Mechanistic Synthesis of LLM Steering, Fine-Tuning, and Test-Time Compute

**Alethia Research Group**  
*May 27, 2026*

---

## Abstract
We synthesize the empirical findings across four phases of mechanistic interpretability research: Contrastive Neuron Attribution (CNA) scaling laws, universal activation-variance blacklists, factual context-repair, Layer-Frozen Safety Fine-Tuning (LFSFT), and Group Relative Policy Optimization (GRPO) cognitive monologue training. We propose a unified theory of transformer alignment: **Periphery Alignment & Central Logic**. Under this paradigm, early and middle layers act as a dense, invariant "central logic engine" encoding raw factual networks, logical operators, and syntactical infrastructure, while the final 10–15% of layers serve as a "periphery alignment filter" controlling behavioral routing, safety thresholds, and output formatting. We discuss how this explains capability-preservation dynamics in LFSFT, the transferability of variance blacklists, the categorical routing of factual steering, and the dramatic capability restoration observed when GRPO unlocks test-time compute via monologue scratchpads.

---

## 1. The Periphery Alignment & Central Logic Theory

A core question in modern AI alignment is: *How and where does a language model store its capabilities versus its alignment behaviors?*

Our consolidated findings suggest a clear structural division within transformer LLMs:

```
               [Transformer Layers L0 — L28]
  +------------------------------------------------------+
  | L00 - L23: Central Logic Engine (85% Depth)           |
  | - Factual Knowledge Graphs (ROME/MEMIT)               |
  | - Mathematical & Coding Rules (GSM8K/HumanEval)      |
  | - Polyfunctional Infrastructure (54% Conserved)       |
  +------------------------------------------------------+
                             |
                             v
  +------------------------------------------------------+
  | L24 - L27: Periphery Alignment Filter (15% Depth)    |
  | - Safety Refusal Circuits (Peaks at 96-97% Depth)    |
  | - Sycophancy Elicitation Circuits                    |
  | - Conversational Formatting (ChatML, System Prompts) |
  +------------------------------------------------------+
```

1. **The Central Logic Engine (L0 to $\approx$ 85% Depth):**
   This region forms the foundational computer of the model. It contains the core factual representations, mathematical operations, code syntax rules, and the general capability matrix. It is highly dense, tightly coupled, and invariant.
2. **The Periphery Alignment Filter (Last 10–15% of Layers):**
   This final section acts as a semantic gateway or behavioral router. Rather than storing core knowledge, it determines *how* that knowledge is routed and presented (e.g., whether to refuse a query, whether to agree sycophantically with a user, or how to format code outputs).

This structural division explains the key phenomena observed across our research phases.

---

## 2. CNA Scaling & LFSFT: Proving the Periphery Filter

Our scaling sweeps and Layer-Frozen Safety Fine-Tuning (LFSFT) experiments provide direct causal proof of this periphery filter:

* **Universal Late-Layer Localization:** Contrastive Neuron Attribution (CNA) localized safety refusal circuits to a sharp peak at **96–97% model depth** across Qwen2.5 (1.5B, 3B, 7B) and Phi-3 (3.8B). This localization is model-agnostic and scale-invariant.
* **LFSFT Capability Preservation:** By freezing layers L0–L23 and updating only L24–L27 during safety tuning, we preserved mathematical reasoning capabilities (**62.0% GSM-8K accuracy** compared to **58.0%** for full-parameter SFT). Because the central logic engine was frozen, it was completely protected from the destructive gradient noise of safety tuning.
* **The Formatting Trade-off (HumanEval):** LFSFT failed to improve coding scores on HumanEval (scoring 32.5% vs. 47.5% for full SFT). This is because code block generation is a format-heavy instruction capability. To format code blocks correctly for automatic parsers, the model requires formatting adaptations in earlier layers. Freezing L0–L23 prevented these formatting updates from propagating, demonstrating that formatting is a complex central-to-periphery capability whereas safety is a pure late-layer filter.

---

## 3. Pretraining Invariance: The Conserved Infrastructure

Our activation-variance blacklist experiments reveal that the model's core infrastructure is formed during pretraining and remains structurally unmodified by alignment:

* **Universal Blacklists:** By analyzing neuron activation variance across diverse prompts, we identified "infrastructure" neurons that fire consistently regardless of semantic content. These neurons are heavily concentrated in the final layer (**71% in Layer 27**), demonstrating that the periphery layer is highly congested with polyfunctional nodes.
* **Base-to-Instruct Conservation:** We observed a **54% Jaccard overlap** between the blacklist of Qwen2.5-1.5B (Base) and Qwen2.5-1.5B-Instruct. Given the millions of parameters, this high overlap proves that core routing pathways are established during pre-training.
* **Alignment as Final Routing:** Post-pretraining alignment (SFT/RLHF) does not rebuild the model's infrastructure. Instead, it alters the final routing of tokens in the late-layer periphery, leaving the base infrastructure conserved.

## 4. Semantic Categories vs. Token Strings: Context-Repair vs. Target Substitution

When steering factual recall using signed logit-diff attribution, we documented two distinct behavioral outcomes:

* **Semantic Context-Repair:** When steered away from a fact (e.g., Germany $\rightarrow$ *Bonn*), the model does not output nonsense. Instead, it rewrites the context to make the steered token factually correct (*"Bonn served as the capital of West Germany..."*).
* **Direct Target Substitution:** When steered to high-proximity sibling tokens (e.g., Japan $\rightarrow$ *Seoul*), the model confidently asserts a direct factual falsehood (*"The capital of Japan is Seoul"*) without attempting to rewrite the context.

**Model-Specific Divergence:** Crucially, this direct target substitution behavior (such as outputting *"The capital of Japan is Seoul"*) was **specifically observed on Phi-3-mini**. On the same prompt, Qwen-1.5B successfully performed context-repair by shifting the semantic frame to Taiwan and outputting Taipei.

### The Causal Competition Model
These outcomes represent a competition between the **steered local circuit** and the model's **global semantic coherence constraints**:
* **Context-Repair (Balanced Intervention):** The model attempts to resolve the factual contradiction by shifting to a nearby semantic frame where the steered token fits factually.
* **Target Substitution (Dominant Intervention):** When the steered circuit has a clean, high-dimensional projection directly to a high-proximity token within the same category (like `Tokyo` $\rightarrow$ `Seoul` under `capital cities`) and overrides the model's global coherence pathways, it directly outputs the steered falsehood.

This proves that factual circuits operate on categorical groups, but can be forced into direct target substitution under high steering strengths (as seen in Phi-3-mini).

---

## 5. Test-Time Compute: GRPO & Monologue Scaling

Our Phase 4 GRPO training logs and test-set evaluations show how unlocking test-time compute interacts with the Central Logic Engine:

### The "Ablation vs. Expansion" Duality
There is a beautiful duality between LFSFT (Layer-Frozen training) and GRPO (test-time compute training):
* **LFSFT** protects capabilities by *limiting weight updates* (freezing the central engine).
* **GRPO** unlocks capabilities by *expanding token sequence space* (generating a `<think>` monologue).

### Mechanistic Role of the Monologue
Why does letting the model think inside `<think>...</think>` tags restore its performance on logical tasks (e.g., Example 2's bolt calculation, where it scored 100% correct in zero-shot with tags, but failed in few-shot without tags)?
1. **Parallel-to-Sequential Conversion:** Standard forward passes try to map the question vector to the correct answer token in a single, parallel step-wise pass through the layers. For complex logic, this single step runs out of representational capacity, leading to memory hallucinations (e.g., converting bolts to yards).
2. **Monologue as State Memory:** The `<think>` monologue converts this parallel computation into a sequential path. The intermediate tokens generated inside `<think>` serve as externalized residual stream updates. The model updates its active state memory with each generated token, allowing the Central Logic Engine to perform simple, sequential steps (calculating white fiber first, then adding blue and white, then outputting the sum).
3. **Step-GRPO Decay Efficiency:** Step-GRPO's decaying reward penalty ($\gamma^{\text{steps}}$) acts as a regularization constraint, training the model to find the most compact sequential path rather than inflating the monologue with redundant tokens.

---

## 6. Strategic Conclusion: Auditable Democratic Alignment

This unified theory supports a new paradigm for open-source AI safety:

* **Auditable Open-Source Weights:** Since safety and behavior circuits are highly sparse and localized (e.g., 200 neurons in a 1.5B model), developers can ship a `safety_circuit.json` along with the model weights. Third-party auditors can verify alignment integrity directly by testing the causal effect of these specified neurons, preventing silent jailbreaks or backdoor modifications.
* **Low-Compute Alignment:** Since LFSFT restricts training backpropagation to the late-layer periphery, safety alignment can be achieved on consumer hardware (T4 GPUs) in a few hours. This democratizes alignment, allowing diverse communities to safely fine-tune base models without degrading their core reasoning engines.
* **Sequential Training Pipelines:** Future models should follow a sequential pipeline: (1) Full SFT on early/middle layers to build instruction-following and capability formats, followed by (2) Layer-Frozen safety alignment and Step-GRPO monologue tuning on the late-layer periphery to enforce behavior gates without capability degradation.
