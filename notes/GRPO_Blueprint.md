# Master Research and Implementation Blueprint: Group Relative Policy Optimization Cognitive Optimization for Resource-Constrained Environments

## Phase Positioning and Master Roadmap Integration

The development of high-performing, resource-efficient language models requires structured progression from static alignment to active, inference-time cognitive compute allocation. Within the master roadmap, Phase 4 represents a critical shift from static token-steering architectures toward active, self-correcting cognitive policy optimization. Prior phases established baseline models and validated sparse intervention mechanisms, but they lacked the capacity to autonomously generate structured internal deliberation traces to solve complex problems.

By utilizing Group Relative Policy Optimization (GRPO), Phase 4 introduces a framework to train compact models containing 1.5B to 3B parameters to execute detailed internal monologues without the memory-heavy overhead of traditional actor-critic networks. This phase establishes the core cognitive engine that subsequent merging and sub-ternary quantization phases will scale into edge-compatible deployment environments.

### Phase Roadmap

| Phase | Designation | Functional Focus | Implementation Environment | Status |
|-------|-------------|-----------------|---------------------------|--------|
| Phase 1 | Contrastive Neuron Attribution (CNA) — Core | Mapping sparse behavioral subnetworks and polyfunctional circuits. | Qwen2.5-1.5B-Instruct via neural-steering library. | Complete |
| Phase 2 | CNA Scale-Up & Cross-Model Analysis | Verifying behavioral circuit patterns at larger scales. | 7B to 72B parameter architectures (Qwen vs. Llama). | Complete |
| Phase 3 | Universal Blacklist Optimization | Isolating and filtering polyfunctional circuit contaminants. | Automated cross-model transfer heuristics. | Complete |
| Phase 4 | GRPO Cognitive Monologue Optimization | Eliciting structured internal monologues on micro-budgets. | 1.5B to 3B parameters via Google Colab with Unsloth and TRL. | **Planned** |
| Phase 5 | Model Merging & FrankenMoE | Stacking specialized capabilities onto CPU-bound runtimes. | CPU-native mergekit pipelines. | Planned |
| Phase B | Sub-Ternary Quantization | Extreme model compression for mobile edge nodes. | BitNet b1.58 and BTC-LLM optimization. | Stretch |

---

## Google Colab & Single-GPU Notebook Architectures

Running online reinforcement learning on a single consumer GPU or free-tier Google Colab notebook requires a highly optimized memory configuration. Traditional reinforcement learning frameworks use independent, distributed server nodes for generating responses and training. To run the entire GRPO pipeline inside a single notebook environment, the architecture is refactored into **Colocate Mode**.

### Dependencies and Environment Initialization

To configure Google Colab (such as the free Tesla T4 GPU with 15GB VRAM or an L4 GPU with 24GB VRAM) for Phase 4, the environment must be initialized with optimized compilation libraries:

```bash
# Install Unsloth, vLLM, and TRL dependencies optimized for notebook environments
!pip install --upgrade -qqq uv
!uv pip install -qqq "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!uv pip install -qqq datasets trl transformers accelerate peft bitsandbytes
```

To prevent memory fragmentation during high-throughput rollout generations, configure PyTorch's allocator flags directly inside the notebook:

```python
import os
import torch

# Enable Unsloth's Standby feature to save 30%+ VRAM during GRPO training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### Single-GPU Colocate Mode and Standby Optimizations

In Colocate Mode, the training engine and the inference engine (vLLM) run inside the exact same Python process and share the same GPU VRAM pool.

```
┌────────────────────────────────────────────────────────┐
│                Single GPU Memory Pool (VRAM)           │
│                                                        │
│  ┌───────────────────────┐  vLLM Standby Swapping      │
│  │    Training Engine    │ ◄─────────────────────────┐ │
│  │  - LoRA Policy Model  │                           │ │
│  │  - Backward Gradients │                           │ │
│  │  - Optimizer States   │                           │ │
│  └───────────────────────┘                           │ │
│                                                      │ │
│  ┌───────────────────────┐  Offloads parameters &    │ │
│  │  vLLM Rollout Engine  │  KV Cache to System RAM ──┘ │
│  │  - Shared FP8 Weights │                             │
│  │  - Active KV Cache    │                             │
│  └───────────────────────┘                             │
└────────────────────────────────────────────────────────┘
                          ▼
              ┌──────────────────────┐
              │   System RAM (Host)  │
              │  - vLLM Sleep State  │
              └──────────────────────┘
```

Normally, this causes severe memory contention. Since both frameworks must maintain copies of the model weights, they will quickly trigger an out-of-memory (OOM) error. To resolve this, Unsloth's Standby feature implements a dynamic sleep-state transition:

1. **Rollout Phase:** The GRPOTrainer pauses backpropagation. vLLM wakes up, reclaims the allocated VRAM buffer, and processes parallel completions at high throughput.
2. **Transition Phase:** Once the completions are generated, vLLM transitions to a standby state. It offloads its parameter buffers, temporary activations, and KV cache out of active GPU VRAM into system host RAM.
3. **Training Phase:** GRPOTrainer consumes the freed VRAM space to perform forward passes, compute relative advantages, and execute backward propagation updates on the active LoRA policy.
4. **Weight Sync:** After the gradients are updated, the newly computed weights are synchronized with the standby vLLM instance using direct memory reference, avoiding any disk I/O overhead.

### Minimal GRPOTrainer Notebook Configuration

Configuring the GRPOTrainer in Colocate Mode requires pointing the rollout handler to the in-process vLLM engine:

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Load model in 4-bit quantization to drastically reduce memory footprint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,          # Patches vLLM for inline execution
    gpu_memory_utilization=0.95   # Higher headroom allowed via Standby
)

# Apply PEFT LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth"
)

# Define TRL GRPO arguments for single-GPU colocate training
training_args = GRPOConfig(
    output_dir="./grpo_cot_output",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,   # Simulates larger batch sizes without OOMs
    num_generations=8,               # Group size for calculating relative advantage
    max_prompt_length=512,
    max_completion_length=1024,
    use_vllm=True,
    vllm_mode="colocate",            # Required for single-GPU execution
    logging_steps=5,
    save_steps=50,
    optim="paged_adamw_8bit"         # Extremely memory-efficient optimizer
)
```

---

## Mathematical Profiling of Memory and Resource Utilization

Running reinforcement learning on notebook GPUs like the Tesla T4 (15GB VRAM) requires precise modeling of VRAM allocation. If memory parameters are set incorrectly, the pipeline will encounter out-of-memory (OOM) errors during the backward pass.

### Algorithmic VRAM Demand Equations

The overall VRAM requirements (`M_total`) during GRPO training are defined by the accumulation of model parameter states, the pre-allocated vLLM generation cache, and peak training activation buffers:

**Model parameter memory** (`M_model`) is defined by the precision of the active policy weights (`W_active`), the frozen reference model weights (`W_ref`), and the active optimizer states (`O`). When utilizing LoRA adapters, only the adapter parameters require optimizer tracking:

```
M_model = W_active + W_ref + O_LoRA
```

For a 3B model using 4-bit quantization, the active base weights require approximately **1.5 GB** of VRAM, while the frozen reference weights occupy another **1.5 GB**. Unsloth's weight-sharing feature eliminates the "double model" memory overhead entirely, holding only a single 4-bit copy of the model in VRAM.

**vLLM pre-allocation memory** (`M_rollout`) is determined by the `gpu_memory_utilization` constant (`U`) and the total physical memory of the card (`V_total`):

```
M_rollout = U × V_total
```

**Training activation memory** (`M_activation`) scales dynamically with the active batch size (`b`), sequence length (`s`), layer count (`L`), hidden dimensionality (`h`), and the number of generated trajectories per prompt (`N`):

```
M_activation = f(b, s, L, h, N)
```

For Qwen2.5-3B-Instruct (`h = 2048`, `L = 36`) running with `b = 1`, `s = 1536`, and `N = 8` group generations:

```
M_activation ≈ calculated peak activation budget
```

Without standby memory management, the combined footprint of `M_rollout` (usually reserving ~60% of GPU capacity upfront) and `M_activation` easily exceeds the 15GB ceiling of a T4 GPU.

### Notebook-Focused Memory Optimization Mechanisms

- **Unsloth Memory Compression:** Unsloth's custom Triton kernels reduce memory usage by up to 80% compared to native Hugging Face and Flash Attention 2 implementations. On 20K context windows with 8 generations, standard frameworks require 510.8 GB of VRAM, whereas Unsloth executes the same operations using only 54.3 GB.
- **Embedding Offloading:** Setting `offload_embedding=True` during initialization offloads embedding tables and output head layers to system RAM, reducing active GPU VRAM by an additional 1.08 GB.
- **Flattened Sequence Chunking:** Rather than materializing logits for the entire sequence space at once, Unsloth introduces chunking across the sequence dimension. Controlled via `unsloth_logit_chunk_multiplier`, it flattens sequence batches into manageable sub-segments, preventing exponential memory spikes during backpropagation.
- **FP8 Activation and KV Cache:** Utilizing FP8 precision in vLLM doubles the potential throughput of the generation backend. Although a single token generation pass is roughly 10% slower due to quantization scaling, the reduced memory footprint allows twice as many sequences to run in parallel without triggering VRAM spikes.

---

## Mathematical Formulation of Relative Policy Algorithms

Traditional proximal policy optimization relies on a learned critic model to compute expected state values, consuming substantial VRAM and introducing optimization instability. GRPO resolves this by sampling a group of completions for each prompt and normalizing their rewards to calculate advantages directly.

### Standard Group Relative Policy Optimization

During standard GRPO training, the policy generates a group of `G` completions `{o_1, o_2, ..., o_G}` for each input prompt `x`. The completions are scored using a set of deterministic reward functions, and the group-relative advantage `A_i` for each completion `o_i` is computed as:

```
A_i = (r_i - mean(r)) / std(r)
```

If a completion achieves a score higher than the group average, it receives a positive advantage signal; if it performs worse, it receives a negative signal. The model parameters `θ` are updated by maximizing the clipped surrogate objective, penalized by a Kullback-Leibler (KL) divergence constraint to prevent the policy from drifting too far from the reference model:

```
L_GRPO(θ) = E[min(ρ_i · A_i, clip(ρ_i, 1-ε, 1+ε) · A_i)] - β · D_KL
```

where the probability ratio is defined as:

```
ρ_i = π_θ(o_i | x) / π_ref(o_i | x)
```

The KL divergence `D_KL` is computed using a low-variance estimator:

```
D_KL ≈ (π_θ / π_ref) - log(π_θ / π_ref) - 1
```

### Posterior Group Relative Policy Optimization

Standard GRPO assigns the same advantage value uniformly to all tokens generated within a completion. If the overall score is high, every token—including incorrect steps or redundant loops—is positively reinforced. Conversely, if a single mistake at the final step causes execution failure, correct steps are penalized. Furthermore, when all generated samples in a group achieve identical correct outcomes, the standard deviation becomes zero, causing the relative advantage to collapse and erasing the learning gradient.

**Posterior-GRPO (P-GRPO)** resolves this by conditioning step-level process rewards directly on the final task success. The advantage calculation is mean-centered but does not divide by the standard deviation, preserving the scale of the gradients even when all samples are correct:

```
A_i = r_i - mean(r)
```

The process reward is allocated conditionally: the model is only rewarded for elegant, concise internal monologues if the final code or math output executes successfully against the test suite. If the final execution fails, the process reward is zeroed out. This suppresses the model's tendency to write plausible-sounding but logically flawed explanations to exploit step-level rewards.

### Step Group Relative Policy Optimization and Decaying Rewards

To prevent the model from generating excessively long cognitive traces—a behavior known as **monologue redundancy**—Step-GRPO shifts the optimization objective from raw tokens to semantic step increments. The system monitors the output and segments the generation whenever a cognitive transition token (e.g., "Wait", "Hmm") is detected.

To train the model to stop thinking and output the final answer when it has achieved sufficient confidence, Step-GRPO uses a **decaying reward policy**. If the model terminates its internal monologue and produces a correct answer at step `j`, the reward decays exponentially based on step depth:

```
R_j = γ^j · 𝟙[output == y*]
```

where `γ ∈ (0, 1)` is the temporal decay factor and `y*` is the ground truth. This decay structure penalizes overthinking and redundant steps, forcing the model to favor concise problem-solving paths.

---

## Strategic Hypotheses for Cognitive Optimization

Executing Phase 4 under strict hardware constraints requires testing several core hypotheses regarding model size thresholds, memory tradeoffs, and reward architectures.

- **Hypothesis 1 — Minimum Scale for Cognitive Monologue Emergence:** A model size of 1.5B parameters represents the absolute lower limit for generating structured thinking monologues. Models below this threshold (e.g., 0.5B parameters) lack the parameter capacity to sustain structured `<think>...</think>` formatting under reinforcement signals and will collapse into empty tags or infinite repetitive loops when subjected to GRPO training.

- **Hypothesis 2 — Memory Swapping Latency Tradeoff:** Enabling colocated sleep-mode memory offloading (Standby) allows 3B models to be trained on a single 15GB–24GB notebook GPU, but the latency of swapping model states to and from system RAM increases training wall-clock time by roughly 15% compared to running without Standby. However, disabling Standby limits maximum sequence length capacity to less than 2K context, whereas Standby enables context scaling up to 20K+ on the same single-GPU hardware.

- **Hypothesis 3 — Multi-Objective Reward Hacking Sensitivity:** Jointly optimizing formatting alignment (`R_format`) and task correctness (`R_correct`) via scalar summation causes the model to exploit formatting rules at the expense of accuracy. Applying Multi-Objective GRPO (MO-GRPO) with automatic variance reweighting or Posterior-GRPO (P-GRPO) prevents this divergence, ensuring both formatting and accuracy are optimized.

- **Hypothesis 4 — Self-Supervised Step Guidance via Entropy Progress:** Integrating Entropy-Progress Aligned GRPO (EP-GRPO) allows the model to leverage its own internal information flow—using token entropy to locate critical decision pivots—to generate dense, step-level rewards. This self-supervised approach will match the convergence stability of models guided by external Process Reward Models (PRMs) while eliminating the compute and memory overhead of running an auxiliary model.

---

## Quantitative Performance Expectations and Convergence Trajectories

### Analytical Accuracy and Token Efficiency Benchmarks

Models trained with Step-GRPO or S-GRPO exhibit a superior balance between accuracy and token efficiency compared to standard GRPO baseline runs.

| Model & Framework Configuration | AIME 2024 Accuracy | Training Samples Used | Average Token Reduction | Convergence Speedup |
|----------------------------------|-------------------|----------------------|------------------------|---------------------|
| Standard Baseline GRPO (Qwen 1.5B) | ~15.4% | 40,000+ | 0% (Baseline) | 1.0x (Baseline) |
| Step-GRPO (Qwen 3-8B) | ~36.7% | 11,000 | 32.0%–43.0% | 2.5x faster |
| S-GRPO Decaying Reward (Qwen 3B) | ~38.2% | 15,000 | 35.4%–61.1% | 2.2x faster |

Step-GRPO achieves 36.7% accuracy on AIME 2024 using only 11,000 training samples — a massive improvement over traditional reinforcement learning baselines that require over 40,000 samples to reach comparable performance. Because S-GRPO penalizes overthinking through decaying rewards, it achieves a 35.4%–61.1% reduction in generated sequence lengths, directly translating to lower compute costs and faster inference times.

### Compute and Temporal Requirements in Notebook Runtimes

| Hardware Tier | Cost Environment | Model Size & Precision | Total Training Time | Projected Cost |
|---------------|-----------------|----------------------|--------------------:|----------------|
| 1x Tesla T4 (15GB) | Google Colab Free | Qwen 1.5B (4-bit QLoRA) | 4.0–6.0 hours | Free / Included |
| 1x L4 (24GB) | Google Colab Pro | Qwen 3B (4-bit QLoRA) | 3.5–5.0 hours | ~$0.30/hr |
| 1x RTX 4090 (24GB) | Local / Ephemeral Cloud | Llama 3.2 3B (16-bit LoRA) | 6.0–12.0 hours | ~$0.55/hr |
| 1x H200 (141GB) | Dedicated High-VRAM Cloud | Qwen 14B (FP8 LoRA + Standby) | 3.0–5.0 hours | ~$2.50/hr |

Using Unsloth's QLoRA memory optimizations combined with Standby features, a standard reasoning training run converges in less than 100 steps for format-priming, and under 500 steps for full logical steering. This keeps Phase 4 completely feasible within the standard execution time limit of a single Google Colab session.

---

## Long-Term Strategic Horizon: The Autonomous Progression

Completing Phase 4 establishes a foundational capability: a small model that can autonomously generate correct internal monologues to solve complex logical problems. This capability forms the core engine for Phase 5 (FrankenMoE capability stacking) and bridges the gap toward fully autonomous scientific agents.

```
┌────────────────────────────────────────────────────────┐
│       Phase 4: GRPO Cognitive Optimization             │
│       - Compact 1.5B–3B scale with structured monologue│
└───────────────────────┬────────────────────────────────┘
                        │ Out-of-distribution transfer
                        ▼
┌────────────────────────────────────────────────────────┐
│       Phase 5: FrankenMoE & Model Merging              │
│       - Stacks Phase 4 cognitive engine with custom    │
│         domains                                        │
└───────────────────────┬────────────────────────────────┘
                        │ Agentic orchestration
                        ▼
┌────────────────────────────────────────────────────────┐
│       Full Autonomy (Aletheia Horizon)                 │
│       - Iterative generation, verification, revision   │
│       - Source verification and tool-use loop          │
└────────────────────────────────────────────────────────┘
```

The long-term development target is aligned with Google's **Aletheia** math agent architecture, which utilizes advanced cognitive models to go beyond Olympiad-level math to tackle PhD-level exercises and open research problems. Aletheia achieves its performance by separating generation from verification:

- **The Generator (Idea Generation):** Proposes initial mathematical proofs, structural outlines, and cross-domain connections.
- **The Verifier (Critical Evaluation):** A logical verifier that reads candidate steps, checks for calculations, identifies logical gaps, and guides iterative revision.

This iterative loop enables autonomous scientific progress. The Aletheia agent successfully wrote a complete mathematics paper (Feng26) calculating structural constants in arithmetic geometry without any human intervention. It also solved four open questions on Bloom's Erdős Conjectures database.

By mastering Phase 4, the localized 1.5B–3B model becomes capable of acting as a lightweight, low-compute generator and verifier within a localized agentic stack, allowing researchers to run continuous, offline discovery loops on consumer hardware.

---

## Nuanced Conclusions and Deployment Recommendations

The transition of the Phase 4 cognitive optimization plan into active execution requires adhering to several strict system recommendations to ensure stability and cost efficiency:

- **Cold-Start Safeguard:** Do not initiate GRPO training directly on a base model. If the model cannot already follow basic instructions, the probability of generating a correct response is zero, meaning the relative advantage calculation will collapse into a zero-gradient state. Always initialize from a high-quality, supervised fine-tuned (SFT) instruct baseline.

- **Two-Stage Reward Scaling:** Implement a staged training approach. During the first 50–100 steps, set formatting rewards (`R_format`) high and correctness rewards (`R_correct`) low. This trains the model to consistently output well-formed `<think>...</think>` structures before optimizing for factual correctness.

- **Notebook Disconnection Defenses:** Because Google Colab has active session timeouts, configure persistent checkpoint saving to Google Drive every 50 steps. If the notebook runtime disconnects, the trainer can resume from the last saved state dict without losing progress.

- **Active Anti-Hacking Guardrails:** To prevent the model from generating verbose, redundant monologues to exploit step-level rewards, training should incorporate Posterior-GRPO (P-GRPO) to zero out process rewards on failed trajectories, alongside Step-GRPO's decaying step rewards to penalize overthinking.

---

## Works Cited

1. Aletheia Unveiled: Google's Autonomous Mathematical Research AI | Atal Upadhyay — https://atalupadhyay.wordpress.com/2026/02/19/aletheia-unveiled-googles-autonomous-mathematical-research-ai/
2. Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning — arXiv — https://arxiv.org/html/2604.16890v1
3. GRPO Trainer — Hugging Face — https://huggingface.co/docs/trl/grpo_trainer
4. Tutorial: Scaling Reinforcement Learning with verl on GKE — Google Developer Forums — https://discuss.google.dev/t/tutorial-scaling-reinforcement-learning-with-verl-on-gke/336370
5. vLLM Serving for GRPO Training — Axolotl — https://docs.axolotl.ai/docs/vllm_serving.html
6. GRPO Fine-Tuning on GPU Cloud: Train Reasoning Models — https://www.spheron.network/blog/grpo-fine-tuning-gpu-cloud/
7. Transformers Reinforcement Learning — vLLM Documentation — https://docs.vllm.ai/en/latest/training/trl/
8. Train your own R1 reasoning model locally (GRPO) — Unsloth — https://unsloth.ai/blog/r1-reasoning
9. [Bug] GRPO Training VRAM usage increases with each step. OOM Error · Issue #3864 · unslothai/unsloth — GitHub — https://github.com/unslothai/unsloth/issues/3864
10. Stop Guessing: A Systematic Guide to Fixing CUDA Out of Memory Errors in GRPO Training — https://home.mlops.community/public/blogs/stop-guessing-a-systematic-guide-to-fixing-cuda-out-of-memory-errors-in-grpo-training
11. Reinforcement Learning (RL) Guide — Unsloth Documentation — https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
12. How to train your LLM to reason like DeepSeek: GRPO reinforcement learning using Unsloth! — Medium — https://medium.com/mitb-for-all/how-to-train-your-llm-to-reason-grpo-reinforcement-learning-using-unsloth-64af5e82ac3c
13. ToolBrain: A Flexible Reinforcement Learning Framework for Agentic Tools — arXiv — https://arxiv.org/html/2510.00023v1
14. EP-GRPO: Entropy-Progress Aligned Group Relative Policy Optimization with Implicit Process Guidance — arXiv — https://arxiv.org/html/2605.04960v1
15. Posterior-GRPO: Rewarding Reasoning Processes in Code Generation — OpenReview — https://openreview.net/forum?id=koTNDE8hl0
16. ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models — GitHub — https://github.com/facebookresearch/threadweaver
17. [Revue de papier] S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models — https://www.themoonlight.io/fr/review/s-grpo-early-exit-via-reinforcement-learning-in-reasoning-models
18. [Literature Review] Dynamic Early Exit in Reasoning Models — Moonlight — https://www.themoonlight.io/en/review/dynamic-early-exit-in-reasoning-models
19. MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems — arXiv — https://arxiv.org/html/2509.22047v1
20. Step-GRPO: Enhancing Reasoning Quality and Efficiency via Structured PRM-Based Reinforcement Learning — AAAI — https://ojs.aaai.org/index.php/AAAI/article/view/40441
21. DRM: Diffusion-based Reward Model With Step-wise Guidance — arXiv — https://arxiv.org/html/2605.25661v1
22. xuyongfu/open-r1-20250207: Fully open reproduction of DeepSeek-R1 — GitHub — https://github.com/xuyongfu/open-r1-20250207
23. Gemini Deep Think: Redefining the Future of Scientific Research — Google DeepMind — https://deepmind.google/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/
24. [2602.10177] Towards Autonomous Mathematics Research — arXiv — https://arxiv.org/abs/2602.10177
25. Towards Autonomous Mathematics Research — arXiv — https://arxiv.org/html/2602.10177v1
26. The Engineering Handbook for GRPO + LoRA with Verl — Hugging Face — https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl
27. A better way of extracting information with LLMs: GRPO Experiment — Infinite Lambda — https://infinitelambda.com/extract-info-llm-grpo/
