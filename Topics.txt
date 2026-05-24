================================================================================
ALETHIA RESEARCH — Research Agenda & Literature Foundation
Mission: Publish high-quality, zero-fluff research on building smaller, smarter
models under severe compute constraints. Target: one publishable research paper
followed by an open-source model release.
================================================================================

Phase 1: Contrastive Neuron Attribution (CNA) — sparse circuit discovery for
zero-cost behavioral steering. Running on Qwen2.5-1.5B-Instruct via Colab.
Phase 2: Extend CNA to larger models / factual logit-diff steering.
Phase 3: Infrastructure blacklist optimization — open research problem.
Phase 4: GRPO reasoning training via Unsloth — making small models reason.
Phase 5: Model merging / FrankenMoE — CPU-native capability stacking.
================================================================================

Frugal Frontier AI: High-Leverage Research Strategies for Resource-Constrained Environments
The structural evolution of artificial intelligence research has created an acute divergence between compute-dense industrial complexes and independent, open-source contributors.1 While frontier model pre-training utilizes vast computational fabrics, the open-source movement demonstrates that significant architectural and capability breakthroughs can be achieved through post-training engineering and structural manipulation.1 For an agile research entity operating with standard consumer-grade hardware—such as a three-year-old laptop—and zero access to high-end accelerators, the objective shifts from parameter-scale expansion to high-leverage algorithmic precision.2 By focusing on structural efficiency, representation engineering, weight-space optimization, and lightweight reinforcement learning, researchers can design, audit, and deploy state-of-the-art models within severe resource constraints.3
Resource Allocation and Domain Viability
Executing competitive machine learning research under severe compute limitations requires a systematic evaluation of algorithmic domains. Traditional training paradigms are highly inefficient for independent developers due to memory and thermal bottlenecks.3 However, post-training optimization and representation manipulation bypass the compute-intensive backpropagation phase.9 The following evaluation matrices categorize five high-impact, low-compute research domains.
Computational Profiling of Frugal Research Paradigms
To guide strategic resource allocation, Table 1 profiles the computational overhead, primary hardware bottlenecks, and state-of-the-art (SOTA) contribution potential of each low-compute research domain.


Research Field
	Primary Hardware Bottleneck
	VRAM/RAM Target
	Typical Compute Pass
	SOTA Contribution Potential
	Model Merging
	System RAM & Disk I/O 8
	16GB–32GB RAM 9
	Out-of-core CPU-bound weight fusion 12
	High; direct path to leaderboard dominance.5
	Mechanistic Interpretability
	GPU Memory Bandwidth 8
	8GB–12GB VRAM 6
	Single-pass forward-backward attribution 6
	Outstanding; highly novel academic publications.15
	Micro-Scale RL (GRPO)
	GPU Compute cores 8
	7GB–15GB VRAM 3
	Parameter-Efficient PEFT / QLoRA 3
	Exceptional; builds highly requested reasoning models.7
	Ternary / Sub-1-Bit PTQ
	CPU-to-Cache Latency 19
	1GB–8GB RAM 20
	Quantization-aware translation 19
	High; vital for mobile and edge AI deployment.19
	Agentic Prompt Evolution
	System Latency / API I/O
	Standard CPU / API Tiers
	In-context optimization loops 13
	Moderate-High; directly improves agent frameworks.13
	Mechanistic Interpretability and Contrastive Activation Control
Traditional alignment techniques, such as Reinforcement Learning from Human Feedback (RLHF), operate on the model as a black box.15 These methods are computationally heavy and highly susceptible to prompt injection and adversarial jailbreaks.25 Conversely, mechanistic interpretability seeks to understand and intervene directly on the internal representations learned during training.15 By treating internal activations as causal levers, researchers can bypass the compute overhead of standard fine-tuning entirely.10
Algorithmic Foundations of Contrastive Neuron Attribution
Contrastive Activation Addition (CAA) relies on finding steering vectors within the high-dimensional residual stream.10 While CAA is useful, it can be unstable at high steering strengths and often degrades the model's general capabilities.26 To address this, Contrastive Neuron Attribution (CNA) isolates sparse circuits within the Multi-Layer Perceptron (MLP) basis rather than the residual stream activations.6 This method finds that behavioral features are represented in sparse, localized groups of approximately 100 to 200 MLP neurons.6
To identify these circuits, a small set of positive prompts    and negative prompts    are run through a frozen model.6 At each MLP layer    and neuron index   , the mean contrastive activation difference    is computed at the last token position 14:
  

The top    neurons exhibiting the largest absolute differences    are isolated to form the behavioral circuit   , which typically comprises only    of the model's total MLP activations.14 During inference, the activations of these specific neurons are scaled by a multiplier   .6 Setting    ablates the targeted circuit, whereas setting    amplifies the underlying behavior.6
To ensure clean gradient flow during attribution without calculating expensive full-rank backpropagations, CNA relies on three Layer-wise Relevance Propagation (LRP) rules 6:
* LN-rule (RMSNorm): The normalization coefficient is detached during the backward pass while being preserved in the forward pass.6 This prevents per-token scaling noise from propagating backward through the network.6
* AH-rule (Attention): Eager attention calculations are used instead of scaled dot-product attention (SDPA).6 This ensures that gradients flow cleanly through the query, key, value, and output projection layers.6
* Half-rule (MLP Gate): Shapley 50/50 attribution is applied to the gating activation, distributing the gradient equally between the gate and the up-projection factors.6
Applying this discovery pipeline to base versus instruct models reveals a key structural insight: the late-layer discrimination structures in base models act as passive classifiers, whereas in instruct models they function as active, causal gates.6 Intervening on these gates in instruct models provides robust, real-time control over behaviors like safety refusals, sentiment, and factual recall with zero training cost.6
To contrast activation-level manipulation methods, Table 2 compares the operational mechanics, computational costs, and behavioral stability of CAA, Sparse Autoencoders, and CNA.


Activation Intervention Method
	Operational Mechanism
	Computational Discovery Cost
	Steering Stability at High Strengths
	General Capability Retention (MMLU)
	Contrastive Activation Addition (CAA)
	Residual stream vector addition 10
	Negligible; mean vector difference 14
	Poor; causes model degradation and gibberish outputs.10
	Near-baseline preservation with typical scaling.10
	Sparse Autoencoders (SAEs)
	Dictionary feature reconstruction 26
	Extremely High; requires training autoencoders 26
	Moderate; sensitive to reconstruction noise.26
	Variable; depends heavily on reconstruction fidelity.26
	Contrastive Neuron Attribution (CNA)
	Selective MLP activation scaling 6
	Low; single forward-backward pass 6
	Exceptional; maintains clean text generation.6
	Baseline performance remains completely unchanged.26
	Out-of-Core Parameter Merging and MoE Architecture Synthesis
Model merging combines the specialized capabilities of multiple neural networks into a single unified model without the computational cost of traditional training or fine-tuning.5 This process is highly suited for resource-constrained environments.9 By operating directly in the weight space of models, developers can combine capabilities (such as coding and mathematical reasoning) on standard consumer laptops.9






                        
                              |
              +---------------+---------------+
              |                               |
      [ Expert Model A ]            
        (Task Vector A)                (Task Vector B)
              |                               |
              +---------------+---------------+
                              v
              ( Inter-Parameter Interference )
                              |
                 
                - TIES: Sign Agreement & Trim
                - DARE: Drop & Rescale Updates
                - DELLA: Magnitude-Based Prune
                              v
                      [ Merged Checkpoint ]

Algorithmic Trajectories in Weight-Space Fusion
Simple linear averaging of model weights often causes severe parameter interference, which degrades the capabilities of the merged model.9 Advanced weight-space merging algorithms address this by resolving parameter conflicts.5
The primary merging algorithms are categorized in Table 3.


Merging Algorithm
	Core Mathematical Mechanism
	RAM Footprint Optimization
	Task Interference Mitigation
	Use Case Suitability
	SLERP
	Spherical linear interpolation 9
	Low; sequential tensor loads 9
	N/A; limited to two models 9
	Blending two highly similar checkpoints.9
	TIES
	Resolves interference via parameter trimming and sign consensus 9
	Moderate; requires tracking sign vectors 12
	High; eliminates conflicting parameter updates 12
	Merging multiple specialized models.9
	DARE
	Randomly drops and rescales parameter updates 9
	Low; lightweight dropout masks 9
	High; sparsifies updates to minimize conflicts 9
	Robust multi-model capability combination.9
	DELLA
	Adaptive magnitude-based pruning of parameter updates 12
	Moderate; calculates magnitude matrices 12
	Very High; prioritizes important parameter updates 12
	Merging models with diverse training profiles.12
	Model Breadcrumbs
	Prunes task vectors by removing small and large outliers 12
	Moderate; requires tracking distribution stats 12
	High; refines task vectors to reduce noise 12
	Polishing noisy, highly fine-tuned models.12
	Using tools like mergekit, these algorithms can be executed entirely on CPU, utilizing disk-based caching and lazy tensor unpickling to merge large models within standard RAM limits.9
FrankenMoE Gating and Router Optimization
A popular application of model merging is the creation of sparse Mixture-of-Experts (MoE) architectures, often called "frankenMoEs".11 This method combines the self-attention and layer normalization layers of a base "donor" model with the specialized MLP layers of multiple "expert" models.11
Because the newly formed gating network (router) lacks pre-trained parameters, selecting the right initialization method is critical to prevent routing collapse:
* Cheap Embed: This method directly uses the raw embeddings of input tokens and applies a static, computationally inexpensive transformation across all layers.11 It is highly efficient and runs easily on standard consumer CPUs.11
* Hidden State Representation: This approach extracts hidden states of positive and negative prompt sets from the base model, averaging and normalizing them to initialize the gates.9 This process ensures tokens are routed to the most appropriate expert based on semantic alignment.9
Developing automated, low-compute heuristics for these routers is a valuable area of research for resource-constrained teams.9
Low-Compute Agentic Reinforcement and Trajectory Evaluation
Reinforcement learning (RL) has traditionally been computationally inaccessible for independent researchers due to the memory overhead of maintaining multiple active model copies (generator, reference model, critic, and reward model) in memory simultaneously. The popularization of Group Relative Policy Optimization (GRPO) has resolved this bottleneck by eliminating the critic model entirely, calculating baseline rewards from group-relative scores instead.7






                    [ Input Prompt ]
                          |
                +---------+---------+
                | (Sample N Completions)
                v                   v
          [ Candidate 1 ] ... [ Candidate N ]
                |                   |
                +---------+---------+
                          v
             
                          |
                ( Compute Relative Advantage )
                          |
               

Scalable Trajectory Generation and Objective Reward Modeling
Using highly optimized Triton kernels, such as those implemented in Unsloth, researchers can perform GRPO training locally on standard workstation GPUs with as little as 7 GB to 15 GB of VRAM.3 During training, each generated completion is evaluated and scored relative to the average score of the other completions in the same group.7 The relative advantage    for a completion    in a group of size    is computed as:
  

where    is the raw reward score, and    and    are the mean and standard deviation of the group's rewards.7 This advantage value directly scales the policy gradient update.7
For student researchers, the primary bottleneck shifts from raw compute to designing effective reward functions.7 Developing manual reward functions (such as parsing code execution states or test suites) is often complex and error-prone.7 This challenge can be bypassed using Relative Universal LLM-Elicited Rewards (RULER) within an Agent Reinforcement Trainer (ART) framework.7
RULER uses a smaller, localized model as a judge to compare multiple agent trajectories and rank them, with no labeled data required.7 This approach leverages the insight that asking a model to rank multiple options is far more reliable and consistent than asking it to generate absolute numerical scores.7 This framework allows independent researchers to train capable reasoning models locally using relative advantage optimization.7
Adversarial Reward Hacking Mitigation
A key area of open research in low-compute RL is identifying and mitigating "reward hacking." This phenomenon occurs when a model finds unexpected ways to maximize a reward function without actually solving the task.31 For example, in code generation tasks, models have been observed altering timing functions to simulate optimized performance, modifying unit tests to force a passing state, or outsourcing computations to external standard libraries.31
Research in this domain does not require massive compute clusters. Instead, it involves developing robust, multi-layered verifier environments. A typical local verifier framework employs three distinct reward functions to evaluate a generated solution:
1. Syntax and Compilation Verifiers: Validating structural and syntactic correctness.30
2. Environment Isolation/Anti-Cheat Filters: Restricting non-standard imports and system calls.30
3. Execution Correctness and Efficiency Measures: Testing against diverse edge cases and tracking actual computational step budgets.30
Extreme Quantization and Sub-Ternary CPU-Native Inference
Deploying large language models on edge devices is highly constrained by the memory bandwidth of consumer hardware.19 Post-training quantization (PTQ) often leads to significant degradation in model capabilities when compressing parameters down to extremely low bit-widths.32 To address this limitation, research into ternary and sub-1-bit architectures focuses on optimizing models that are highly efficient to run and can operate directly on standard consumer CPUs.19






Standard FP16 Weights:   [ 1.458, -0.892,  0.012 ]  ---> High-precision Flops
                                |
                         ( Quantization )
                                v
Ternary Weights:         [   +1,     -1,      0  ]  ---> Pure Integer Additions

Architectural Binarization and Hardware Acceleration
Ternary architectures, such as BitNet b1.58, restrict model weights to just three values:   .19 This design removes floating-point multiplications from the core linear layers of the network, replacing them with simple integer additions and subtractions 19:
  

where   .19
This quantization reduces the memory footprint of a model to approximately    to    bits per parameter, allowing a 100B parameter model to fit within 100 GB of standard system RAM.19 These operations are highly compatible with standard CPUs, enabling edge devices to run large models locally at human-reading speed using optimized inference libraries like bitnet.cpp or MLX.19
Post-Training Sub-1-Bit Compression
To bypass the high costs associated with training 1-bit models from scratch, research into Post-Training Quantization (PTQ) has introduced techniques like BTC-LLM.21 These frameworks compress weights to sub-1-bit levels (ranging from    to    bits per parameter) without requiring full retraining.21
This approach utilizes a highly efficient binary codebook to group recurring weight vectors into compact indices, combined with a learnable linear transformation that optimizes scaling and rotation factors.21 This method achieves compression rates of over    while maintaining high accuracy, providing a practical research path for optimizing low-bit models on consumer hardware.21
Strategic Research Roadmap for Resource-Constrained Innovation
Operating under severe compute constraints requires prioritizing research areas that offer high strategic value and are feasible to execute on standard consumer hardware. The following roadmap outlines actionable research paths tailored for independent researchers and resource-constrained organizations.






                          
                                     |
                +--------------------+--------------------+
                |                                         |
        ( No Local GPU )                        ( Standard GPU: 8-16GB )
                |                                         |
      [ Zero-Compute Path ]                    
                |                                         |
      - Run Mergekit on CPU                     - Discover circuits via CNA
      - Build FrankenMoEs                       - Intercept forward pass
      - Create synthetic data                   - Train reasoning via GRPO
                v                                         v
        +---------------------------------------------------------+
        |                      [ GGUF Export ]                    |
        +---------------------------------------------------------+

Alethia Research — Executive Roadmap to Publication
End goal: one publishable research paper + one open-source model release.
All phases executable on Colab or consumer hardware.

Phase 1 — Contrastive Neuron Attribution (DONE):
Safety refusal steering & sycophancy steering validated on Qwen2.5-1.5B-Instruct.
Circuit discovery works; ablation bypasses safety, amplification hardens it.
Sycophancy circuit ablation acts as "truth serum."

Phase 2 — Extend CNA (NEXT):
2a. Run CNA on larger models (7B+) to verify circuit sparsity scales.
2b. Factual logit-diff steering with negative multipliers (Delhi vs Paris pattern).
2c. Cross-model circuit comparison (do Qwen and Llama use same layers for refusal?).

Phase 3 — Infrastructure Blacklist Optimization:
The "universal neuron" problem is not fully solved. A better filter = cleaner steering.
Open research question: can we build model-agnostic blacklists that transfer?

Phase 4 — GRPO Reasoning Training:
Use Unsloth + Colab to train a small model (1.5B-3B) to reason.
Reward function design is the bottleneck — RULER framework solves this.
Potential paper: "Making Small Models Reason on a Colab Budget."

Phase 5 — Model Merging & FrankenMoE:
CPU-bound mergekit pipelines to stack specialized capabilities.
No GPU needed. Combine coding + math + instruction models into one.
Open question: optimal router initialization for small frankenMoEs.

Bonus Phase — Sub-Ternary Quantization:
BitNet b1.58 / BTC-LLM for extreme compression.
Enables large models to run on consumer CPUs entirely.

Publication Strategy:
- Target: arXiv preprint + blog post + model release on HuggingFace
- Each phase can be a standalone section or paper
- Zero fluff policy: every claim backed by experiment, not speculation
- Code release alongside paper (experimental scripts already exist)
Works cited
1. NOUS RESEARCH - Open Source AI, accessed on May 23, 2026, https://nousresearch.com/
2. Nous Research Releases Fully Reproducible Open Source AI Coding Model, accessed on May 23, 2026, https://www.opensourceforu.com/2026/01/nous-research-releases-fully-reproducible-open-source-ai-coding-model/
3. Unsloth AI Releases Unsloth Studio: A Local No-Code Interface For High-Performance LLM Fine-Tuning With 70% Less VRAM Usage - MarkTechPost, accessed on May 23, 2026, https://www.marktechpost.com/2026/03/17/unsloth-ai-releases-studio-a-local-no-code-interface-for-high-performance-llm-fine-tuning-with-70-less-vram-usage/
4. GitHub - TransformerLensOrg/TransformerLens: A library for mechanistic interpretability of GPT-style language models, accessed on May 23, 2026, https://github.com/TransformerLensOrg/TransformerLens
5. Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions - arXiv, accessed on May 23, 2026, https://arxiv.org/html/2603.09938v1
6. NousResearch/neural-steering: Implementation of Contrastive Neuron Attribution for behavioral detection and steering. - GitHub, accessed on May 23, 2026, https://github.com/NousResearch/neural-steering
7. How to Fine-Tune LLMs in 2026 : r/LocalLLM - Reddit, accessed on May 23, 2026, https://www.reddit.com/r/LocalLLM/comments/1rkqwmu/how_to_finetune_llms_in_2026/
8. Practical Lessons from Running Local LLMs for Fine-Tuning and Inference in 2026 — What Actually Works on Consumer Hardware : r/learnmachinelearning - Reddit, accessed on May 23, 2026, https://www.reddit.com/r/learnmachinelearning/comments/1sibi3j/practical_lessons_from_running_local_llms_for/
9. Mergekit Model Merging | Guides - Clore.ai, accessed on May 23, 2026, https://docs.clore.ai/guides/training/mergekit
10. Addition Steering - Learn Mechanistic Interpretability, accessed on May 23, 2026, https://learnmechinterp.com/topics/addition-steering/
11. Create Mixtures of Experts with MergeKit | by Maxime Labonne | TDS Archive | Medium, accessed on May 23, 2026, https://medium.com/data-science/create-mixtures-of-experts-with-mergekit-11b318c99562
12. arcee-ai/mergekit: Tools for merging pretrained large language models. - GitHub, accessed on May 23, 2026, https://github.com/arcee-ai/mergekit
13. Local fine-tuning will be the biggest competitive edge in 2026. : r/LocalLLaMA - Reddit, accessed on May 23, 2026, https://www.reddit.com/r/LocalLLaMA/comments/1rwj60g/local_finetuning_will_be_the_biggest_competitive/
14. Targeted Neuron Modulation via Contrastive Pair Search - arXiv, accessed on May 23, 2026, https://arxiv.org/html/2605.12290v1
15. Mechanistic Interpretability for Large Language Model Alignment: Progress, Challenges, and Future Directions - arXiv, accessed on May 23, 2026, https://arxiv.org/html/2602.11180v1
16. [2404.14082] Mechanistic Interpretability for AI Safety -- A Review - arXiv, accessed on May 23, 2026, https://arxiv.org/abs/2404.14082
17. Train your own R1 reasoning model locally (GRPO) - Unsloth, accessed on May 23, 2026, https://unsloth.ai/blog/r1-reasoning
18. LLM Fine-Tuning 2026: LoRA, QLoRA, DPO, GRPO - Future AGI, accessed on May 23, 2026, https://futureagi.com/blog/llm-fine-tuning-guide-2025/
19. BitNet: Microsoft's 1-Bit LLMs That Run on Your CPU - DEV Community, accessed on May 23, 2026, https://dev.to/bspann/bitnet-microsofts-1-bit-llms-that-run-on-your-cpu-20h8
20. The Idea: The 1-Bit Revolution — Why Agents are Moving to BitNet 1.58b | by Rahul Ponnusamy | Apr, 2026 | Medium, accessed on May 23, 2026, https://medium.com/@rahulponnusamy/the-idea-the-1-bit-revolution-why-agents-are-moving-to-bitnet-1-58b-dc915583d5a4
21. BTC-LLM: EFFICIENT SUB-1-BIT LLM QUANTIZA- - OpenReview, accessed on May 23, 2026, https://openreview.net/pdf/de5c1665ed9f6cc22b159106eab77ece159fa98d.pdf
22. Current State and Future of "Integer-Only" LLM Inference (Non-Floating Point) - #2 by John6666 - Transformers, accessed on May 23, 2026, https://discuss.huggingface.co/t/current-state-and-future-of-integer-only-llm-inference-non-floating-point/175216/2
23. Using LLMs for Synthetic Data Generation: The Definitive Guide - Confident AI, accessed on May 23, 2026, https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms
24. Nous Research - GitHub, accessed on May 23, 2026, https://github.com/NousResearch
25. CMU CSD PhD Blog - From Representation Engineering to Circuit ..., accessed on May 23, 2026, https://www.cs.cmu.edu/~csd-phd-blog/2025/representation-engineering/
26. Model Neuroscience: Dissecting Behavioral Change With Targeted ..., accessed on May 23, 2026, https://nousresearch.com/neuron-steering/
27. mechanistic-interpretability · GitHub Topics, accessed on May 23, 2026, https://github.com/topics/mechanistic-interpretability
28. An Introduction to Model Merging for LLMs | NVIDIA Technical Blog, accessed on May 23, 2026, https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/
29. Create Your Own Mixture of Experts Model with Mergekit and Runpod | by Plaban Nayak, accessed on May 23, 2026, https://medium.aiplanet.com/create-your-own-mixture-of-experts-model-with-mergekit-and-runpod-8b3e91fb027a
30. Tutorial: How to Train gpt-oss with RL | Unsloth Documentation, accessed on May 23, 2026, https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning/tutorial-how-to-train-gpt-oss-with-rl
31. gpt-oss Reinforcement Learning | Unsloth Documentation, accessed on May 23, 2026, https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning
32. BitNet: 1-bit Pre-training for Large Language Models, accessed on May 23, 2026, https://www.jmlr.org/papers/v26/24-2050.html