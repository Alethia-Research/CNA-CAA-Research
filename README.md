# Alethia Research — CNA / CAA Steering Suite

Contrastive Neuron Attribution (CNA) for finding and steering sparse behavioral circuits in LLMs without fine-tuning. Maps safety refusal, sycophancy, and factual recall circuits across multiple model scales (1.5B–72B). Includes blacklist calibration for model-agnostic circuit transfer and variance-based optimization.

Brand: [Alethia Research](Brand_Doc.md) — *Aletheia: the uncovering of what is hidden.*

## Project Structure

```
├── src/                               # Core experiment code
│   ├── cna_steering_experiment.py     # Main cross-model CNA steering suite
│   ├── advanced_steering_suite.py     # Config-driven variant sweeps
│   ├── generalization_7b.py           # Circuit generalization on 7B models
│   ├── calibrate_blacklist.py         # Variance-based blacklist calibration
│   ├── cross_model_blacklist_test.py  # Cross-model blacklist transfer tests
│   ├── verify_blacklist.py            # Blacklist verification/viz utilities
│   ├── calculate_alpha.py             # Scaling law α estimation
│   ├── phi3_factual_steering.py       # Phi-3 factual logit-diff steering
│   ├── phi3_mini_testing.py           # Phi-3 mini experiment runner
│   └── qwen3b_testing.py              # Qwen 3B experiment runner
├── data/
│   ├── blacklists/                    # Universal neuron blacklists
│   │   ├── blacklist_1.5b.json
│   │   ├── blacklist_7b.json
│   │   ├── blacklist_7b_topk1000.json
│   │   └── Phase-3/                   # Variance-calibrated blacklists
│   ├── circuits/                      # Discovered circuit indices
│   │   └── safety_1.5b.json
│   └── results/                       # Experimental results
│       ├── generalization_7b.json
│       ├── phi3_factual.json
│       ├── phi3_mini_results.json
│       └── qwen3b_factual_results.json
├── findings/
│   ├── FINDINGS.md                    # Full experimental findings (paper-ready)
│   └── Paper_1_Draft.md               # arXiv paper draft
├── logs/
│   ├── 1.5b_experiments.md
│   ├── 7b_experiments.md
│   ├── cross_model_report.md          # Cross-model comparison results
│   └── phi3_factual_notes.md
├── notes/
│   ├── cna_analysis.md
│   ├── cna_overview.md
│   └── research_agenda.md             # Research roadmap & literature survey
├── scratch/                           # Analysis & scratch scripts
├── docs/superpowers/specs/            # Research roadmap specs
├── index.html                         # Interactive findings dashboard
├── Brand_Doc.md                       # Brand identity guidelines
└── LICENSE
```

## Method

CNA identifies behavioral circuits by contrasting MLP activations between positive and negative prompt sets (5 each). The top ~200 contrastive neurons form the circuit. At inference, their activations are scaled by multiplier `m`:

- **`m=0.0`** — Ablate: removes the behavior
- **`m=1.0`** — Baseline: unchanged behavior
- **`m=2.0`** — Amplify: strengthens the behavior

No training required. No SAEs needed.

**Blacklist Calibration:** Variance-based method selects model-agnostic safety neurons by analyzing attribution stability across prompts. Blacklists from small models (1.5B) transfer predictably to larger models (7B), enabling single-pass circuit discovery at scale.

**Scaling Law:** Safety circuit ablation follows a predictable scaling curve — bypass requires ~200 neurons in 1.5B, projected ~2500 in 7B. Refusal dissolves monotonically. Circuits peak at 96–97% model depth regardless of architecture or scale.

## Key Findings

| Behavior | Ablate (m=0.0) | Amplify (m=2.0) |
|---|---|---|
| **Safety Refusal** | Jailbreak — model complies with harmful requests | Safety hardening — more assertive refusal |
| **Sycophancy** | Truth serum — model directly corrects false premises | More evasion / polite deflection |
| **Factual Recall** | Logit-diff steering swaps factual answers (e.g., Delhi ↔ Paris) | — |

**Cross-model:** Blacklists transfer across Qwen/Llama architectures. CNA maintains generation coherence where CAA degenerates to garbage output. CAA collapse language fingerprints training data distribution.

## Running

```bash
# Main cross-model steering suite
python src/cna_steering_experiment.py --help

# Advanced steering sweeps
python src/advanced_steering_suite.py --help

# Blacklist calibration
python src/calibrate_blacklist.py --help

# Cross-model blacklist transfer test
python src/cross_model_blacklist_test.py --help
```

Models tested: Qwen2.5 (1.5B–72B), Llama 3.2 (1B/3B), Llama 3.1 (8B), Phi-3-mini.

## Research Roadmap

1. **CNA circuit discovery** — Done ✓ (1.5B safety, sycophancy, factual)
2. **Cross-model scaling & factual steering** — Done ✓ (7B generalization, blacklist transfer)
3. **Blacklist optimization** — Done ✓ (variance calibration, Phase-3)
4. **GRPO reasoning training** — Upcoming
5. **Model merging / FrankenMoE** — Upcoming

Target: arXiv preprint + open-source model release.

## Citation

```bibtex
@misc{alethia2026cna,
  title={Contrastive Neuron Attribution: Sparse Circuit Steering Without Fine-Tuning},
  author={Alethia Research},
  year={2026}
}
```
