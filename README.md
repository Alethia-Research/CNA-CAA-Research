# Alethia Research — CNA / CAA Steering Suite

Contrastive Neuron Attribution (CNA) for finding and steering sparse behavioral circuits in LLMs without fine-tuning. Maps safety refusal, sycophancy, and factual recall circuits across multiple model scales.

## Project Structure

```
├── cna_steering_experiment.py        # Main cross-model CNA steering suite (Qwen/Llama 1.5B→72B)
├── cna_steering_analysis.md           # Findings analysis: safety, sycophancy, truthfulness
├── CNA_ LLM Neuron Attribution and Steering.md  # Full research write-up
├── Topics.txt                         # Research agenda & literature foundation
├── Adv_Steering/
│   ├── advanced_steering_suite.py     # Advanced steering with config-driven variant sweeps
│   ├── cna_cross_model_report.md      # Cross-model comparison results table
│   ├── generalization_7b.py           # Circuit generalization testing on 7B models
│   ├── custom_blacklist.json          # Universal neuron blacklist filters
│   ├── custom_blacklist_7B_Qwen.json
│   ├── custom_blacklist_Qwen7B_topk1000.json
│   ├── safety_1.5b.json               # Safety circuit neuron indices (1.5B)
│   ├── Logs.md / logs-7B.md           # Experiment logs
│   └── .antigravitycli/               # Experiment artifacts
├── findings/FINDINGS.md               # Paper-ready experimental findings
├── docs/superpowers/                  # Research roadmap specs
└── LICENSE
```

## Method

CNA identifies behavioral circuits by contrasting MLP activations between positive and negative prompt sets (5 each). The top ~200 contrastive neurons form the circuit. At inference, their activations are scaled by multiplier `m`:

- **`m=0.0`** — Ablate: removes the behavior
- **`m=1.0`** — Baseline: unchanged behavior
- **`m=2.0`** — Amplify: strengthens the behavior

No training required. No SAEs needed.

## Key Findings

| Behavior | Ablate (m=0.0) | Amplify (m=2.0) |
|---|---|---|
| **Safety Refusal** | Jailbreak — model complies with harmful requests | Safety hardening — more assertive refusal |
| **Sycophancy** | Truth serum — model directly corrects false premises | More evasion / polite deflection |
| **Factual Recall** | Logit-diff steering swaps factual answers (e.g., Delhi ↔ Paris) | — |

Circuits are sparse (~200 neurons out of millions), localized in late MLP layers, and **transfer predictably across model scales**.

## Running

```bash
# Cross-model steering suite (Colab-friendly)
python cna_steering_experiment.py --help

# Advanced steering sweeps
python Adv_Steering/advanced_steering_suite.py --help
```

Models tested: Qwen2.5 (1.5B–72B), Llama 3.2 (1B/3B), Llama 3.1 (8B), Phi-3-mini.

## Research Roadmap

1. **CNA circuit discovery** — Done ✓
2. **Cross-model scaling & factual steering** — Done ✓
3. **Blacklist optimization** — Open problem
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
