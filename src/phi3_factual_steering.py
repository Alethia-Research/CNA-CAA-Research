#!/usr/bin/env python3
"""
Phi-3 Bidirectional Factual Steering — subtoken_idx=1 fix for SentencePiece tokenizer.
Non-interactive. Upload to Colab and run: !python Adv_Steering/phi3_factual_steering.py
"""

import os
import sys
import gc
import json
import torch

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Setup ──────────────────────────────────────────────────────────────────────
repo_name = "neural-steering"
if not os.path.exists(repo_name):
    print(f"[*] Cloning {repo_name}...")
    os.system(f"git clone https://github.com/NousResearch/neural-steering.git")

repo_path = os.path.abspath(repo_name)
if repo_path not in sys.path:
    sys.path.append(repo_path)

# Auto-find advanced_steering_suite.py (same directory as this script or CWD)
try:
    _self_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _self_dir = os.path.abspath(".")
_suite_candidates = [
    os.path.join(_self_dir, "advanced_steering_suite.py"),
    os.path.join(os.getcwd(), "Adv_Steering", "advanced_steering_suite.py"),
    os.path.join(os.getcwd(), "advanced_steering_suite.py"),
]
_suite_path = None
for p in _suite_candidates:
    if os.path.exists(p):
        _suite_path = p
        break
if not _suite_path:
    raise FileNotFoundError("advanced_steering_suite.py not found — upload it first")
print(f"[+] Found suite at: {_suite_path}")
_suite_dir = os.path.dirname(_suite_path)
if _suite_dir not in sys.path:
    sys.path.insert(0, _suite_dir)

try:
    from neuron_steer import NeuronSteerer
except ImportError:
    from neuron_steer.core import NeuronSteerer

from advanced_steering_suite import discover_factual_circuit_signed, steer_with_signed_circuit

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/Phi-3-mini-4k-instruct"
TOP_K        = 100
MAX_TOKENS   = 15
SUBTOKEN_IDX = 1   # SentencePiece: index 0 = space token, index 1 = actual token

PAIRS = [
    ("The capital of France is",                  " Paris",   " London"),
    ("The capital of Germany is",                 " Berlin",  " Paris"),
    ("The capital of Japan is",                   " Tokyo",   " Seoul"),
    ("The largest planet in the solar system is", " Jupiter", " Saturn"),
    ("Water freezes at",                          " 0",       " 100"),
]

# ── Load model ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"Loading {MODEL_NAME} on {device}")
print(f"{'='*60}")
steerer = NeuronSteerer(MODEL_NAME, device=device, dtype=torch.bfloat16, auto_blacklist=False)
print("[+] Model loaded")

# ── Run pairs ─────────────────────────────────────────────────────────────────
results = []

for prompt, correct, target in PAIRS:
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Prompt:  {prompt}")
    print(f"Correct: {correct!r}  Target: {target!r}")
    print(f"{'='*60}")

    cf, cb = discover_factual_circuit_signed(
        steerer, prompt, target, correct,
        top_k=TOP_K, verbose=True, subtoken_idx=SUBTOKEN_IDX
    )

    if cf is None:
        print("  [-] Attribution failed — skipping")
        results.append({"prompt": prompt, "correct": correct, "target": target, "status": "attribution_failed"})
        continue

    base = steer_with_signed_circuit(steerer, prompt, cf, multiplier=0.0, max_new_tokens=MAX_TOKENS).strip()
    fwd  = steer_with_signed_circuit(steerer, prompt, cf, multiplier=2.0, max_new_tokens=MAX_TOKENS).strip()
    bwd  = steer_with_signed_circuit(steerer, prompt, cb, multiplier=2.0, max_new_tokens=MAX_TOKENS).strip()

    print(f"\n  BASELINE (m=0.0):                   {base}")
    print(f"  FORWARD  m=2.0 ({correct.strip()} -> {target.strip()}): {fwd}")
    print(f"  BACKWARD m=2.0 ({target.strip()} -> {correct.strip()}): {bwd}")

    results.append({
        "prompt": prompt,
        "correct": correct,
        "target": target,
        "baseline": base,
        "forward_m2": fwd,
        "backward_m2": bwd,
        "forward_neurons": len(cf),
        "backward_neurons": len(cb),
    })

# ── Save ───────────────────────────────────────────────────────────────────────
out = "phi3_factual_results.json"
with open(out, "w") as f:
    json.dump({"model": MODEL_NAME, "top_k": TOP_K, "subtoken_idx": SUBTOKEN_IDX, "pairs": results}, f, indent=2)
print(f"\n[+] Saved {out}")
