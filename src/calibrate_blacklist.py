#!/usr/bin/env python3
"""
Alethia Research — Universal Blacklist Calibration Script
Phase 3: Computes model-agnostic infrastructure neurons based on activation variance
across 30 diverse semantic prompts.
"""

import os
import sys
import argparse
import time
import json
import torch

# ── Setup Path to neural-steering repository ──────────────────────────────────
repo_name = "neural-steering"
repo_path = os.path.abspath(repo_name)
if not os.path.exists(repo_path):
    print(f"[*] Cloning {repo_name}...")
    os.system(f"git clone https://github.com/NousResearch/neural-steering.git")

if repo_path not in sys.path:
    sys.path.append(repo_path)

try:
    from neuron_steer import NeuronSteerer
except ImportError:
    try:
        from neuron_steer.core import NeuronSteerer
    except ImportError:
        print("[-] Error: Could not find neural-steering repository imports.")
        sys.exit(1)


# ── Diverse Prompts (30 distinct domains) ──────────────────────────────────────
DIVERSE_PROMPTS = [
    "Write a Python function to sort a list of dictionary items by key.",
    "What is the process of photosynthesis and how does chlorophyll participate?",
    "Explain the historical significance of the Magna Carta in democratic systems.",
    "How do you bake a classic French chocolate soufflé? Provide instructions.",
    "What is the theory of general relativity and how does it describe space-time?",
    "Write a formal business cover letter requesting a software engineering interview.",
    "Explain how the immune system defends the human body against viral infections.",
    "Describe the architectural features of the Gothic cathedral of Notre-Dame.",
    "How does a database transaction maintain ACID properties under concurrency?",
    "Write a short, atmospheric description of a rainy street in Kyoto at night.",
    "What is the difference between a stock and a bond? Explain standard concepts.",
    "How do you tie a double windsor knot for a professional tie?",
    "Explain the role of oceans in global carbon sequestration and weather.",
    "What are the main functions of a central bank, and how does it set rates?",
    "Write a step-by-step tutorial on how to configure a simple Nginx reverse proxy.",
    "How do planetary orbits relate to Kepler's three laws of motion?",
    "Describe the standard structure of a protein molecule, primary to quaternary.",
    "Who wrote the play Hamlet, and what are the main themes of the tragedy?",
    "What are the primary differences between SQL and NoSQL database systems?",
    "Explain the economic concept of supply and demand in market equilibrium.",
    "Write a brief summary of the environmental impact of microplastics in oceans.",
    "How does quantum tunneling occur in semiconductors and what are its uses?",
    "Explain how to safely change a flat tire on a standard passenger car.",
    "What is the standard procedure for administering basic first aid for burns?",
    "Describe the cellular structure and function of the mitochondria in detail.",
    "Write a poem about the transition from autumn to winter using natural imagery.",
    "What is the significance of the Rosetta Stone in deciphering hieroglyphs?",
    "Explain the differences between supervised, unsupervised, and RL models.",
    "How does the human digestive system break down food and absorb nutrients?",
    "What is the chemical formula for water, and how do hydrogen bonds work?"
]


def calibrate_variance_blacklist(steerer, top_n=100, verbose=True):
    """
    Hooks MLP down_proj activations, runs 30 prompts, computes variance
    across prompts, and returns the top_n highest variance neurons.
    """
    model = steerer.model
    tokenizer = steerer.tokenizer
    device = steerer.device
    layers = steerer._layers_ref
    n_layers = len(layers)

    embed_device = model.get_input_embeddings().weight.device
    
    # Store activations per layer: layer_idx -> list of [intermediate_size] tensors
    act_cache_accum = {i: [] for i in range(2, n_layers)}

    print(f"[*] Running forward passes on {len(DIVERSE_PROMPTS)} prompts to record activations...")
    
    for idx, prompt in enumerate(DIVERSE_PROMPTS, 1):
        if verbose and idx % 5 == 0:
            print(f"    Processing prompt {idx}/{len(DIVERSE_PROMPTS)}...")
        
        try:
            formatted = steerer._format_prompt(prompt)
        except Exception:
            formatted = prompt
            
        enc = tokenizer(formatted, return_tensors="pt").to(embed_device)
        input_ids = enc.input_ids

        # Temporary hooks for this forward pass
        layer_acts = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook_fn(module, args):
                # args[0] shape: [1, seq_len, intermediate_size]
                # Cache activation at the last token position
                layer_acts[layer_idx] = args[0][0, -1].detach().float().cpu()
            return hook_fn

        for li in range(2, n_layers):
            h = layers[li].mlp.down_proj.register_forward_pre_hook(make_hook(li))
            hooks.append(h)

        try:
            with torch.no_grad():
                model(input_ids)
        except Exception as e:
            print(f"[-] Forward pass failed on prompt {idx}: {e}")
        finally:
            for h in hooks:
                h.remove()

        # Save from this prompt to overall cache
        for li in range(2, n_layers):
            if li in layer_acts:
                act_cache_accum[li].append(layer_acts[li])

    print("[*] Computing activation variance per neuron...")
    all_neurons_var = []
    
    for li in range(2, n_layers):
        if not act_cache_accum[li]:
            continue
        
        # Stack shape: [num_prompts, intermediate_size]
        stacked = torch.stack(act_cache_accum[li])
        # Compute variance along prompt dimension (dim=0)
        variance = stacked.var(dim=0, unbiased=False)  # shape: [intermediate_size]
        
        for ni in range(variance.shape[0]):
            all_neurons_var.append((li, ni, variance[ni].item()))

    # Sort descending by variance score
    all_neurons_var.sort(key=lambda x: -x[2])

    top_neurons = all_neurons_var[:top_n]
    
    if verbose:
        print(f"\n[+] Top 10 High-Variance (Infrastructure) Neurons:")
        for idx, (li, ni, var) in enumerate(top_neurons[:10], 1):
            print(f"    Rank {idx:2d}: Layer {li:2d}, Neuron {ni:5d} (Variance: {var:.4f})")
            
        # Display layer concentration of the blacklist
        layer_counts = {}
        for li, ni, _ in top_neurons:
            layer_counts[li] = layer_counts.get(li, 0) + 1
        print(f"\n[+] Blacklist layer distribution:")
        for li in sorted(layer_counts.keys()):
            print(f"    Layer {li:2d}: {layer_counts[li]} neurons")

    return [[li, ni] for li, ni, _ in top_neurons]


def main():
    parser = argparse.ArgumentParser(description="Calibrate universal activation variance blacklist")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace model identifier")
    parser.add_argument("--top_n", type=int, default=100, help="Number of neurons to blacklist")
    parser.add_argument("--output_dir", type=str, default="data/blacklists", help="Directory to save blacklist JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Universal Blacklist Calibration: {args.model}")
    print(f"{'='*70}")

    # Load model using NeuronSteerer
    kwargs = {}
    if "7b" in args.model.lower() or "8b" in args.model.lower() or "70b" in args.model.lower():
        kwargs["load_in_4bit"] = True
        print("[*] Detecting large model scale — enabling 4-bit load parameter")
        
    start_time = time.time()
    steerer = NeuronSteerer(args.model, device=args.device, dtype=torch.bfloat16, auto_blacklist=False, **kwargs)
    print(f"[+] Model loaded in {time.time() - start_time:.2f} seconds.")

    # Fix CPU embeddings to GPU transfer hook if they are on different devices
    embed_layer = steerer.model.get_input_embeddings()
    if embed_layer.weight.device.type == "cpu" and args.device == "cuda":
        print("[*] Registering CPU-to-GPU forward hook on input embeddings to prevent device mismatch hangs...")
        def embed_forward_hook(module, inputs, output):
            return output.to(args.device)
        embed_layer.register_forward_hook(embed_forward_hook)

    # Run calibration
    blacklist = calibrate_variance_blacklist(steerer, top_n=args.top_n, verbose=True)

    # Save to file
    os.makedirs(args.output_dir, exist_ok=True)
    clean_name = args.model.split("/")[-1].lower()
    filename = os.path.join(args.output_dir, f"blacklist_{clean_name}.json")
    
    with open(filename, "w") as f:
        json.dump(blacklist, f, indent=2)

    print(f"\n[+] Successfully saved universal blacklist of {args.top_n} neurons to:")
    print(f"    {filename}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
