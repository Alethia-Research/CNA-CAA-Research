#!/usr/bin/env python3
"""
Extracts layer-wise hidden state centroids from a base language model 
for different domain experts, resolving FrankenMoE routing collapse.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CentroidExtractor")

# Default domain representative prompts
DEFAULT_DOMAIN_SAMPLES = {
    "safety": [
        "I cannot fulfill this request.", 
        "unauthorized and illegal",
        "safety guidelines and ethical standards", 
        "refuse to provide instructions"
    ],
    "reasoning": [
        "Let's analyze this step-by-step.", 
        "Therefore, the final answer is",
        "Wait, let me double check the calculation.", 
        "Looking at the transition step"
    ]
}

def parse_args():
    parser = argparse.ArgumentParser(description="FrankenMoE Centroid Extractor")
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen2.5-1.5B-Instruct",
        help="Base model path or Hugging Face repository ID."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="layer_centroids.json",
        help="Output JSON file to store centroids."
    )
    parser.add_argument(
        "--samples_file",
        type=str,
        default=None,
        help="Path to JSON file containing domain-specific text samples (optional)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)."
    )
    return parser.parse_args()

def load_domain_samples(samples_file: str) -> Dict[str, List[str]]:
    if not samples_file:
        logger.info("No custom samples file provided. Using default representative prompts.")
        return DEFAULT_DOMAIN_SAMPLES
    
    if not os.path.exists(samples_file):
        logger.warning(f"Custom samples file '{samples_file}' not found. Falling back to defaults.")
        return DEFAULT_DOMAIN_SAMPLES
    
    try:
        with open(samples_file, "r") as f:
            samples = json.load(f)
        logger.info(f"Loaded custom domain samples from {samples_file}")
        return samples
    except Exception as e:
        logger.error(f"Error loading custom samples: {e}. Falling back to defaults.")
        return DEFAULT_DOMAIN_SAMPLES

def get_hidden_states(model_name: str, domain_samples: Dict[str, List[str]], device: str, output_path: str):
    logger.info(f"Loading tokenizer and base model '{model_name}' on device '{device}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model in float16 for memory efficiency
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()
    
    # Resolve the layers object depending on model architecture
    # Standard architectures: Llama -> model.layers, Qwen2/Qwen2.5 -> model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        logger.error("Could not automatically resolve layers in model architecture.")
        sys.exit(1)
        
    n_layers = len(layers)
    logger.info(f"Detected {n_layers} layers in model.")
    
    # Structure: layer_index -> list of centroids for each domain
    centroids_per_layer = {l: [] for l in range(n_layers)}
    domains = list(domain_samples.keys())
    
    for domain in domains:
        texts = domain_samples[domain]
        logger.info(f"Extracting activations for domain '{domain}' using {len(texts)} prompts...")
        hidden_accum = {l: [] for l in range(n_layers)}
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            act_cache = {}
            hooks = []
            
            # Hook to capture input activations of MLP layer
            def get_activation_hook(layer_idx):
                def hook_fn(module, input_args, output_args):
                    # input_args is a tuple where the first element is the input hidden state tensor
                    if isinstance(input_args, tuple):
                        tensor_in = input_args[0]
                    else:
                        tensor_in = input_args
                    act_cache[layer_idx] = tensor_in.detach().cpu().numpy()
                return hook_fn
            
            # Register forward hooks across all layers
            for l in range(n_layers):
                # Try registering on standard MLP module
                if hasattr(layers[l], "mlp"):
                    target_module = layers[l].mlp
                elif hasattr(layers[l], "ffn"):
                    target_module = layers[l].ffn
                else:
                    logger.error(f"Cannot locate MLP/FFN module in layer {l}")
                    sys.exit(1)
                    
                h = target_module.register_forward_hook(get_activation_hook(l))
                hooks.append(h)
            
            with torch.no_grad():
                model(**inputs)
                
            # Remove hooks immediately after forward pass
            for h in hooks:
                h.remove()
                
            # Process and store activations
            for l in range(n_layers):
                if l in act_cache:
                    # act_cache[l] shape: (batch_size=1, sequence_length, hidden_dim)
                    # Average over the sequence dimension to obtain a representative hidden state
                    mean_state = np.mean(act_cache[l], axis=1).squeeze()
                    hidden_accum[l].append(mean_state)
        
        # Fit K-Means clusterer for each layer to find the single domain centroid
        logger.info(f"Computing centroids for domain '{domain}'...")
        for l in range(n_layers):
            stacked_activations = np.array(hidden_accum[l]) # Shape: (num_prompts, hidden_dim)
            
            # Extract 1 cluster centroid (geometric center) using MiniBatchKMeans for speed
            kmeans = MiniBatchKMeans(
                n_clusters=1,
                random_state=42,
                batch_size=max(2, len(texts)),
                n_init="auto"
            )
            kmeans.fit(stacked_activations)
            centroid = kmeans.cluster_centers_[0].tolist()
            centroids_per_layer[l].append(centroid)
            
    # Save the computed centroids list
    with open(output_path, "w") as f:
        json.dump({
            "domains": domains,
            "centroids": centroids_per_layer
        }, f, indent=2)
        
    logger.info(f"[+] Successfully extracted and saved centroids to '{output_path}'")

def main():
    args = parse_args()
    samples = load_domain_samples(args.samples_file)
    get_hidden_states(args.model_name, samples, args.device, args.output_json)

if __name__ == "__main__":
    main()
