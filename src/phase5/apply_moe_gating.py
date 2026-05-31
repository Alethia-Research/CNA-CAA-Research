#!/usr/bin/env python3
"""
Low-memory gate weight injector.
Modifies FrankenMoE router gates directly in the .safetensors shard files on disk.
Bypasses CPU RAM limits by avoiding loading the entire model into PyTorch memory.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LowMemGatingApplier")

def parse_args():
    parser = argparse.ArgumentParser(description="Low-Memory FrankenMoE Router Injector")
    parser.add_argument(
        "--moe_model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the merged FrankenMoE model shards."
    )
    parser.add_argument(
        "--centroids_json",
        type=str,
        default="layer_centroids.json",
        help="Path to the JSON file containing the extracted centroids."
    )
    parser.add_argument(
        "--domain_mapping",
        type=str,
        default="safety:0,reasoning:1",
        help="Comma-separated mapping of domain name to expert index (e.g. safety:0,reasoning:1)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save the modified model. If not specified, overrides the input model in-place (recommended to save memory)."
    )
    return parser.parse_args()

def parse_domain_mapping(mapping_str: str) -> Dict[str, int]:
    mapping = {}
    try:
        parts = mapping_str.split(",")
        for part in parts:
            domain, idx = part.split(":")
            mapping[domain.strip().lower()] = int(idx.strip())
        return mapping
    except Exception as e:
        logger.error(f"Failed to parse domain mapping string '{mapping_str}': {e}")
        sys.exit(1)

def main():
    args = parse_args()
    domain_to_idx = parse_domain_mapping(args.domain_mapping)
    output_dir = args.output_dir if args.output_dir else args.moe_model_dir

    # 1. Load the layer-wise centroids
    logger.info(f"Loading centroids from '{args.centroids_json}'...")
    if not os.path.exists(args.centroids_json):
        logger.error(f"Centroids file '{args.centroids_json}' does not exist.")
        sys.exit(1)
        
    with open(args.centroids_json, "r") as f:
        centroid_data = json.load(f)
    
    available_domains = [d.lower() for d in centroid_data["domains"]]
    centroids = centroid_data["centroids"] # Dictionary: str(layer_idx) -> list of centroid vectors
    
    # Verify mapping aligns with available domains
    for domain in domain_to_idx.keys():
        if domain not in available_domains:
            logger.error(f"Mapped domain '{domain}' not found in centroids file domains: {available_domains}")
            sys.exit(1)
            
    domain_indices_in_json = {domain: available_domains.index(domain) for domain in domain_to_idx.keys()}

    # 2. Find all .safetensors files in the model directory
    model_path = Path(args.moe_model_dir)
    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        logger.error(f"No .safetensors files found in directory '{args.moe_model_dir}'.")
        sys.exit(1)
        
    logger.info(f"Found {len(safetensors_files)} safetensors shard files to check.")

    # Create output directory if different from input
    if output_dir != args.moe_model_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Copy non-safetensors metadata files (config.json, tokenizer.json, etc.)
        for file_path in model_path.iterdir():
            if file_path.suffix != ".safetensors" and file_path.is_file():
                import shutil
                shutil.copy(file_path, Path(output_dir) / file_path.name)

    # 3. Process each shard file independently to keep memory usage near zero
    for shard_path in safetensors_files:
        logger.info(f"Scanning shard: {shard_path.name}...")
        
        # Load the dictionary of tensors from the file
        tensors = load_file(shard_path)
        modified = False
        
        for name in list(tensors.keys()):
            # Match only routed expert gate weights
            if "shared_expert" in name:
                continue
            if len(tensors[name].shape) != 2:
                continue
            if name.endswith("gate.weight") or name.endswith("wg.weight"):
                # Isolate the layer index from the name (e.g. 'model.layers.12.mlp.gate.weight' -> 12)
                parts = name.split(".")
                layer_idx = None
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        break
                
                if layer_idx is None:
                    continue
                    
                str_layer = str(layer_idx)
                if str_layer not in centroids:
                    logger.warning(f"No centroid data found for layer {layer_idx}. Skipping.")
                    continue
                    
                layer_centroids = centroids[str_layer]
                param_tensor = tensors[name]
                num_experts, hidden_dim = param_tensor.shape
                
                logger.info(f"  -> Found gate for Layer {layer_idx} (shape: {num_experts}x{hidden_dim}) in {shard_path.name}")
                
                # Create a copy to edit
                param_tensor_edited = param_tensor.clone()
                
                for domain, expert_idx in domain_to_idx.items():
                    if expert_idx >= num_experts:
                        logger.error(f"Mapped expert index {expert_idx} exceeds model's num_experts ({num_experts})")
                        sys.exit(1)
                        
                    json_idx = domain_indices_in_json[domain]
                    centroid_vec = layer_centroids[json_idx]
                    
                    if len(centroid_vec) != hidden_dim:
                        logger.error(f"Dimension mismatch at Layer {layer_idx}: centroid dimension is {len(centroid_vec)} but gate hidden_dim is {hidden_dim}")
                        sys.exit(1)
                    
                    # Convert centroid list to tensor and calculate L2 norm
                    centroid_tensor = torch.tensor(centroid_vec, dtype=param_tensor.dtype)
                    l2_norm = torch.linalg.norm(centroid_tensor, ord=2)
                    
                    # Normalize and assign to weight tensor
                    normalized_centroid = centroid_tensor / (l2_norm + 1e-8)
                    param_tensor_edited[expert_idx].copy_(normalized_centroid)
                    
                tensors[name] = param_tensor_edited
                modified = True
                logger.info(f"  [+] Updated layer {layer_idx} gating weights inside {shard_path.name}")
        
        # Save shard back if any parameters were modified
        if modified:
            dest_path = Path(output_dir) / shard_path.name
            logger.info(f"Saving updated shard to: {dest_path}...")
            save_file(tensors, dest_path)
        elif output_dir != args.moe_model_dir:
            # If not overriding in place, copy unmodified shards directly
            import shutil
            dest_path = Path(output_dir) / shard_path.name
            logger.info(f"Copying unmodified shard to: {dest_path}...")
            shutil.copy(shard_path, dest_path)

    logger.info("[+] FrankenMoE gating update completed successfully in low-memory mode!")

if __name__ == "__main__":
    main()
