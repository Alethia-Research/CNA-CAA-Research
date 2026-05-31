#!/usr/bin/env python3
"""
Python wrapper to execute memory-optimized Periphery-Restricted Della Merging (PRM-DELLA)
on resource-constrained systems like Google Colab Free Tier.
"""

import os
import sys
import argparse
import subprocess
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="PRM-DELLA Merging Executor")
    parser.add_argument(
        "--config",
        type=str,
        default="src/phase5/prm_della_config.yml",
        help="Path to the mergekit yaml configuration file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./merged_model_prm_della",
        help="Directory to save the merged model."
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Execute tensor math on CUDA GPU VRAM."
    )
    parser.add_argument(
        "--low-cpu-memory",
        action="store_true",
        default=True,
        help="Enable low-CPU memory optimization flags in mergekit."
    )
    parser.add_argument(
        "--out-shard-size",
        type=str,
        default="1B",
        help="Write sharded model weights to disk in small steps."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Environment variables to control OS and PyTorch memory allocations
    print("[*] Configuring memory optimization variables...")
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Verify mergekit-yaml is in PATH
    mergekit_cmd = shutil.which("mergekit-yaml")
    if not mergekit_cmd:
        print("[!] Error: 'mergekit-yaml' command not found in PATH.")
        print("[!] Please run: pip install mergekit")
        sys.exit(1)

    # 2. Build the mergekit command
    cmd = [
        mergekit_cmd,
        args.config,
        args.output,
        "--out-shard-size", args.out_shard_size
    ]

    if args.cuda:
        cmd.append("--cuda")
    if args.low_cpu_memory:
        cmd.append("--low-cpu-memory")

    cmd.append("--write-model-card")

    print(f"[*] Command to execute: {' '.join(cmd)}")
    print("[*] Starting merge process. This may take several minutes depending on hardware...")

    # 3. Execute subprocess
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[+] Model merge completed successfully. Merged model saved to: {args.output}")
    except subprocess.CalledProcessError as e:
        print(f"[!] Merge process failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[!] Merge process interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
