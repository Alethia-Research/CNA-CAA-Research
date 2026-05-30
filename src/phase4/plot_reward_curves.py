import os
import json
import glob
import argparse

def parse_trainer_states(output_dir):
    """
    Search for all trainer_state.json files under output_dir,
    extract logs, de-duplicate by step, and sort.
    """
    # Look for trainer_state.json recursively under output_dir
    pattern = os.path.join(output_dir, "**", "trainer_state.json")
    state_files = glob.glob(pattern, recursive=True)
    
    # Also check the output_dir root itself
    state_files.append(os.path.join(output_dir, "trainer_state.json"))
    
    # Filter only valid existing files
    state_files = [f for f in state_files if os.path.isfile(f)]
    
    if not state_files:
        print(f"[-] No trainer_state.json files found in {output_dir}")
        return []
    
    print(f"[*] Found {len(state_files)} trainer_state.json files:")
    for f in state_files:
        print(f"  - {f}")
        
    combined_history = {}
    
    for f_path in state_files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            
            history = state_data.get("log_history", [])
            for log in history:
                step = log.get("step")
                if step is not None:
                    # Keep the record with the most key information
                    if step not in combined_history or len(log) > len(combined_history[step]):
                        combined_history[step] = log
        except Exception as e:
            print(f"[-] Warning: Failed to parse {f_path}: {e}")
            
    # Sort history by step
    sorted_steps = sorted(combined_history.keys())
    sorted_history = [combined_history[step] for step in sorted_steps]
    
    print(f"[+] Successfully extracted {len(sorted_history)} logged training steps.")
    return sorted_history

def main():
    parser = argparse.ArgumentParser(description="Plot GRPO reward curves from trainer state files.")
    parser.add_argument("--output_dir", type=str, default="grpo_cot_output", help="Directory where checkpoints are saved.")
    parser.add_argument("--output_plot", type=str, default="findings/figures/reward_curve_step_grpo.png", help="Path to save the generated plot.")
    parser.add_argument("--transition_step", type=int, default=100, help="Step at which formatting-to-correctness transition occurred.")
    args = parser.parse_args()
    
    history = parse_trainer_states(args.output_dir)
    if not history:
        print("[-] Error: No log history extracted. Exiting.")
        return
        
    # Extract keys
    steps = []
    rewards = []
    rewards_std = []
    
    # Attempt to locate reward columns
    # We check common naming conventions
    reward_keys = ["rewards/reward_fn/mean", "reward", "rewards/reward_fn"]
    std_keys = ["rewards/reward_fn/std", "reward_std", "rewards/reward_std"]
    
    found_reward_key = None
    found_std_key = None
    
    # Introspect keys from the first log entry that contains a reward
    for log in history:
        for rk in reward_keys:
            if rk in log:
                found_reward_key = rk
                break
        for sk in std_keys:
            if sk in log:
                found_std_key = sk
                break
        if found_reward_key:
            break
            
    if not found_reward_key:
        # Fallback: find any key containing 'reward' and not 'std'
        for log in history:
            for k in log.keys():
                if "reward" in k.lower() and "std" not in k.lower():
                    found_reward_key = k
                    print(f"[*] Fallback reward key detected: '{found_reward_key}'")
                    break
            if found_reward_key:
                break
                
    if not found_std_key:
        # Fallback: find any key containing 'reward' and 'std'
        for log in history:
            for k in log.keys():
                if "reward" in k.lower() and "std" in k.lower():
                    found_std_key = k
                    print(f"[*] Fallback reward standard deviation key detected: '{found_std_key}'")
                    break
            if found_std_key:
                break

    print(f"[*] Parsing logs using reward key: '{found_reward_key}' and std key: '{found_std_key}'")

    for log in history:
        step = log.get("step")
        # Ensure we have a valid step and at least the reward value
        if step is not None and found_reward_key in log:
            steps.append(step)
            rewards.append(float(log[found_reward_key]))
            if found_std_key and found_std_key in log:
                rewards_std.append(float(log[found_std_key]))
            else:
                rewards_std.append(0.0)

    if not steps:
        print("[-] Error: No reward values found in logs. Exiting.")
        return

    # Try plotting
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = np.array(steps)
        rewards = np.array(rewards)
        rewards_std = np.array(rewards_std)
        
        # Plot mean rewards
        ax.plot(steps, rewards, label='Mean Reward (R_j)', color='#1f77b4', linewidth=2.5)
        
        # Plot standard deviation band if it exists and has non-zero elements
        if np.any(rewards_std > 0):
            ax.fill_between(steps, rewards - rewards_std, rewards + rewards_std, 
                            color='#1f77b4', alpha=0.15, label='Reward Std Dev')
            
        # Draw transition threshold line
        if args.transition_step in steps or (steps[0] <= args.transition_step <= steps[-1]):
            ax.axvline(x=args.transition_step, color='#d62728', linestyle='--', linewidth=2, 
                       label=f'Stage Transition (Step {args.transition_step})')
            
            # Annotate regions
            y_min, y_max = ax.get_ylim()
            ax.text(args.transition_step * 0.4, y_min + (y_max - y_min) * 0.85, 
                    "Stage 1:\nFormat Priming", color='#555555', fontsize=11, fontweight='semibold', ha='center')
            ax.text(args.transition_step + (steps[-1] - args.transition_step) * 0.5, y_min + (y_max - y_min) * 0.85, 
                    "Stage 2:\nCorrectness Focus", color='#555555', fontsize=11, fontweight='semibold', ha='center')
            
        ax.set_title("LF-GRPO Two-Stage Convergence and Reward Curves", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Global Training Step", fontsize=12, labelpad=10)
        ax.set_ylabel("Group Relative Reward Score", fontsize=12, labelpad=10)
        ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e0e0e0')
        ax.set_xlim(steps[0], steps[-1])
        
        # Save figure
        os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300)
        plt.close()
        print(f"[+] Reward curve plot successfully saved to: {args.output_plot}")
        
    except ImportError:
        print("[-] Warning: matplotlib or numpy is not installed. Saving text log data fallback instead of plot image.")
        text_log_path = args.output_plot.replace(".png", "_data.txt")
        os.makedirs(os.path.dirname(text_log_path), exist_ok=True)
        with open(text_log_path, "w", encoding="utf-8") as f:
            f.write("Step\tMeanReward\tStdReward\n")
            for s, r, sd in zip(steps, rewards, rewards_std):
                f.write(f"{s}\t{r:.6f}\t{sd:.6f}\n")
        print(f"[+] Text data log saved to: {text_log_path}")

if __name__ == "__main__":
    main()
