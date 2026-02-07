#!/usr/bin/env python
"""
Plot evaluation results from evaluation_metrics.log
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re


def parse_metrics_log(filepath):
    """Parse the evaluation metrics log file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('episode')]
    
    data = []
    for line in lines:
        parts = re.split(r'\s*\|\s*', line)
        if len(parts) < 5:
            continue
        
        try:
            data.append({
                'episode': int(parts[0]),
                'ts': int(parts[1]),
                'seed': int(parts[2]),
                'attempt': int(parts[3]),
                'reward': float(parts[4]),
                'cap': float(parts[5]) if len(parts) > 5 else 0,
                'step': float(parts[6]) if len(parts) > 6 else 0,
                'mdl': float(parts[7]) if len(parts) > 7 else 0,
                'wait': float(parts[8]) if len(parts) > 8 else 0,
                'comp': float(parts[9]) if len(parts) > 9 else 0,
                'nsv': float(parts[10]) if len(parts) > 10 else 0,
            })
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse line: {line[:80]}")
            continue
    
    df = pd.DataFrame(data)
    return df.sort_values('ts').reset_index(drop=True)


def ma(data, window):
    """Moving average that preserves array length."""
    data = np.array(data, dtype=float)
    window = min(window, len(data))
    result = np.convolve(data, np.ones(window)/window, mode='same')
    half_window = window // 2
    for i in range(half_window):
        result[i] = np.mean(data[:i+1])
        result[-(i+1)] = np.mean(data[-(i+1):])
    return result


def parse_baseline_log(filepath):
    """Parse baseline log file to extract mean and std for each policy."""
    baselines = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        in_summary = False
        for line in f:
            line = line.strip()
            
            # Detect summary section
            if line.startswith('# SUMMARY STATISTICS'):
                in_summary = True
                continue
            
            # Parse summary statistics
            if in_summary and line and not line.startswith('pol'):
                parts = re.split(r'\s+', line)
                if len(parts) >= 2:
                    pol_name = parts[0]
                    # Extract rew±std (e.g., "3.78±1.59")
                    rew_field = parts[1]
                    if '±' in rew_field:
                        try:
                            mean_str, std_str = rew_field.split('±')
                            baselines[pol_name] = {
                                'mean': float(mean_str),
                                'std': float(std_str)
                            }
                        except ValueError:
                            pass
    
    return baselines


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_eval_results.py <evaluation_metrics.log> [--ma-window WINDOW] [--baseline-log BASELINE_LOG]")
        print("Example: python plot_eval_results.py eval_results/evaluation_20260206_231327/evaluation_metrics.log --baseline-log baseline_train_seeds_v2000_ms1200_mwd240_mtd900_cap2.log")
        sys.exit(1)
    
    metrics_log = sys.argv[1]
    ma_window = 10
    baseline_log = None
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[2:]):
        if arg == '--ma-window' and i + 2 < len(sys.argv):
            ma_window = int(sys.argv[i + 3])
        elif arg == '--baseline-log' and i + 2 < len(sys.argv):
            baseline_log = sys.argv[i + 3]
    
    print(f"Loading evaluation metrics from: {metrics_log}")
    df = parse_metrics_log(metrics_log)
    print(f"Loaded {len(df)} evaluation records\n")
    
    # Load baseline data if provided
    baselines = {}
    if baseline_log:
        import os
        if os.path.exists(baseline_log):
            baselines = parse_baseline_log(baseline_log)
            print(f"Loaded baselines from: {baseline_log}")
            for pol, stats in baselines.items():
                print(f"  {pol}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            print()
        else:
            print(f"[WARN] Baseline log not found: {baseline_log}\n")
    
    # Output directory (same as log file)
    import os
    output_dir = os.path.dirname(metrics_log)
    log_name = os.path.basename(metrics_log)
    log_path = os.path.relpath(metrics_log)
    
    # Plot 1: Mean reward vs training steps
    print("Generating plots...")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    # Group by timestep
    grouped = df.groupby('ts')['reward'].agg(['mean', 'std', 'count']).reset_index()
    ts = grouped['ts'].values
    means = grouped['mean'].values
    sems = grouped['std'].values / np.sqrt(grouped['count'].values)

    max_ts = 0
    max_mean = 0.0
    ts_min = int(ts.min()) if len(ts) else 0
    ts_max = int(ts.max()) if len(ts) else 0
    unique_ts = int(len(ts))
    if len(means) > 0:
        max_idx = int(np.argmax(means))
        max_ts = int(ts[max_idx])
        max_mean = float(means[max_idx])
        print(f"Rows: {len(df)}")
        print(f"TS min/max: {ts_min} / {ts_max} (unique: {unique_ts})")
        print(f"Max mean reward: {max_mean:.4f} at ts={max_ts}")
        print(f"Mean reward (overall): {df['reward'].mean():.4f}")
        print(f"Std reward (overall): {df['reward'].std():.4f}")

    # Export grouped data used for the plot
    grouped_out = os.path.join(output_dir, 'reward_vs_timesteps_data.csv')
    grouped.to_csv(grouped_out, index=False)
    print(f"[OK] Saved {grouped_out}")
    
    ax.errorbar(ts, means, yerr=sems, fmt='o-', alpha=0.6, 
                label='Mean Reward (Trained)', color='#3498db', capsize=5, markersize=6)
    
    # Moving average
    if len(means) > 1:
        ma_rew = ma(means, ma_window)
        ax.plot(ts, ma_rew, 'r-', lw=2.5, alpha=0.7, label=f'Moving Average (w={ma_window})')
    
    # Add baseline horizontal lines
    if baselines:
        ts_range = [ts.min(), ts.max()] if len(ts) else [0, 1]
        baseline_colors = {'random': '#d62728', 'greedy': '#ff7f0e', 'unique': '#8c564b'}
        baseline_labels = {'random': 'Random', 'greedy': 'Greedy', 'unique': 'Greedy-Unique'}
        
        for pol, stats in baselines.items():
            color = baseline_colors.get(pol, '#95a5a6')
            label = baseline_labels.get(pol, pol.capitalize())
            mean_val = stats['mean']
            std_val = stats['std']
            
            # Horizontal line for mean
            ax.axhline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'{label} Baseline')
            
            # Dotted lines for ±1 std
            ax.axhline(mean_val + std_val, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axhline(mean_val - std_val, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Evaluation Reward', fontsize=11, fontweight='bold')
    ax.set_title('Model Performance vs Training Steps', fontsize=12, fontweight='bold')
    ax.text(0.01, 0.02, f"log: {log_path}\nmax mean: {max_mean:.3f} @ ts={max_ts}\nts: {ts_min}-{ts_max} (n={unique_ts})",
            transform=ax.transAxes, fontsize=9, color='#555555')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'reward_vs_timesteps.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved {output_file}")
    plt.close()
    
    # Plot 2: Mean reward by seed
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    seed_data = df.groupby('seed')['reward'].agg(['mean', 'std', 'count']).reset_index()
    seeds = seed_data['seed'].values
    means = seed_data['mean'].values
    sems = seed_data['std'].values / np.sqrt(seed_data['count'].values)
    
    ax.bar(range(len(seeds)), means, yerr=sems, capsize=5, alpha=0.7, color='#27ae60', edgecolor='black', linewidth=1.5)
    
    # Add baseline horizontal lines
    if baselines:
        baseline_colors = {'random': '#d62728', 'greedy': '#ff7f0e', 'unique': '#8c564b'}
        baseline_labels = {'random': 'Random', 'greedy': 'Greedy', 'unique': 'Greedy-Unique'}
        
        for pol, stats in baselines.items():
            color = baseline_colors.get(pol, '#95a5a6')
            label = baseline_labels.get(pol, pol.capitalize())
            mean_val = stats['mean']
            
            ax.axhline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'{label} Baseline')
    
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'{int(s)}' for s in seeds], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Seed', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
    ax.set_title('Mean Reward by Seed (Averaged over All Models)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'reward_by_seed.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved {output_file}")
    plt.close()
    
    # Plot 3: Per-seed plots
    seeds_unique = sorted(df['seed'].unique())
    for seed in seeds_unique:
        seed_data = df[df['seed'] == seed].sort_values('ts')
        
        if len(seed_data) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
        ax.set_facecolor('#fafafa')
        
        ts = seed_data['ts'].values
        rew = seed_data['reward'].values
        
        ax.plot(ts, rew, 'o-', alpha=0.6, color='#3498db', markersize=5, label='Reward')
        
        # Moving average
        if len(rew) > 1:
            ma_rew = ma(rew, ma_window)
            ax.plot(ts, ma_rew, 'r-', lw=2.5, alpha=0.7, label=f'Moving Average (w={ma_window})')
        
        ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('Evaluation Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'Model Performance for Seed {int(seed)}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'reward_seed{int(seed)}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] Saved {len(seeds_unique)} per-seed plots")
    
    # Plot 4: Component breakdown
    try:
        eval_sorted = df.sort_values('ts').reset_index(drop=True)
        all_reward = eval_sorted['reward'].values
        all_cap = eval_sorted['cap'].values
        all_mdl = eval_sorted['mdl'].values
        all_wait = eval_sorted['wait'].values
        all_comp = eval_sorted['comp'].values
        
        if len(all_reward) > 0:
            fig = plt.figure(figsize=(14, 8), facecolor='#fafafa')
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
            
            eps = np.arange(len(all_reward))
            
            # Total reward
            ax = fig.add_subplot(gs[0, 0])
            ax.set_facecolor('#fafafa')
            ax.plot(eps, all_reward, 'o-', alpha=0.7, color='#2c3e50', markersize=4)
            ax.fill_between(eps, all_reward, alpha=0.3, color='#2c3e50')
            ax.set_xlabel('Evaluation Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Total Reward', fontsize=10, fontweight='bold')
            ax.set_title('Total Reward', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Capacity
            ax = fig.add_subplot(gs[0, 1])
            ax.set_facecolor('#fafafa')
            ax.plot(eps, all_cap, 'o-', alpha=0.7, color='#e74c3c', markersize=4)
            ax.fill_between(eps, all_cap, alpha=0.3, color='#e74c3c')
            ax.set_xlabel('Evaluation Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Capacity Reward', fontsize=10, fontweight='bold')
            ax.set_title('Capacity Utilization', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Component breakdown
            ax = fig.add_subplot(gs[1, :])
            ax.set_facecolor('#fafafa')
            baseline = np.zeros(len(eps))
            ax.fill_between(eps, baseline, baseline + all_mdl,
                            color='#e74c3c', alpha=0.4, label='Middleware')
            ax.fill_between(eps, baseline + all_mdl, 
                            baseline + all_mdl + all_wait,
                            color='#f39c12', alpha=0.4, label='Wait')
            ax.fill_between(eps, baseline + all_mdl + all_wait,
                            baseline + all_mdl + all_wait + all_comp,
                            color='#27ae60', alpha=0.4, label='Completion')
            
            ax.set_xlabel('Evaluation Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Reward', fontsize=10, fontweight='bold')
            ax.set_title('Reward Component Breakdown', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(alpha=0.25, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, 'reward_components.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved {output_file}")
            plt.close()
    except Exception as e:
        print(f"[WARN] Could not generate component breakdown plot: {e}")
    
    print(f"\n[OK] All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
