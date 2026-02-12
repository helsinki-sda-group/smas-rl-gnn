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
        lines = [l.rstrip('\n') for l in f if l.strip()]

    # Find header: first line with at least 2 columns separated by '|' and containing 'rew' or 'reward' or 'pol' or 'seed'
    header_line = None
    header_idx = None
    for idx, line in enumerate(lines):
        line_stripped = line.strip().lower()
        if '|' in line and (
            'rew' in line_stripped or 'reward' in line_stripped or 'pol' in line_stripped or 'seed' in line_stripped
        ):
            # Must have at least 2 columns
            parts = [c.strip() for c in line.split('|') if c.strip()]
            if len(parts) >= 2:
                header_line = line
                header_idx = idx
                break

    if header_line is None:
        raise ValueError("Could not find header in metrics log file.")

    # Parse header columns: split first segment by whitespace, then others by whitespace as well
    segs = [seg.strip() for seg in header_line.split('|')]
    columns = []
    for i, seg in enumerate(segs):
        if i == 0:
            columns += seg.lower().split()
        else:
            columns += seg.lower().split()
    columns = [c for c in columns if c]

    # Read data lines
    data = []
    for line in lines[header_idx+1:]:
        if not line.strip() or line.strip().startswith('#'):
            continue
        segs = [seg.strip() for seg in line.split('|')]
        parts = []
        for i, seg in enumerate(segs):
            if i == 0:
                parts += seg.split()
            else:
                parts += seg.split()
        if len(parts) < 2:
            continue
        # Pad parts if missing columns
        if len(parts) < len(columns):
            parts += [''] * (len(columns) - len(parts))
        row = {}
        for col, val in zip(columns, parts):
            val = val.strip()
            # Try to cast to int or float if possible
            if col in ('episode', 'ts', 'seed', 'attempt'):
                try:
                    row[col] = int(val)
                except Exception:
                    row[col] = 0
            else:
                try:
                    row[col] = float(val)
                except Exception:
                    row[col] = val
        data.append(row)

    df = pd.DataFrame(data)
    # Use 'ts' if present, else fallback to 'timestep' or 'step' for sorting
    sort_col = None
    for candidate in ['ts', 'timestep', 'step']:
        if candidate in df.columns:
            sort_col = candidate
            break
    if sort_col is None:
        sort_col = df.columns[0]  # fallback to first column
    return df.sort_values(sort_col).reset_index(drop=True)


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
    import re
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Find the summary table (last one in file)
    summary_start = None
    for i, line in enumerate(lines):
        if line.startswith('pol') and 'rew±std' in line:
            summary_start = i
    if summary_start is not None:
        metric_names = ['rew', 'cap', 'step', 'mdl', 'wait', 'comp', 'nsv']
        for line in lines[summary_start+1:]:
            if not line or line.startswith('#'):
                break
            if '±' not in line:
                continue
            # Use regex to extract: policy name, then all value±std pairs
            m = re.match(r'^(\w+)', line)
            if not m:
                continue
            pol_name = m.group(1)
            # Find all value±std pairs (including negatives and decimals)
            pairs = re.findall(r'([-+]?\d*\.\d+|\d+)[\s]*±[\s]*([-+]?\d*\.\d+|\d+)', line)
            baseline = {}
            for idx, metric in enumerate(metric_names):
                if idx < len(pairs):
                    mean_str, std_str = pairs[idx]
                    baseline[f'{metric}_mean'] = float(mean_str)
                    baseline[f'{metric}_std'] = float(std_str)
            # For backward compatibility, also set 'mean' and 'std' for reward
            if 'rew_mean' in baseline and 'rew_std' in baseline:
                baseline['mean'] = baseline['rew_mean']
                baseline['std'] = baseline['rew_std']
            baselines[pol_name] = baseline
    return baselines


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_eval_results.py <evaluation_metrics.log> [--ma-window WINDOW] [--baseline-log BASELINE_LOG]")
        print("Example: python plot_eval_results.py eval_results/evaluation_20260206_231327/evaluation_metrics.log --baseline-log baseline_train_seeds_v2000_ms1200_mwd240_mtd900_cap2.log")
        sys.exit(1)
    
    metrics_log = sys.argv[1]
    ma_window = 10
    baseline_log = None
    baseline_std = False

    # Parse arguments
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--ma-window' and i + 1 < len(args):
            ma_window = int(args[i + 1])
            i += 2
        elif arg == '--baseline-log' and i + 1 < len(args):
            baseline_log = args[i + 1]
            i += 2
        elif arg == '--baseline-std':
            baseline_std = True
            i += 1
        else:
            i += 1
    
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
                if 'mean' in stats and 'std' in stats:
                    print(f"  {pol}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                else:
                    print(f"  {pol}: keys={list(stats.keys())} (no 'mean'/'std')")
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
    grouped = df.groupby('ts')['rew'].agg(['mean', 'std', 'count']).reset_index()
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
        print(f"Mean reward (overall): {df['rew'].mean():.4f}")
        print(f"Std reward (overall): {df['rew'].std():.4f}")

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
        random_baseline_mean = None
        for pol, stats in baselines.items():
            color = baseline_colors.get(pol, '#95a5a6')
            label = baseline_labels.get(pol, pol.capitalize())
            mean_val = stats['mean']
            std_val = stats['std']
            if pol == 'random':
                random_baseline_mean = mean_val
            # Horizontal line for mean
            ax.axhline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'{label} Baseline')
            # Dotted lines for ±1 std only if baseline_std is set
            if baseline_std:
                ax.axhline(mean_val + std_val, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
                ax.axhline(mean_val - std_val, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
        # Set ylim if no baseline_std and random baseline present
        if not baseline_std and random_baseline_mean is not None:
            ax.set_ylim(bottom=random_baseline_mean - 0.8)
    
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

    seed_data = df.groupby('seed')['rew'].agg(['mean', 'std', 'count']).reset_index()
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
        rew = seed_data['rew'].values
        ax.plot(ts, rew, 'o-', alpha=0.6, color='#3498db', markersize=5, label='Reward')
        # Moving average
        if len(rew) > 1:
            ma_rew = ma(rew, ma_window)
            ax.plot(ts, ma_rew, 'r-', lw=2.5, alpha=0.7, label=f'Moving Average (w={ma_window})')
        # Add baseline horizontal lines
        if baselines:
            baseline_colors = {'random': '#d62728', 'greedy': '#ff7f0e', 'unique': '#8c564b'}
            baseline_labels = {'random': 'Random', 'greedy': 'Greedy', 'unique': 'Greedy-Unique'}
            for pol, stats in baselines.items():
                mean_val = stats.get('mean', None)
                if mean_val is not None:
                    color = baseline_colors.get(pol, '#95a5a6')
                    label = baseline_labels.get(pol, pol.capitalize())
                    ax.axhline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'{label} Baseline')
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
    
    # --- Additional Plots: reward_mdl, reward_wait, reward_comp ---
    for reward_key, ylabel, fname, baseline_key, color in [
        ("mdl", "Middleware Reward", "reward_mdl_vs_timesteps.png", "mdl", "#e74c3c"),
        ("wait", "Wait Reward", "reward_wait_vs_timesteps.png", "wait", "#f39c12"),
        ("comp", "Completion Reward", "reward_comp_vs_timesteps.png", "comp", "#27ae60"),
    ]:
        grouped = df.groupby('ts')[reward_key].agg(['mean', 'std', 'count']).reset_index()
        ts = grouped['ts'].values
        means = grouped['mean'].values
        sems = grouped['std'].values / np.sqrt(grouped['count'].values)

        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
        ax.set_facecolor('#fafafa')
        ax.errorbar(ts, means, yerr=sems, fmt='o-', alpha=0.6, label=f'Mean {reward_key} (Trained)', color=color, capsize=5, markersize=6)
        if len(means) > 1:
            ma_vals = ma(means, ma_window)
            ax.plot(ts, ma_vals, 'r-', lw=2.5, alpha=0.7, label=f'Moving Average (w={ma_window})')

        # Baseline horizontal lines (if available)
        if baselines:
            ts_range = [ts.min(), ts.max()] if len(ts) else [0, 1]
            baseline_colors = {'random': '#d62728', 'greedy': '#ff7f0e', 'unique': '#8c564b'}
            baseline_labels = {'random': 'Random', 'greedy': 'Greedy', 'unique': 'Greedy-Unique'}
            for pol, stats in baselines.items():
                mean_val = stats.get(f'{reward_key}_mean', None)
                std_val = stats.get(f'{reward_key}_std', None)
                if mean_val is not None:
                    color2 = baseline_colors.get(pol, '#95a5a6')
                    label2 = baseline_labels.get(pol, pol.capitalize())
                    ax.axhline(mean_val, color=color2, linestyle='--', linewidth=2.5, alpha=0.9, label=f'{label2} Baseline')
                    if baseline_std and std_val is not None:
                        ax.axhline(mean_val + std_val, color=color2, linestyle=':', linewidth=1.5, alpha=0.7)
                        ax.axhline(mean_val - std_val, color=color2, linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{ylabel} vs Training Steps', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        output_file = os.path.join(output_dir, fname)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved {output_file}")
        plt.close()
    print(f"\n[OK] All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
