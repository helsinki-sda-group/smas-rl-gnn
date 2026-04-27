#!/usr/bin/env python3
"""
Standalone script to plot training results from:
1. training_metrics_*.log (episode rewards and components)
2. train_output.txt (PPO training metrics)

Usage:
    python plot_training_results.py <metrics_log> <train_output> [--ma-window WINDOW]

Example:
    python plot_training_results.py training_metrics_v2000_ms1200_mwd240_mtd900_cap2.log train_output.txt
    python plot_training_results.py training_metrics_v2000_ms1200_mwd240_mtd900_cap2.log train_output.txt --ma-window 50
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from scipy.ndimage import uniform_filter1d
import re
from omegaconf import OmegaConf


def _normalize_metric_key(name: str) -> str | None:
    name = name.strip().lower()
    mapping = {
        "rew": "rew",
        "cap": "cap",
        "step": "step",
        "mdl": "deadline",
        "dln": "deadline",
        "deadline": "deadline",
        "wait": "wait",
        "trav": "travel",
        "travel": "travel",
        "comp": "comp",
        "nsv": "nsv",
    }
    return mapping.get(name)


def _extract_header_tokens(lines: list[str]) -> list[str]:
    for line in lines:
        if line.startswith("pol") and "|" in line:
            parts = re.split(r'\s*\|\s*', line)
            if len(parts) >= 2:
                return parts[1].split()
    return []


def parse_metrics_log(filepath):
    """Parse the training metrics log file containing episode rewards and components."""
    with open(filepath, 'r') as f:
        raw_lines = [l.strip() for l in f.readlines() if l.strip()]

    header_tokens = _extract_header_tokens(raw_lines)
    lines = [l for l in raw_lines if not l.startswith('pol')]

    data = []
    for line in lines:
        parts = re.split(r'\s*\|\s*', line)
        if len(parts) < 2:
            continue

        seg0 = parts[0].split()
        seg1 = parts[1].split()

        if len(seg0) < 2 or len(seg1) < 4:
            continue

        ts_val = None
        if len(seg0) >= 3:
            try:
                ts_val = int(seg0[2])
            except Exception:
                ts_val = None

        row = {
            "pol": int(seg0[0]),
            "seed": int(seg0[1]),
            "ts": ts_val,
            "rew": 0.0,
            "cap": 0.0,
            "step": 0.0,
            "deadline": 0.0,
            "wait": 0.0,
            "travel": 0.0,
            "comp": 0.0,
            "nsv": 0.0,
        }

        if header_tokens and len(header_tokens) == len(seg1):
            for name, val in zip(header_tokens, seg1):
                key = _normalize_metric_key(name)
                if key is None:
                    continue
                try:
                    row[key] = float(val)
                except Exception:
                    row[key] = 0.0
        else:
            try:
                row["rew"] = float(seg1[0])
                row["cap"] = float(seg1[1])
                row["step"] = float(seg1[2])
                row["deadline"] = float(seg1[3]) if len(seg1) > 3 else 0.0
                row["wait"] = float(seg1[4]) if len(seg1) > 4 else 0.0
                row["travel"] = float(seg1[5]) if len(seg1) > 5 else 0.0
                row["comp"] = float(seg1[6]) if len(seg1) > 6 else 0.0
                row["nsv"] = float(seg1[7]) if len(seg1) > 7 else 0.0
            except Exception:
                continue

        data.append(row)

    df = pd.DataFrame(data)
    if "ts" in df.columns and df["ts"].notna().any():
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    else:
        df = df.sort_values('pol').reset_index(drop=True)
    return df


def load_reward_type(config_path: str | None) -> str | None:
    if not config_path:
        return None
    try:
        cfg = OmegaConf.load(config_path)
        reward_type = str(getattr(getattr(cfg, "env", {}), "reward_params", {}).get("reward_type", ""))
        reward_type = reward_type.strip().lower()
        if reward_type in {"default", "missed_deadline"}:
            return "deadline"
        if reward_type in {"wait_travel", "wait+travel", "wait-travel", "travel"}:
            return "wait_travel"
        if reward_type:
            return reward_type
        return None
    except Exception:
        return None


def parse_train_output(filepath):
    """Parse the PPO training output log file."""
    with open(filepath, 'r') as f:
        log_txt = f.read()
    
    train_data = []
    blocks = re.split(r'-{30,}', log_txt)
    
    for block in blocks:
        row = {}
        patterns = {
            'iterations': r'iterations\s+\|\s+(\d+)',
            'total_timesteps': r'total_timesteps\s+\|\s+(\d+)',
            'ep_rew_mean': r'ep_rew_mean\s+\|\s+([\d.]+)',
            'approx_kl': r'approx_kl\s+\|\s+([\d.e+-]+)',
            'clip_fraction': r'clip_fraction\s+\|\s+([\d.]+)',
            'entropy_loss': r'entropy_loss\s+\|\s+([-\d.]+)',
            'explained_variance': r'explained_variance\s+\|\s+([-\d.]+)',
            'value_loss': r'value_loss\s+\|\s+([\d.e+]+)',
            'learning_rate': r'learning_rate\s+\|\s+([\d.e-]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, block)
            if match:
                row[key] = float(match.group(1))
        
        if 'iterations' in row and 'entropy_loss' in row:
            train_data.append(row)
    
    return pd.DataFrame(train_data)


def ma(data, window):
    """Moving average that preserves array length."""
    data = data.astype(float)
    window = min(window, len(data))
    
    # Use 'same' mode to keep the same length, then handle edges
    result = np.convolve(data, np.ones(window)/window, mode='same')
    
    # Fix edge artifacts by using 'nearest' style padding
    half_window = window // 2
    for i in range(half_window):
        result[i] = np.mean(data[:i+half_window+1])
        result[-(i+1)] = np.mean(data[-(i+half_window+1):])
    
    return result


def plot_reward_components(df, reward_type: str | None, output_file='reward_components.png'):
    """Create reward component breakdown plots."""
    use_ts = 'ts' in df.columns and df['ts'].notna().any()
    eps = df['ts'].values if use_ts else df['pol'].values
    rew = df['rew'].values
    deadline = df['deadline'].values
    wait = df['wait'].values
    travel = df['travel'].values
    comp = df['comp'].values

    if reward_type is None:
        reward_type = "wait_travel" if np.any(travel != 0.0) else "deadline"
    
    N = len(eps)
    
    fig = plt.figure(figsize=(18, 12), facecolor='#fafafa')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # ── Total reward ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor('#fafafa')
    
    # Adjust scatter size based on number of episodes
    scatter_size = 12 if N < 500 else 8 if N < 1500 else 6
    scatter_alpha = 0.2 if N < 500 else 0.15 if N < 1500 else 0.1
    
    #ax.scatter(eps, rew, s=scatter_size, alpha=scatter_alpha, color='#aec6cf', zorder=1, label='Raw')
    ax.plot(eps, ma(rew, 200), lw=2.5, color='#e74c3c', alpha=0.85, label='MA 20')
    
    if N > 1000:
        ax.plot(eps, ma(rew, 50), lw=3, color='#922b21', alpha=0.90, label='MA 50')
    
    ax.axhline(rew.mean(), color='#2c3e50', lw=2, ls='--', alpha=0.6,
              label=f'Mean: {rew.mean():.2f}')
    
    # Trend line
    z = np.polyfit(eps, rew, 1)
    ax.plot(eps, np.polyval(z, eps), lw=1.8, color='#27ae60', alpha=0.5,
            ls=':', label=f'Trend: {z[0]:+.5f}/ep')
    
    if use_ts:
        x_min, x_max = float(np.min(eps)), float(np.max(eps))
        pad = max(1.0, 0.02 * (x_max - x_min))
        ax.set_xlim(x_min - pad, x_max + pad)
    else:
        ax.set_xlim(-10, len(eps) + 10)
    ax.set_xlabel('Timestep' if use_ts else 'Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Reward', fontsize=11, fontweight='bold')
    ax.set_title(f'Training Reward - {N} Episodes', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Stats box
    q1m = rew[:N//4].mean()
    q4m = rew[3*N//4:].mean()
    last_n = min(100, N//10)
    txt = (f'Mean:  {rew.mean():.2f}    Last {last_n}:  {rew[-last_n:].mean():.2f}\n'
           f'Std:   {rew.std():.2f}    Q1→Q4:    {q1m:.2f}→{q4m:.2f} ({100*(q4m-q1m)/q1m:+.1f}%)')
    ax.text(0.98, 0.04, txt, transform=ax.transAxes, fontsize=9,
            va='bottom', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))
    
    # ── Completion reward ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('#fafafa')
    #ax.plot(eps, comp, 'o', markersize=scatter_size//2, alpha=scatter_alpha, color='#27ae60')
    ax.plot(eps, ma(comp, 20), lw=2.5, color='#27ae60', alpha=0.9)
    ax.axhline(comp.mean(), color='#16a085', lw=2, ls='--', alpha=0.6)
    ax.set_ylabel('Completion Reward', fontsize=10, fontweight='bold')
    ax.set_title(f'Completion (mean: +{comp.mean():.2f})', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── Penalty component ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('#fafafa')
    primary_penalty = travel if reward_type == "wait_travel" else deadline
    primary_label = "Travel Penalty" if reward_type == "wait_travel" else "Deadline Penalty"
    primary_color = '#9b59b6' if reward_type == "wait_travel" else '#e74c3c'
    ax.plot(eps, ma(primary_penalty, 20), lw=2.5, color=primary_color, alpha=0.9)
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.axhline(primary_penalty.mean(), color=primary_color, lw=2, ls='--', alpha=0.6)
    ax.set_ylabel(primary_label, fontsize=10, fontweight='bold')
    ax.set_title(f'{primary_label} (mean: {primary_penalty.mean():.2f})', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── Wait penalty ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor('#fafafa')
    #ax.plot(eps, wait, 'o', markersize=scatter_size//2, alpha=scatter_alpha, color='#f39c12')
    ax.plot(eps, ma(wait, 20), lw=2.5, color='#f39c12', alpha=0.9)
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.axhline(wait.mean(), color='#e67e22', lw=2, ls='--', alpha=0.6)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Wait Penalty', fontsize=10, fontweight='bold')
    ax.set_title(f'Wait (mean: {wait.mean():.2f})', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── Stacked area ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor('#fafafa')
    
    baseline = np.zeros(len(eps))
    if reward_type == "wait_travel":
        ax.fill_between(eps, baseline, baseline + travel,
                        color='#9b59b6', alpha=0.4, label='Travel')
        ax.fill_between(eps, baseline + travel, baseline + travel + wait,
                        color='#f39c12', alpha=0.4, label='Wait')
        ax.fill_between(eps, baseline + travel + wait, baseline + travel + wait + comp,
                        color='#27ae60', alpha=0.4, label='Completion')
    else:
        ax.fill_between(eps, baseline, baseline + deadline,
                        color='#e74c3c', alpha=0.4, label='Deadline')
        ax.fill_between(eps, baseline + deadline, baseline + deadline + wait,
                        color='#f39c12', alpha=0.4, label='Wait')
        ax.fill_between(eps, baseline + deadline + wait, baseline + deadline + wait + comp,
                        color='#27ae60', alpha=0.4, label='Completion')
    
    # Subsample total line for large datasets
    stride = max(1, N // 500)
    ax.plot(eps[::stride], rew[::stride], lw=1.5, color='#2c3e50', alpha=0.7, 
            label='Total', zorder=10)
    ax.axhline(0, color='black', lw=1, alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=10, fontweight='bold')
    ax.set_title(f'Component Breakdown ({reward_type})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_file}")


def plot_ppo_metrics(train_df, output_file='ppo_metrics.png'):
    """Create PPO training metrics plots."""
    if len(train_df) == 0:
        print("⚠ No PPO training data found in train_output.txt")
        return
    
    if 'total_timesteps' in train_df.columns and train_df['total_timesteps'].notna().any():
        iters = train_df['total_timesteps'].values
        x_label = 'Timesteps'
    else:
        raw_iters = train_df['iterations'].values
        iters = []
        offset = 0
        prev = None
        for val in raw_iters:
            if prev is not None and val <= prev:
                offset += prev
            iters.append(val + offset)
            prev = val
        iters = np.array(iters, dtype=float)
        x_label = 'Iteration'
    ev = train_df['explained_variance'].values
    vl = train_df['value_loss'].values
    ent = train_df['entropy_loss'].values
    kl = train_df['approx_kl'].values
    cf = train_df['clip_fraction'].values
    ep_rew = train_df['ep_rew_mean'].values
    
    fig = plt.figure(figsize=(18, 12), facecolor='#fafafa')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Adjust marker size based on number of iterations
    marker_size = 7 if len(iters) < 50 else 4 if len(iters) < 150 else 3
    
    # ── Episode reward ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor('#fafafa')
    
    ax.plot(iters, ep_rew, 'o-', lw=2, markersize=marker_size, color='#2c3e50', alpha=0.7)
    
    z = np.polyfit(iters, ep_rew, 1)
    ax.plot(iters, np.polyval(z, iters), '--', lw=2, color='#27ae60', alpha=0.6,
            label=f'Trend: {z[0]:+.5f}/iter')
    ax.axhline(ep_rew.mean(), color='#e74c3c', lw=2, ls='--', alpha=0.5)
    
    ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
    ax.set_title(f'PPO Training Progress - {len(iters)} Iterations', 
                fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Stats
    txt = (f'Start:  {ep_rew[0]:.2f}\n'
           f'End:    {ep_rew[-1]:.2f}\n'
           f'Change: {ep_rew[-1]-ep_rew[0]:+.2f} ({100*(ep_rew[-1]-ep_rew[0])/ep_rew[0]:+.1f}%)')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
            va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))
    
    # ── Explained variance ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('#fafafa')
    
    ax.axhspan(-0.25, 0, alpha=0.08, color='red')
    ax.axhspan(0, 0.5, alpha=0.08, color='gold')
    ax.axhspan(0.5, 1, alpha=0.08, color='green')
    
    ax.plot(iters, ev, 'o-', lw=2, markersize=marker_size, color='#e74c3c', alpha=0.7)
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.axhline(0.5, color='green', lw=1.5, ls='--', alpha=0.5)

    ax.set_ylim(-0.25, 1)
    
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Explained Variance', fontsize=10, fontweight='bold')
    ax.set_title('Value Function Quality', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── Value loss ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('#fafafa')
    
    ax.plot(iters, vl, 'o-', lw=2, markersize=marker_size, color='#3498db', alpha=0.7)
    
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Value Loss', fontsize=10, fontweight='bold')
    ax.set_title('Value Loss', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── Entropy ─────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor('#fafafa')
    
    ax.plot(iters, ent, 'o-', lw=2, markersize=marker_size, color='#9b59b6', alpha=0.7)
    ax.axhline(-1.0, color='orange', lw=1.5, ls='--', alpha=0.5, label='Caution')
    ax.axhline(-0.5, color='red', lw=1.5, ls='--', alpha=0.5, label='Collapse')
    
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Entropy Loss', fontsize=10, fontweight='bold')
    ax.set_title('Exploration (Entropy)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── KL + Clip fraction ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor('#fafafa')
    ax2 = ax.twinx()
    
    l1 = ax.plot(iters, kl, 'o-', lw=2, markersize=marker_size, color='#e67e22', 
                alpha=0.7, label='Approx KL')
    ax.axhline(0.01, color='#d35400', lw=1.5, ls='--', alpha=0.4, label='Target: 0.01')
    ax.axhline(0.001, color='#f39c12', lw=1, ls=':', alpha=0.3)
    
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Approx KL', fontsize=10, fontweight='bold', color='#e67e22')
    ax.tick_params(axis='y', labelcolor='#e67e22')
    ax.set_yscale('log')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    
    l2 = ax2.plot(iters, cf, 's-', lw=1.5, markersize=marker_size-1, color='#16a085', 
                 alpha=0.7, label='Clip Fraction')
    ax2.axhline(0.1, color='#1abc9c', lw=1.5, ls='--', alpha=0.4)
    ax2.set_ylabel('Clip Fraction', fontsize=10, fontweight='bold', color='#16a085')
    ax2.tick_params(axis='y', labelcolor='#16a085')
    ax2.spines['top'].set_visible(False)
    
    # Combined legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', fontsize=9)
    
    ax.set_title('Policy Update Magnitude', fontsize=11, fontweight='bold')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_file}")


def print_summary(df, train_df):
    """Print summary statistics."""
    N = len(df)
    rew = df['rew'].values
    
    print("\n" + "="*70)
    print(f"SUMMARY - {N} Episodes")
    print("="*70)
    
    print(f"\nReward Statistics:")
    print(f"  Mean:  {rew.mean():6.2f}")
    print(f"  Std:   {rew.std():5.2f}")
    print(f"  Range: [{rew.min():6.2f}, {rew.max():6.2f}]")
    
    q1m = rew[:N//4].mean()
    q4m = rew[3*N//4:].mean()
    print(f"\n  Q1 mean:  {q1m:.2f}")
    print(f"  Q4 mean:  {q4m:.2f}")
    print(f"  Q1→Q4:    {100*(q4m-q1m)/q1m:+.1f}%")
    
    if len(train_df) > 0:
        print(f"\nPPO Training:")
        print(f"  Iterations:  {len(train_df)}")
        print(f"  Reward:      {train_df['ep_rew_mean'].iloc[0]:.2f} → {train_df['ep_rew_mean'].iloc[-1]:.2f}")
        
        ev = train_df['explained_variance'].values
        print(f"  Expl. Var:   {ev[0]:.3f} → {ev[-1]:.3f}")
        print(f"  Positive EV: {100*np.mean(ev > 0):.0f}%")
        
        kl = train_df['approx_kl'].values
        print(f"  Approx KL:   [{kl.min():.2e}, {kl.max():.2e}]")
        print(f"  Frozen:      {100*np.mean(kl < 1e-6):.0f}%")
    
    print("="*70 + "\n")


def plot_reward_by_seed(df, ma_window=20, output_prefix='reward_seed'):
    """Create separate reward plots for each seed."""
    seeds = df['seed'].unique()
    print(f"\n  Found {len(seeds)} unique seeds: {sorted(seeds)}")
    use_ts = 'ts' in df.columns and df['ts'].notna().any()
    
    for seed in sorted(seeds):
        seed_df = df[df['seed'] == seed].copy()
        if use_ts:
            seed_df = seed_df.sort_values('ts').drop_duplicates(subset=['ts'], keep='last').reset_index(drop=True)
        else:
            seed_df = seed_df.sort_values('pol').reset_index(drop=True)
        
        eps = seed_df['ts'].values if use_ts else seed_df['pol'].values
        rew = seed_df['rew'].values
        
        if len(eps) == 0:
            continue
        
        N = len(eps)
        
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='#fafafa')
        ax.set_facecolor('#fafafa')
        
        # Adjust scatter size based on number of episodes
        scatter_size = 12 if N < 100 else 8 if N < 300 else 6
        scatter_alpha = 0.25 if N < 100 else 0.18 if N < 300 else 0.12
        
        #ax.scatter(eps, rew, s=scatter_size, alpha=scatter_alpha, color='#aec6cf', zorder=1, label='Raw')
        ax.plot(eps, ma(rew, ma_window), lw=2.5, color='#e74c3c', alpha=0.85, label=f'MA {ma_window}')
        
        # Mean line
        ax.axhline(rew.mean(), color='#2c3e50', lw=2, ls='--', alpha=0.6,
                  label=f'Mean: {rew.mean():.2f}')
        
        # Trend line
        z = np.polyfit(eps, rew, 1)
        ax.plot(eps, np.polyval(z, eps), lw=1.8, color='#27ae60', alpha=0.5,
                ls=':', label=f'Trend: {z[0]:+.5f}/ep')
        
        pad = 5 if not use_ts else max(1.0, 0.02 * (eps.max() - eps.min()))
        ax.set_xlim(eps.min() - pad, eps.max() + pad)
        ax.set_xlabel('Timestep' if use_ts else 'Episode', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'Training Reward - Seed {seed} ({N} Episodes)', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Stats box
        first_n = min(50, N//4)
        last_n = min(50, N//4)
        txt = (f'Mean:  {rew.mean():.2f}    First {first_n}:  {rew[:first_n].mean():.2f}\n'
               f'Std:   {rew.std():.2f}    Last {last_n}:   {rew[-last_n:].mean():.2f}\n'
               f'Range: [{rew.min():.2f}, {rew.max():.2f}]')
        ax.text(0.98, 0.04, txt, transform=ax.transAxes, fontsize=9,
                va='bottom', ha='right', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))
        
        output_file = f'{output_prefix}{seed}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot training results from metrics log and PPO output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('metrics_log', help='Path to training_metrics_*.log file')
    parser.add_argument('train_output', help='Path to train_output.txt file')
    parser.add_argument('--config', type=str, default='configs/rp_gnn.yaml',
                       help='Path to YAML config for reward_type (default: configs/rp_gnn.yaml)')
    parser.add_argument('--ma-window', type=int, default=20, 
                       help='Moving average window size for per-seed plots (default: 20)')
    
    args = parser.parse_args()
    
    print(f"Loading {args.metrics_log}...")
    df = parse_metrics_log(args.metrics_log)
    print(f"  ✓ Loaded {len(df)} episodes")
    
    print(f"\nLoading {args.train_output}...")
    train_df = parse_train_output(args.train_output)
    print(f"  ✓ Loaded {len(train_df)} training iterations")

    reward_type = load_reward_type(args.config)
    
    # Print summary
    print_summary(df, train_df)
    
    # Generate plots
    print("Generating plots...")
    plot_reward_components(df, reward_type=reward_type, output_file='reward_components.png')
    plot_ppo_metrics(train_df, output_file='ppo_metrics.png')
    
    print(f"\nGenerating per-seed plots (MA window: {args.ma_window})...")
    plot_reward_by_seed(df, ma_window=args.ma_window, output_prefix='reward_seed')
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
