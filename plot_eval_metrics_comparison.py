#!/usr/bin/env python
"""
Plot selected metrics from evaluation logs, comparing different noop values and evaluation types.

Usage:
    python plot_eval_metrics_comparison.py eval_noops --metrics pkr cmr --ma 10
    python plot_eval_metrics_comparison.py eval_noops --metrics all --ma 10
"""

import sys
import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Metric mapping: short name -> (full column name, display name, ylabel)
METRIC_MAP = {
    'pkr': ('pkr', 'Pickup Rate', 'Rate'),
    'pkvr': ('pkvr', 'Pickup Violated Rate', 'Rate'),
    'cmr': ('cmr', 'Completion Rate', 'Rate'),
    'mtt': ('mtt', 'Mean Travel Time', 'Time (s)'),
    'noop': ('noop', 'NOOP Fraction', 'Fraction'),
    'mcand': ('mcand', 'Mean Candidates per Taxi', 'Count'),
    'cne_fr': ('cne_fr', 'Candidate Non-Empty Fraction', 'Fraction'),
    'cne_mn': ('cne_mn', 'Candidate Mean Non-Empty', 'Count'),
    'ovrlap': ('ovrlap', 'Overlap Rate', 'Rate'),
    'shared': ('shared', 'Mean Shared Tasks per Step', 'Count'),
}


def parse_filename(filename):
    """
    Parse evaluation log filename to extract noop value and evaluation type.
    
    Format: evaluation_metrics_noop[value]_[type].log
    Returns: (noop_value, eval_type) or (None, None) if parsing fails
    """
    pattern = r'evaluation_metrics_noop([-+]?\d*\.?\d+)_(deterministic|stochastic)\.log'
    match = re.match(pattern, filename)
    if match:
        noop_value = float(match.group(1))
        eval_type = match.group(2)
        return noop_value, eval_type
    return None, None


def parse_metrics_log(filepath):
    """Parse the evaluation metrics log file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]

    # Find header: first line with at least 2 columns separated by '|'
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
        raise ValueError(f"Could not find header in metrics log file: {filepath}")

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
                    # Handle fraction format (e.g., "12/15" -> extract numerator/denominator or convert to float)
                    if '/' in val:
                        # For now, keep as string; can be parsed later if needed
                        row[col] = val
                    else:
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
    if window < 1:
        return data
    result = np.convolve(data, np.ones(window)/window, mode='same')
    half_window = window // 2
    for i in range(half_window):
        result[i] = np.mean(data[:i+1])
        result[-(i+1)] = np.mean(data[-(i+1):])
    return result


def plot_metric(metric_key, all_data, ma_window, output_dir):
    """
    Plot a single metric across all evaluation logs.
    
    Args:
        metric_key: short metric name (e.g., 'pkr')
        all_data: dict of {(noop_value, eval_type): dataframe}
        ma_window: moving average window size
        output_dir: directory to save plots
    """
    if metric_key not in METRIC_MAP:
        print(f"[WARN] Unknown metric: {metric_key}, skipping")
        return
    
    col_name, display_name, ylabel = METRIC_MAP[metric_key]
    
    # Check if metric exists in any dataframe
    has_metric = False
    for df in all_data.values():
        if col_name in df.columns:
            has_metric = True
            break
    
    if not has_metric:
        print(f"[WARN] Metric '{col_name}' not found in any log files, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    # Color schemes for different noop values and line styles for eval types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = {'deterministic': '-', 'stochastic': '--'}
    
    # Sort by noop value for consistent coloring
    sorted_keys = sorted(all_data.keys(), key=lambda x: (x[0], x[1]))
    
    for idx, (noop_val, eval_type) in enumerate(sorted_keys):
        df = all_data[(noop_val, eval_type)]
        
        if col_name not in df.columns:
            continue
        
        # Group by timestep and compute mean/std over seeds
        grouped = df.groupby('ts')[col_name].agg(['mean', 'std', 'count']).reset_index()
        ts = grouped['ts'].values
        means = grouped['mean'].values
        stds = grouped['std'].values
        counts = grouped['count'].values
        sems = stds / np.sqrt(counts)
        
        if len(means) == 0:
            continue
        
        color = colors[idx % len(colors)]
        linestyle = linestyles.get(eval_type, '-')
        label = f'noop={noop_val:.1f} ({eval_type})'
        
        # Plot mean with error bands
        ax.plot(ts, means, color=color, linestyle=linestyle, linewidth=2, alpha=0.8, label=label)
        ax.fill_between(ts, means - sems, means + sems, color=color, alpha=0.2)
        
        # Plot moving average if window > 1
        if ma_window > 1 and len(means) > 1:
            ma_vals = ma(means, ma_window)
            ax.plot(ts, ma_vals, color=color, linestyle=linestyle, linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{display_name} vs Training Steps', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{metric_key}_vs_timesteps.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot evaluation metrics from multiple evaluation logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_eval_metrics_comparison.py eval_noops --metrics pkr cmr --ma 10
  python plot_eval_metrics_comparison.py eval_noops --metrics all --ma 10
  python plot_eval_metrics_comparison.py eval_noops --metrics pkr pkvr mtt noop --ma 5
        """
    )
    parser.add_argument('folder', type=str, help='Folder containing evaluation_metrics_*.log files')
    parser.add_argument('--metrics', nargs='+', required=True, 
                        help='Metrics to plot (or "all" for all available metrics)')
    parser.add_argument('--ma', type=int, default=10, help='Moving average window size (default: 10)')
    
    args = parser.parse_args()
    
    folder = args.folder
    ma_window = args.ma
    
    # Determine which metrics to plot
    if 'all' in args.metrics:
        metrics_to_plot = list(METRIC_MAP.keys())
    else:
        metrics_to_plot = args.metrics
    
    print(f"Folder: {folder}")
    print(f"Metrics to plot: {', '.join(metrics_to_plot)}")
    print(f"Moving average window: {ma_window}")
    print()
    
    # Find all evaluation log files
    log_pattern = os.path.join(folder, 'evaluation_metrics_noop*.log')
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"[ERROR] No evaluation log files found in {folder}")
        sys.exit(1)
    
    print(f"Found {len(log_files)} evaluation log files:")
    
    # Parse all log files
    all_data = {}
    for log_file in log_files:
        filename = os.path.basename(log_file)
        noop_val, eval_type = parse_filename(filename)
        
        if noop_val is None or eval_type is None:
            print(f"  [SKIP] {filename} - could not parse filename")
            continue
        
        print(f"  Loading {filename} (noop={noop_val}, type={eval_type})...")
        try:
            df = parse_metrics_log(log_file)
            all_data[(noop_val, eval_type)] = df
            print(f"    Loaded {len(df)} records")
        except Exception as e:
            print(f"    [ERROR] Could not parse: {e}")
    
    if not all_data:
        print(f"[ERROR] No valid log files could be parsed")
        sys.exit(1)
    
    print()
    print(f"Successfully loaded {len(all_data)} evaluation logs")
    print()
    
    # Create output directory
    output_dir = os.path.join(folder, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each metric
    print("Generating plots...")
    for metric in metrics_to_plot:
        plot_metric(metric, all_data, ma_window, output_dir)
    
    print()
    print(f"[OK] All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
