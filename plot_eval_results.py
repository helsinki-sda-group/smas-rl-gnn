#!/usr/bin/env python
"""
Plot evaluation results from evaluation_metrics.log
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors
import re


COORDINATION_METRICS = {
    "ecr": {
        "label": "Empty Candidate Rate",
        "filename": "empty_cand_rate_vs_ts.png",
    },
    "unop": {
        "label": "Unforced NOOP Rate",
        "filename": "unforced_noop_rate_vs_ts.png",
    },
    "ncpr": {
        "label": "Nonconflicting Proposal Rate",
        "filename": "nonconf_prop_rate_vs_ts.png",
    },
    "psur": {
        "label": "Proposal Survival Rate",
        "filename": "prop_survival_rate_vs_ts.png",
    },
    "offpr": {
        "label": "Off-Proposal Assignment Rate",
        "filename": "offprop_assign_rate_vs_ts.png",
    },
}

CONFLICT_METRICS = {
    "ctot": {
        "label": "Conflicts Total",
        "filename": "conflicts_total_vs_ts.png",
        "ylim": None,
    },
    "crat": {
        "label": "Conflict Ratio",
        "filename": "conflict_ratio_vs_ts.png",
        "ylim": (-0.02, 1.02),
    },
    "catx": {
        "label": "Avg Taxis per Conflict",
        "filename": "avg_taxis_per_conflict_vs_ts.png",
        "ylim": None,
    },
}

REWARD_METRICS = {
    "rew": {"label": "Mean Evaluation Reward", "filename": "reward_vs_timesteps.png"},
    "trav": {"label": "Travel Reward", "filename": "reward_trav_vs_timesteps.png"},
    "dln": {"label": "Deadline Reward", "filename": "reward_dln_vs_timesteps.png"},
    "wait": {"label": "Wait Reward", "filename": "reward_wait_vs_timesteps.png"},
    "comp": {"label": "Completion Reward", "filename": "reward_comp_vs_timesteps.png"},
}

BASELINE_COLOR_OVERRIDES = {
    "random": "#D62728",
    "unique": "#8C564B",
    "greedy": "#FF7F0E",
    "pickup_distance": "#1F77B4",
    "pickup_deadline": "#2CA02C",
    "pickup_deadline_distance": "#17BECF",
    "predicted_reward": "#9467BD",
    "predicted_reward_joint": "#E377C2",
    "predicted_reward_joint_competition": "#BCBD22",
    "proposal_joint_competition": "#7F7F7F",
    "capacity": "#0072B2",
    "closest": "#56B4E9",
    "closest_then_capacity": "#009E73",
    "hungarian": "#D55E00",
}

BASELINE_LABEL_OVERRIDES = {
    "random": "Random",
    "unique": "Greedy-Unique",
    "greedy": "Greedy",
    "pickup_distance": "Pickup-Distance",
    "pickup_deadline": "Pickup-Deadline",
    "pickup_deadline_distance": "Pickup-Deadline-Distance",
    "predicted_reward": "Predicted-Reward",
    "predicted_reward_joint": "Predicted-Reward-Joint",
    "predicted_reward_joint_competition": "Predicted-Reward-Joint-Competition",
    "proposal_joint_competition": "Proposal-Joint-Competition",
    "closest_then_capacity": "Closest-Then-Capacity",
    "hungarian": "Hungarian",
}

BASELINE_ALIAS_MAP = {
    "ctc": "closest_then_capacity",
    "closest_than_capacity": "closest_then_capacity",
    "reward_joint": "predicted_reward_joint",
}


def normalize_baseline_policy_name(policy_name):
    raw = str(policy_name).strip().lower()
    return BASELINE_ALIAS_MAP.get(raw, raw)


def build_baseline_styles(policy_names):
    """Build deterministic color/label styles for baseline policies."""
    styles = {}
    fallback_palette = [mcolors.to_hex(c) for c in plt.get_cmap('tab20').colors]
    used_colors = set(BASELINE_COLOR_OVERRIDES.values())
    fallback_idx = 0

    for policy_name in sorted(policy_names, key=lambda x: str(x).lower()):
        canonical = normalize_baseline_policy_name(policy_name)
        color = BASELINE_COLOR_OVERRIDES.get(canonical)
        if color is None:
            while fallback_palette[fallback_idx % len(fallback_palette)] in used_colors:
                fallback_idx += 1
            color = fallback_palette[fallback_idx % len(fallback_palette)]
            fallback_idx += 1
        used_colors.add(color)
        label = BASELINE_LABEL_OVERRIDES.get(canonical, canonical.replace('_', '-').title())
        styles[str(policy_name)] = {"color": color, "label": label}
    return styles


def metric_plot_spec(metric_key):
    """Return display metadata for a metric key."""
    if metric_key in REWARD_METRICS:
        spec = REWARD_METRICS[metric_key]
        return spec['label'], spec['filename'], None
    if metric_key in COORDINATION_METRICS:
        spec = COORDINATION_METRICS[metric_key]
        return spec['label'], spec['filename'], (-0.02, 1.02)
    if metric_key in CONFLICT_METRICS:
        spec = CONFLICT_METRICS[metric_key]
        return spec['label'], spec['filename'], spec.get('ylim')
    return metric_key, f'{metric_key}_vs_timesteps.png', None


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
    # Preferred path: parse per-row metrics and compute mean/std by policy.
    # This supports newly added metrics such as ecr/unop/ncpr/psur/offpr.
    try:
        df = parse_metrics_log(filepath)
        if not df.empty and 'pol' in df.columns:
            # Prefer actual per-seed baseline rows and ignore summary/table artifacts.
            # Baseline logs commonly use ts=0, so do not filter by ts>0.
            if 'seed' in df.columns:
                seed_num = pd.to_numeric(df['seed'], errors='coerce').fillna(0)
                df_seed_rows = df[seed_num > 0].copy()
                if not df_seed_rows.empty:
                    df = df_seed_rows
            metric_names = ['rew', 'cap', 'step', 'dln', 'wait', 'trav', 'comp', 'nsv', 'ecr', 'unop', 'ncpr', 'psur', 'offpr', 'ctot', 'crat', 'catx']
            available_metrics = [m for m in metric_names if m in df.columns]
            out = {}
            if available_metrics:
                for pol in sorted(str(p) for p in df['pol'].dropna().unique()):
                    if pol.upper() in {'MEAN', 'STD'}:
                        continue
                    sub = df[df['pol'].astype(str) == pol]
                    if sub.empty:
                        continue
                    stats = {}
                    for m in available_metrics:
                        vals = pd.to_numeric(sub[m], errors='coerce').dropna()
                        if len(vals) == 0:
                            continue
                        stats[f'{m}_mean'] = float(vals.mean())
                        stats[f'{m}_std'] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
                    if 'rew_mean' in stats and 'rew_std' in stats:
                        stats['mean'] = stats['rew_mean']
                        stats['std'] = stats['rew_std']
                    if stats:
                        out[pol] = stats
            if out:
                return out
    except Exception:
        pass

    # Fallback path: parse legacy summary rows that use ± formatting.
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
        metric_names = ['rew', 'cap', 'step', 'dln', 'wait', 'trav', 'comp', 'nsv', 'ecr', 'unop', 'ncpr', 'psur', 'offpr', 'ctot', 'crat', 'catx']
        for line in lines[summary_start+1:]:
            if not line or line.startswith('#'):
                break
            # Keep fallback strict: only parse compact table rows, not narrative summaries.
            if '|' not in line:
                continue
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


def parse_compare_log_spec(spec):
    """Parse LABEL=PATH compare-log specs."""
    if '=' in spec:
        label, path = spec.split('=', 1)
        return label, path
    path = spec
    label = os.path.basename(os.path.dirname(os.path.dirname(path))) or os.path.basename(path)
    return label, path


def group_metric_by_ts(df, metric_key):
    """Aggregate a metric by timestep within one evaluation log."""
    grouped = df.groupby('ts')[metric_key].agg(['mean', 'std', 'count']).reset_index()
    grouped['std'] = grouped['std'].fillna(0.0)
    return grouped


def infer_metric_key_from_csv_path(csv_path):
    """Infer metric key from exported *_data.csv filename."""
    stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
    base = stem[:-5] if stem.endswith('_data') else stem

    if base in {'reward_vs_timesteps', 'rew_vs_timesteps'}:
        return 'rew'

    m = re.match(r'^(?:reward_)?(rew|trav|dln|wait|comp)_vs_timesteps$', base)
    if m:
        return m.group(1)

    coord_stems = {os.path.splitext(v['filename'])[0]: k for k, v in COORDINATION_METRICS.items()}
    if base in coord_stems:
        return coord_stems[base]

    conflict_stems = {os.path.splitext(v['filename'])[0]: k for k, v in CONFLICT_METRICS.items()}
    if base in conflict_stems:
        return conflict_stems[base]

    return None


def load_metric_series_from_csv(csv_path):
    """Load one metric CSV as series dicts with keys: label, mean, std."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    if 'ts' not in df.columns:
        for alt in ('timestep', 'step'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'ts'})
                break
    if 'ts' not in df.columns:
        return []

    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    df = df.dropna(subset=['ts'])
    if df.empty:
        return []

    # Long format: ts + mean (+ optional label/std/count)
    if 'mean' in df.columns:
        df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
        df = df.dropna(subset=['mean'])
        if df.empty:
            return []
        if 'label' in df.columns:
            out = []
            for label, group in df.groupby('label'):
                mean_series = group.groupby('ts')['mean'].mean().sort_index()
                std_series = None
                if 'std' in group.columns:
                    group_std = pd.to_numeric(group['std'], errors='coerce')
                    group_with_std = group.assign(_std=group_std)
                    std_series = group_with_std.groupby('ts')['_std'].mean().sort_index()
                    std_series = std_series.reindex(mean_series.index)
                if not mean_series.empty:
                    out.append({'label': str(label), 'mean': mean_series, 'std': std_series})
            return out
        mean_series = df.groupby('ts')['mean'].mean().sort_index()
        std_series = None
        if 'std' in df.columns:
            df_std = pd.to_numeric(df['std'], errors='coerce')
            df_with_std = df.assign(_std=df_std)
            std_series = df_with_std.groupby('ts')['_std'].mean().sort_index()
            std_series = std_series.reindex(mean_series.index)
        return [{'label': 'Mean', 'mean': mean_series, 'std': std_series}] if not mean_series.empty else []

    # Wide format: ts + one column per run/label.
    candidate_cols = []
    excluded = {'ts', 'std', 'count', 'count_runs'}
    for col in df.columns:
        if col in excluded:
            continue
        num_vals = pd.to_numeric(df[col], errors='coerce')
        if num_vals.notna().any():
            candidate_cols.append(col)

    out = []
    for col in candidate_cols:
        vals = pd.to_numeric(df[col], errors='coerce')
        tmp = pd.DataFrame({'ts': df['ts'], 'val': vals}).dropna(subset=['val'])
        if tmp.empty:
            continue
        series = tmp.groupby('ts')['val'].mean().sort_index()
        if not series.empty:
                out.append({'label': str(col), 'mean': series, 'std': None})
    return out


def plot_metric_from_csv(csv_path, metric_key, output_png, baselines, baseline_std, mean_run):
    """Plot one metric from an exported *_data.csv file."""
    series_items = load_metric_series_from_csv(csv_path)
    if not series_items:
        print(f"[INFO] No usable series in {csv_path} — skipping")
        return False

    ylabel, _, y_limits = metric_plot_spec(metric_key)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')

    if mean_run and len(series_items) > 1:
        frames = [pd.DataFrame({entry['label']: entry['mean']}).sort_index() for entry in series_items]
        merged = pd.concat(frames, axis=1, sort=True).sort_index()
        run_mean = merged.mean(axis=1, skipna=True)
        run_std = merged.std(axis=1, skipna=True).fillna(0.0)
        ts = run_mean.index.to_numpy(dtype=float)
        ax.plot(ts, run_mean.values, color='#3498db', linewidth=2.5, alpha=0.95, label='Mean Across Runs')
        ax.fill_between(ts, (run_mean - run_std).values, (run_mean + run_std).values,
                        color='#3498db', alpha=0.2, label='Run Std')
    else:
        cmap = plt.get_cmap('tab10')
        for idx, entry in enumerate(series_items):
            label = str(entry['label'])
            mean_series = entry['mean']
            ts = mean_series.index.to_numpy(dtype=float)
            vals = mean_series.to_numpy(dtype=float)
            color = cmap(idx % 10)
            ax.plot(ts, vals, linewidth=2.0, alpha=0.9, color=color, label=label)

            std_series = entry.get('std')
            if std_series is not None:
                std_vals = pd.to_numeric(std_series, errors='coerce').to_numpy(dtype=float)
                std_vals = np.nan_to_num(std_vals, nan=0.0)
                if np.any(std_vals > 0):
                    ax.fill_between(ts, vals - std_vals, vals + std_vals, color=color, alpha=0.18, label=f'{label} Std')

    add_baseline_lines(ax, baselines, metric_key, baseline_std)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    if mean_run and len(series_items) > 1:
        mode_label = 'Mean ± Std Across Runs'
    elif any(item.get('std') is not None for item in series_items):
        mode_label = 'Series from CSV (with Std)'
    else:
        mode_label = 'Series from CSV'
    ax.set_title(f'{ylabel} vs Training Steps ({mode_label})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {output_png}")
    return True


def plot_from_csv_directory(csv_dir, output_dir, baselines, baseline_std, mean_run):
    """Regenerate plots from existing *_data.csv files in a directory tree."""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '**', '*_data.csv'), recursive=True))
    if not csv_files:
        raise ValueError(f'No *_data.csv files found under: {csv_dir}')

    plotted = 0
    for csv_path in csv_files:
        metric_key = infer_metric_key_from_csv_path(csv_path)
        if metric_key is None:
            continue

        rel_path = os.path.relpath(csv_path, csv_dir)
        rel_dir = os.path.dirname(rel_path)
        png_name = os.path.basename(csv_path).replace('_data.csv', '.png')
        output_png = os.path.join(output_dir, rel_dir, png_name)
        if plot_metric_from_csv(csv_path, metric_key, output_png, baselines, baseline_std, mean_run):
            plotted += 1

    if plotted == 0:
        raise ValueError(
            'No plottable CSV files were recognized. Expected names like '
            "reward_vs_timesteps_data.csv, reward_wait_vs_timesteps_data.csv, "
            "coord/*_data.csv, conflict/*_data.csv, or rew_vs_timesteps_data.csv."
        )
    print(f"[OK] Generated {plotted} plot(s) from CSV data in {csv_dir}")


def add_baseline_lines(ax, baselines, reward_key, baseline_std):
    """Add baseline mean lines and optional std bands for a metric."""
    if not baselines:
        return

    styles = build_baseline_styles(list(baselines.keys()))
    mean_key = 'mean' if reward_key == 'rew' else f'{reward_key}_mean'
    std_key = 'std' if reward_key == 'rew' else f'{reward_key}_std'

    for pol, stats in baselines.items():
        mean_val = stats.get(mean_key)
        std_val = stats.get(std_key)
        if mean_val is None:
            continue
        style = styles.get(str(pol), {"color": '#95a5a6', "label": str(pol).capitalize()})
        color = style['color']
        label = style['label']
        ax.axhline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'{label} Baseline')
        if baseline_std and std_val is not None:
            ax.axhline(mean_val + std_val, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axhline(mean_val - std_val, color=color, linestyle=':', linewidth=1.5, alpha=0.7)


def plot_aggregate_metric(
    run_frames,
    reward_key,
    ylabel,
    fname,
    output_dir,
    baselines,
    baseline_std,
    mean_run,
    y_limits=None,
    use_baselines=True,
    subdirectory=None,
):
    """Plot one metric across multiple evaluation runs."""
    available = []
    for label, df in run_frames:
        if reward_key not in df.columns:
            continue
        grouped = group_metric_by_ts(df, reward_key)
        series = grouped[['ts', 'mean']].rename(columns={'mean': label}).set_index('ts')
        available.append(series)

    if not available:
        print(f"[INFO] Column '{reward_key}' not present in any compare log — skipping {fname}")
        return

    merged = pd.concat(available, axis=1, sort=True).sort_index()
    ts = merged.index.to_numpy()

    save_dir = os.path.join(output_dir, subdirectory) if subdirectory else output_dir
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')

    if mean_run:
        run_mean = merged.mean(axis=1, skipna=True)
        run_std = merged.std(axis=1, skipna=True).fillna(0.0)
        ax.plot(ts, run_mean.values, color='#3498db', linewidth=2.5, alpha=0.95, label='Mean Across Runs')
        ax.fill_between(ts, (run_mean - run_std).values, (run_mean + run_std).values,
                        color='#3498db', alpha=0.2, label='Run Std')
        export_df = pd.DataFrame({
            'ts': ts,
            'mean': run_mean.values,
            'std': run_std.values,
            'count_runs': merged.notna().sum(axis=1).values,
        })
    else:
        cmap = plt.get_cmap('tab10')
        for idx, col in enumerate(merged.columns):
            ax.plot(ts, merged[col].values, linewidth=2.0, alpha=0.9,
                    color=cmap(idx % 10), label=col)
        export_df = merged.reset_index()

    if use_baselines:
        add_baseline_lines(ax, baselines, reward_key, baseline_std)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    title_mode = 'Mean ± Std Across Evaluation Runs' if mean_run else 'Separate Evaluation Runs'
    ax.set_title(f'{ylabel} vs Training Steps ({title_mode})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_file = os.path.join(save_dir, fname)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved {output_file}")
    plt.close()

    csv_name = os.path.splitext(fname)[0] + '_data.csv'
    export_path = os.path.join(save_dir, csv_name)
    export_df.to_csv(export_path, index=False)
    print(f"[OK] Saved {export_path}")


def _plot_single_coordination_metric(df, metric_key, spec, ma_window, coord_dir, baselines, baseline_std):
    if metric_key not in df.columns:
        print(f"[INFO] Column '{metric_key}' not present — skipping {spec['label']} plot")
        return False

    grouped = group_metric_by_ts(df, metric_key)
    if grouped.empty:
        print(f"[INFO] Column '{metric_key}' has no grouped rows — skipping {spec['label']} plot")
        return False

    ts = grouped['ts'].values
    means = grouped['mean'].values
    sems = grouped['std'].values / np.sqrt(np.maximum(grouped['count'].values, 1))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    ax.errorbar(
        ts,
        means,
        yerr=sems,
        fmt='o-',
        alpha=0.6,
        color='#2980b9',
        capsize=5,
        markersize=6,
        label=f"Mean {spec['label']}",
    )
    if len(means) > 1:
        ma_vals = ma(means, ma_window)
        ax.plot(ts, ma_vals, 'r-', lw=2.5, alpha=0.7, label=f'Moving Average (w={ma_window})')

    # Baselines for coordination metrics are shown when available in baseline log.
    add_baseline_lines(ax, baselines, metric_key, baseline_std)

    ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax.set_ylabel(spec['label'], fontsize=11, fontweight='bold')
    ax.set_title(f"{spec['label']} vs Training Steps", fontsize=12, fontweight='bold')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(coord_dir, spec['filename'])
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out_png}")

    out_csv = os.path.join(coord_dir, f"{os.path.splitext(spec['filename'])[0]}_data.csv")
    grouped[['ts', 'mean', 'std', 'count']].to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")
    return True


def _plot_single_coordination_overview(df, ma_window, coord_dir):
    available = [k for k in COORDINATION_METRICS if k in df.columns]
    if not available:
        print("[INFO] No coordination columns present — skipping coordination overview")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='#fafafa')
    axes_arr = axes.ravel()

    for idx, key in enumerate(available):
        spec = COORDINATION_METRICS[key]
        grouped = group_metric_by_ts(df, key)
        ts = grouped['ts'].values
        means = grouped['mean'].values
        sems = grouped['std'].values / np.sqrt(np.maximum(grouped['count'].values, 1))

        ax = axes_arr[idx]
        ax.set_facecolor('#fafafa')
        ax.errorbar(ts, means, yerr=sems, fmt='o-', alpha=0.6, color='#2980b9', capsize=4, markersize=4)
        if len(means) > 1:
            ma_vals = ma(means, ma_window)
            ax.plot(ts, ma_vals, 'r-', lw=2.0, alpha=0.7)
        ax.set_title(spec['label'], fontsize=10, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=9)
        ax.set_ylabel(spec['label'], fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(len(available), len(axes_arr)):
        fig.delaxes(axes_arr[idx])

    fig.tight_layout()
    out_png = os.path.join(coord_dir, 'coord_metrics_vs_ts.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def _plot_single_conflict_metric(df, metric_key, spec, ma_window, conflict_dir, baselines, baseline_std):
    if metric_key not in df.columns:
        print(f"[INFO] Column '{metric_key}' not present — skipping {spec['label']} plot")
        return False

    grouped = group_metric_by_ts(df, metric_key)
    if grouped.empty:
        print(f"[INFO] Column '{metric_key}' has no grouped rows — skipping {spec['label']} plot")
        return False

    ts = grouped['ts'].values
    means = grouped['mean'].values
    sems = grouped['std'].values / np.sqrt(np.maximum(grouped['count'].values, 1))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    ax.errorbar(
        ts,
        means,
        yerr=sems,
        fmt='o-',
        alpha=0.6,
        color='#8e44ad',
        capsize=5,
        markersize=6,
        label=f"Mean {spec['label']}",
    )
    if len(means) > 1:
        ma_vals = ma(means, ma_window)
        ax.plot(ts, ma_vals, 'r-', lw=2.5, alpha=0.7, label=f'Moving Average (w={ma_window})')

    add_baseline_lines(ax, baselines, metric_key, baseline_std)

    if spec.get('ylim') is not None:
        ax.set_ylim(*spec['ylim'])

    ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax.set_ylabel(spec['label'], fontsize=11, fontweight='bold')
    ax.set_title(f"{spec['label']} vs Training Steps", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(conflict_dir, spec['filename'])
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved {out_png}")

    out_csv = os.path.join(conflict_dir, f"{os.path.splitext(spec['filename'])[0]}_data.csv")
    grouped[['ts', 'mean', 'std', 'count']].to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")
    return True


def _plot_compare_coordination_overview(run_frames, output_dir, mean_run):
    available = []
    for key in COORDINATION_METRICS:
        if any(key in df.columns for _, df in run_frames):
            available.append(key)
    if not available:
        print("[INFO] No coordination columns present in compare logs — skipping coordination overview")
        return

    coord_dir = os.path.join(output_dir, 'coord')
    os.makedirs(coord_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='#fafafa')
    axes_arr = axes.ravel()

    for idx, key in enumerate(available):
        spec = COORDINATION_METRICS[key]
        ax = axes_arr[idx]
        ax.set_facecolor('#fafafa')

        plotted_any = False
        available_series = []
        for label, df in run_frames:
            if key not in df.columns:
                continue
            grouped = group_metric_by_ts(df, key)
            series = grouped[['ts', 'mean']].rename(columns={'mean': label}).set_index('ts')
            available_series.append(series)

        if available_series:
            merged = pd.concat(available_series, axis=1, sort=True).sort_index()
            ts = merged.index.to_numpy(dtype=float)
            if mean_run:
                run_mean = merged.mean(axis=1, skipna=True)
                run_std = merged.std(axis=1, skipna=True).fillna(0.0)
                ax.plot(ts, run_mean.values, color='#3498db', linewidth=2.2, alpha=0.95, label='Mean Across Runs')
                ax.fill_between(ts, (run_mean - run_std).values, (run_mean + run_std).values, color='#3498db', alpha=0.2)
            else:
                cmap = plt.get_cmap('tab10')
                for j, col in enumerate(merged.columns):
                    ax.plot(ts, merged[col].values, linewidth=1.8, alpha=0.9, color=cmap(j % 10), label=str(col))
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        ax.set_title(spec['label'], fontsize=10, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=9)
        ax.set_ylabel(spec['label'], fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=7, loc='best')

    for idx in range(len(available), len(axes_arr)):
        fig.delaxes(axes_arr[idx])

    fig.tight_layout()
    out_png = os.path.join(coord_dir, 'coord_metrics_vs_ts.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def plot_aggregate_runs(compare_logs, output_dir, baselines, baseline_std, mean_run):
    """Plot family-level comparisons across multiple evaluation runs."""
    run_frames = []
    for spec in compare_logs:
        label, path = parse_compare_log_spec(spec)
        if not os.path.exists(path):
            print(f"[WARN] Compare log not found: {path}")
            continue
        df = parse_metrics_log(path)
        print(f"Loaded compare log for {label}: {path} ({len(df)} rows)")
        run_frames.append((label, df))

    if not run_frames:
        raise ValueError('No valid compare logs were provided.')

    plot_aggregate_metric(
        run_frames,
        'rew',
        'Mean Evaluation Reward',
        'reward_vs_timesteps.png',
        output_dir,
        baselines,
        baseline_std,
        mean_run,
    )

    for reward_key, ylabel, fname in [
        ('trav', 'Travel Reward', 'reward_trav_vs_timesteps.png'),
        ('dln', 'Deadline Reward', 'reward_dln_vs_timesteps.png'),
        ('wait', 'Wait Reward', 'reward_wait_vs_timesteps.png'),
        ('comp', 'Completion Reward', 'reward_comp_vs_timesteps.png'),
    ]:
        plot_aggregate_metric(
            run_frames,
            reward_key,
            ylabel,
            fname,
            output_dir,
            baselines,
            baseline_std,
            mean_run,
        )

    for metric_key, spec in COORDINATION_METRICS.items():
        plot_aggregate_metric(
            run_frames,
            metric_key,
            spec['label'],
            spec['filename'],
            output_dir,
            baselines,
            baseline_std,
            mean_run,
            y_limits=(-0.02, 1.02),
            use_baselines=True,
            subdirectory='coord',
        )

    for metric_key, spec in CONFLICT_METRICS.items():
        plot_aggregate_metric(
            run_frames,
            metric_key,
            spec['label'],
            spec['filename'],
            output_dir,
            baselines,
            baseline_std,
            mean_run,
            y_limits=spec.get('ylim'),
            use_baselines=True,
            subdirectory='conflict',
        )

    _plot_compare_coordination_overview(run_frames, output_dir, mean_run)


def main():
    has_help_flag = any(a in ('-h', '--help') for a in sys.argv[1:])
    if len(sys.argv) < 2 or has_help_flag:
        print("Usage: python plot_eval_results.py <evaluation_metrics.log> [--ma-window WINDOW] [--ma WINDOW] [--baseline-log BASELINE_LOG] [--baseline-std] [--output-dir DIR]")
        print("   or: python plot_eval_results.py --compare-log LABEL=PATH [--compare-log LABEL=PATH ...] [--mean-run] [--baseline-log BASELINE_LOG] [--output-dir DIR]")
        print("   or: python plot_eval_results.py --csv-dir DIR [--mean-run] [--baseline-log BASELINE_LOG] [--baseline-std] [--output-dir DIR]")
        print("Example: python plot_eval_results.py eval_results/evaluation_20260206_231327/evaluation_metrics.log --baseline-log baseline_train_seeds_v2000_ms1200_mwd240_mtd900_cap2.log")
        print("Example: python plot_eval_results.py --csv-dir ablation_results/eval_comp_plots --baseline-log metrics_v15_ms1200_mwd240_mtd900_cap2.log")
        sys.exit(0 if has_help_flag else 1)

    metrics_log = None
    ma_window = 10
    baseline_log = None
    baseline_std = False
    output_dir_arg = None
    compare_logs = []
    mean_run = False
    csv_dir = None

    # Parse arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ('--ma-window', '--ma') and i + 1 < len(args):
            ma_window = int(args[i + 1])
            i += 2
        elif arg in ('--baseline-log', '-baseline-log') and i + 1 < len(args):
            baseline_log = args[i + 1]
            i += 2
        elif arg == '--baseline-std':
            baseline_std = True
            i += 1
        elif arg == '--compare-log' and i + 1 < len(args):
            compare_logs.append(args[i + 1])
            i += 2
        elif arg.startswith('--compare-log='):
            compare_logs.append(arg.split('=', 1)[1])
            i += 1
        elif arg in ('--mean-run', '--mean_run'):
            mean_run = True
            i += 1
        elif arg == '--csv-dir' and i + 1 < len(args):
            csv_dir = args[i + 1]
            i += 2
        elif arg.startswith('--csv-dir='):
            csv_dir = arg.split('=', 1)[1]
            i += 1
        elif arg == '--output-dir' and i + 1 < len(args):
            output_dir_arg = args[i + 1]
            i += 2
        elif arg.startswith('--'):
            i += 1
        else:
            if metrics_log is None:
                metrics_log = arg
            i += 1

    mode_count = sum([1 if metrics_log else 0, 1 if compare_logs else 0, 1 if csv_dir else 0])
    if mode_count == 0:
        raise ValueError(
            'Expected one input mode: evaluation_metrics.log, --compare-log LABEL=PATH, or --csv-dir DIR.'
        )
    if mode_count > 1:
        raise ValueError(
            'Please choose a single input mode: either a metrics log, or --compare-log, or --csv-dir.'
        )
    
    # Load baseline data if provided
    baselines = {}
    if baseline_log:
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
    
    # Output directory: --output-dir overrides, otherwise same dir as log file
    if output_dir_arg:
        output_dir = output_dir_arg
        os.makedirs(output_dir, exist_ok=True)
    else:
        if metrics_log:
            output_dir = os.path.dirname(metrics_log)
        elif compare_logs:
            output_dir = os.getcwd()
        else:
            output_dir = os.path.join(csv_dir, 'replots')
        os.makedirs(output_dir, exist_ok=True)

    if csv_dir:
        csv_dir_abs = os.path.abspath(csv_dir)
        if not os.path.isdir(csv_dir_abs):
            raise ValueError(f'CSV directory does not exist: {csv_dir}')
        print(f"Generating plots from existing CSV data in: {csv_dir_abs}")
        plot_from_csv_directory(csv_dir_abs, output_dir, baselines, baseline_std, mean_run)
        print(f"\n[OK] CSV-based plots saved to {output_dir}")
        return

    if compare_logs:
        print("Generating aggregate comparison plots...")
        plot_aggregate_runs(compare_logs, output_dir, baselines, baseline_std, mean_run)
        print(f"\n[OK] Aggregate plots saved to {output_dir}")
        return

    print(f"Loading evaluation metrics from: {metrics_log}")
    df = parse_metrics_log(metrics_log)
    print(f"Loaded {len(df)} evaluation records\n")
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
        random_baseline_mean = None
        for pol, stats in baselines.items():
            mean_val = stats.get('mean', None)
            if mean_val is None:
                continue
            if pol == 'random':
                random_baseline_mean = mean_val
        add_baseline_lines(ax, baselines, 'rew', baseline_std)
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
        add_baseline_lines(ax, baselines, 'rew', baseline_std=False)

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
            add_baseline_lines(ax, baselines, 'rew', baseline_std=False)
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
    # Columns: rew=total, comp=completion, wait=wait, trav=travel (wait_travel),
    #          dln=deadline (deadline reward type). cap/step not plotted.
    try:
        eval_sorted = df.sort_values('ts').reset_index(drop=True)
        # Determine travel/deadline column
        travel_col = 'trav' if 'trav' in df.columns else ('dln' if 'dln' in df.columns else None)
        required = ['rew', 'wait', 'comp']
        if travel_col:
            required.append(travel_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for component breakdown: {missing}")

        all_reward = eval_sorted['rew'].values
        all_wait   = eval_sorted['wait'].values
        all_comp   = eval_sorted['comp'].values
        all_travel = eval_sorted[travel_col].values if travel_col else np.zeros(len(all_reward))
        travel_label = 'Travel' if travel_col == 'trav' else ('Deadline' if travel_col == 'dln' else 'N/A')

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

            # Completion component
            ax = fig.add_subplot(gs[0, 1])
            ax.set_facecolor('#fafafa')
            ax.plot(eps, all_comp, 'o-', alpha=0.7, color='#27ae60', markersize=4)
            ax.fill_between(eps, all_comp, alpha=0.3, color='#27ae60')
            ax.set_xlabel('Evaluation Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Completion Reward', fontsize=10, fontweight='bold')
            ax.set_title('Completion Component', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Stacked component breakdown
            ax = fig.add_subplot(gs[1, :])
            ax.set_facecolor('#fafafa')
            baseline = np.zeros(len(eps))
            ax.fill_between(eps, baseline, baseline + all_travel,
                            color='#e74c3c', alpha=0.4, label=travel_label)
            ax.fill_between(eps, baseline + all_travel,
                            baseline + all_travel + all_wait,
                            color='#f39c12', alpha=0.4, label='Wait')
            ax.fill_between(eps, baseline + all_travel + all_wait,
                            baseline + all_travel + all_wait + all_comp,
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
    
    # --- Additional Plots: trav/dln, wait, comp vs timesteps ---
    for reward_key, ylabel, fname, baseline_key, color in [
        ("trav", "Travel Reward",   "reward_trav_vs_timesteps.png",  "trav", "#e74c3c"),
        ("dln",  "Deadline Reward", "reward_dln_vs_timesteps.png",   "dln",  "#e74c3c"),
        ("wait", "Wait Reward",     "reward_wait_vs_timesteps.png",  "wait", "#f39c12"),
        ("comp", "Completion Reward","reward_comp_vs_timesteps.png", "comp", "#27ae60"),
    ]:
        if reward_key not in df.columns:
            print(f"[INFO] Column '{reward_key}' not present in log — skipping {fname}")
            continue
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
            add_baseline_lines(ax, baselines, reward_key, baseline_std)

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

    coord_dir = os.path.join(output_dir, 'coord')
    os.makedirs(coord_dir, exist_ok=True)
    any_coord = False
    for metric_key, spec in COORDINATION_METRICS.items():
        any_coord = _plot_single_coordination_metric(df, metric_key, spec, ma_window, coord_dir, baselines, baseline_std) or any_coord

    if any_coord:
        _plot_single_coordination_overview(df, ma_window, coord_dir)
    else:
        try:
            os.rmdir(coord_dir)
        except Exception:
            pass

    conflict_dir = os.path.join(output_dir, 'conflict')
    os.makedirs(conflict_dir, exist_ok=True)
    any_conflict = False
    for metric_key, spec in CONFLICT_METRICS.items():
        any_conflict = _plot_single_conflict_metric(df, metric_key, spec, ma_window, conflict_dir, baselines, baseline_std) or any_conflict

    if not any_conflict:
        try:
            os.rmdir(conflict_dir)
        except Exception:
            pass

    print(f"\n[OK] All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
