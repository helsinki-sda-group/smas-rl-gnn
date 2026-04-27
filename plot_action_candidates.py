#!/usr/bin/env python3
"""
Plot action/candidate and assignment outcome columns from training_metrics_*.log

Plots action/candidate columns:
    noop, overld, mcand, cne_fr, cne_mn, dstep, macmr, msd, ovrlap, shared

Also plots assignment outcome rates:
    cmr, obsr, anpr, pncr

with optional moving-average smoothing.

Supports multiple log files for comparison.

Usage:
    python plot_action_candidates.py <metrics_log> [<metrics_log2> ...] [--window 5] [--out out_dir]

Examples:
    python plot_action_candidates.py training_metrics_v1250_ms2400_mwd240_mtd900_cap3.log --window 10 --out action_plots
    python plot_action_candidates.py run1_metrics.log run2_metrics.log run3_metrics.log --window 10 --out action_comparison
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Columns in the action/candidate block
ACTION_CANDIDATE_COLS = [
    "noop",
    "overld",
    "mcand",
    "cne_fr",
    "cne_mn",
    "dstep",
    "macmr",
    "msd",
    "ovrlap",
    "shared",
]

OUTCOME_RATE_COLS = [
    "cmr",
    "obsr",
    "anpr",
    "pncr",
]

# Human-readable labels
COL_LABELS = {
    "noop": "NO-OP Fraction",
    "overld": "Overload Assignment Fraction",
    "mcand": "Mean Candidates per Taxi",
    "cne_fr": "Candidate Non-empty Fraction",
    "cne_mn": "Candidate Mean (Non-empty)",
    "dstep": "Decision Steps",
    "macmr": "Macro Reward Mean",
    "msd": "Macro Steps Done",
    "ovrlap": "Overlap Rate",
    "shared": "Mean Shared Tasks per Step",
    "cmr": "Completion Rate (CMR)",
    "obsr": "Obsolete Rate (OBSR)",
    "anpr": "Assigned Never Picked Rate (ANPR)",
    "pncr": "Picked Not Completed Rate (PNCR)",
}


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    """Apply centered moving average smoothing."""
    if window <= 1:
        return series.copy()
    return series.rolling(window=window, center=True, min_periods=1).mean()


def load_log(log_path: Path, label: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Load a metrics log file and return dataframe + label."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    print(f"[INFO] Reading {log_path}")
    df = pd.read_csv(log_path, sep=r"\s+")
    
    # Use provided label or derive from filename
    if label is None:
        label = log_path.stem
    
    return df, label


def plot_single_column(
    data_dict: Dict[str, pd.DataFrame],
    col: str,
    window: int,
    out_dir: Path,
) -> None:
    """Plot a single column across multiple runs."""
    # Check if column exists in any dataframe
    if not any(col in df.columns for df in data_dict.values()):
        print(f"[WARN] Column '{col}' not found in any log file; skipping.")
        return

    plt.figure(figsize=(11, 6))
    
    for label, df in data_dict.items():
        if col not in df.columns:
            continue
        
        x = df.index
        y_raw = df[col].astype(float)
        y_smooth = smooth_series(y_raw, window) if window > 1 else y_raw
        
        # Plot smoothed line
        plt.plot(x, y_smooth, linewidth=2, label=label, marker='o', markersize=4)
        
        # Optionally plot raw as transparent background
        if window > 1:
            plt.plot(x, y_raw, alpha=0.15, linewidth=0.5, color='gray')
    
    plt.xlabel("Episode")
    plt.ylabel(col)
    plt.title(COL_LABELS.get(col, col))
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = out_dir / f"{col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved {out_path}")


def plot_multi_column(
    data_dict: Dict[str, pd.DataFrame],
    cols: List[str],
    window: int,
    out_dir: Path,
    title_suffix: str = "",
) -> None:
    """Plot multiple columns on same axes (one line per run)."""
    # Filter to valid columns
    valid_cols = [c for c in cols if any(c in df.columns for df in data_dict.values())]
    if not valid_cols:
        print("[WARN] No valid columns found for multi-plot; skipping.")
        return

    plt.figure(figsize=(13, 7))
    
    for label, df in data_dict.items():
        # For multi-column view, plot one metric per run (e.g., noop)
        # Show first available column
        if valid_cols and valid_cols[0] in df.columns:
            x = df.index
            y_raw = df[valid_cols[0]].astype(float)
            y_smooth = smooth_series(y_raw, window) if window > 1 else y_raw
            plt.plot(x, y_smooth, linewidth=1.5, label=label, marker='o', markersize=3)
    
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title(f"Action/Candidate Metrics{title_suffix}")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = out_dir / "all_action_candidates.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved {out_path}")


def plot_grouped(
    data_dict: Dict[str, pd.DataFrame],
    window: int,
    out_dir: Path,
) -> None:
    """Plot groups of related metrics across multiple runs."""
    groups = {
        "noop_overld": ["noop", "overld"],
        "candidates": ["mcand", "cne_fr", "cne_mn"],
        "decision": ["dstep", "macmr", "msd"],
        "sharing": ["ovrlap", "shared"],
    }
    
    for group_key, cols in groups.items():
        valid_cols = [c for c in cols if any(c in df.columns for df in data_dict.values())]
        if not valid_cols:
            continue
        
        n_cols = len(valid_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, valid_cols):
            for label, df in data_dict.items():
                if col not in df.columns:
                    continue
                
                x = df.index
                y_raw = df[col].astype(float)
                y_smooth = smooth_series(y_raw, window) if window > 1 else y_raw
                
                ax.plot(x, y_smooth, linewidth=1.5, label=label, marker='o', markersize=3)
            
            ax.set_xlabel("Episode")
            ax.set_ylabel(col)
            ax.set_title(COL_LABELS.get(col, col))
            ax.legend(loc='best', fontsize='small')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = out_dir / f"group_{group_key}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[OK] Saved {out_path}")


def plot_outcome_rates_grouped(
    data_dict: Dict[str, pd.DataFrame],
    window: int,
    out_dir: Path,
) -> None:
    """Plot cmr/obsr/anpr/pncr in a single grouped figure (2x2 subplots)."""
    valid_cols = [c for c in OUTCOME_RATE_COLS if any(c in df.columns for df in data_dict.values())]
    if not valid_cols:
        print("[WARN] No outcome-rate columns found; skipping grouped outcome plot.")
        return

    n = len(valid_cols)
    nrows, ncols = (2, 2) if n > 2 else (1, n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8) if n > 2 else (6 * n, 4))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for i, col in enumerate(valid_cols):
        ax = axes_arr[i]
        for label, df in data_dict.items():
            if col not in df.columns:
                continue

            x = df.index
            y_raw = df[col].astype(float)
            y_smooth = smooth_series(y_raw, window) if window > 1 else y_raw

            ax.plot(x, y_smooth, linewidth=1.8, label=label, marker='o', markersize=3)
            if window > 1:
                ax.plot(x, y_raw, alpha=0.12, linewidth=0.5, color='gray')

        ax.set_xlabel("Episode")
        ax.set_ylabel(col)
        ax.set_title(COL_LABELS.get(col, col))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize='small')

    for j in range(n, len(axes_arr)):
        fig.delaxes(axes_arr[j])

    plt.tight_layout()
    out_path = out_dir / "group_outcome_rates.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot action/candidate and outcome-rate columns from training_metrics_*.log with smoothing. Support multiple log files for comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (single file):
  python plot_action_candidates.py training_metrics_v1250_ms2400_mwd240_mtd900_cap3.log
  python plot_action_candidates.py training_metrics_v1250_ms2400_mwd240_mtd900_cap3.log --window 10 --out action_plots

Examples (multiple files for comparison):
  python plot_action_candidates.py run1_metrics.log run2_metrics.log run3_metrics.log
  python plot_action_candidates.py run1_metrics.log run2_metrics.log --window 5 --out comparison_plots
        """,
    )
    parser.add_argument(
        "metrics_log",
        nargs='+',
        type=str,
        help="Path(s) to training_metrics_*.log file(s) (one or more)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Moving average window size for smoothing (default: 5, use 1 for no smoothing)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="action_candidate_plots",
        help="Output directory for plots (default: action_candidate_plots)",
    )

    args = parser.parse_args()
    out_dir = Path(args.out)

    # Load all log files
    data_dict: Dict[str, pd.DataFrame] = {}
    for log_path_str in args.metrics_log:
        try:
            df, label = load_log(Path(log_path_str))
            data_dict[label] = df
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return 1
        except Exception as e:
            print(f"[ERROR] Could not parse {log_path_str}: {e}")
            return 1

    if not data_dict:
        print("[ERROR] No valid log files loaded")
        return 1

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    # Validate columns
    all_cols = set()
    for df in data_dict.values():
        all_cols.update(df.columns)
    
    expected_cols = ACTION_CANDIDATE_COLS + OUTCOME_RATE_COLS
    missing = [c for c in expected_cols if c not in all_cols]
    if missing:
        print(f"[WARN] Missing columns across all logs: {missing}")

    # Generate plots
    print(f"[INFO] Generating plots (smoothing window={args.window}, {len(data_dict)} run(s))...")
    
    # Individual plots
    print("[INFO] Plotting individual columns...")
    for col in ACTION_CANDIDATE_COLS:
        plot_single_column(data_dict, col, args.window, out_dir)

    print("[INFO] Plotting outcome-rate columns...")
    for col in OUTCOME_RATE_COLS:
        plot_single_column(data_dict, col, args.window, out_dir)
    
    # Combined plot
    print("[INFO] Plotting all action/candidate columns together...")
    plot_multi_column(data_dict, ACTION_CANDIDATE_COLS, args.window, out_dir)
    
    # Grouped plots
    print("[INFO] Plotting grouped metrics...")
    plot_grouped(data_dict, args.window, out_dir)

    print("[INFO] Plotting grouped outcome rates...")
    plot_outcome_rates_grouped(data_dict, args.window, out_dir)

    print(f"[OK] All plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
