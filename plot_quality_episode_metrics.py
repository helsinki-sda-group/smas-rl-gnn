"""
Plot episode-level quality metrics from quality_episode_metrics.csv.

Usage:
    python plot_quality_episode_metrics.py \
        --metrics runs/my_run/quality_episode_metrics.csv \
        --out plots/quality

Multiple files can be passed to overlay runs:
    python plot_quality_episode_metrics.py \
        --metrics run1/quality_episode_metrics.csv run2/quality_episode_metrics.csv \
        --label-from run_id \
        --out plots/quality
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. pip install pandas")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required. pip install matplotlib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# metric groups
# ---------------------------------------------------------------------------

METRIC_GROUPS: dict[str, list[str]] = {
    "reward_components": [
        "rew_completion_sum",
        "rew_wait_event_pickup_sum",
        "rew_wait_obsolete_pickup_sum",
        "rew_wait_terminal_never_picked_sum",
        "rew_travel_event_dropoff_sum",
        "rew_travel_terminal_picked_not_dropped_sum",
        "rew_deadline_pickup_lateness_sum",
        "rew_deadline_dropoff_lateness_sum",
        "rew_obsolete_total_sum",
        "rew_terminal_total_sum",
        "macro_reward_mean",
        "macro_reward_sum",
    ],
    "task_rates": [
        "task_completed_rate",
        "task_obsolete_rate",
        "task_never_picked_rate",
        "task_picked_not_dropped_rate",
        "task_obs_dropoff_rate",
    ],
    "task_counts": [
        "task_total_count",
        "task_completed_count",
        "task_obsolete_count",
        "task_never_picked_count",
        "task_picked_not_dropped_count",
    ],
    "task_wait_time": [
        "task_wait_time_mean",
        "task_wait_time_std",
        "task_wait_time_p50",
        "task_wait_time_p90",
        "task_wait_time_p95",
    ],
    "task_travel_time": [
        "task_travel_time_mean",
        "task_travel_time_std",
        "task_travel_time_p50",
        "task_travel_time_p90",
        "task_travel_time_p95",
    ],
    "pooling": [
        "pool_mean_onboard",
        "pool_max_onboard",
        "pool_frac_multi_pax",
        "pool_frac_empty",
    ],
    "decisions": [
        "dec_noop_rate",
        "dec_mean_candidates",
        "dec_noop_no_candidates_count",
        "dec_noop_with_candidates_count",
    ],
    "conflicts": [
        "conf_total",
        "conf_winner_pickup_rate",
        "conf_resolver_override_rate",
        "conf_policy_matches_resolver_rate",
    ],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _x_col(df: pd.DataFrame) -> str:
    """Determine x-axis column: ts if available, else episode."""
    if "ts" in df.columns and df["ts"].notna().any():
        return "ts"
    return "episode"


def _plot_group(
    ax: "plt.Axes",
    dataframes: list[pd.DataFrame],
    labels: list[str],
    metrics: list[str],
    x_col: str,
    title: str,
) -> None:
    """Plot one metric group onto ax."""
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for m_idx, metric in enumerate(metrics):
        for d_idx, (df, lbl) in enumerate(zip(dataframes, labels)):
            if metric not in df.columns:
                continue
            col_idx = (m_idx * len(dataframes) + d_idx) % len(colors)
            x = df[x_col].values
            y = df[metric].astype(float).values
            line_label = metric if len(labels) == 1 else f"{lbl} / {metric}"
            ax.plot(x, y, label=line_label, color=colors[col_idx], alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(x_col, fontsize=8)
    ax.legend(fontsize=6, loc="best", ncol=1)
    ax.grid(True, alpha=0.3)


def _get_label(df: pd.DataFrame, fallback: str) -> str:
    if "run_id" in df.columns and df["run_id"].notna().any():
        return str(df["run_id"].iloc[0])
    if "config_id" in df.columns and df["config_id"].notna().any():
        return str(df["config_id"].iloc[0])
    return fallback


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quality_episode_metrics CSVs")
    parser.add_argument(
        "--metrics", nargs="+", required=True,
        help="Path(s) to quality_episode_metrics.csv file(s)."
    )
    parser.add_argument(
        "--out", default="plots/quality",
        help="Output directory for plots (default: plots/quality)."
    )
    parser.add_argument(
        "--label-from", choices=["run_id", "config_id", "filename"], default="run_id",
        help="Column to use as legend label for each input file."
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Subset of metric groups to plot. Default: all."
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for saved figures (default: 150)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataframes: list[pd.DataFrame] = []
    labels: list[str] = []

    for path in args.metrics:
        if not os.path.isfile(path):
            print(f"[WARN] File not found: {path} — skipping.")
            continue
        df = pd.read_csv(path)
        dataframes.append(df)
        if args.label_from == "filename":
            lbl = Path(path).stem
        else:
            lbl = _get_label(df, Path(path).stem)
        labels.append(lbl)

    if not dataframes:
        print("No valid input files. Exiting.")
        sys.exit(1)

    # Use the first df's x_col as reference
    x_col = _x_col(dataframes[0])

    # Sort each df by x_col
    for i, df in enumerate(dataframes):
        if x_col in df.columns:
            dataframes[i] = df.sort_values(x_col)

    groups_to_plot = args.groups if args.groups else list(METRIC_GROUPS.keys())

    for group_name in groups_to_plot:
        if group_name not in METRIC_GROUPS:
            print(f"[WARN] Unknown metric group: {group_name} — skipping.")
            continue
        metrics = METRIC_GROUPS[group_name]
        # filter to metrics actually present in at least one dataframe
        present = [m for m in metrics if any(m in df.columns for df in dataframes)]
        if not present:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_group(ax, dataframes, labels, present, x_col, group_name)
        fig.tight_layout()
        out_path = out_dir / f"quality_{group_name}.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"Done. Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
