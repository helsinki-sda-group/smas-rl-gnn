"""
Plot episode-level quality metrics from quality_episode_metrics.csv.

Usage:
    python plot_quality_episode_metrics.py \
        --metrics runs/my_run/quality_episode_metrics.csv \
        --out plots/quality \
        --smooth-window 500 \
        --plot_std

Multiple files (max 5) can be passed to overlay runs:
    python plot_quality_episode_metrics.py \
        --metrics run1/quality_episode_metrics.csv run2/quality_episode_metrics.csv \
        --label-from run_id \
        --out plots/quality
"""
from __future__ import annotations

import argparse
import os
import re
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
    "reward_big_components": [
        "rew_completion_agg_sum",
        "rew_wait_agg_sum",
        "rew_deadline_agg_sum",
        "rew_travel_agg_sum",
    ],
    "reward_wait_subcomponents": [
        "rew_wait_event_pickup_sum",
        "rew_wait_obsolete_pickup_sum",
        "rew_wait_terminal_never_picked_sum",
    ],
    "reward_deadline_subcomponents": [
        "rew_deadline_pickup_lateness_sum",
        "rew_deadline_dropoff_lateness_sum",
    ],
    "reward_travel_subcomponents": [
        "rew_travel_event_dropoff_sum",
        "rew_travel_terminal_picked_not_dropped_sum",
    ],
    "task_rates": [
        "task_completed_rate",
        "task_obsolete_rate",
        "task_never_picked_rate",
        "task_picked_not_dropped_rate",
        "task_obs_dropoff_rate",
    ],
    "task_counts": [
        "task_completed_count",
        "task_obsolete_count",
        "task_never_picked_count",
        "task_picked_not_dropped_count",
    ],
    "task_wait_time": [
        "task_wait_time_mean",
    ],
    "task_travel_time": [
        "task_travel_time_mean",
    ],
    "pooling_absolute": [
        "pool_mean_onboard",
        "pool_max_onboard",
    ],
    "pooling_rates": [
        "pool_frac_multi_pax",
        "pool_frac_empty",
    ],
    "decisions_rates": [
        "dec_noop_rate",
    ],
    "decisions_absolute": [
        "dec_mean_candidates",
        "dec_noop_no_candidates_count",
        "dec_noop_with_candidates_count",
    ],
    "conflicts_absolute": [
        "conf_total",
    ],
    "conflicts_rates": [
        "conf_winner_pickup_rate",
        "conf_resolver_override_rate",
        "conf_policy_matches_resolver_rate",
    ],
}

MAX_JOBS = 5
LINE_STYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]

WAIT_SUBCOMPONENTS = [
    "rew_wait_event_pickup_sum",
    "rew_wait_obsolete_pickup_sum",
    "rew_wait_terminal_never_picked_sum",
]
DEADLINE_SUBCOMPONENTS = [
    "rew_deadline_pickup_lateness_sum",
    "rew_deadline_dropoff_lateness_sum",
]
TRAVEL_SUBCOMPONENTS = [
    "rew_travel_event_dropoff_sum",
    "rew_travel_terminal_picked_not_dropped_sum",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _x_col(df: pd.DataFrame) -> str:
    """Determine x-axis column: ts if available, else episode."""
    if "ts" in df.columns and df["ts"].notna().any():
        return "ts"
    return "episode"


def _is_percentile_metric(metric: str) -> bool:
    return bool(re.search(r"percentile|p\d+$", metric))


def _aggregate_sum(df: pd.DataFrame, columns: list[str], target_col: str) -> None:
    present_cols = [c for c in columns if c in df.columns]
    if not present_cols:
        return
    numeric = df[present_cols].apply(pd.to_numeric, errors="coerce")
    df[target_col] = numeric.sum(axis=1)


def _add_reward_aggregates(df: pd.DataFrame) -> None:
    completion_source = None
    for candidate in ["rew_valid_dropoff_sum", "rew_completion_sum"]:
        if candidate in df.columns:
            completion_source = candidate
            break
    if completion_source is not None:
        df["rew_completion_agg_sum"] = pd.to_numeric(df[completion_source], errors="coerce")
    _aggregate_sum(df, WAIT_SUBCOMPONENTS, "rew_wait_agg_sum")
    _aggregate_sum(df, DEADLINE_SUBCOMPONENTS, "rew_deadline_agg_sum")
    _aggregate_sum(df, TRAVEL_SUBCOMPONENTS, "rew_travel_agg_sum")


def _rolling_stats_by_timestep(x: np.ndarray, y: np.ndarray, window_timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    if window_timesteps <= 1:
        return y, np.zeros_like(y)

    mean = np.full_like(y, np.nan, dtype=float)
    std = np.full_like(y, np.nan, dtype=float)
    left = 0
    for right in range(len(y)):
        while left <= right and (x[right] - x[left]) > window_timesteps:
            left += 1
        window = y[left:right + 1]
        mean[right] = np.nanmean(window)
        std[right] = np.nanstd(window)
    return mean, std


def _run_name_from_metrics_path(path: str) -> str:
    parent_name = Path(path).resolve().parent.name
    if parent_name.startswith("job_"):
        run = parent_name[len("job_"):]
        run = re.sub(r"_\d+$", "", run)
        run = re.sub(r"-[0-9]+(_|$)", r"\1", run)
        run = re.sub(r"__+", "_", run).strip("_")
        if run:
            return run
    return Path(path).stem


def _subfolder_name(metrics_paths: list[str]) -> str:
    names: list[str] = []
    seen: set[str] = set()
    for path in metrics_paths:
        name = _run_name_from_metrics_path(path)
        if name not in seen:
            names.append(name)
            seen.add(name)
    return "__".join(names) if names else "quality_runs"


def _plot_group(
    ax: "plt.Axes",
    dataframes: list[pd.DataFrame],
    labels: list[str],
    metrics: list[str],
    x_col: str,
    title: str,
    smooth_window: int,
    plot_std: bool,
) -> None:
    """Plot one metric group onto ax."""
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for m_idx, metric in enumerate(metrics):
        for d_idx, (df, lbl) in enumerate(zip(dataframes, labels)):
            if metric not in df.columns:
                continue
            color = colors[m_idx % len(colors)]
            linestyle = LINE_STYLES[d_idx % len(LINE_STYLES)]
            x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                continue
            x = x[valid]
            y = y[valid]

            if smooth_window > 1:
                y_mean, y_std = _rolling_stats_by_timestep(x, y, smooth_window)
            else:
                y_mean = y
                y_std = np.zeros_like(y)

            line_label = metric if len(labels) == 1 else f"{lbl} / {metric}"
            ax.plot(x, y_mean, label=line_label, color=color, linestyle=linestyle, alpha=0.9)
            if plot_std and smooth_window > 1:
                ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)
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
    parser.add_argument(
        "--smooth-window", type=int, default=500,
        help="Smoothing window in timesteps (default: 500)."
    )
    parser.add_argument(
        "--plot_std", dest="plot_std", action="store_true", default=True,
        help="Plot standard deviation as a shaded range when smoothing window > 1."
    )
    parser.add_argument(
        "--no-plot_std", dest="plot_std", action="store_false",
        help="Disable standard deviation shaded ranges."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.metrics) > MAX_JOBS:
        print(f"ERROR: At most {MAX_JOBS} jobs can be compared at once.")
        sys.exit(1)

    out_subdir = _subfolder_name(args.metrics)
    out_dir = Path(args.out) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataframes: list[pd.DataFrame] = []
    labels: list[str] = []

    for path in args.metrics:
        if not os.path.isfile(path):
            print(f"[WARN] File not found: {path} — skipping.")
            continue
        df = pd.read_csv(path)
        _add_reward_aggregates(df)
        dataframes.append(df)
        if args.label_from == "filename":
            lbl = Path(path).stem
        else:
            lbl = _get_label(df, Path(path).stem)
        labels.append(lbl)

    if not dataframes:
        print("No valid input files. Exiting.")
        sys.exit(1)

    if len(dataframes) > MAX_JOBS:
        print(f"ERROR: Found more than {MAX_JOBS} valid jobs.")
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
        metrics = [m for m in METRIC_GROUPS[group_name] if not _is_percentile_metric(m)]
        # filter to metrics actually present in at least one dataframe
        present = [m for m in metrics if any(m in df.columns for df in dataframes)]
        if not present:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_group(
            ax=ax,
            dataframes=dataframes,
            labels=labels,
            metrics=present,
            x_col=x_col,
            title=group_name,
            smooth_window=max(1, int(args.smooth_window)),
            plot_std=bool(args.plot_std),
        )
        fig.tight_layout()
        out_path = out_dir / f"quality_{group_name}.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"Done. Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
