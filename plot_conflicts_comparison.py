#!/usr/bin/env python3
"""Plot grouped conflict metrics comparison across multiple runs.

Usage:
    python plot_conflicts_comparison.py run1/conflicts.log run2/conflicts.log --labels "1 hop,2 hop" --window 10 --out conflicts_comparison
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLS = [
    "episode",
    "conflicts_total",
    "tasks_total",
    "conflict_ratio",
    "winner_pickup",
    "winner_margin",
    "resolver_override",
    "resolver_override_rate",
    "avg_margin_win",
    "avg_margin_lose",
    "avg_margin_gap",
]


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _smooth_std(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return series.rolling(window=window, center=True, min_periods=1).std(ddof=0).fillna(0.0)


def _linestyle_for_label(label: str) -> str:
    """Return matplotlib linestyle based on architecture keywords in the label."""
    lbl = label.lower()
    if "1hop_critic" in lbl:
        return "--"  # dashed
    if "1hop" in lbl:
        return ":"   # dotted
    return "-"       # solid (2hop and default)


def _method_from_label(label: str) -> str:
    """Extract method key from '<method>-<run_idx>[...]' labels."""
    # Remove only run-index tokens like '-1' or '-12' when followed by end or '_'.
    # Examples: '1hop-1_ctc' -> '1hop_ctc', '1hop-3' -> '1hop'.
    return re.sub(r"-\d+(?=$|_)", "", label)


def _aggregate_series_by_method_conflict(
    data: Dict[str, pd.DataFrame],
    metric: str,
    window: int,
    mean_runs: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return label -> (x, mean_curve, std_curve)."""
    run_series: Dict[str, pd.Series] = {}
    for label, df in data.items():
        if metric not in df.columns:
            continue
        y = df[metric].astype(float)
        y_smooth = _smooth(y, window)
        run_series[label] = pd.Series(y_smooth.values, index=df["episode"].values)

    if not run_series:
        return {}

    if not mean_runs:
        out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for label, s in run_series.items():
            std_s = _smooth_std(s.astype(float), window)
            out[label] = (
                s.index.to_numpy(dtype=float),
                s.to_numpy(dtype=float),
                std_s.to_numpy(dtype=float),
            )
        return out

    grouped: Dict[str, List[pd.Series]] = {}
    for label, s in run_series.items():
        grouped.setdefault(_method_from_label(label), []).append(s)

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method, series_list in grouped.items():
        wide = pd.concat(series_list, axis=1)
        mean_s = wide.mean(axis=1, skipna=True)
        std_s = wide.std(axis=1, ddof=0, skipna=True).fillna(0.0)
        out[method] = (
            mean_s.index.to_numpy(dtype=float),
            mean_s.to_numpy(dtype=float),
            std_s.to_numpy(dtype=float),
        )
    return out


def _compute_ylim_from_smoothed_conflict(data: Dict[str, pd.DataFrame], metric: str, window: int) -> Tuple[float, float]:
    """Compute ylim based on min/max of smoothed values across all runs."""
    all_smooth = []
    for df in data.values():
        if metric not in df.columns:
            continue
        y = df[metric]
        y_smooth = _smooth(y, window)
        all_smooth.extend(y_smooth.dropna().values)
    
    if not all_smooth:
        return (0, 1)
    
    min_val = float(np.nanmin(all_smooth))
    max_val = float(np.nanmax(all_smooth))
    margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
    return (min_val - margin, max_val + margin)


def _load_conflict_logs(paths: List[str], labels: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for i, path_str in enumerate(paths):
        p = Path(path_str)
        label = labels[i] if labels else p.parent.name or p.stem

        if not p.exists():
            print(f"[WARN] Missing conflicts log, skipping: {p}")
            continue

        try:
            df = pd.read_csv(p)
        except Exception as exc:
            print(f"[WARN] Failed to read {p}: {exc}")
            continue

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"[WARN] Missing columns in {p}: {missing}")
            continue

        for col in REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Backward-compatible fallback for older conflicts logs.
        if "p_win_given_ego_action" not in df.columns:
            denom = df["conflicts_total"].replace(0, pd.NA)
            df["p_win_given_ego_action"] = (df["winner_pickup"] / denom).fillna(0.0)
        else:
            df["p_win_given_ego_action"] = pd.to_numeric(df["p_win_given_ego_action"], errors="coerce").fillna(0.0)

        if "p_win_given_high_logit" not in df.columns:
            denom = df["conflicts_total"].replace(0, pd.NA)
            df["p_win_given_high_logit"] = (df["winner_margin"] / denom).fillna(0.0)
        else:
            df["p_win_given_high_logit"] = pd.to_numeric(df["p_win_given_high_logit"], errors="coerce").fillna(0.0)

        df = df.dropna(subset=["episode"]).sort_values("episode").reset_index(drop=True)
        if df.empty:
            print(f"[WARN] Empty conflicts data, skipping: {p}")
            continue

        unique_label = label
        suffix = 2
        while unique_label in out:
            unique_label = f"{label}_{suffix}"
            suffix += 1
        out[unique_label] = df

    return out


def _plot_group(
    data: Dict[str, pd.DataFrame],
    metrics: List[str],
    title: str,
    out_path: Path,
    window: int,
    plot_std: bool,
    mean_runs: bool,
) -> None:
    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.0 * nrows))
    axes_arr = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for idx, metric in enumerate(metrics):
        ax = axes_arr[idx]
        plotted = _aggregate_series_by_method_conflict(data, metric, window, mean_runs)
        for label, (x, y_mean, y_std) in plotted.items():
            ax.plot(x, y_mean, linewidth=1.5, linestyle=_linestyle_for_label(label), label=label)
            if plot_std:
                ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.12)

        ax.set_title(metric)
        ax.set_xlabel("episode")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)
        
        # Apply dynamic ylim based on smoothed values for all metrics
        if plotted:
            min_val = min(float(np.nanmin(y_mean)) for _, (_, y_mean, _) in plotted.items())
            max_val = max(float(np.nanmax(y_mean)) for _, (_, y_mean, _) in plotted.items())
            margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            ylim = (min_val - margin, max_val + margin)
        else:
            ylim = _compute_ylim_from_smoothed_conflict(data, metric, window)
        ax.set_ylim(ylim)
        ax.legend(loc="upper left", fontsize="x-small", framealpha=0.65)

    for idx in range(n, len(axes_arr)):
        fig.delaxes(axes_arr[idx])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grouped comparison plots for multiple conflicts.log files")
    parser.add_argument("conflicts_logs", nargs="+", help="Paths to conflicts.log files")
    parser.add_argument("--labels", type=str, default="", help="Comma-separated labels for inputs")
    parser.add_argument("--window", type=int, default=10, help="Moving average window")
    parser.add_argument("--out", type=str, default="conflicts_comparison", help="Output directory")
    parser.add_argument(
        "--plot-std",
        dest="plot_std",
        action="store_true",
        default=True,
        help="Plot std band using the same window as smoothing (default: enabled)",
    )
    parser.add_argument(
        "--no-plot-std",
        dest="plot_std",
        action="store_false",
        help="Disable std band plotting",
    )
    parser.add_argument(
        "--mean-runs",
        dest="mean_runs",
        action="store_true",
        default=True,
        help="Group runs by method and plot mean/std across runs (default: enabled)",
    )
    parser.add_argument(
        "--no-mean-runs",
        dest="mean_runs",
        action="store_false",
        help="Disable run grouping and plot each run separately",
    )
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()] if args.labels.strip() else []
    if labels and len(labels) != len(args.conflicts_logs):
        print(
            f"[ERROR] --labels count ({len(labels)}) does not match number of logs ({len(args.conflicts_logs)})"
        )
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_conflict_logs(args.conflicts_logs, labels)
    if not data:
        print("[ERROR] No valid conflicts logs loaded")
        return 1

    _plot_group(
        data,
        ["conflicts_total", "tasks_total", "conflict_ratio", "resolver_override_rate"],
        "Conflict Volume and Rates",
        out_dir / "group_conflict_volume_rates.png",
        args.window,
        args.plot_std,
        args.mean_runs,
    )
    _plot_group(
        data,
        ["winner_pickup", "winner_margin", "resolver_override"],
        "Conflict Winner Signals",
        out_dir / "group_conflict_winner_signals.png",
        args.window,
        args.plot_std,
        args.mean_runs,
    )
    _plot_group(
        data,
        ["avg_margin_win", "avg_margin_lose", "avg_margin_gap"],
        "Conflict Margin Statistics",
        out_dir / "group_conflict_margins.png",
        args.window,
        args.plot_std,
        args.mean_runs,
    )
    _plot_group(
        data,
        ["p_win_given_ego_action", "p_win_given_high_logit"],
        "Conflict Win Probability Diagnostics",
        out_dir / "group_conflict_win_probabilities.png",
        args.window,
        args.plot_std,
        args.mean_runs,
    )

    print(f"[OK] Wrote conflict comparison plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
