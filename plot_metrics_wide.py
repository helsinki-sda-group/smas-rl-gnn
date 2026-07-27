#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ID_COLUMNS = {"source_file", "instance", "resolver", "pol"}
DEFAULT_OUTPUT_DIR = "metrics_wide_plots"
RATIO_METRICS = {"crat", "conf_ratio"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aggregated metrics from metrics_wide.csv. "
            "If no plot-type flags are provided, all available plot types are generated."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("metrics_wide.csv"),
        help="Path to metrics_wide.csv (default: ./metrics_wide.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Directory for saved plots (default: ./{DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help="Metric names to plot, or 'all' to plot every metric found in the CSV",
    )
    parser.add_argument(
        "--resolver-cmp",
        action="store_true",
        help="Create one bar-chart figure per metric with one subplot per resolver",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        metrics = detect_metrics(reader.fieldnames)
        rows: list[dict[str, object]] = []
        for raw_row in reader:
            parsed_row: dict[str, object] = {}
            for key, value in raw_row.items():
                parsed_row[key] = parse_cell(value)
            rows.append(parsed_row)

    if not rows:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    return rows, metrics


def detect_metrics(fieldnames: list[str]) -> list[str]:
    metrics: list[str] = []
    for fieldname in fieldnames:
        if fieldname in ID_COLUMNS or fieldname.endswith("_std"):
            continue
        metrics.append(fieldname)
    return metrics


def parse_cell(value: str | None) -> object:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return None
    if stripped.upper() == "NA":
        return None

    try:
        return float(stripped)
    except ValueError:
        return stripped


def select_metrics(requested_metrics: list[str], available_metrics: list[str]) -> list[str]:
    if not requested_metrics or requested_metrics == ["all"]:
        return available_metrics

    requested_set = {metric.lower() for metric in requested_metrics}
    selected = [metric for metric in available_metrics if metric.lower() in requested_set]
    missing = sorted(requested_set.difference(metric.lower() for metric in selected))
    if missing:
        raise ValueError(f"Unknown metric(s): {', '.join(missing)}")
    return selected


def resolve_plot_types(args: argparse.Namespace) -> list[str]:
    requested_types: list[str] = []
    if args.resolver_cmp:
        requested_types.append("resolver_cmp")
    if requested_types:
        return requested_types
    return ["resolver_cmp"]


def safe_number(value: object) -> float | None:
    if value is None or not isinstance(value, (int, float)):
        return None
    if math.isnan(value):
        return None
    return float(value)


def metric_limits(rows: list[dict[str, object]], metric: str) -> tuple[float, float]:
    if metric.lower() in RATIO_METRICS:
        return (-0.02, 1.02)

    lower = math.inf
    upper = -math.inf

    for row in rows:
        mean = safe_number(row.get(metric))
        std = safe_number(row.get(f"{metric}_std")) or 0.0
        if mean is None:
            continue
        lower = min(lower, mean - std, 0.0)
        upper = max(upper, mean + std, 0.0)

    if lower is math.inf or upper is -math.inf:
        return (-1.0, 1.0)

    if math.isclose(lower, upper):
        padding = abs(lower) * 0.05 or 1.0
        return lower - padding, upper + padding

    padding = (upper - lower) * 0.05
    return lower - padding, upper + padding


def plot_resolver_cmp(rows: list[dict[str, object]], metrics: list[str], output_dir: Path) -> list[Path]:
    resolver_dir = output_dir / "resolver_cmp"
    resolver_dir.mkdir(parents=True, exist_ok=True)

    resolvers = sorted({str(row["resolver"]) for row in rows if row.get("resolver") is not None})
    saved_paths: list[Path] = []

    for metric in metrics:
        y_min, y_max = metric_limits(rows, metric)
        figure, axes = plt.subplots(
            nrows=1,
            ncols=len(resolvers),
            figsize=(max(5 * len(resolvers), 6), 5),
            squeeze=False,
            sharey=True,
        )
        axis_row = axes[0]
        figure.suptitle(f"{metric}: comparison across resolvers", fontsize=14)

        for axis, resolver in zip(axis_row, resolvers):
            resolver_rows = [row for row in rows if row.get("resolver") == resolver]
            resolver_rows.sort(key=lambda row: str(row.get("pol", "")))

            policies = [str(row["pol"]) for row in resolver_rows if row.get("pol") is not None]
            means = [safe_number(row.get(metric)) for row in resolver_rows]
            stds = [(safe_number(row.get(f"{metric}_std")) or 0.0) for row in resolver_rows]

            filtered = [
                (policy, mean, std)
                for policy, mean, std in zip(policies, means, stds)
                if mean is not None
            ]

            if not filtered:
                axis.set_visible(False)
                continue

            policies = [item[0] for item in filtered]
            means = [item[1] for item in filtered]
            stds = [item[2] for item in filtered]
            positions = list(range(len(policies)))

            axis.bar(
                positions,
                means,
                yerr=stds,
                capsize=4,
                color="#4C78A8",
                edgecolor="#2F3E4E",
                alpha=0.9,
            )
            axis.set_title(resolver)
            axis.set_xticks(positions)
            axis.set_xticklabels(policies, rotation=45, ha="right")
            axis.set_ylim(y_min, y_max)
            axis.grid(axis="y", alpha=0.3, linestyle="--")
            axis.set_axisbelow(True)

        axis_row[0].set_ylabel(metric)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        output_path = resolver_dir / f"{metric}_resolver_cmp.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def main() -> int:
    args = parse_args()
    rows, available_metrics = load_rows(args.csv_path)
    metrics = select_metrics(args.metrics, available_metrics)
    plot_types = resolve_plot_types(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    if "resolver_cmp" in plot_types:
        saved_paths.extend(plot_resolver_cmp(rows, metrics, output_dir))

    print(f"Saved {len(saved_paths)} plot(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())