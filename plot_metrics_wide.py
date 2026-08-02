#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


ID_COLUMNS = {"source_file", "scenario", "instance", "protocol", "resolver", "route_construction", "admission_aware", "pol"}
DEFAULT_OUTPUT_DIR = "metrics_wide_plots"
ROUTE_CONSTRUCTION_OUTPUT_SUBDIR = "route_construction_cmp"
ROUTE_MODE_ORDER = ["nearest", "reward_aligned"]
ROUTE_MODE_COLORS = {
    "nearest": "#4C78A8",
    "reward_aligned": "#F58518",
}
PROTOCOL_CMP_OUTPUT_SUBDIR = "protocol_cmp"
RATIO_METRICS = {"crat", "conf_ratio"}
GROUPED_BAR_WIDTH = 0.11
RESOLVER_ORDER_BASE = [
    "capacity",
    "closest_then_capacity",
    "predicted_reward",
    "predicted_reward_joint",
    "hungarian",
]
RESOLVER_COLOR_MAP = {
    "capacity": "#0072B2",
    "closest_then_capacity": "#56B4E9",
    "predicted_reward": "#009E73",
    "predicted_reward_joint": "#E69F00",
    "hungarian": "#D55E00",
}
COLORBLIND_PALETTE = [
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#F0E442",
    "#D55E00",
    "#000000",
]
POLICY_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*", "8"]
PROTOCOL_COMPARE_POLICIES: list[tuple[str, set[str]]] = [
    ("predicted_reward", {"predicted_reward"}),
    ("predicted_reward_joint", {"predicted_reward_joint"}),
    ("proposal_joint_completion", {"proposal_joint_completion", "proposal_joint_competition"}),
]
PROTOCOL_COMPARE_ORDER = ["forced", "aa"]
PROTOCOL_COMPARE_DISPLAY = {
    "forced": "f",
    "aa": "aa",
}
WORK_COLUMNS = {
    "work_route_stops",
    "work_insertion_pairs",
    "work_candidate_scan",
    "work_active_proposals",
    "work_competition_factor",
    "work_proposer",
    "work_resolver",
    "work_total",
}
NON_QUALITY_COLUMNS = {
    "work_model",
    "work_warning",
    "num_robots",
    "max_robot_capacity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aggregated metrics from metrics_wide.csv. "
            "If no plot-type flags are provided, grouped resolver/policy/protocol comparison plots are generated."
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
        help=(
            "Create resolver comparison plots. With --grouped (default), this also creates "
            "grouped policy and protocol comparison plots."
        ),
    )
    parser.add_argument(
        "--protocol-cmp",
        action="store_true",
        help=(
            "Create protocol comparison plots (forced vs aa) in protocol_cmp, with "
            "resolver subplots and grouped bars for supported policies"
        ),
    )
    parser.add_argument(
        "--grouped",
        type=lambda value: str(value).strip().lower() not in {"false", "0", "no", "off"},
        default=True,
        help="Generate grouped resolver/policy/protocol comparison plots (default: true)",
    )
    parser.add_argument(
        "--exclude-resolvers",
        nargs="+",
        default=[],
        help="Resolver names to exclude from grouped plots",
    )
    parser.add_argument(
        "--exclude-policies",
        nargs="+",
        default=[],
        help="Policy names to exclude from grouped plots",
    )
    parser.add_argument(
        "--route-construction",
        action="store_true",
        help="Compare metrics across route_construction methods (nearest vs reward_aligned) for each policy/resolver pair",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario to plot, or 'all' to create one scenario subfolder per available scenario",
    )
    parser.add_argument(
        "--work-cmp",
        action="store_true",
        help="Create quality-versus-estimated-work scatter plots from enriched work columns",
    )
    parser.add_argument(
        "--work-x",
        choices=["relative", "total"],
        default="relative",
        help="Work axis scaling mode for --work-cmp (default: relative)",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Draw Pareto frontier for work-comparison plots",
    )
    parser.add_argument(
        "--annotate-work",
        action="store_true",
        help="Annotate work-comparison points with policy labels",
    )
    parser.add_argument(
        "--work-x-padding",
        type=float,
        default=0.06,
        help="Log-space padding fraction for --work-cmp x-axis limits (default: 0.06)",
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
        if fieldname in WORK_COLUMNS or fieldname in NON_QUALITY_COLUMNS:
            continue
        metrics.append(fieldname)
    return metrics


def _scenario_of_row(row: dict[str, object]) -> str:
    value = row.get("scenario")
    text = str(value).strip().lower() if value is not None else ""
    return text if text else "unknown"


def _scenario_folder_name(scenario: str) -> str:
    text = str(scenario or "").strip().lower()
    return text if text else "unknown"


def _selected_scenarios(rows: list[dict[str, object]], scenario_arg: str) -> list[str]:
    available = sorted({_scenario_of_row(row) for row in rows})
    requested = str(scenario_arg or "all").strip().lower()
    if not requested or requested == "all":
        return available
    if requested not in available:
        raise ValueError(f"Unknown scenario: {scenario_arg}. Available scenarios: {', '.join(available)}")
    return [requested]


def _rows_for_scenario(rows: list[dict[str, object]], scenario: str) -> list[dict[str, object]]:
    return [row for row in rows if _scenario_of_row(row) == scenario]


def _normalize_name_set(values: list[str]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        for chunk in str(value).split(","):
            text = chunk.strip().lower()
            if text:
                normalized.add(text)
    return normalized


def _normalize_policy_key(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("-", "_")
    text = "_".join(part for part in text.split())
    return text


def _normalize_policy_set(values: list[str]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        for chunk in str(value).split(","):
            key = _normalize_policy_key(chunk)
            if key:
                normalized.add(key)
    return normalized


def _filtered_rows(
    rows: list[dict[str, object]],
    *,
    exclude_resolvers: set[str],
    exclude_policies: set[str],
) -> list[dict[str, object]]:
    filtered_rows: list[dict[str, object]] = []
    for row in rows:
        resolver = str(row.get("resolver", "")).strip().lower()
        policy = _normalize_policy_key(row.get("pol", ""))
        if resolver in exclude_resolvers:
            continue
        if policy in exclude_policies:
            continue
        filtered_rows.append(row)
    return filtered_rows


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
    has_explicit_flags = False

    if args.route_construction:
        has_explicit_flags = True
        requested_types.append("route_construction_cmp")
    if args.resolver_cmp:
        has_explicit_flags = True
        if args.grouped:
            requested_types.append("resolver_cmp_grouped")
            requested_types.append("policy_cmp_grouped")
            requested_types.append("protocol_cmp_grouped")
        else:
            requested_types.append("resolver_cmp")
    if args.protocol_cmp:
        has_explicit_flags = True
        requested_types.append("protocol_cmp_grouped")
    if args.work_cmp:
        has_explicit_flags = True
        requested_types.append("work_cmp")

    if has_explicit_flags:
        return requested_types
    if args.grouped:
        return ["resolver_cmp_grouped", "policy_cmp_grouped", "protocol_cmp_grouped"]
    return ["resolver_cmp"]


def _route_construction_of_row(row: dict[str, object]) -> str:
    value = row.get("route_construction")
    text = str(value).strip() if value is not None else ""
    return text if text else "nearest"


def _route_construction_alias_of_row(row: dict[str, object]) -> str:
    route_construction = _route_construction_of_row(row)
    if route_construction == "nearest":
        return "nr"
    if route_construction == "reward_aligned":
        return "ra"
    return route_construction.replace(" ", "_").replace("-", "_") or "unknown"


def _protocol_of_row(row: dict[str, object]) -> str:
    value = row.get("protocol")
    text = str(value).strip().lower() if value is not None else ""
    return text if text in {"forced", "admission"} else "forced"


def _protocol_alias_of_row(row: dict[str, object]) -> str:
    protocol = _protocol_of_row(row)
    if protocol == "admission":
        return "aa"
    if protocol == "forced":
        return "forced"
    return protocol


def _resolver_combo_of_row(row: dict[str, object]) -> tuple[str, str]:
    return _route_construction_alias_of_row(row), _protocol_alias_of_row(row)


def _resolver_combo_folder_name(rows: list[dict[str, object]]) -> str:
    combos = sorted({_resolver_combo_of_row(row) for row in rows})
    if len(combos) != 1:
        joined = ", ".join(f"{route}_{protocol}" for route, protocol in combos) if combos else "<none>"
        raise ValueError(
            "resolver comparison rows must share one route_construction/protocol combo, "
            f"but found: {joined}"
        )
    route_alias, protocol_alias = combos[0]
    return f"{route_alias}_{protocol_alias}"


def _subplots_layout(count: int) -> tuple[int, int]:
    nrows = 2 if count > 3 else 1
    ncols = max(1, math.ceil(count / nrows))
    return nrows, ncols


def _visible_axis_grid(axes: list[list[plt.Axes]]) -> list[plt.Axes]:
    return [axis for axis_row in axes for axis in axis_row]


def _display_label(name: str) -> str:
    return str(name).replace("_", " ")


def _metric_title(metric: str) -> str:
    return "Reward" if str(metric).strip().lower() == "rew" else str(metric)


def _metric_ylabel(metric: str) -> str:
    return "Reward" if str(metric).strip().lower() == "rew" else str(metric)


def _ordered_resolvers(labels: list[str]) -> list[str]:
    present = [str(label).strip() for label in labels if str(label).strip()]
    present_set = set(present)

    known_non_hungarian = [
        resolver
        for resolver in RESOLVER_ORDER_BASE
        if resolver != "hungarian" and resolver in present_set
    ]
    unknown = sorted(
        resolver
        for resolver in present
        if resolver not in set(RESOLVER_ORDER_BASE)
    )
    ordered = known_non_hungarian + unknown
    if "hungarian" in present_set:
        ordered.append("hungarian")
    return ordered


def _canonical_resolver_name(resolver: object) -> str:
    text = str(resolver or "").strip().lower()
    if text in {"closest", "ctc", "closest_then_capacity"}:
        return "closest_then_capacity"
    return text


def _series_color_map(labels: list[str], preferred: dict[str, str] | None = None) -> dict[str, str]:
    color_map: dict[str, str] = {}
    used_colors: set[str] = set()
    preferred = preferred or {}

    for label in labels:
        if label in preferred:
            color_map[label] = preferred[label]
            used_colors.add(preferred[label])

    palette_iter = [color for color in COLORBLIND_PALETTE if color not in used_colors] + COLORBLIND_PALETTE
    palette_index = 0
    for label in labels:
        if label in color_map:
            continue
        color_map[label] = palette_iter[palette_index % len(palette_iter)]
        palette_index += 1
    return color_map


def _grouped_figure_width(num_groups: int) -> float:
    return max(8.0, 1.1 * max(num_groups, 1) + 4.0)


def _metric_limits_from_grouped_series(series_values: dict[str, list[tuple[float | None, float]]]) -> tuple[float, float]:
    lower = math.inf
    upper = -math.inf

    for values in series_values.values():
        for mean, std in values:
            if mean is None or (isinstance(mean, float) and math.isnan(mean)):
                continue
            std_value = float(std or 0.0)
            lower = min(lower, float(mean) - std_value, 0.0)
            upper = max(upper, float(mean) + std_value, 0.0)

    if lower is math.inf or upper is -math.inf:
        return (-1.0, 1.0)
    if math.isclose(lower, upper):
        padding = abs(lower) * 0.05 or 1.0
        return lower - padding, upper + padding
    padding = (upper - lower) * 0.05
    return lower - padding, upper + padding


def _protocol_cmp_policy_label(policy: str) -> str:
    if policy == "proposal_joint_completion":
        return "proposal joint completion"
    return _display_label(policy)


def _protocol_cmp_series_for_resolver(
    rows: list[dict[str, object]],
    resolver: str,
    metric: str,
) -> tuple[list[str], dict[str, list[tuple[float | None, float]]]]:
    protocol_display_order = [PROTOCOL_COMPARE_DISPLAY[protocol] for protocol in PROTOCOL_COMPARE_ORDER]
    series_values: dict[str, list[tuple[float | None, float]]] = {
        protocol_label: []
        for protocol_label in protocol_display_order
    }
    selected_policy_labels: list[str] = []

    for canonical_policy, aliases in PROTOCOL_COMPARE_POLICIES:
        per_protocol_values: dict[str, tuple[float | None, float]] = {}
        for protocol in PROTOCOL_COMPARE_ORDER:
            matched_rows = [
                row for row in rows
                if _canonical_resolver_name(row.get("resolver")) == resolver
                and str(row.get("pol", "")).strip().lower() in aliases
                and _protocol_alias_of_row(row) == protocol
            ]
            means = [safe_number(row.get(metric)) for row in matched_rows]
            stds = [safe_number(row.get(f"{metric}_std")) or 0.0 for row in matched_rows]
            valid = [(mean, std) for mean, std in zip(means, stds) if mean is not None]
            if not valid:
                per_protocol_values[protocol] = (None, 0.0)
                continue
            valid_means = [float(mean) for mean, _ in valid]
            valid_stds = [float(std) for _, std in valid]
            per_protocol_values[protocol] = (
                sum(valid_means) / len(valid_means),
                sum(valid_stds) / len(valid_stds) if valid_stds else 0.0,
            )

        if any(per_protocol_values[protocol][0] is None for protocol in PROTOCOL_COMPARE_ORDER):
            continue

        selected_policy_labels.append(_protocol_cmp_policy_label(canonical_policy))
        for protocol in PROTOCOL_COMPARE_ORDER:
            protocol_label = PROTOCOL_COMPARE_DISPLAY[protocol]
            series_values[protocol_label].append(per_protocol_values[protocol])

    return selected_policy_labels, series_values


def _policy_marker_map(policies: list[str]) -> dict[str, str]:
    ordered = sorted({str(policy).strip() for policy in policies if str(policy).strip()})
    return {
        policy: POLICY_MARKERS[index % len(POLICY_MARKERS)]
        for index, policy in enumerate(ordered)
    }


def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    efficient: list[tuple[float, float]] = []
    for point in points:
        x_value, y_value = point
        dominated = False
        for other in points:
            if other is point:
                continue
            other_x, other_y = other
            if (other_x <= x_value and other_y >= y_value) and (other_x < x_value or other_y > y_value):
                dominated = True
                break
        if not dominated:
            efficient.append(point)
    efficient.sort(key=lambda pair: pair[0])
    return efficient


def _work_xlabel(work_x: str) -> str:
    if work_x == "relative":
        return "Relative estimated computational work"
    return "Estimated computational work"


def _work_title(metric: str) -> str:
    metric_display = _metric_title(metric)
    return f"{metric_display} vs estimated computational work"


def _work_footer() -> str:
    return (
        "Analytical operation-count proxy derived from aggregated metrics; "
        "not measured wall-clock runtime."
    )


def set_tight_log_x_limits(
    axis: plt.Axes,
    x_values: list[float],
    *,
    padding_fraction: float = 0.06,
    minimum_log_span: float = 0.20,
) -> None:
    valid_x: list[float] = []
    for value in x_values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value) and numeric_value > 0.0:
            valid_x.append(numeric_value)

    if not valid_x:
        return

    padding_fraction = max(float(padding_fraction), 0.0)
    minimum_log_span = max(float(minimum_log_span), 0.0)

    log_min = math.log10(min(valid_x))
    log_max = math.log10(max(valid_x))

    if math.isclose(log_min, log_max):
        half_span = minimum_log_span / 2.0
        lower_log = log_min - half_span
        upper_log = log_max + half_span
    else:
        log_span = log_max - log_min
        padding = max(log_span * padding_fraction, 0.03)
        lower_log = log_min - padding
        upper_log = log_max + padding

    axis.set_xlim(10.0 ** lower_log, 10.0 ** upper_log)
    axis.set_autoscalex_on(False)


def _apply_grouped_bar_plot(
    axis: plt.Axes,
    *,
    x_labels: list[str],
    series_labels: list[str],
    series_values: dict[str, list[tuple[float | None, float]]],
    series_colors: dict[str, str],
    y_min: float,
    y_max: float,
) -> None:
    if not x_labels or not series_labels:
        axis.set_visible(False)
        return

    positions = list(range(len(x_labels)))
    bar_width = GROUPED_BAR_WIDTH
    offsets = [
        (index - (len(series_labels) - 1) / 2.0) * bar_width
        for index in range(len(series_labels))
    ]

    for index, series_label in enumerate(series_labels):
        values = series_values.get(series_label, [])
        means = [mean for mean, _ in values]
        stds = [std for _, std in values]
        shifted_positions = [position + offsets[index] for position in positions]
        axis.bar(
            shifted_positions,
            means,
            width=bar_width,
            yerr=stds,
            capsize=2.5,
            label=_display_label(series_label),
            color=series_colors.get(series_label, "#4C78A8"),
            edgecolor="#2F3E4E",
            alpha=0.9,
            linewidth=0.6,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(x_labels, rotation=43, ha="right")
    axis.set_ylim(y_min, y_max)
    axis.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
    axis.set_axisbelow(True)


def _aggregate_series_for_grouped_plot(
    rows: list[dict[str, object]],
    *,
    outer_key: str,
    inner_key: str,
    metric: str,
) -> tuple[list[str], list[str], dict[str, list[tuple[float | None, float]]]]:
    x_labels = sorted({str(row.get(outer_key, "")).strip() for row in rows if str(row.get(outer_key, "")).strip()})
    series_labels = sorted({str(row.get(inner_key, "")).strip() for row in rows if str(row.get(inner_key, "")).strip()})
    series_values: dict[str, list[tuple[float | None, float]]] = {}

    for series_label in series_labels:
        values_for_series: list[tuple[float | None, float]] = []
        for x_label in x_labels:
            matched_rows = [
                row for row in rows
                if str(row.get(outer_key, "")).strip() == x_label and str(row.get(inner_key, "")).strip() == series_label
            ]
            means = [safe_number(row.get(metric)) for row in matched_rows]
            stds = [safe_number(row.get(f"{metric}_std")) or 0.0 for row in matched_rows]
            valid = [(mean, std) for mean, std in zip(means, stds) if mean is not None]
            if not valid:
                values_for_series.append((math.nan, 0.0))
                continue
            valid_means = [mean for mean, _ in valid]
            valid_stds = [std for _, std in valid]
            values_for_series.append((sum(valid_means) / len(valid_means), sum(valid_stds) / len(valid_stds) if valid_stds else 0.0))
        series_values[series_label] = values_for_series

    return x_labels, series_labels, series_values


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
    resolver_dir = output_dir / "resolver_cmp" / _resolver_combo_folder_name(rows)
    resolver_dir.mkdir(parents=True, exist_ok=True)

    resolvers = sorted({str(row["resolver"]) for row in rows if row.get("resolver") is not None})
    saved_paths: list[Path] = []
    nrows = 2 if len(resolvers) > 3 else 1
    ncols = math.ceil(len(resolvers) / nrows)

    for metric in metrics:
        y_min, y_max = metric_limits(rows, metric)
        figure, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(max(5 * ncols, 6), 5 * nrows),
            squeeze=False,
            sharey=True,
        )
        axis_grid = [axis for axis_row in axes for axis in axis_row]
        figure.suptitle(f"{metric}: comparison across resolvers", fontsize=14)

        for axis, resolver in zip(axis_grid, resolvers):
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

        for axis in axis_grid[len(resolvers):]:
            axis.set_visible(False)

        axis_grid[0].set_ylabel(metric)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        output_path = resolver_dir / f"{metric}_resolver_cmp.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def plot_resolver_cmp_grouped(rows: list[dict[str, object]], metrics: list[str], output_dir: Path) -> list[Path]:
    resolver_dir = output_dir / "resolver_cmp_grouped" / _resolver_combo_folder_name(rows)
    resolver_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    resolvers = _ordered_resolvers(
        [str(row.get("resolver", "")).strip() for row in rows if str(row.get("resolver", "")).strip()]
    )
    resolver_colors = _series_color_map(resolvers, preferred=RESOLVER_COLOR_MAP)

    for metric in metrics:
        y_min, y_max = metric_limits(rows, metric)
        x_labels, _, series_values = _aggregate_series_for_grouped_plot(
            rows,
            outer_key="pol",
            inner_key="resolver",
            metric=metric,
        )
        figure, axis = plt.subplots(figsize=(_grouped_figure_width(len(x_labels)), 5.6))
        metric_display = _metric_title(metric)
        figure.suptitle(f"{metric_display} by policy and resolver", fontsize=14)
        _apply_grouped_bar_plot(
            axis,
            x_labels=x_labels,
            series_labels=resolvers,
            series_values=series_values,
            series_colors=resolver_colors,
            y_min=y_min,
            y_max=y_max,
        )
        axis.set_ylabel(_metric_ylabel(metric))
        axis.legend(
            title="resolver",
            fontsize=10,
            title_fontsize=10,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
        )
        figure.tight_layout(rect=(0.0, 0.0, 0.84, 0.93))
        output_path = resolver_dir / f"{metric}_resolver_cmp_grouped.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def plot_policy_cmp_grouped(rows: list[dict[str, object]], metrics: list[str], output_dir: Path) -> list[Path]:
    policy_dir = output_dir / "policy_cmp_grouped" / _resolver_combo_folder_name(rows)
    policy_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    policies = sorted({str(row.get("pol", "")).strip() for row in rows if str(row.get("pol", "")).strip()})
    policy_colors = _series_color_map(policies)
    ordered_resolvers = _ordered_resolvers(
        [str(row.get("resolver", "")).strip() for row in rows if str(row.get("resolver", "")).strip()]
    )

    for metric in metrics:
        y_min, y_max = metric_limits(rows, metric)
        x_labels, _, series_values = _aggregate_series_for_grouped_plot(
            rows,
            outer_key="resolver",
            inner_key="pol",
            metric=metric,
        )
        ordered_x_labels = [resolver for resolver in ordered_resolvers if resolver in set(x_labels)]
        series_values_reordered: dict[str, list[tuple[float | None, float]]] = {}
        x_label_to_index = {label: index for index, label in enumerate(x_labels)}
        for policy, values in series_values.items():
            reordered = [values[x_label_to_index[label]] for label in ordered_x_labels]
            series_values_reordered[policy] = reordered

        figure, axis = plt.subplots(figsize=(_grouped_figure_width(len(ordered_x_labels)), 5.6))
        metric_display = _metric_title(metric)
        figure.suptitle(f"{metric_display} by resolver and policy", fontsize=14)
        _apply_grouped_bar_plot(
            axis,
            x_labels=[_display_label(label) for label in ordered_x_labels],
            series_labels=policies,
            series_values=series_values_reordered,
            series_colors=policy_colors,
            y_min=y_min,
            y_max=y_max,
        )
        axis.set_ylabel(_metric_ylabel(metric))
        axis.legend(
            title="policy",
            fontsize=10,
            title_fontsize=10,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
        )
        figure.tight_layout(rect=(0.0, 0.0, 0.84, 0.93))
        output_path = policy_dir / f"{metric}_policy_cmp_grouped.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def plot_protocol_cmp_grouped(rows: list[dict[str, object]], metrics: list[str], output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    resolvers = _ordered_resolvers(
        [
            _canonical_resolver_name(row.get("resolver"))
            for row in rows
            if _canonical_resolver_name(row.get("resolver"))
        ]
    )

    protocol_labels = [PROTOCOL_COMPARE_DISPLAY[protocol] for protocol in PROTOCOL_COMPARE_ORDER]
    protocol_colors = _series_color_map(protocol_labels, preferred={
        "f": "#4C78A8",
        "aa": "#F58518",
    })

    protocol_dir: Path | None = None

    for metric in metrics:
        resolver_series: list[tuple[str, list[str], dict[str, list[tuple[float | None, float]]]]] = []
        for resolver in resolvers:
            x_labels, series_values = _protocol_cmp_series_for_resolver(rows, resolver, metric)
            if not x_labels:
                continue
            resolver_series.append((resolver, x_labels, series_values))

        if not resolver_series:
            continue

        if protocol_dir is None:
            protocol_dir = output_dir / PROTOCOL_CMP_OUTPUT_SUBDIR
            protocol_dir.mkdir(parents=True, exist_ok=True)

        nrows, ncols = _subplots_layout(len(resolver_series))
        figure, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(max(5 * ncols, 7), 5 * nrows),
            squeeze=False,
            sharey=True,
        )
        axis_grid = _visible_axis_grid(axes)
        metric_display = _metric_title(metric)
        figure.suptitle(f"{metric_display} by resolver and protocol", fontsize=14)

        merged_series_values: dict[str, list[tuple[float | None, float]]] = {label: [] for label in protocol_labels}
        for _, _, series_values in resolver_series:
            for protocol_label in protocol_labels:
                merged_series_values[protocol_label].extend(series_values.get(protocol_label, []))
        y_min, y_max = _metric_limits_from_grouped_series(merged_series_values)

        for index, (resolver, x_labels, series_values) in enumerate(resolver_series):
            axis = axis_grid[index]
            _apply_grouped_bar_plot(
                axis,
                x_labels=x_labels,
                series_labels=protocol_labels,
                series_values=series_values,
                series_colors=protocol_colors,
                y_min=y_min,
                y_max=y_max,
            )
            axis.set_title(_display_label(resolver))
            if index % ncols == 0:
                axis.set_ylabel(_metric_ylabel(metric))

        for axis in axis_grid[len(resolver_series):]:
            axis.set_visible(False)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=8,
                markerfacecolor=protocol_colors.get(protocol_label, "#4C78A8"),
                markeredgecolor="#2F3E4E",
                label=protocol_label,
            )
            for protocol_label in protocol_labels
        ]
        figure.legend(
            handles=legend_handles,
            title="protocol",
            fontsize=10,
            title_fontsize=10,
            loc="upper left",
            bbox_to_anchor=(0.88, 0.98),
            frameon=False,
        )

        figure.tight_layout(rect=(0.0, 0.0, 0.86, 0.93))
        output_path = protocol_dir / f"{metric}_protocol_cmp_grouped.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def plot_work_cmp(
    rows: list[dict[str, object]],
    metrics: list[str],
    output_dir: Path,
    *,
    work_x: str,
    work_x_padding: float,
    pareto: bool,
    annotate_work: bool,
) -> list[Path]:
    work_dir = output_dir / "work_cmp" / _resolver_combo_folder_name(rows)
    work_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    resolver_labels = _ordered_resolvers([
        str(row.get("resolver", "")).strip()
        for row in rows
        if str(row.get("resolver", "")).strip()
    ])
    resolver_colors = _series_color_map(resolver_labels, preferred=RESOLVER_COLOR_MAP)
    policy_markers = _policy_marker_map([
        str(row.get("pol", "")).strip()
        for row in rows
        if str(row.get("pol", "")).strip()
    ])

    for metric in metrics:
        metric_points: list[dict[str, object]] = []
        skipped_rows = 0

        for row in rows:
            quality = safe_number(row.get(metric))
            work_total = safe_number(row.get("work_total"))
            if quality is None or work_total is None or work_total <= 0.0:
                skipped_rows += 1
                continue

            metric_std = safe_number(row.get(f"{metric}_std")) or 0.0
            metric_points.append(
                {
                    "quality": float(quality),
                    "quality_std": float(metric_std),
                    "work_total": float(work_total),
                    "resolver": str(row.get("resolver", "")).strip(),
                    "policy": str(row.get("pol", "")).strip(),
                }
            )

        if not metric_points:
            print(
                f"[warn] No valid rows for work comparison metric '{metric}' in {work_dir}",
                file=sys.stderr,
            )
            continue

        if skipped_rows:
            print(
                f"[warn] Skipped {skipped_rows} row(s) for work comparison metric '{metric}' in {work_dir}",
                file=sys.stderr,
            )

        min_positive_work = min(point["work_total"] for point in metric_points if point["work_total"] > 0.0)
        if min_positive_work <= 0.0:
            print(
                f"[warn] No positive work_total values for metric '{metric}' in {work_dir}",
                file=sys.stderr,
            )
            continue

        for point in metric_points:
            if work_x == "relative":
                point["x_value"] = float(point["work_total"]) / min_positive_work
            else:
                point["x_value"] = float(point["work_total"])

        figure, axis = plt.subplots(figsize=(8.8, 5.8))
        axis.set_xscale("log")

        for point in metric_points:
            resolver = str(point["resolver"])
            policy = str(point["policy"])
            x_value = float(point["x_value"])
            y_value = float(point["quality"])
            y_std = float(point["quality_std"])

            axis.errorbar(
                [x_value],
                [y_value],
                yerr=[y_std],
                fmt="none",
                ecolor=resolver_colors.get(resolver, "#4C78A8"),
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
                zorder=2,
            )
            axis.scatter(
                [x_value],
                [y_value],
                marker=policy_markers.get(policy, "o"),
                color=resolver_colors.get(resolver, "#4C78A8"),
                edgecolors="#2F3E4E",
                linewidths=0.6,
                s=52,
                alpha=0.95,
                zorder=3,
            )
            if annotate_work:
                axis.annotate(
                    _display_label(policy),
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=8,
                )

        if pareto:
            frontier_input = [(float(point["x_value"]), float(point["quality"])) for point in metric_points]
            frontier = _pareto_frontier(frontier_input)
            if len(frontier) >= 2:
                frontier_x = [point[0] for point in frontier]
                frontier_y = [point[1] for point in frontier]
                axis.plot(frontier_x, frontier_y, linestyle="--", linewidth=1.0, color="#222222", zorder=1)

        plotted_x_values = [float(point["x_value"]) for point in metric_points]
        set_tight_log_x_limits(axis, plotted_x_values, padding_fraction=work_x_padding)
        axis.xaxis.set_major_locator(LogLocator(base=10.0))
        axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        axis.xaxis.set_minor_formatter(NullFormatter())

        axis.set_xlabel(_work_xlabel(work_x))
        axis.set_ylabel(_metric_ylabel(metric))
        axis.set_title(_work_title(metric))
        axis.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
        axis.set_axisbelow(True)

        visible_resolvers = [resolver for resolver in resolver_labels if any(str(point["resolver"]) == resolver for point in metric_points)]
        resolver_handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=resolver_colors.get(resolver, "#4C78A8"), markeredgecolor="#2F3E4E", label=_display_label(resolver))
            for resolver in visible_resolvers
        ]
        visible_policies = sorted({str(point["policy"]) for point in metric_points})
        policy_handles = [
            Line2D([0], [0], marker=policy_markers.get(policy, "o"), linestyle="", markersize=7, markerfacecolor="#FFFFFF", markeredgecolor="#2F3E4E", label=_display_label(policy))
            for policy in visible_policies
        ]

        resolver_legend = axis.legend(
            handles=resolver_handles,
            title="resolver",
            fontsize=9,
            title_fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
        )
        axis.add_artist(resolver_legend)
        axis.legend(
            handles=policy_handles,
            title="policy",
            fontsize=9,
            title_fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.48),
            frameon=False,
        )

        figure.text(0.01, 0.01, _work_footer(), fontsize=8, alpha=0.85)
        figure.tight_layout(rect=(0.0, 0.03, 0.81, 0.97))

        output_path = work_dir / f"{metric}_work_cmp.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def plot_route_construction_cmp(rows: list[dict[str, object]], metrics: list[str], output_dir: Path) -> list[Path]:
    policy_labels = sorted({str(row.get("pol", "")).strip() for row in rows if str(row.get("pol", "")).strip()})
    resolver_labels = _ordered_resolvers([
        _canonical_resolver_name(row.get("resolver"))
        for row in rows
        if _canonical_resolver_name(row.get("resolver"))
    ])

    # Only create route_construction_cmp when at least one policy/resolver pair has
    # both nearest and reward_aligned measurements available.
    has_any_pair = False
    for policy in policy_labels:
        for resolver in resolver_labels:
            modes_present = {
                _route_construction_of_row(row)
                for row in rows
                if str(row.get("pol", "")).strip() == policy
                and _canonical_resolver_name(row.get("resolver")) == resolver
            }
            if "nearest" in modes_present and "reward_aligned" in modes_present:
                has_any_pair = True
                break
        if has_any_pair:
            break

    if not has_any_pair:
        return []

    route_dir = output_dir / ROUTE_CONSTRUCTION_OUTPUT_SUBDIR
    route_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for metric in metrics:
        y_min, y_max = metric_limits(rows, metric)
        policy_keys = [_normalize_policy_key(policy) for policy in policy_labels]
        nrows = max(1, len(policy_keys))
        ncols = max(1, len(resolver_labels))
        figure, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(max(4.6 * ncols, 8.0), max(4.0 * nrows, 4.8)),
            squeeze=False,
            sharey=True,
        )
        figure.suptitle(f"{_metric_title(metric)}: route construction (nearest vs reward aligned)", fontsize=14)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=8,
                markerfacecolor=ROUTE_MODE_COLORS[mode],
                markeredgecolor="#2F3E4E",
                label=_display_label(mode),
            )
            for mode in ROUTE_MODE_ORDER
        ]

        for row_index, policy_key in enumerate(policy_keys):
            for col_index, resolver in enumerate(resolver_labels):
                axis = axes[row_index][col_index]
                pair_rows = [
                    row for row in rows
                    if _normalize_policy_key(row.get("pol", "")) == policy_key
                    and _canonical_resolver_name(row.get("resolver")) == resolver
                ]

                means_by_mode: dict[str, float] = {}
                stds_by_mode: dict[str, float] = {}
                for mode in ROUTE_MODE_ORDER:
                    mode_rows = [
                        row for row in pair_rows
                        if _route_construction_of_row(row) == mode
                    ]
                    mode_means = [safe_number(row.get(metric)) for row in mode_rows]
                    mode_stds = [safe_number(row.get(f"{metric}_std")) or 0.0 for row in mode_rows]
                    valid_means = [float(value) for value in mode_means if value is not None]
                    valid_stds = [float(value) for value in mode_stds]

                    if valid_means:
                        means_by_mode[mode] = float(sum(valid_means) / len(valid_means))
                        stds_by_mode[mode] = float(sum(valid_stds) / len(valid_stds)) if valid_stds else 0.0
                    else:
                        # Requested behavior: missing measurements are treated as zero.
                        means_by_mode[mode] = 0.0
                        stds_by_mode[mode] = 0.0

                positions = list(range(len(ROUTE_MODE_ORDER)))
                values = [means_by_mode[mode] for mode in ROUTE_MODE_ORDER]
                errors = [stds_by_mode[mode] for mode in ROUTE_MODE_ORDER]
                colors = [ROUTE_MODE_COLORS.get(mode, "#4C78A8") for mode in ROUTE_MODE_ORDER]

                axis.bar(
                    positions,
                    values,
                    yerr=errors,
                    capsize=4,
                    color=colors,
                    edgecolor="#2F3E4E",
                    alpha=0.92,
                )
                axis.set_xticks(positions)
                axis.set_xticklabels([_display_label(mode) for mode in ROUTE_MODE_ORDER], rotation=0, fontsize=8)
                axis.set_ylim(y_min, y_max)
                axis.grid(axis="y", alpha=0.25, linestyle="--")
                axis.set_axisbelow(True)

                if row_index == 0:
                    axis.set_title(_display_label(resolver), fontsize=10)
                if col_index == 0:
                    axis.set_ylabel(f"{_display_label(policy_key)}\n{_metric_ylabel(metric)}", fontsize=9)

        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.985),
            fontsize=9,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        output_path = route_dir / f"{metric}_route_construction_cmp.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def main() -> int:
    args = parse_args()
    if float(args.work_x_padding) < 0.0:
        raise ValueError("--work-x-padding must be non-negative")

    rows, available_metrics = load_rows(args.csv_path)
    metrics = select_metrics(args.metrics, available_metrics)
    plot_types = resolve_plot_types(args)

    if "work_cmp" in plot_types and "work_total" not in rows[0]:
        raise ValueError(
            "metrics_wide.csv has no work estimates.\n"
            "Run:\n"
            "python estimate_work_metrics.py metrics_wide.csv --in-place"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_resolvers = _normalize_name_set(args.exclude_resolvers)
    exclude_policies = _normalize_policy_set(args.exclude_policies)

    saved_paths: list[Path] = []
    scenarios = _selected_scenarios(rows, args.scenario)
    for scenario in scenarios:
        scenario_rows = _rows_for_scenario(rows, scenario)
        scenario_rows = _filtered_rows(
            scenario_rows,
            exclude_resolvers=exclude_resolvers,
            exclude_policies=exclude_policies,
        )
        scenario_output_dir = output_dir / _scenario_folder_name(scenario)
        scenario_output_dir.mkdir(parents=True, exist_ok=True)

        if not scenario_rows:
            print(f"[warn] No rows remaining for scenario '{scenario}' after filtering", file=sys.stderr)
            continue

        if "resolver_cmp_grouped" in plot_types:
            resolver_combos = sorted({_resolver_combo_of_row(row) for row in scenario_rows})
            for route_alias, protocol_alias in resolver_combos:
                combo_rows = [
                    row for row in scenario_rows
                    if _resolver_combo_of_row(row) == (route_alias, protocol_alias)
                ]
                saved_paths.extend(plot_resolver_cmp_grouped(combo_rows, metrics, scenario_output_dir))
                saved_paths.extend(plot_policy_cmp_grouped(combo_rows, metrics, scenario_output_dir))
        if "protocol_cmp_grouped" in plot_types:
            saved_paths.extend(plot_protocol_cmp_grouped(scenario_rows, metrics, scenario_output_dir))
        if "resolver_cmp" in plot_types:
            resolver_combos = sorted({_resolver_combo_of_row(row) for row in scenario_rows})
            for route_alias, protocol_alias in resolver_combos:
                combo_rows = [
                    row for row in scenario_rows
                    if _resolver_combo_of_row(row) == (route_alias, protocol_alias)
                ]
                saved_paths.extend(plot_resolver_cmp(combo_rows, metrics, scenario_output_dir))
        if "route_construction_cmp" in plot_types:
            saved_paths.extend(plot_route_construction_cmp(scenario_rows, metrics, scenario_output_dir))
        if "work_cmp" in plot_types:
            resolver_combos = sorted({_resolver_combo_of_row(row) for row in scenario_rows})
            for route_alias, protocol_alias in resolver_combos:
                combo_rows = [
                    row for row in scenario_rows
                    if _resolver_combo_of_row(row) == (route_alias, protocol_alias)
                ]
                if not combo_rows:
                    print(
                        f"[warn] No rows for work comparison in scenario='{scenario}' combo='{route_alias}_{protocol_alias}'",
                        file=sys.stderr,
                    )
                    continue
                saved_paths.extend(
                    plot_work_cmp(
                        combo_rows,
                        metrics,
                        scenario_output_dir,
                        work_x=args.work_x,
                        work_x_padding=float(args.work_x_padding),
                        pareto=bool(args.pareto),
                        annotate_work=bool(args.annotate_work),
                    )
                )

    print(f"Saved {len(saved_paths)} plot(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())