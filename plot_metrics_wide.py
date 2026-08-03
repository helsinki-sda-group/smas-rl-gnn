#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from estimate_work_metrics import clip, insertion_pairs, normalize_name, resolver_work


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
    parser.add_argument(
        "--work-cmp-with-rl",
        action="store_true",
        help=(
            "Create an additional reward-vs-work plot with one RL point overlaid on baselines, "
            "including baseline and extended Pareto fronts"
        ),
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("evaluation_metrics.log"),
        help="Evaluation log used to derive the RL point (default: ./evaluation_metrics.log)",
    )
    parser.add_argument(
        "--eval-proposer",
        type=str,
        default="1hop",
        help="RL proposer mode for overlay point (default: 1hop)",
    )
    parser.add_argument(
        "--eval-resolver",
        type=str,
        default="closest",
        help="RL resolver for overlay point (default: closest)",
    )
    parser.add_argument(
        "--eval-num-robots",
        type=float,
        default=6.0,
        help="Robot count used in RL work estimate (default: 6)",
    )
    parser.add_argument(
        "--eval-max-robot-capacity",
        type=float,
        default=2.0,
        help="Max robot capacity used in RL work estimate (default: 2)",
    )
    parser.add_argument(
        "--eval-joint-multiplier",
        type=float,
        default=1.5,
        help="Joint multiplier for RL work estimate (default: 1.5)",
    )
    parser.add_argument(
        "--eval-noop-candidates",
        type=float,
        default=1.0,
        help="Additional NOOP candidate count in actor inference estimate (default: 1)",
    )
    parser.add_argument(
        "--eval-scenario",
        type=str,
        default="randdest",
        help="Scenario where RL overlay plot is generated (default: randdest)",
    )
    parser.add_argument(
        "--eval-route-construction",
        type=str,
        default="nearest",
        help="Route construction for RL overlay plot selection (default: nearest)",
    )
    parser.add_argument(
        "--eval-protocol",
        type=str,
        default="aa",
        choices=["aa", "admission", "forced"],
        help="Protocol combo for RL overlay plot selection (default: aa)",
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


def _work_measure_label(work_measure: str) -> str:
    if work_measure == "parallel":
        return "parallel-time proxy"
    return "total"


def _work_xlabel_for_measure(work_x: str, work_measure: str) -> str:
    base_label = _work_xlabel(work_x)
    if work_measure == "parallel":
        return f"{base_label} (parallel-time proxy)"
    return base_label


def _work_title(metric: str, *, work_measure: str = "total") -> str:
    metric_display = _metric_title(metric)
    if work_measure == "parallel":
        return f"{metric_display} vs est. computational work (parallel-time proxy)"
    return f"{metric_display} vs est. computational work"


def _work_footer() -> str:
    return (
        "Analytical operation-count proxy derived from aggregated metrics; "
        "not measured wall-clock runtime."
    )


def _parallel_time_proxy(
    *,
    work_proposer: float | None,
    work_resolver: float | None,
    num_robots: float | None,
) -> float | None:
    if work_proposer is None or work_resolver is None or num_robots is None:
        return None
    if not math.isfinite(work_proposer) or not math.isfinite(work_resolver) or not math.isfinite(num_robots):
        return None
    if num_robots <= 0.0:
        return None
    return float(work_resolver) + (float(work_proposer) / float(num_robots))


def _resolved_num_robots_for_parallel(row: dict[str, object]) -> float | None:
    direct_num_robots = safe_number(row.get("num_robots"))
    if direct_num_robots is not None and direct_num_robots > 0.0:
        return direct_num_robots

    work_candidate_scan = safe_number(row.get("work_candidate_scan"))
    msd = safe_number(row.get("msd"))
    mcand = safe_number(row.get("mcand"))
    if (
        work_candidate_scan is not None
        and msd is not None
        and mcand is not None
        and msd > 0.0
        and mcand > 0.0
    ):
        inferred_num_robots = work_candidate_scan / (msd * mcand)
        if math.isfinite(inferred_num_robots) and inferred_num_robots > 0.0:
            return inferred_num_robots

    work_active_proposals = safe_number(row.get("work_active_proposals"))
    noop = safe_number(row.get("noop"))
    if (
        work_active_proposals is not None
        and msd is not None
        and noop is not None
        and msd > 0.0
        and (1.0 - float(noop)) > 0.0
    ):
        inferred_num_robots = work_active_proposals / (msd * (1.0 - float(noop)))
        if math.isfinite(inferred_num_robots) and inferred_num_robots > 0.0:
            return inferred_num_robots

    return None


def _work_output_suffix(work_measure: str) -> str:
    if work_measure == "parallel":
        return "_parallel"
    return ""


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


def _route_alias_from_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"nearest", "nr"}:
        return "nr"
    if text in {"reward_aligned", "ra", "reward-aligned", "reward aligned"}:
        return "ra"
    return text.replace(" ", "_").replace("-", "_") or "unknown"


def _protocol_alias_from_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"aa", "admission"}:
        return "aa"
    if text == "forced":
        return "forced"
    return text


def _split_columns_from_header(header_line: str) -> list[str]:
    columns: list[str] = []
    for segment in header_line.split("|"):
        columns.extend(part.strip().lower() for part in segment.strip().split() if part.strip())
    return columns


def _parse_eval_log_rows(log_path: Path) -> list[dict[str, Any]]:
    lines = [
        line.rstrip("\n")
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    header_index: int | None = None
    header_columns: list[str] = []
    for index, line in enumerate(lines):
        low = line.lower()
        if "|" in line and "pol" in low and "seed" in low and "rew" in low and "ts" in low:
            columns = _split_columns_from_header(line)
            if columns:
                header_index = index
                header_columns = columns
                break

    if header_index is None:
        raise ValueError(f"Could not find metrics header in evaluation log: {log_path}")

    parsed_rows: list[dict[str, Any]] = []
    for line in lines[header_index + 1 :]:
        if line.lstrip().startswith("#"):
            continue

        fields: list[str] = []
        for segment in line.split("|"):
            fields.extend(part for part in segment.strip().split() if part)

        if len(fields) < len(header_columns):
            fields.extend([""] * (len(header_columns) - len(fields)))
        if len(fields) > len(header_columns):
            fields = fields[: len(header_columns)]

        row: dict[str, Any] = {}
        for key, raw in zip(header_columns, fields):
            text = str(raw).strip()
            if key in {"seed", "ts"}:
                try:
                    row[key] = int(text)
                except ValueError:
                    row[key] = None
                continue
            try:
                row[key] = float(text)
            except ValueError:
                row[key] = text

        if row.get("ts") is None:
            continue
        if safe_number(row.get("rew")) is None:
            continue
        parsed_rows.append(row)

    if not parsed_rows:
        raise ValueError(f"No usable evaluation rows in: {log_path}")
    return parsed_rows


def _aggregate_eval_by_ts(eval_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in eval_rows:
        grouped.setdefault(int(row["ts"]), []).append(row)

    required_metrics = ["rew", "mcand", "msd", "dstep", "noop", "ovrlap", "shared"]
    out_rows: list[dict[str, float]] = []
    for ts, rows in grouped.items():
        item: dict[str, float] = {"ts": float(ts), "n_seeds": float(len(rows))}
        for metric in required_metrics:
            values: list[float] = []
            for row in rows:
                value = safe_number(row.get(metric))
                if value is not None and math.isfinite(value):
                    values.append(float(value))
            if not values:
                raise ValueError(f"Missing numeric metric '{metric}' at ts={ts}")
            mean_value = sum(values) / float(len(values))
            if len(values) > 1:
                variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
                std_value = math.sqrt(variance)
            else:
                std_value = 0.0
            item[f"{metric}_mean"] = mean_value
            item[f"{metric}_std"] = std_value
        out_rows.append(item)

    out_rows.sort(key=lambda row: row["ts"])
    return out_rows


def _canonical_eval_resolver_name(resolver: str) -> str:
    normalized = normalize_name(resolver)
    if normalized in {"closest", "ctc", "closest_than_capacity", "closest_then_capacity"}:
        return "closest_then_capacity"
    return normalized


def _proposer_actor_work(
    *,
    proposer: str,
    candidate_scan_work: float,
    insertion_pairs_value: float,
    competition_factor: float,
    joint_multiplier: float,
) -> float:
    normalized = normalize_name(proposer)
    if normalized in {"1hop", "gnn_1hop", "one_hop"}:
        return candidate_scan_work
    if normalized in {"2hop", "gnn_2hop", "two_hop"}:
        return 2.0 * candidate_scan_work
    if normalized in {"random", "unique", "pickup_distance", "pickup_deadline"}:
        return candidate_scan_work
    if normalized == "pickup_deadline_distance":
        return 2.0 * candidate_scan_work
    if normalized == "predicted_reward":
        return candidate_scan_work * insertion_pairs_value
    if normalized == "predicted_reward_joint":
        return candidate_scan_work * insertion_pairs_value * joint_multiplier
    if normalized in {"proposal_joint_competition", "predicted_reward_joint_competition"}:
        return candidate_scan_work * insertion_pairs_value * competition_factor
    raise ValueError(
        f"Unsupported eval proposer '{proposer}'. Supported: 1hop/2hop and baseline-like proposer aliases."
    )


def _estimate_rl_point_from_eval(
    *,
    eval_file: Path,
    proposer: str,
    resolver: str,
    num_robots: float,
    max_robot_capacity: float,
    joint_multiplier: float,
    noop_candidates: float,
) -> dict[str, Any]:
    eval_rows = _parse_eval_log_rows(eval_file)
    grouped = _aggregate_eval_by_ts(eval_rows)
    best = max(grouped, key=lambda row: (row["rew_mean"], -row["ts"]))

    R = max(float(num_robots), 0.0)
    M = max(float(best["msd_mean"]), 0.0)
    D = max(float(best["dstep_mean"]), 0.0)
    K = max(float(best["mcand_mean"]), 0.0)
    noop_fraction = clip(float(best["noop_mean"]), 0.0, 1.0)

    route_stops = 2.0 * max(float(max_robot_capacity), 0.0)
    insertion_value = insertion_pairs(route_stops)
    actor_candidate_scan = M * R * (K + max(float(noop_candidates), 0.0))
    active_proposal_work = M * R * (1.0 - noop_fraction)

    ovrlap = max(float(best["ovrlap_mean"]), 0.0)
    shared = max(float(best["shared_mean"]), 0.0)
    competition_factor = 1.0 + (ovrlap * shared)

    actor_work = _proposer_actor_work(
        proposer=proposer,
        candidate_scan_work=actor_candidate_scan,
        insertion_pairs_value=insertion_value,
        competition_factor=competition_factor,
        joint_multiplier=float(joint_multiplier),
    )

    resolver_name = _canonical_eval_resolver_name(resolver)
    resolver_estimate = resolver_work(
        resolver_name=resolver_name,
        active_proposal_work=active_proposal_work,
        insertion_pairs_value=insertion_value,
        joint_multiplier=float(joint_multiplier),
        decisions_per_episode=D,
        num_robots=R,
        mean_candidates=K,
    )
    if resolver_estimate is None:
        raise ValueError(f"Unsupported eval resolver '{resolver}' (normalized: {resolver_name})")

    return {
        "quality": float(best["rew_mean"]),
        "quality_std": float(best["rew_std"]),
        "work_proposer": float(actor_work),
        "work_resolver": float(resolver_estimate),
        "num_robots": R,
        "work_total": float(actor_work) + float(resolver_estimate),
        "resolver": resolver_name,
        "policy": f"rl_{proposer}",
        "ts": int(best["ts"]),
        "n_seeds": int(best["n_seeds"]),
    }


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
            work_proposer = safe_number(row.get("work_proposer"))
            work_resolver = safe_number(row.get("work_resolver"))
            num_robots = _resolved_num_robots_for_parallel(row)
            work_parallel = _parallel_time_proxy(
                work_proposer=work_proposer,
                work_resolver=work_resolver,
                num_robots=num_robots,
            )
            metric_points.append(
                {
                    "quality": float(quality),
                    "quality_std": float(metric_std),
                    "work_total": float(work_total),
                    "work_parallel": work_parallel,
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

        for work_measure in ["total", "parallel"]:
            measure_key = "work_total" if work_measure == "total" else "work_parallel"
            valid_points = [
                point for point in metric_points
                if isinstance(point.get(measure_key), (int, float)) and float(point[measure_key]) > 0.0
            ]
            if not valid_points:
                print(
                    f"[warn] No positive {measure_key} values for metric '{metric}' in {work_dir}",
                    file=sys.stderr,
                )
                continue

            min_positive_work = min(float(point[measure_key]) for point in valid_points)
            for point in valid_points:
                if work_x == "relative":
                    point["x_value"] = float(point[measure_key]) / min_positive_work
                else:
                    point["x_value"] = float(point[measure_key])

            figure, axis = plt.subplots(figsize=(8.8, 5.8))
            axis.set_xscale("log")

            for point in valid_points:
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
                frontier_input = [(float(point["x_value"]), float(point["quality"])) for point in valid_points]
                frontier = _pareto_frontier(frontier_input)
                if len(frontier) >= 2:
                    frontier_x = [point[0] for point in frontier]
                    frontier_y = [point[1] for point in frontier]
                    axis.plot(frontier_x, frontier_y, linestyle="--", linewidth=1.0, color="#222222", zorder=1)

            plotted_x_values = [float(point["x_value"]) for point in valid_points]
            set_tight_log_x_limits(axis, plotted_x_values, padding_fraction=work_x_padding)
            axis.xaxis.set_major_locator(LogLocator(base=10.0))
            axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
            axis.xaxis.set_minor_formatter(NullFormatter())

            axis.set_xlabel(_work_xlabel_for_measure(work_x, work_measure))
            axis.set_ylabel(_metric_ylabel(metric))
            axis.set_title(_work_title(metric, work_measure=work_measure))
            axis.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
            axis.set_axisbelow(True)

            visible_resolvers = [resolver for resolver in resolver_labels if any(str(point["resolver"]) == resolver for point in valid_points)]
            resolver_handles = [
                Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=resolver_colors.get(resolver, "#4C78A8"), markeredgecolor="#2F3E4E", label=_display_label(resolver))
                for resolver in visible_resolvers
            ]
            visible_policies = sorted({str(point["policy"]) for point in valid_points})
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

            output_path = work_dir / f"{metric}_work_cmp{_work_output_suffix(work_measure)}.png"
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(figure)
            saved_paths.append(output_path)

    return saved_paths


def plot_work_cmp_with_rl(
    rows: list[dict[str, object]],
    output_dir: Path,
    *,
    work_x: str,
    work_x_padding: float,
    annotate_work: bool,
    rl_point: dict[str, Any],
) -> list[Path]:
    work_dir = output_dir / "work_cmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    metric = "rew"
    metric_points: list[dict[str, object]] = []
    for row in rows:
        quality = safe_number(row.get(metric))
        work_total = safe_number(row.get("work_total"))
        if quality is None or work_total is None or work_total <= 0.0:
            continue
        quality_std = safe_number(row.get(f"{metric}_std")) or 0.0
        work_proposer = safe_number(row.get("work_proposer"))
        work_resolver = safe_number(row.get("work_resolver"))
        num_robots = _resolved_num_robots_for_parallel(row)
        metric_points.append(
            {
                "quality": float(quality),
                "quality_std": float(quality_std),
                "work_total": float(work_total),
                "work_parallel": _parallel_time_proxy(
                    work_proposer=work_proposer,
                    work_resolver=work_resolver,
                    num_robots=num_robots,
                ),
                "resolver": str(row.get("resolver", "")).strip(),
                "policy": str(row.get("pol", "")).strip(),
            }
        )

    if not metric_points:
        print("[warn] No valid baseline rows for rew_work_cmp_with_rl", file=sys.stderr)
        return []

    resolver_labels = _ordered_resolvers([
        str(point["resolver"]).strip()
        for point in metric_points
        if str(point["resolver"]).strip()
    ])
    resolver_colors = _series_color_map(resolver_labels, preferred=RESOLVER_COLOR_MAP)
    policy_markers = _policy_marker_map([
        str(point["policy"]).strip()
        for point in metric_points
        if str(point["policy"]).strip()
    ])

    visible_resolvers = [resolver for resolver in resolver_labels if any(str(point["resolver"]) == resolver for point in metric_points)]
    resolver_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=resolver_colors.get(resolver, "#4C78A8"),
            markeredgecolor="#2F3E4E",
            label=_display_label(resolver),
        )
        for resolver in visible_resolvers
    ]
    visible_policies = sorted({str(point["policy"]) for point in metric_points})
    policy_legend_label_map = {
        "pickup_deadline": "pickup deadline",
        "pickup_deadline_distance": "pickup deadline dist.",
        "pickup_distance": "pickup distance",
        "predicted_reward": "predicted reward",
        "predicted_reward_joint": "predicted reward joint",
        "proposal_joint_competition": "proposal joint comp.",
        "random": "random",
        "unique": "unique",
    }
    policy_handles = [
        Line2D(
            [0],
            [0],
            marker=policy_markers.get(policy, "o"),
            linestyle="",
            markersize=7,
            markerfacecolor="#FFFFFF",
            markeredgecolor="#2F3E4E",
            label=policy_legend_label_map.get(policy, _display_label(policy)),
        )
        for policy in visible_policies
    ]

    extra_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=10,
            markerfacecolor="#111111",
            markeredgecolor="#FFFFFF",
            label="RL point (inference)",
        ),
        Line2D([0], [0], linestyle="--", color="#333333", linewidth=1.1, label="Pareto baseline"),
        Line2D([0], [0], linestyle="-", color="#C44E52", linewidth=1.2, label="Pareto extended"),
    ]

    saved_paths: list[Path] = []
    for work_measure in ["total", "parallel"]:
        measure_key = "work_total" if work_measure == "total" else "work_parallel"
        valid_points = [
            point for point in metric_points
            if isinstance(point.get(measure_key), (int, float)) and float(point[measure_key]) > 0.0
        ]
        rl_measure_value = float(rl_point["work_total"])
        if work_measure == "parallel":
            rl_parallel_value = _parallel_time_proxy(
                work_proposer=safe_number(rl_point.get("work_proposer")),
                work_resolver=safe_number(rl_point.get("work_resolver")),
                num_robots=safe_number(rl_point.get("num_robots")),
            )
            if rl_parallel_value is None or rl_parallel_value <= 0.0:
                print("[warn] RL point has no valid parallel-time proxy; skipping RL parallel plot", file=sys.stderr)
                continue
            rl_measure_value = float(rl_parallel_value)

        if not valid_points:
            print(f"[warn] No valid baseline rows for rew_work_cmp_with_rl ({work_measure})", file=sys.stderr)
            continue

        min_positive_work = min(float(point[measure_key]) for point in valid_points)
        for point in valid_points:
            if work_x == "relative":
                point["x_value"] = float(point[measure_key]) / min_positive_work
            else:
                point["x_value"] = float(point[measure_key])

        if work_x == "relative":
            rl_x_value = rl_measure_value / min_positive_work
        else:
            rl_x_value = rl_measure_value

        figure, axis = plt.subplots(figsize=(8.8, 5.8))
        axis.set_xscale("log")

        for point in valid_points:
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

        axis.errorbar(
            [rl_x_value],
            [float(rl_point["quality"])],
            yerr=[float(rl_point["quality_std"])],
            fmt="none",
            ecolor="#111111",
            elinewidth=1.0,
            capsize=2.8,
            capthick=1.0,
            zorder=5,
        )
        axis.scatter(
            [rl_x_value],
            [float(rl_point["quality"])],
            marker="*",
            color="#111111",
            edgecolors="#FFFFFF",
            linewidths=0.7,
            s=170,
            alpha=0.98,
            zorder=6,
        )
        if annotate_work:
            axis.annotate(
                f"RL ({rl_point['policy']}, ts={rl_point['ts']})",
                (rl_x_value, float(rl_point["quality"])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        baseline_frontier = _pareto_frontier(
            [(float(point["x_value"]), float(point["quality"])) for point in valid_points]
        )
        if len(baseline_frontier) >= 2:
            axis.plot(
                [point[0] for point in baseline_frontier],
                [point[1] for point in baseline_frontier],
                linestyle="--",
                linewidth=1.1,
                color="#333333",
                zorder=1,
            )

        extended_points = [(float(point["x_value"]), float(point["quality"])) for point in valid_points]
        extended_points.append((float(rl_x_value), float(rl_point["quality"])))
        extended_frontier = _pareto_frontier(extended_points)
        if len(extended_frontier) >= 2:
            axis.plot(
                [point[0] for point in extended_frontier],
                [point[1] for point in extended_frontier],
                linestyle="-",
                linewidth=1.2,
                color="#C44E52",
                zorder=1,
            )

        plotted_x_values = [float(point["x_value"]) for point in valid_points] + [float(rl_x_value)]
        set_tight_log_x_limits(axis, plotted_x_values, padding_fraction=work_x_padding)
        axis.xaxis.set_major_locator(LogLocator(base=10.0))
        axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        axis.xaxis.set_minor_formatter(NullFormatter())

        axis.set_xlabel(_work_xlabel_for_measure(work_x, work_measure))
        axis.set_ylabel("Reward")
        axis.set_title(_work_title(metric, work_measure=work_measure) + " (with RL point)")
        axis.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
        axis.set_axisbelow(True)

        resolver_legend = axis.legend(
            handles=resolver_handles,
            title="resolver",
            fontsize=8,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
        )
        axis.add_artist(resolver_legend)

        policy_legend = axis.legend(
            handles=policy_handles,
            title="policy",
            fontsize=8,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=(0.99, 0.52),
            frameon=False,
        )
        axis.add_artist(policy_legend)

        axis.legend(
            handles=extra_handles,
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(0.99, 0.16),
            frameon=False,
        )

        figure.text(0.01, 0.01, _work_footer(), fontsize=8, alpha=0.85)
        figure.tight_layout(rect=(0.0, 0.03, 0.82, 0.97))

        output_path = work_dir / f"rew_work_cmp{_work_output_suffix(work_measure)}_with_rl.png"
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

    eval_target_scenario = str(args.eval_scenario or "").strip().lower()
    eval_target_route_alias = _route_alias_from_text(str(args.eval_route_construction or "nearest"))
    eval_target_protocol_alias = _protocol_alias_from_text(str(args.eval_protocol or "aa"))
    rl_plot_requested = bool(args.work_cmp_with_rl)
    rl_plot_emitted = False

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

                if (
                    rl_plot_requested
                    and not rl_plot_emitted
                    and scenario == eval_target_scenario
                    and route_alias == eval_target_route_alias
                    and protocol_alias == eval_target_protocol_alias
                ):
                    eval_file_path = Path(args.eval_file).expanduser().resolve()
                    rl_point = _estimate_rl_point_from_eval(
                        eval_file=eval_file_path,
                        proposer=str(args.eval_proposer),
                        resolver=str(args.eval_resolver),
                        num_robots=float(args.eval_num_robots),
                        max_robot_capacity=float(args.eval_max_robot_capacity),
                        joint_multiplier=float(args.eval_joint_multiplier),
                        noop_candidates=float(args.eval_noop_candidates),
                    )
                    saved_paths.extend(
                        plot_work_cmp_with_rl(
                            combo_rows,
                            scenario_output_dir,
                            work_x=args.work_x,
                            work_x_padding=float(args.work_x_padding),
                            annotate_work=bool(args.annotate_work),
                            rl_point=rl_point,
                        )
                    )
                    rl_plot_emitted = True

    if rl_plot_requested and not rl_plot_emitted:
        print(
            "[warn] RL overlay plot was requested but no matching scenario/route/protocol subset was found: "
            f"scenario='{eval_target_scenario}', route='{eval_target_route_alias}', protocol='{eval_target_protocol_alias}'",
            file=sys.stderr,
        )

    print(f"Saved {len(saved_paths)} plot(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())