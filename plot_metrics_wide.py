#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import re
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
TIME_PHASE_PREFIXES = [
    "env_pre_controller",
    "pre_step_sync",
    "proposal",
    "resolution",
    "commit_dispatch",
    "simulation",
    "post_step_logging",
    "other",
]
TIME_TOTAL_PHASE = "decision_total"
TIME_CMP_SUBDIR = "time_cmp"
TIME_CMP_AUDIT_CSV = "time_cmp_data.csv"
TIME_PHASE_AUDIT_CSV = "time_phase_data.csv"
TIME_DIAG_RUNS_CSV = "proposal_resolver_diagnostics_runs.csv"
TIME_DIAG_GROUPS_CSV = "proposal_resolver_diagnostics_groups.csv"
TIME_DIAG_MATCHED_REPLICATES_CSV = "proposal_resolver_matched_replicates.csv"
TIME_DIAG_MATCHED_SUMMARY_CSV = "proposal_resolver_matched_summary.csv"
TIME_DIAG_WORKLOAD_FIELDS = [
    "num_robots",
    "mcand",
    "cne_fr",
    "cne_mn",
    "msd",
    "dstep",
    "noop",
    "overld",
    "ovrlap",
    "shared",
    "work_candidate_scan",
    "work_active_proposals",
]
TIME_PHASE_COLORS = {
    "env_pre_controller": "#4C78A8",
    "pre_step_sync": "#72B7B2",
    "proposal": "#54A24B",
    "resolution": "#ECA02C",
    "commit_dispatch": "#EECA3B",
    "simulation": "#B279A2",
    "post_step_logging": "#FF9DA6",
    "other": "#9D755D",
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
        "--time",
        action="store_true",
        help="Create quality-versus-measured-latency plots from timing summary CSV logs",
    )
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=None,
        help="Directory to recursively search for timing_summary*.csv files",
    )
    parser.add_argument(
        "--timing-files",
        nargs="+",
        type=Path,
        default=None,
        help="Exact timing summary CSV files to use",
    )
    parser.add_argument(
        "--annotate-time",
        action="store_true",
        help="Annotate time-comparison scatter points with proposer/policy labels",
    )
    parser.add_argument(
        "--time-linear",
        action="store_true",
        help="Use a linear x-axis for measured-time scatter plots (default is log)",
    )
    parser.add_argument(
        "--strict-time-metadata",
        action="store_true",
        help="Fail if important timing metadata differ across compared timing groups",
    )
    parser.add_argument(
        "--time-diagnose-proposer-resolver",
        action="store_true",
        help="Add diagnostics to explain proposer-latency spread across resolvers (implies --time)",
    )
    parser.add_argument(
        "--time-phase-stat",
        choices=["mean", "total", "both"],
        default="both",
        help="Phase-stat plot set: mean, total, or both (default: both)",
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
    parser.add_argument(
        "--rl-eval-dirs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "One or more evaluation_<date>_<time> directories (RL saved-model evaluation runs) to overlay "
            "on --time plots (resolver/proposer combination plots, decision-phase breakdowns, and rew_time_cmp). "
            "Each directory is expected to contain --rl-quality-file and (optionally) --rl-timing-file."
        ),
    )
    parser.add_argument(
        "--rl-mode",
        choices=["mean", "best", "distinct"],
        default="best",
        help=(
            "How to combine multiple --rl-eval-dirs runs: 'best' picks the run with the highest quality-window "
            "reward (timing from that run only); 'mean' averages the quality-window point across runs and pools "
            "all runs' timing rows; 'distinct' keeps every run as its own labeled series (default: best)"
        ),
    )
    parser.add_argument(
        "--rl-quality-window",
        type=int,
        default=5,
        help="Sliding window size (in episodes) used to pick the best-reward quality window per RL run (default: 5)",
    )
    parser.add_argument(
        "--rl-time-scope",
        choices=["all", "window"],
        default="all",
        help=(
            "Which RL episodes contribute measured timing data: 'all' uses every non-warmup measured episode "
            "(recommended: more samples, latency is not expected to depend on the reward window); 'window' "
            "restricts timing rows to the same episodes selected for the quality window (default: all)"
        ),
    )
    parser.add_argument(
        "--rl-quality-file",
        type=str,
        default="evaluation_metrics.log",
        help="Filename (relative to each --rl-eval-dirs entry) with per-episode quality metrics (default: evaluation_metrics.log)",
    )
    parser.add_argument(
        "--rl-timing-file",
        type=str,
        default="timing_summary.csv",
        help="Filename (relative to each --rl-eval-dirs entry) with per-episode measured timing (default: timing_summary.csv)",
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
    if args.time:
        has_explicit_flags = True
        requested_types.append("time_cmp")
    if args.time_diagnose_proposer_resolver:
        has_explicit_flags = True
        if "time_cmp" not in requested_types:
            requested_types.append("time_cmp")
        requested_types.append("time_diag")

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
    if text in {"closest", "ctc", "closest_than_capacity", "closest_then_capacity"}:
        return "closest_then_capacity"
    if text in {"reward_aligned", "reward-aligned", "ra"}:
        return "ra"
    if text in {"nearest", "nr"}:
        return "nr"
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
    if text in {"aa", "admission", "admission_aware"}:
        return "aa"
    if text in {"forced", "f"}:
        return "forced"
    return text


def _canonical_policy_join_name(value: object) -> str:
    text = normalize_name(str(value or ""))
    if text in {"proposal_joint_competition", "proposal_joint_completion"}:
        return "proposal_joint_completion"
    return text


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _rank_values(values: list[float]) -> list[float]:
    indexed = list(enumerate(values))
    sorted_values = sorted(indexed, key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and math.isclose(sorted_values[j + 1][1], sorted_values[i][1], rel_tol=0.0, abs_tol=1e-12):
            j += 1
        rank_value = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_values[k][0]] = rank_value
        i = j + 1
    return ranks


def _pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    mean_x = sum(x_values) / float(len(x_values))
    mean_y = sum(y_values) / float(len(y_values))
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / (den_x * den_y)


def _spearman_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return _pearson_correlation(_rank_values(x_values), _rank_values(y_values))


def _tokenized_text(value: str) -> str:
    text = normalize_name(value)
    text = text.replace("/", "_").replace("\\", "_").replace(".", "_")
    return re.sub(r"_+", "_", text)


def _find_timing_files(
    *,
    timing_dir: Path | None,
    timing_files: list[Path] | None,
    csv_parent: Path,
) -> list[Path]:
    if timing_files:
        found: list[Path] = []
        for file_path in timing_files:
            path = file_path.expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Timing file not found: {path}")
            found.append(path)
        return found

    search_root = timing_dir.expanduser().resolve() if timing_dir is not None else csv_parent.resolve()
    if not search_root.exists() or not search_root.is_dir():
        raise FileNotFoundError(f"Timing directory not found: {search_root}")
    files = sorted(path.resolve() for path in search_root.rglob("timing_summary*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No timing summary CSV files found under: {search_root}")
    return files


def _load_timing_rows(timing_files: list[Path]) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for csv_path in timing_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Timing CSV has no header: {csv_path}")
            for row_index, raw_row in enumerate(reader, start=2):
                parsed: dict[str, object] = {"source_timing_file": str(csv_path)}
                parsed["source_timing_row_id"] = f"{csv_path}#row{row_index}"
                for key, value in raw_row.items():
                    parsed[key] = parse_cell(value)
                out_rows.append(parsed)
    if not out_rows:
        raise ValueError("No timing rows were loaded from timing summary files")
    return out_rows


def _infer_scenario_from_path(path_text: str, known_scenarios: set[str]) -> str | None:
    if not known_scenarios:
        return None
    normalized_path = f"_{_tokenized_text(path_text)}_"
    matches = [scenario for scenario in known_scenarios if f"_{scenario}_" in normalized_path]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


def _infer_route_alias_from_path(path_text: str) -> str | None:
    normalized = f"_{_tokenized_text(path_text)}_"
    if any(token in normalized for token in ["_nearest_", "_nr_"]):
        return "nr"
    if any(token in normalized for token in ["_reward_aligned_", "_rewardaligned_", "_ra_"]):
        return "ra"
    return None


def _infer_protocol_alias_from_path(path_text: str) -> str | None:
    normalized = f"_{_tokenized_text(path_text)}_"
    if any(token in normalized for token in ["_admission_aware_", "_admission_", "_aa_"]):
        return "aa"
    if any(token in normalized for token in ["_forced_", "_f_"]):
        return "forced"
    return None


def _canonical_timing_scenario(value: object, known_scenarios: set[str]) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = _tokenized_text(raw)
    return normalized if normalized in known_scenarios else None


def _canonical_timing_route(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    alias = _route_alias_from_text(raw)
    return alias if alias in {"nr", "ra"} else None


def _canonical_timing_protocol(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    alias = _protocol_alias_from_text(raw)
    return alias if alias in {"aa", "forced"} else None


def _csv_row_phase_mean_ms(row: dict[str, object], phase_prefix: str) -> float | None:
    mean_value = safe_number(row.get(f"{phase_prefix}_mean_ms"))
    if mean_value is not None:
        return float(mean_value)

    total_value = safe_number(row.get(f"{phase_prefix}_total_ms"))
    measured = safe_number(row.get("measured_decisions"))
    if total_value is not None and measured is not None and measured > 0.0:
        return float(total_value) / float(measured)
    return None


def _canonicalize_timing_rows(
    timing_rows: list[dict[str, object]],
    *,
    known_scenarios: set[str],
    allowed_routes: set[str],
    allowed_protocols: set[str],
) -> list[dict[str, object]]:
    canonical_rows: list[dict[str, object]] = []

    scenario_unique = next(iter(known_scenarios)) if len(known_scenarios) == 1 else None
    route_unique = next(iter(allowed_routes)) if len(allowed_routes) == 1 else None
    protocol_unique = next(iter(allowed_protocols)) if len(allowed_protocols) == 1 else None

    for row in timing_rows:
        source_file = str(row.get("source_timing_file", "")).strip()
        row_scenario = _canonical_timing_scenario(row.get("scenario"), known_scenarios)
        if row_scenario is None:
            row_scenario = _infer_scenario_from_path(source_file, known_scenarios)
        if row_scenario is None:
            row_scenario = scenario_unique

        row_route = _canonical_timing_route(row.get("route_construction") or row.get("route"))
        if row_route is None:
            row_route = _infer_route_alias_from_path(source_file)
        if row_route is None and route_unique is not None:
            row_route = route_unique

        row_protocol = _canonical_timing_protocol(row.get("protocol"))
        if row_protocol is None:
            row_protocol = _infer_protocol_alias_from_path(source_file)
        if row_protocol is None and protocol_unique is not None:
            row_protocol = protocol_unique

        if row_scenario is None or row_route is None or row_protocol is None:
            print(
                "[warn] Skipping timing row due to ambiguous context: "
                f"file={source_file}, scenario={row_scenario}, route={row_route}, protocol={row_protocol}",
                file=sys.stderr,
            )
            continue

        resolver = _canonical_resolver_name(row.get("resolver"))
        proposer_raw = row.get("proposer") if row.get("proposer") is not None else row.get("policy")
        proposer = _canonical_policy_join_name(proposer_raw)
        if not resolver or not proposer:
            print(
                "[warn] Skipping timing row with missing proposer/resolver: "
                f"file={source_file}, proposer={proposer_raw}, resolver={row.get('resolver')}",
                file=sys.stderr,
            )
            continue

        measured_decisions = safe_number(row.get("measured_decisions"))
        if measured_decisions is None or measured_decisions <= 0.0:
            print(
                "[warn] Skipping timing row with non-positive measured_decisions: "
                f"file={source_file}",
                file=sys.stderr,
            )
            continue

        canonical_row = dict(row)
        canonical_row["scenario_canonical"] = row_scenario
        canonical_row["route_canonical"] = row_route
        canonical_row["protocol_canonical"] = row_protocol
        canonical_row["resolver_canonical"] = resolver
        canonical_row["proposer_canonical"] = proposer
        canonical_row["method_canonical"] = normalize_name(str(row.get("method") or "na")) or "na"
        canonical_row["inference_mode_canonical"] = normalize_name(str(row.get("inference_mode") or "na")) or "na"
        canonical_row["seed_canonical"] = str(row.get("seed") if row.get("seed") is not None else "na")
        canonical_row["episode_canonical"] = str(row.get("episode") if row.get("episode") is not None else "na")
        canonical_row["timing_protocol_canonical"] = normalize_name(str(row.get("timing_protocol") or "na")) or "na"
        canonical_rows.append(canonical_row)

    return canonical_rows


def _detect_duplicate_timing_rows(rows: list[dict[str, object]]) -> None:
    seen: dict[tuple[str, ...], list[str]] = {}
    for row in rows:
        key = (
            str(row.get("scenario_canonical") or ""),
            str(row.get("route_canonical") or ""),
            str(row.get("protocol_canonical") or ""),
            str(row.get("method_canonical") or ""),
            str(row.get("proposer_canonical") or ""),
            str(row.get("resolver_canonical") or ""),
            str(row.get("inference_mode_canonical") or ""),
            str(row.get("seed_canonical") or ""),
            str(row.get("episode_canonical") or ""),
            str(row.get("timing_protocol_canonical") or ""),
        )
        seen.setdefault(key, []).append(str(row.get("source_timing_file") or "<unknown>"))

    duplicates = {
        key: sorted(set(files))
        for key, files in seen.items()
        if len(files) > 1
    }
    if not duplicates:
        return

    lines = ["Duplicate timing records detected after canonicalization:"]
    for key, files in sorted(duplicates.items()):
        key_display = "/".join(key)
        lines.append(f"- key={key_display}")
        for file_name in files:
            lines.append(f"    source={file_name}")
    lines.append("Pass the intended files explicitly with --timing-files.")
    raise ValueError("\n".join(lines))


def _metadata_string(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else "<missing>"


def _aggregate_timing_groups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["scenario_canonical"]),
            str(row["route_canonical"]),
            str(row["protocol_canonical"]),
            str(row["proposer_canonical"]),
            str(row["resolver_canonical"]),
            str(row["method_canonical"]),
            str(row["inference_mode_canonical"]),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key, group_rows in grouped.items():
        scenario, route, protocol, proposer, resolver, method, inference_mode = key

        measured_values = [safe_number(row.get("measured_decisions")) for row in group_rows]
        measured = [float(value) for value in measured_values if value is not None and value > 0.0]
        measured_sum = float(sum(measured))
        if measured_sum <= 0.0:
            continue

        proposal_totals = [safe_number(row.get("proposal_total_ms")) for row in group_rows]
        resolution_totals = [safe_number(row.get("resolution_total_ms")) for row in group_rows]
        proposal_sum = sum(float(value) for value in proposal_totals if value is not None)
        resolution_sum = sum(float(value) for value in resolution_totals if value is not None)

        proposal_run_means = [
            float(value)
            for value in [safe_number(row.get("proposal_mean_ms")) for row in group_rows]
            if value is not None
        ]
        resolution_run_means = [
            float(value)
            for value in [safe_number(row.get("resolution_mean_ms")) for row in group_rows]
            if value is not None
        ]
        allocation_run_means: list[float] = []
        for row in group_rows:
            p = safe_number(row.get("proposal_mean_ms"))
            r = safe_number(row.get("resolution_mean_ms"))
            if p is not None and r is not None:
                allocation_run_means.append(float(p) + float(r))

        phase_means: dict[str, float | None] = {}
        phase_stds: dict[str, float] = {}
        phase_totals: dict[str, float] = {}
        phase_missing: list[str] = []

        all_phase_prefixes = TIME_PHASE_PREFIXES + ["decision"]
        for phase_prefix in all_phase_prefixes:
            phase_total_num = 0.0
            phase_total_den = 0.0
            phase_run_means: list[float] = []
            phase_total_acc = 0.0
            for row in group_rows:
                measured_decisions = safe_number(row.get("measured_decisions"))
                if measured_decisions is None or measured_decisions <= 0.0:
                    continue
                total_value = safe_number(row.get(f"{phase_prefix}_total_ms"))
                if total_value is not None:
                    phase_total_num += float(total_value)
                    phase_total_den += float(measured_decisions)
                    phase_total_acc += float(total_value)
                run_mean = _csv_row_phase_mean_ms(row, phase_prefix)
                if run_mean is not None:
                    phase_run_means.append(float(run_mean))

            if phase_total_den > 0.0:
                phase_means[phase_prefix] = phase_total_num / phase_total_den
            else:
                phase_means[phase_prefix] = None
                phase_missing.append(phase_prefix)
            phase_stds[phase_prefix] = _sample_std(phase_run_means)
            phase_totals[phase_prefix] = phase_total_acc

        if phase_missing:
            print(
                "[warn] Missing phase totals for some timing columns in group "
                f"scenario={scenario} route={route} protocol={protocol} proposer={proposer} resolver={resolver}: "
                f"{', '.join(phase_missing)}",
                file=sys.stderr,
            )

        named_phase_sum = sum(
            float(phase_means[prefix])
            for prefix in TIME_PHASE_PREFIXES
            if phase_means.get(prefix) is not None
        )
        decision_mean = phase_means.get("decision")
        accounting_diff = (float(decision_mean) - named_phase_sum) if decision_mean is not None else math.nan

        aggregated = {
            "scenario": scenario,
            "route": route,
            "protocol": protocol,
            "proposer": proposer,
            "resolver": resolver,
            "method": method,
            "inference_mode": inference_mode,
            "n_timing_runs": len(group_rows),
            "n_measured_decisions": int(round(measured_sum)),
            "proposal_time_ms": (proposal_sum / measured_sum),
            "resolution_time_ms": (resolution_sum / measured_sum),
            "allocation_time_ms": ((proposal_sum + resolution_sum) / measured_sum),
            "proposal_time_std_ms": _sample_std(proposal_run_means),
            "resolution_time_std_ms": _sample_std(resolution_run_means),
            "allocation_time_std_ms": _sample_std(allocation_run_means),
            "timing_protocol_values": sorted({_metadata_string(row.get("timing_protocol")) for row in group_rows}),
            "device_values": sorted({_metadata_string(row.get("device")) for row in group_rows}),
            "torch_num_threads_values": sorted({_metadata_string(row.get("torch_num_threads")) for row in group_rows}),
            "omp_num_threads_values": sorted({_metadata_string(row.get("omp_num_threads")) for row in group_rows}),
            "mkl_num_threads_values": sorted({_metadata_string(row.get("mkl_num_threads")) for row in group_rows}),
            "cpu_model_values": sorted({_metadata_string(row.get("cpu_model")) for row in group_rows}),
            "host_name_values": sorted({_metadata_string(row.get("host_name")) for row in group_rows}),
            "source_timing_files": sorted({str(row.get("source_timing_file") or "") for row in group_rows}),
            "named_phase_sum_ms": named_phase_sum,
            "accounting_difference_ms": accounting_diff,
        }

        for phase_prefix in TIME_PHASE_PREFIXES:
            aggregated[f"{phase_prefix}_mean_ms"] = phase_means.get(phase_prefix)
            aggregated[f"{phase_prefix}_std_ms"] = phase_stds.get(phase_prefix, 0.0)
            aggregated[f"{phase_prefix}_total_ms"] = phase_totals.get(phase_prefix, 0.0)

        aggregated["decision_total_mean_ms"] = phase_means.get("decision")
        aggregated["decision_total_std_ms"] = phase_stds.get("decision", 0.0)
        aggregated["decision_total_total_ms"] = phase_totals.get("decision", 0.0)
        out.append(aggregated)

    return out


def _build_metrics_time_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        _scenario_of_row(row),
        _route_alias_from_text(str(row.get("route_construction") or "nearest")),
        _protocol_alias_of_row(row),
        _canonical_policy_join_name(row.get("pol")),
        _canonical_resolver_name(row.get("resolver")),
    )


def _build_timing_time_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("scenario")),
        str(row.get("route")),
        str(row.get("protocol")),
        str(row.get("proposer")),
        str(row.get("resolver")),
    )


def _join_metrics_with_timing(
    metrics_rows: list[dict[str, object]],
    timing_groups: list[dict[str, object]],
) -> tuple[list[dict[str, object]], set[tuple[str, str, str, str, str]], set[tuple[str, str, str, str, str]]]:
    timing_index: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = {}
    for timing_group in timing_groups:
        timing_index.setdefault(_build_timing_time_key(timing_group), []).append(timing_group)

    joined_rows: list[dict[str, object]] = []
    matched_timing_keys: set[tuple[str, str, str, str, str]] = set()
    missing_metric_keys: set[tuple[str, str, str, str, str]] = set()

    for metric_row in metrics_rows:
        key = _build_metrics_time_key(metric_row)
        candidates = timing_index.get(key, [])
        if not candidates:
            missing_metric_keys.add(key)
            continue

        method_inference_pairs = {(str(c.get("method")), str(c.get("inference_mode"))) for c in candidates}
        if len(method_inference_pairs) > 1:
            pair_list = ", ".join(sorted(f"{method}:{mode}" for method, mode in method_inference_pairs))
            raise ValueError(
                "Ambiguous timing groups map to one metrics row key "
                f"{key}: {pair_list}. Select one timing subset with --timing-files."
            )

        chosen = candidates[0]
        matched_timing_keys.add(key)
        merged = dict(metric_row)
        for field_name, field_value in chosen.items():
            merged[f"timing_{field_name}"] = field_value
        joined_rows.append(merged)

    unmatched_timing_keys = set(timing_index.keys()).difference(matched_timing_keys)
    return joined_rows, missing_metric_keys, unmatched_timing_keys


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


# ---------------------------------------------------------------------------
# RL saved-model evaluation overlay for --time plots (rew_time_cmp, decision-phase
# breakdowns, proposal-latency diagnostics). Unlike --work-cmp-with-rl (an analytical
# work estimate), this uses REAL measured timing_summary.csv rows produced by
# eval_saved_models.py, so the RL point is injected as ordinary rows into the same
# quality/timing pipeline used for baselines and flows through every combination plot.
# ---------------------------------------------------------------------------

RL_POLICY_PREFIX = "rl_"


def _is_rl_policy_label(value: object) -> bool:
    return str(value or "").strip().lower().startswith(RL_POLICY_PREFIX)


def _parse_rl_quality_rows(log_path: Path) -> list[dict[str, Any]]:
    """Parse per-episode rows from an evaluation_metrics.log-style file, preserving file order.

    Uses the same header/row layout as _parse_eval_log_rows but keeps 'pol' as text and
    drops MEAN/STD summary rows so only real per-episode data remains (needed for the
    sliding quality-window selection, which relies on episode order).
    """
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
        raise ValueError(f"Could not find metrics header in RL evaluation log: {log_path}")

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
            if key == "pol":
                row[key] = text
                continue
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

        if str(row.get("pol", "")).strip().lower() in {"mean", "std"}:
            continue
        if row.get("seed") is None or safe_number(row.get("rew")) is None:
            continue
        parsed_rows.append(row)

    if not parsed_rows:
        raise ValueError(f"No usable per-episode rows in RL evaluation log: {log_path}")
    return parsed_rows


def _infer_rl_attempt_indices(rows: list[dict[str, Any]]) -> None:
    """Reconstruct the 0-based attempt (episode) index per seed from row order.

    eval_saved_models.py appends rows in `for seed: for attempt in range(eval_runs)` order,
    matching the 'episode' column written to timing_summary.csv, so this lets us correlate
    quality rows (which carry no explicit episode index) with timing rows.
    """
    counters: dict[int, int] = {}
    for row in rows:
        seed = int(row.get("seed") or 0)
        attempt = counters.get(seed, 0)
        row["_attempt"] = attempt
        counters[seed] = attempt + 1


def _rl_numeric_quality_columns(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {"pol", "seed", "ts", "_attempt"}
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded or key in seen:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                columns.append(key)
                seen.add(key)
    return columns


def _rolling_best_window_rows(rows: list[dict[str, Any]], *, metric: str, window: int) -> list[dict[str, Any]]:
    """Slide a window of `window` consecutive episodes and keep the window with the highest mean `metric`."""
    window = max(1, int(window))
    if len(rows) <= window:
        return list(rows)

    values = [safe_number(row.get(metric)) for row in rows]
    best_start = 0
    best_mean = -math.inf
    for start in range(0, len(rows) - window + 1):
        window_values = [value for value in values[start : start + window] if value is not None]
        if not window_values:
            continue
        mean_value = sum(window_values) / float(len(window_values))
        if mean_value > best_mean:
            best_mean = mean_value
            best_start = start
    return rows[best_start : best_start + window]


def _quality_point_from_rows(rows: list[dict[str, Any]], numeric_columns: list[str]) -> dict[str, float]:
    point: dict[str, float] = {"n": float(len(rows))}
    for column in numeric_columns:
        values = [safe_number(row.get(column)) for row in rows]
        valid = [float(value) for value in values if value is not None and math.isfinite(value)]
        if not valid:
            continue
        mean_value = sum(valid) / float(len(valid))
        if len(valid) > 1:
            variance = sum((value - mean_value) ** 2 for value in valid) / float(len(valid))
            std_value = math.sqrt(variance)
        else:
            std_value = 0.0
        point[column] = mean_value
        point[f"{column}_std"] = std_value
    return point


def _find_rl_run_file(eval_dir: Path, file_name: str) -> Path | None:
    direct = eval_dir / file_name
    if direct.is_file():
        return direct
    matches = sorted(eval_dir.rglob(file_name))
    return matches[0] if matches else None


def _load_rl_run(
    eval_dir: Path,
    *,
    quality_file_name: str,
    timing_file_name: str,
    quality_window: int,
) -> dict[str, Any]:
    eval_dir = eval_dir.expanduser().resolve()
    quality_path = _find_rl_run_file(eval_dir, quality_file_name)
    if quality_path is None:
        raise FileNotFoundError(f"Could not find '{quality_file_name}' under RL eval dir: {eval_dir}")

    rows = _parse_rl_quality_rows(quality_path)
    _infer_rl_attempt_indices(rows)
    numeric_columns = _rl_numeric_quality_columns(rows)
    window_rows = _rolling_best_window_rows(rows, metric="rew", window=quality_window)

    timing_path = _find_rl_run_file(eval_dir, timing_file_name)
    timing_rows = _load_timing_rows([timing_path]) if timing_path is not None else []
    if timing_path is None:
        print(f"[warn] No '{timing_file_name}' found under RL eval dir: {eval_dir}; time plots will skip this run", file=sys.stderr)

    return {
        "name": eval_dir.name,
        "dir": eval_dir,
        "rows": rows,
        "window_rows": window_rows,
        "window_attempts": {(int(row.get("seed") or 0), int(row.get("_attempt") or 0)) for row in window_rows},
        "quality_all": _quality_point_from_rows(rows, numeric_columns),
        "quality_window": _quality_point_from_rows(window_rows, numeric_columns),
        "timing_rows": timing_rows,
    }


def _rl_timing_rows_for_scope(run: dict[str, Any], *, scope: str) -> list[dict[str, object]]:
    timing_rows = run["timing_rows"]
    if scope != "window":
        return timing_rows
    window_attempts = run["window_attempts"]
    filtered = [
        row
        for row in timing_rows
        if (int(safe_number(row.get("seed")) or 0), int(safe_number(row.get("episode")) or 0)) in window_attempts
    ]
    if not filtered:
        print(
            f"[warn] --rl-time-scope=window matched no timing rows for run '{run['name']}'; falling back to all timing rows",
            file=sys.stderr,
        )
        return timing_rows
    return filtered


def _eval_protocol_to_metrics_text(protocol: str) -> str:
    """metrics_wide.csv rows only recognize the raw 'protocol' values 'admission'/'forced'
    (unlike timing CSVs, which also accept the 'aa' alias); convert --eval-protocol accordingly."""
    return "admission" if _protocol_alias_from_text(protocol) == "aa" else "forced"


def _build_rl_quality_row(
    quality_point: dict[str, float],
    *,
    label: str,
    resolver: str,
    route_construction: str,
    protocol: str,
    scenario: str,
    source_name: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "source_file": source_name,
        "scenario": scenario,
        "instance": "rl_eval",
        "protocol": _eval_protocol_to_metrics_text(protocol),
        "resolver": resolver,
        "route_construction": route_construction,
        "admission_aware": "true" if _protocol_alias_from_text(protocol) == "aa" else "false",
        "pol": label,
    }
    row.update(quality_point)
    return row


def _build_rl_timing_rows(
    raw_timing_rows: list[dict[str, object]],
    *,
    label: str,
    route_construction: str,
    protocol: str,
    scenario: str,
    episode_offset: int = 0,
) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for raw_row in raw_timing_rows:
        row = dict(raw_row)
        row["proposer"] = label
        row["scenario"] = scenario
        row["route_construction"] = route_construction
        row["protocol"] = protocol
        if episode_offset:
            episode_value = safe_number(row.get("episode")) or 0.0
            row["episode"] = int(episode_value) + episode_offset
        out_rows.append(row)
    return out_rows


def _resolver_from_rl_timing_rows(timing_rows: list[dict[str, object]], fallback: str) -> str:
    for row in timing_rows:
        resolver = str(row.get("resolver") or "").strip()
        if resolver:
            return resolver
    return fallback


def load_rl_time_overlay(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build synthetic quality rows + raw timing rows representing RL saved-model evaluation run(s).

    Returned rows use the same column conventions as metrics_wide.csv rows / timing_summary.csv rows, so
    callers can simply concatenate them onto the normal quality/timing row lists before the rest of the
    --time pipeline runs; every per-combination plot then includes the RL point automatically.
    """
    eval_dirs = list(args.rl_eval_dirs or [])
    if not eval_dirs:
        return [], []

    quality_window = int(args.rl_quality_window)
    runs = [
        _load_rl_run(
            eval_dir,
            quality_file_name=str(args.rl_quality_file),
            timing_file_name=str(args.rl_timing_file),
            quality_window=quality_window,
        )
        for eval_dir in eval_dirs
    ]

    scenario = str(args.eval_scenario)
    route_construction = str(args.eval_route_construction)
    protocol = str(args.eval_protocol)
    base_label = f"{RL_POLICY_PREFIX}{args.eval_proposer}"

    quality_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []

    if args.rl_mode == "distinct":
        for index, run in enumerate(runs, start=1):
            label = f"{base_label}_r{index}"
            resolver = _resolver_from_rl_timing_rows(run["timing_rows"], fallback=str(args.eval_resolver))
            quality_rows.append(
                _build_rl_quality_row(
                    run["quality_window"],
                    label=label,
                    resolver=resolver,
                    route_construction=route_construction,
                    protocol=protocol,
                    scenario=scenario,
                    source_name=run["name"],
                )
            )
            scoped_timing = _rl_timing_rows_for_scope(run, scope=str(args.rl_time_scope))
            timing_rows.extend(
                _build_rl_timing_rows(
                    scoped_timing,
                    label=label,
                    route_construction=route_construction,
                    protocol=protocol,
                    scenario=scenario,
                )
            )
        return quality_rows, timing_rows

    if args.rl_mode == "best":
        best_run = max(runs, key=lambda run: float(run["quality_window"].get("rew", -math.inf)))
        resolver = _resolver_from_rl_timing_rows(best_run["timing_rows"], fallback=str(args.eval_resolver))
        quality_rows.append(
            _build_rl_quality_row(
                best_run["quality_window"],
                label=base_label,
                resolver=resolver,
                route_construction=route_construction,
                protocol=protocol,
                scenario=scenario,
                source_name=best_run["name"],
            )
        )
        scoped_timing = _rl_timing_rows_for_scope(best_run, scope=str(args.rl_time_scope))
        timing_rows.extend(
            _build_rl_timing_rows(
                scoped_timing,
                label=base_label,
                route_construction=route_construction,
                protocol=protocol,
                scenario=scenario,
            )
        )
        return quality_rows, timing_rows

    # "mean": average the per-run quality windows, pool all runs' timing rows.
    numeric_columns = sorted({
        key
        for run in runs
        for key in run["quality_window"]
        if not key.endswith("_std") and key != "n"
    })
    pooled_window_rows = [run["quality_window"] for run in runs]
    mean_point: dict[str, float] = {"n": float(sum(row.get("n", 0.0) for row in pooled_window_rows))}
    for column in numeric_columns:
        run_means = [row[column] for row in pooled_window_rows if column in row]
        if not run_means:
            continue
        mean_value = sum(run_means) / float(len(run_means))
        std_value = _sample_std(run_means) if len(run_means) > 1 else 0.0
        mean_point[column] = mean_value
        mean_point[f"{column}_std"] = std_value

    resolver = str(args.eval_resolver)
    for run in runs:
        candidate = _resolver_from_rl_timing_rows(run["timing_rows"], fallback="")
        if candidate:
            resolver = candidate
            break

    quality_rows.append(
        _build_rl_quality_row(
            mean_point,
            label=base_label,
            resolver=resolver,
            route_construction=route_construction,
            protocol=protocol,
            scenario=scenario,
            source_name="+".join(run["name"] for run in runs),
        )
    )
    for run_index, run in enumerate(runs):
        scoped_timing = _rl_timing_rows_for_scope(run, scope=str(args.rl_time_scope))
        timing_rows.extend(
            _build_rl_timing_rows(
                scoped_timing,
                label=base_label,
                route_construction=route_construction,
                protocol=protocol,
                scenario=scenario,
                # All pooled runs share the same label, so episodes must be offset per run to
                # avoid colliding (seed, episode) keys across otherwise-identical run structures.
                episode_offset=run_index * 1_000_000,
            )
        )
    return quality_rows, timing_rows


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


def _check_timing_metadata(
    timing_groups: list[dict[str, object]],
    *,
    strict: bool,
    context_label: str,
) -> None:
    important = [
        "device_values",
        "timing_protocol_values",
        "torch_num_threads_values",
        "omp_num_threads_values",
        "mkl_num_threads_values",
        "cpu_model_values",
    ]
    differing_lines: list[str] = []
    for field_name in important:
        all_values = sorted({value for group in timing_groups for value in group.get(field_name, [])})
        if len(all_values) > 1:
            differing_lines.append(f"{field_name}={all_values}")

    if not differing_lines:
        return

    message = (
        f"Timing metadata differ across compared groups ({context_label}):\n"
        + "\n".join(f"- {line}" for line in differing_lines)
    )
    if strict:
        raise ValueError(message)
    print(f"[warn] {message}", file=sys.stderr)


def _time_combo_output_dir(base_output_dir: Path, scenario: str, route: str, protocol: str) -> Path:
    out_dir = base_output_dir / _scenario_folder_name(scenario) / TIME_CMP_SUBDIR / f"{route}_{protocol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_time_cmp_audit_csv(
    *,
    output_path: Path,
    joined_rows: list[dict[str, object]],
    metrics: list[str],
    source_metrics_file: Path,
) -> Path:
    header = [
        "scenario",
        "route",
        "protocol",
        "method",
        "inference_mode",
        "proposer",
        "resolver",
        "n_timing_runs",
        "n_measured_decisions",
        "proposal_time_ms",
        "proposal_time_std_ms",
        "resolution_time_ms",
        "resolution_time_std_ms",
        "allocation_time_ms",
        "allocation_time_std_ms",
        "quality_metric",
        "quality_mean",
        "quality_std",
        "timing_protocol",
        "device",
        "torch_num_threads",
        "omp_num_threads",
        "mkl_num_threads",
        "source_timing_files",
        "source_metrics_file",
    ]

    rows_out: list[dict[str, object]] = []
    for row in joined_rows:
        for metric in metrics:
            quality_mean = safe_number(row.get(metric))
            if quality_mean is None:
                continue
            quality_std = safe_number(row.get(f"{metric}_std")) or 0.0
            rows_out.append(
                {
                    "scenario": _scenario_of_row(row),
                    "route": _route_alias_from_text(str(row.get("route_construction") or "nearest")),
                    "protocol": _protocol_alias_of_row(row),
                    "method": row.get("timing_method", "na"),
                    "inference_mode": row.get("timing_inference_mode", "na"),
                    "proposer": row.get("timing_proposer", ""),
                    "resolver": row.get("timing_resolver", ""),
                    "n_timing_runs": int(safe_number(row.get("timing_n_timing_runs")) or 0),
                    "n_measured_decisions": int(safe_number(row.get("timing_n_measured_decisions")) or 0),
                    "proposal_time_ms": safe_number(row.get("timing_proposal_time_ms")),
                    "proposal_time_std_ms": safe_number(row.get("timing_proposal_time_std_ms")) or 0.0,
                    "resolution_time_ms": safe_number(row.get("timing_resolution_time_ms")),
                    "resolution_time_std_ms": safe_number(row.get("timing_resolution_time_std_ms")) or 0.0,
                    "allocation_time_ms": safe_number(row.get("timing_allocation_time_ms")),
                    "allocation_time_std_ms": safe_number(row.get("timing_allocation_time_std_ms")) or 0.0,
                    "quality_metric": metric,
                    "quality_mean": float(quality_mean),
                    "quality_std": float(quality_std),
                    "timing_protocol": ";".join(row.get("timing_timing_protocol_values", [])),
                    "device": ";".join(row.get("timing_device_values", [])),
                    "torch_num_threads": ";".join(row.get("timing_torch_num_threads_values", [])),
                    "omp_num_threads": ";".join(row.get("timing_omp_num_threads_values", [])),
                    "mkl_num_threads": ";".join(row.get("timing_mkl_num_threads_values", [])),
                    "source_timing_files": ";".join(row.get("timing_source_timing_files", [])),
                    "source_metrics_file": str(source_metrics_file),
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
    return output_path


def _write_time_phase_audit_csv(output_path: Path, timing_groups: list[dict[str, object]]) -> Path:
    header = [
        "scenario",
        "route",
        "protocol",
        "method",
        "inference_mode",
        "proposer",
        "resolver",
        "n_timing_runs",
        "n_measured_decisions",
    ]
    for phase in TIME_PHASE_PREFIXES + ["decision_total"]:
        header.append(f"{phase}_mean_ms")
    for phase in TIME_PHASE_PREFIXES + ["decision_total"]:
        header.append(f"{phase}_std_ms")
    header.extend(["named_phase_sum_ms", "accounting_difference_ms"])
    for phase in TIME_PHASE_PREFIXES + ["decision_total"]:
        header.append(f"{phase}_total_ms")
    header.append("source_timing_files")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for group in timing_groups:
            row: dict[str, object] = {
                "scenario": group.get("scenario"),
                "route": group.get("route"),
                "protocol": group.get("protocol"),
                "method": group.get("method"),
                "inference_mode": group.get("inference_mode"),
                "proposer": group.get("proposer"),
                "resolver": group.get("resolver"),
                "n_timing_runs": group.get("n_timing_runs"),
                "n_measured_decisions": group.get("n_measured_decisions"),
                "named_phase_sum_ms": group.get("named_phase_sum_ms"),
                "accounting_difference_ms": group.get("accounting_difference_ms"),
                "source_timing_files": ";".join(group.get("source_timing_files", [])),
            }
            for phase in TIME_PHASE_PREFIXES:
                row[f"{phase}_mean_ms"] = group.get(f"{phase}_mean_ms")
                row[f"{phase}_std_ms"] = group.get(f"{phase}_std_ms")
                row[f"{phase}_total_ms"] = group.get(f"{phase}_total_ms")
            row["decision_total_mean_ms"] = group.get("decision_total_mean_ms")
            row["decision_total_std_ms"] = group.get("decision_total_std_ms")
            row["decision_total_total_ms"] = group.get("decision_total_total_ms")
            writer.writerow(row)
    return output_path


def plot_time_cmp(
    joined_rows: list[dict[str, object]],
    metrics: list[str],
    output_dir: Path,
    *,
    pareto: bool,
    annotate_time: bool,
    time_linear: bool,
) -> list[Path]:
    saved_paths: list[Path] = []
    resolver_labels = _ordered_resolvers([
        str(row.get("timing_resolver", "")).strip()
        for row in joined_rows
        if str(row.get("timing_resolver", "")).strip()
    ])
    resolver_colors = _series_color_map(resolver_labels, preferred=RESOLVER_COLOR_MAP)
    policy_markers = _policy_marker_map([
        str(row.get("timing_proposer", "")).strip()
        for row in joined_rows
        if str(row.get("timing_proposer", "")).strip()
    ])
    # RL points always get a distinct star marker, regardless of alphabetical marker assignment.
    for proposer_label in list(policy_markers):
        if _is_rl_policy_label(proposer_label):
            policy_markers[proposer_label] = "*"

    for metric in metrics:
        points: list[dict[str, object]] = []
        for row in joined_rows:
            quality = safe_number(row.get(metric))
            quality_std = safe_number(row.get(f"{metric}_std")) or 0.0
            x_value = safe_number(row.get("timing_allocation_time_ms"))
            x_std = safe_number(row.get("timing_allocation_time_std_ms")) or 0.0
            if quality is None or x_value is None:
                continue
            if not time_linear and x_value <= 0.0:
                continue
            points.append(
                {
                    "x": float(x_value),
                    "x_std": float(x_std),
                    "y": float(quality),
                    "y_std": float(quality_std),
                    "resolver": str(row.get("timing_resolver", "")).strip(),
                    "proposer": str(row.get("timing_proposer", "")).strip(),
                }
            )

        if not points:
            print(f"[warn] No matched points for time metric '{metric}'", file=sys.stderr)
            continue

        figure, axis = plt.subplots(figsize=(8.8, 5.8))
        if not time_linear:
            axis.set_xscale("log")

        for point in points:
            resolver = str(point["resolver"])
            proposer = str(point["proposer"])
            axis.errorbar(
                [float(point["x"])],
                [float(point["y"])],
                xerr=[float(point["x_std"])],
                yerr=[float(point["y_std"])],
                fmt="none",
                ecolor=resolver_colors.get(resolver, "#4C78A8"),
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
                zorder=2,
            )
            is_rl = _is_rl_policy_label(proposer)
            axis.scatter(
                [float(point["x"])],
                [float(point["y"])],
                marker=policy_markers.get(proposer, "o"),
                color=resolver_colors.get(resolver, "#4C78A8"),
                edgecolors="#111111" if is_rl else "#2F3E4E",
                linewidths=1.1 if is_rl else 0.6,
                s=190 if is_rl else 52,
                alpha=0.98 if is_rl else 0.95,
                zorder=6 if is_rl else 3,
            )
            if annotate_time or is_rl:
                axis.annotate(
                    _display_label(proposer),
                    (float(point["x"]), float(point["y"])),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=8,
                )

        if pareto:
            frontier = _pareto_frontier([(float(p["x"]), float(p["y"])) for p in points])
            if len(frontier) >= 2:
                axis.plot(
                    [pair[0] for pair in frontier],
                    [pair[1] for pair in frontier],
                    linestyle="--",
                    linewidth=1.0,
                    color="#222222",
                    zorder=1,
                )

        if not time_linear:
            set_tight_log_x_limits(axis, [float(p["x"]) for p in points], padding_fraction=0.06)
            axis.xaxis.set_major_locator(LogLocator(base=10.0))
            axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
            axis.xaxis.set_minor_formatter(NullFormatter())

        axis.set_xlabel("Measured proposal + resolver latency per decision (ms)")
        axis.set_ylabel(_metric_ylabel(metric))
        if str(metric).strip().lower() == "rew":
            axis.set_title("Reward vs measured allocation latency")
        else:
            axis.set_title(f"{_metric_title(metric)} vs measured allocation latency")
        axis.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
        axis.set_axisbelow(True)

        visible_resolvers = [resolver for resolver in resolver_labels if any(str(point["resolver"]) == resolver for point in points)]
        resolver_handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=resolver_colors.get(resolver, "#4C78A8"), markeredgecolor="#2F3E4E", label=_display_label(resolver))
            for resolver in visible_resolvers
        ]
        visible_proposers = sorted({str(point["proposer"]) for point in points})
        proposer_handles = [
            Line2D([0], [0], marker=policy_markers.get(proposer, "o"), linestyle="", markersize=7, markerfacecolor="#FFFFFF", markeredgecolor="#2F3E4E", label=_display_label(proposer))
            for proposer in visible_proposers
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
            handles=proposer_handles,
            title="proposer/policy",
            fontsize=9,
            title_fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.48),
            frameon=False,
        )

        figure.text(
            0.01,
            0.01,
            "Measured wall-clock latency from timing-summary logs; allocation latency includes proposal generation and conflict resolution only.",
            fontsize=8,
            alpha=0.85,
        )
        figure.tight_layout(rect=(0.0, 0.03, 0.81, 0.97))
        output_path = output_dir / f"{metric}_time_cmp.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def _combination_rows_for_barplots(joined_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    combo_map: dict[tuple[str, str], dict[str, object]] = {}
    for row in joined_rows:
        proposer = str(row.get("timing_proposer", "")).strip()
        resolver = str(row.get("timing_resolver", "")).strip()
        if not proposer or not resolver:
            continue
        key = (proposer, resolver)
        combo_map[key] = {
            "proposer": proposer,
            "resolver": resolver,
            "proposal_time_ms": safe_number(row.get("timing_proposal_time_ms")),
            "proposal_time_std_ms": safe_number(row.get("timing_proposal_time_std_ms")) or 0.0,
            "resolution_time_ms": safe_number(row.get("timing_resolution_time_ms")),
            "resolution_time_std_ms": safe_number(row.get("timing_resolution_time_std_ms")) or 0.0,
            "allocation_time_ms": safe_number(row.get("timing_allocation_time_ms")),
            "allocation_time_std_ms": safe_number(row.get("timing_allocation_time_std_ms")) or 0.0,
            "decision_total_mean_ms": safe_number(row.get("timing_decision_total_mean_ms")),
        }
        for phase in TIME_PHASE_PREFIXES:
            combo_map[key][f"{phase}_mean_ms"] = safe_number(row.get(f"timing_{phase}_mean_ms"))
            combo_map[key][f"{phase}_std_ms"] = safe_number(row.get(f"timing_{phase}_std_ms")) or 0.0
            combo_map[key][f"{phase}_total_ms"] = safe_number(row.get(f"timing_{phase}_total_ms")) or 0.0
        combo_map[key]["decision_total_total_ms"] = safe_number(row.get("timing_decision_total_total_ms")) or 0.0
    return list(combo_map.values())


def plot_time_component_comparisons(joined_rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    combos = _combination_rows_for_barplots(joined_rows)
    if not combos:
        return saved_paths

    resolvers = _ordered_resolvers([str(row["resolver"]) for row in combos])
    proposers = sorted({str(row["proposer"]) for row in combos})
    proposer_index = {proposer: idx for idx, proposer in enumerate(proposers)}
    combo_lookup = {(str(row["proposer"]), str(row["resolver"])): row for row in combos}

    nrows, ncols = _subplots_layout(len(resolvers))
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(max(5 * ncols, 8), 5 * nrows), squeeze=False, sharey=True)
    axis_grid = _visible_axis_grid(axes)
    for idx, resolver in enumerate(resolvers):
        axis = axis_grid[idx]
        positions = list(range(len(proposers)))
        proposal_vals: list[float] = []
        resolution_vals: list[float] = []
        allocation_stds: list[float] = []
        for proposer in proposers:
            row = combo_lookup.get((proposer, resolver), {})
            proposal_vals.append(float(row.get("proposal_time_ms") or 0.0))
            resolution_vals.append(float(row.get("resolution_time_ms") or 0.0))
            allocation_stds.append(float(row.get("allocation_time_std_ms") or 0.0))
        axis.bar(positions, proposal_vals, color="#54A24B", edgecolor="#2F3E4E", linewidth=0.6, label="proposal")
        axis.bar(positions, resolution_vals, bottom=proposal_vals, color="#ECA02C", edgecolor="#2F3E4E", linewidth=0.6, label="resolution")
        total_vals = [p + r for p, r in zip(proposal_vals, resolution_vals)]
        axis.errorbar(positions, total_vals, yerr=allocation_stds, fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2.2)
        axis.set_title(_display_label(resolver))
        axis.set_xticks(positions)
        axis.set_xticklabels([_display_label(p) for p in proposers], rotation=43, ha="right")
        axis.grid(axis="y", alpha=0.2, linestyle="-")
        axis.set_axisbelow(True)
        if idx % ncols == 0:
            axis.set_ylabel("Latency per decision (ms)")
    for axis in axis_grid[len(resolvers):]:
        axis.set_visible(False)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor="#54A24B", markeredgecolor="#2F3E4E", label="proposal"),
        Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor="#ECA02C", markeredgecolor="#2F3E4E", label="resolution"),
    ]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.86, 0.98), frameon=False)
    figure.suptitle("Proposal vs resolution measured latency by resolver", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 0.84, 0.94))
    out1 = output_dir / "proposal_resolution_time_cmp.png"
    figure.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(figure)
    saved_paths.append(out1)

    resolver_colors = _series_color_map(resolvers, preferred=RESOLVER_COLOR_MAP)
    figure2, axis2 = plt.subplots(figsize=(_grouped_figure_width(len(proposers)), 5.6))
    bar_width = GROUPED_BAR_WIDTH
    offsets = [(i - (len(resolvers) - 1) / 2.0) * bar_width for i in range(len(resolvers))]
    base_positions = list(range(len(proposers)))
    for i, resolver in enumerate(resolvers):
        vals: list[float] = []
        stds: list[float] = []
        for proposer in proposers:
            row = combo_lookup.get((proposer, resolver), {})
            vals.append(float(row.get("proposal_time_ms") or 0.0))
            stds.append(float(row.get("proposal_time_std_ms") or 0.0))
        shifted = [x + offsets[i] for x in base_positions]
        axis2.bar(shifted, vals, width=bar_width, yerr=stds, capsize=2.5, color=resolver_colors.get(resolver, "#4C78A8"), edgecolor="#2F3E4E", linewidth=0.6, label=_display_label(resolver))
    axis2.set_xticks(base_positions)
    axis2.set_xticklabels([_display_label(p) for p in proposers], rotation=43, ha="right")
    axis2.set_ylabel("Proposal latency per decision (ms)")
    axis2.set_title("Proposal latency by proposer and resolver")
    axis2.grid(axis="y", alpha=0.2, linestyle="-")
    axis2.legend(title="resolver", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    figure2.tight_layout(rect=(0.0, 0.0, 0.82, 0.97))
    out2 = output_dir / "proposal_time_by_resolver.png"
    figure2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(figure2)
    saved_paths.append(out2)

    proposer_markers = _policy_marker_map(proposers)
    proposer_colors = _series_color_map(proposers)
    figure3, axis3 = plt.subplots(figsize=(_grouped_figure_width(len(resolvers)), 5.6))
    offsets3 = [(i - (len(proposers) - 1) / 2.0) * bar_width for i in range(len(proposers))]
    base_positions3 = list(range(len(resolvers)))
    for i, proposer in enumerate(proposers):
        vals = []
        stds = []
        for resolver in resolvers:
            row = combo_lookup.get((proposer, resolver), {})
            vals.append(float(row.get("resolution_time_ms") or 0.0))
            stds.append(float(row.get("resolution_time_std_ms") or 0.0))
        shifted = [x + offsets3[i] for x in base_positions3]
        axis3.bar(shifted, vals, width=bar_width, yerr=stds, capsize=2.5, color=proposer_colors.get(proposer, "#4C78A8"), edgecolor="#2F3E4E", linewidth=0.6, label=_display_label(proposer))
    axis3.set_xticks(base_positions3)
    axis3.set_xticklabels([_display_label(r) for r in resolvers], rotation=43, ha="right")
    axis3.set_ylabel("Resolution latency per decision (ms)")
    axis3.set_title("Resolution latency by resolver and proposer")
    axis3.grid(axis="y", alpha=0.2, linestyle="-")
    axis3.legend(title="proposer/policy", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    figure3.tight_layout(rect=(0.0, 0.0, 0.82, 0.97))
    out3 = output_dir / "resolution_time_by_proposer.png"
    figure3.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close(figure3)
    saved_paths.append(out3)

    return saved_paths


def plot_time_phase_breakdown_mean(joined_rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    combos = _combination_rows_for_barplots(joined_rows)
    if not combos:
        return saved_paths

    resolvers = _ordered_resolvers([str(row["resolver"]) for row in combos])
    proposers = sorted({str(row["proposer"]) for row in combos})
    lookup = {(str(row["proposer"]), str(row["resolver"])): row for row in combos}

    nrows, ncols = _subplots_layout(len(resolvers))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(max(5 * ncols, 8), 5 * nrows), squeeze=False, sharey=True)
    axis_grid = _visible_axis_grid(axes)
    for idx, resolver in enumerate(resolvers):
        axis = axis_grid[idx]
        xs = list(range(len(proposers)))
        bottoms = [0.0] * len(proposers)
        for phase in TIME_PHASE_PREFIXES:
            vals: list[float] = []
            for proposer in proposers:
                row = lookup.get((proposer, resolver), {})
                vals.append(float(row.get(f"{phase}_mean_ms") or 0.0))
            axis.bar(xs, vals, bottom=bottoms, color=TIME_PHASE_COLORS.get(phase, "#4C78A8"), edgecolor="#2F3E4E", linewidth=0.5, label=phase)
            bottoms = [a + b for a, b in zip(bottoms, vals)]

        decision_vals: list[float] = []
        for proposer in proposers:
            row = lookup.get((proposer, resolver), {})
            decision_vals.append(float(row.get("decision_total_mean_ms") or 0.0))
        axis.plot(xs, decision_vals, linestyle="", marker="_", markersize=12, color="#111111", label="decision_total")

        # Show allocation phase values directly so they remain readable against large simulation bars.
        for i, proposer in enumerate(proposers):
            row = lookup.get((proposer, resolver), {})
            proposal_val = float(row.get("proposal_mean_ms") or row.get("proposal_time_ms") or 0.0)
            resolution_val = float(row.get("resolution_mean_ms") or row.get("resolution_time_ms") or 0.0)
            axis.text(i, proposal_val + 0.01, f"P:{proposal_val:.2f}", fontsize=7, rotation=90, va="bottom", ha="center")
            axis.text(i, proposal_val + resolution_val + 0.01, f"R:{resolution_val:.2f}", fontsize=7, rotation=90, va="bottom", ha="center")

        axis.set_title(_display_label(resolver))
        axis.set_xticks(xs)
        axis.set_xticklabels([_display_label(p) for p in proposers], rotation=43, ha="right")
        axis.grid(axis="y", alpha=0.2, linestyle="-")
        axis.set_axisbelow(True)
        if idx % ncols == 0:
            axis.set_ylabel("Mean latency per decision (ms)")

    for axis in axis_grid[len(resolvers):]:
        axis.set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=TIME_PHASE_COLORS.get(phase, "#4C78A8"), markeredgecolor="#2F3E4E", label=phase)
        for phase in TIME_PHASE_PREFIXES
    ]
    legend_handles.append(Line2D([0], [0], marker="_", linestyle="", markersize=12, color="#111111", label="decision_total"))
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.86, 0.98), frameon=False)
    fig.suptitle("Decision-phase latency breakdown by proposer-resolver combination (resolver-dependent trajectories)", fontsize=14)
    fig.text(
        0.01,
        0.01,
        "Phase times are observed under resolver-dependent trajectories and are not fixed-input microbenchmarks.",
        fontsize=8,
        alpha=0.85,
    )
    fig.tight_layout(rect=(0.0, 0.03, 0.84, 0.94))
    out1 = output_dir / "decision_phase_breakdown_by_combination.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(out1)

    phase_values: dict[str, list[float]] = {phase: [] for phase in TIME_PHASE_PREFIXES + ["decision_total"]}
    rl_phase_values: dict[str, list[float]] = {phase: [] for phase in TIME_PHASE_PREFIXES + ["decision_total"]}
    for row in combos:
        target = rl_phase_values if _is_rl_policy_label(row.get("proposer")) else phase_values
        for phase in TIME_PHASE_PREFIXES:
            value = safe_number(row.get(f"{phase}_mean_ms"))
            if value is not None:
                target[phase].append(float(value))
        decision_val = safe_number(row.get("decision_total_mean_ms"))
        if decision_val is not None:
            target["decision_total"].append(float(decision_val))

    named_means = [sum(phase_values[phase]) / len(phase_values[phase]) if phase_values[phase] else 0.0 for phase in TIME_PHASE_PREFIXES]
    named_stds = [_sample_std(phase_values[phase]) if phase_values[phase] else 0.0 for phase in TIME_PHASE_PREFIXES]
    decision_mean = (sum(phase_values["decision_total"]) / len(phase_values["decision_total"])) if phase_values["decision_total"] else 0.0
    decision_std = _sample_std(phase_values["decision_total"]) if phase_values["decision_total"] else 0.0
    has_rl_overall = any(rl_phase_values[phase] for phase in TIME_PHASE_PREFIXES)
    rl_named_means = [sum(rl_phase_values[phase]) / len(rl_phase_values[phase]) if rl_phase_values[phase] else 0.0 for phase in TIME_PHASE_PREFIXES]
    rl_named_stds = [_sample_std(rl_phase_values[phase]) if rl_phase_values[phase] else 0.0 for phase in TIME_PHASE_PREFIXES]
    rl_decision_mean = (sum(rl_phase_values["decision_total"]) / len(rl_phase_values["decision_total"])) if rl_phase_values["decision_total"] else 0.0

    fig2, axis2 = plt.subplots(figsize=(10.0, 5.6))
    positions = list(range(len(TIME_PHASE_PREFIXES)))
    bar_width = 0.38 if has_rl_overall else 0.68
    baseline_positions = [p - bar_width / 2.0 for p in positions] if has_rl_overall else positions
    axis2.bar(
        baseline_positions,
        named_means,
        width=bar_width,
        yerr=named_stds,
        capsize=2.5,
        color=[TIME_PHASE_COLORS.get(phase, "#4C78A8") for phase in TIME_PHASE_PREFIXES],
        edgecolor="#2F3E4E",
        linewidth=0.6,
        label="baseline" if has_rl_overall else None,
    )
    if has_rl_overall:
        rl_positions = [p + bar_width / 2.0 for p in positions]
        axis2.bar(
            rl_positions,
            rl_named_means,
            width=bar_width,
            yerr=rl_named_stds,
            capsize=2.5,
            color=[TIME_PHASE_COLORS.get(phase, "#4C78A8") for phase in TIME_PHASE_PREFIXES],
            edgecolor="#111111",
            linewidth=1.1,
            hatch="//",
            label="RL",
        )
        axis2.axhline(rl_decision_mean, linestyle=":", color="#111111", linewidth=1.2, label="RL decision_total mean")
    axis2.axhline(decision_mean, linestyle="--", color="#111111", linewidth=1.0, label="decision_total mean" if not has_rl_overall else "baseline decision_total mean")
    axis2.set_xticks(positions)
    axis2.set_xticklabels([_display_label(phase) for phase in TIME_PHASE_PREFIXES], rotation=35, ha="right")
    axis2.set_ylabel("Mean latency per decision (ms)")
    axis2.set_title("Average decision-phase latency across proposer-resolver combinations")
    axis2.grid(axis="y", alpha=0.2, linestyle="-")
    axis2.legend(frameon=False)
    named_sum = sum(named_means)
    accounting_diff = decision_mean - named_sum
    fig2.text(
        0.01,
        0.01,
        "Each baseline proposer-resolver combination contributes equally; RL (if present) is averaged separately and shown as a second bar, not mixed into the baseline average.",
        fontsize=8,
        alpha=0.85,
    )
    fig2.text(
        0.01,
        0.04,
        f"sum of named phase means={named_sum:.4f} ms | direct decision_total mean={decision_mean:.4f} ms | accounting difference={accounting_diff:.4f} ms",
        fontsize=8,
        alpha=0.85,
    )
    fig2.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    out2 = output_dir / "decision_phase_breakdown_overall.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(out2)

    return saved_paths


def plot_time_phase_totals(joined_rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    combos = _combination_rows_for_barplots(joined_rows)
    if not combos:
        return []

    baseline_combos = [row for row in combos if not _is_rl_policy_label(row.get("proposer"))]
    rl_combos = [row for row in combos if _is_rl_policy_label(row.get("proposer"))]

    totals_ms = {phase: 0.0 for phase in TIME_PHASE_PREFIXES + ["decision_total"]}
    for row in baseline_combos:
        for phase in TIME_PHASE_PREFIXES:
            totals_ms[phase] += float(row.get(f"{phase}_total_ms") or 0.0)
        totals_ms["decision_total"] += float(row.get("decision_total_total_ms") or 0.0)

    rl_totals_ms = {phase: 0.0 for phase in TIME_PHASE_PREFIXES + ["decision_total"]}
    for row in rl_combos:
        for phase in TIME_PHASE_PREFIXES:
            rl_totals_ms[phase] += float(row.get(f"{phase}_total_ms") or 0.0)
        rl_totals_ms["decision_total"] += float(row.get("decision_total_total_ms") or 0.0)
    has_rl = bool(rl_combos)

    max_total_ms = max(list(totals_ms.values()) + list(rl_totals_ms.values())) if totals_ms else 0.0
    use_seconds = max_total_ms >= 5000.0
    scale = 0.001 if use_seconds else 1.0
    unit = "s" if use_seconds else "ms"

    labels = TIME_PHASE_PREFIXES + ["decision_total"]
    values = [totals_ms[label] * scale for label in labels]
    colors = [TIME_PHASE_COLORS.get(label, "#4C78A8") if label != "decision_total" else "#111111" for label in labels]

    fig, axis = plt.subplots(figsize=(10.0, 5.6))
    positions = list(range(len(labels)))
    bar_width = 0.38 if has_rl else 0.68
    baseline_positions = [p - bar_width / 2.0 for p in positions] if has_rl else positions
    axis.bar(baseline_positions, values, width=bar_width, color=colors, edgecolor="#2F3E4E", linewidth=0.6, label="baseline" if has_rl else None)
    if has_rl:
        rl_values = [rl_totals_ms[label] * scale for label in labels]
        rl_positions = [p + bar_width / 2.0 for p in positions]
        axis.bar(rl_positions, rl_values, width=bar_width, color=colors, edgecolor="#111111", linewidth=1.1, hatch="//", label="RL")
        axis.legend(frameon=False)
    axis.set_xticks(positions)
    axis.set_xticklabels([_display_label(label) for label in labels], rotation=35, ha="right")
    axis.set_ylabel(f"Total measured time ({unit})")
    axis.set_title("Total measured time by decision phase")
    axis.grid(axis="y", alpha=0.2, linestyle="-")
    fig.text(
        0.01,
        0.01,
        "Totals depend on the number of measured decisions and timing runs and should not be used to compare scenarios with different measurement volumes. "
        "RL (if present) is summed separately and shown as a second bar, not mixed into the baseline total.",
        fontsize=8,
        alpha=0.85,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    out = output_dir / "decision_phase_totals_overall.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [out]


def _timing_group_key_from_canonical_row(row: dict[str, object]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("scenario_canonical") or ""),
        str(row.get("route_canonical") or ""),
        str(row.get("protocol_canonical") or ""),
        str(row.get("method_canonical") or ""),
        str(row.get("inference_mode_canonical") or ""),
        str(row.get("proposer_canonical") or ""),
        str(row.get("resolver_canonical") or ""),
    )


def _timing_group_key_from_aggregated_group(group: dict[str, object]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(group.get("scenario") or ""),
        str(group.get("route") or ""),
        str(group.get("protocol") or ""),
        str(group.get("method") or ""),
        str(group.get("inference_mode") or ""),
        str(group.get("proposer") or ""),
        str(group.get("resolver") or ""),
    )


def _source_row_identity(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    row_id = str(row.get("source_timing_row_id") or "").strip()
    if row_id:
        return (row_id, "", "", "", "")
    return (
        str(row.get("source_timing_file") or ""),
        str(row.get("seed_canonical") or ""),
        str(row.get("episode_canonical") or ""),
        str(row.get("method_canonical") or ""),
        str(row.get("inference_mode_canonical") or ""),
    )


def _ensure_one_to_one_row_assignment(canonical_rows: list[dict[str, object]]) -> None:
    assignment: dict[tuple[str, str, str, str, str], set[tuple[str, str, str, str, str, str, str]]] = {}
    for row in canonical_rows:
        identity = _source_row_identity(row)
        assignment.setdefault(identity, set()).add(_timing_group_key_from_canonical_row(row))

    conflicts = {
        identity: groups
        for identity, groups in assignment.items()
        if len(groups) > 1
    }
    if not conflicts:
        return

    lines = ["One source timing row was assigned to multiple canonical proposer-resolver groups:"]
    for identity, groups in sorted(conflicts.items()):
        lines.append(f"- row={identity}")
        for group in sorted(groups):
            lines.append(f"    group={group}")
    raise ValueError("\n".join(lines))


def _format_compact_list(values: list[str]) -> str:
    clean = sorted({str(value).strip() for value in values if str(value).strip()})
    return ";".join(clean)


def _proposal_group_means_or_error(
    group_rows: list[dict[str, object]],
    *,
    group_key: tuple[str, str, str, str, str, str, str],
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    measured_pairs: list[tuple[float, float, float]] = []
    run_level_means: list[float] = []
    for row in group_rows:
        measured_decisions = safe_number(row.get("measured_decisions"))
        proposal_total = safe_number(row.get("proposal_total_ms"))
        proposal_mean = safe_number(row.get("proposal_mean_ms"))
        if measured_decisions is not None and measured_decisions > 0.0 and proposal_total is not None and proposal_mean is not None:
            measured_pairs.append((float(measured_decisions), float(proposal_total), float(proposal_mean)))
            run_level_means.append(float(proposal_mean))

    if not measured_pairs:
        return (None, None, None, 0.0, None, None)

    weighted_n = sum(m * mean for m, _, mean in measured_pairs)
    weighted_d = sum(m for m, _, _ in measured_pairs)
    pooled_n = sum(total for _, total, _ in measured_pairs)

    pooled = pooled_n / weighted_d if weighted_d > 0.0 else None
    weighted = weighted_n / weighted_d if weighted_d > 0.0 else None
    if pooled is not None and weighted is not None and not math.isclose(pooled, weighted, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "Proposal pooled mean mismatch against weighted run-level means for group "
            f"{group_key}: pooled={pooled}, weighted={weighted}."
        )

    unweighted = (sum(run_level_means) / float(len(run_level_means))) if run_level_means else None
    std = _sample_std(run_level_means)
    min_v = min(run_level_means) if run_level_means else None
    max_v = max(run_level_means) if run_level_means else None
    return pooled, weighted, unweighted, std, min_v, max_v


def _pooled_mean_from_total(
    group_rows: list[dict[str, object]],
    *,
    total_field: str,
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in group_rows:
        measured_decisions = safe_number(row.get("measured_decisions"))
        total_value = safe_number(row.get(total_field))
        if measured_decisions is None or measured_decisions <= 0.0 or total_value is None:
            continue
        numerator += float(total_value)
        denominator += float(measured_decisions)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _pooled_mean_from_total_or_mean(
    group_rows: list[dict[str, object]],
    *,
    total_field: str,
    mean_field: str,
) -> float | None:
    pooled_from_total = _pooled_mean_from_total(group_rows, total_field=total_field)
    if pooled_from_total is not None:
        return pooled_from_total

    numerator = 0.0
    denominator = 0.0
    for row in group_rows:
        measured_decisions = safe_number(row.get("measured_decisions"))
        mean_value = safe_number(row.get(mean_field))
        if measured_decisions is None or measured_decisions <= 0.0 or mean_value is None:
            continue
        numerator += float(mean_value) * float(measured_decisions)
        denominator += float(measured_decisions)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _metrics_join_key_fields(row: dict[str, object]) -> tuple[str, str, str, str, str, str, str]:
    return (
        _scenario_of_row(row),
        _route_alias_from_text(str(row.get("route_construction") or "nearest")),
        _protocol_alias_of_row(row),
        normalize_name(str(row.get("timing_method") or row.get("method") or "na")) or "na",
        normalize_name(str(row.get("timing_inference_mode") or row.get("inference_mode") or "na")) or "na",
        _canonical_policy_join_name(row.get("timing_proposer") if row.get("timing_proposer") is not None else row.get("pol")),
        _canonical_resolver_name(row.get("timing_resolver") if row.get("timing_resolver") is not None else row.get("resolver")),
    )


def _diagnostic_workload_values(metric_row: dict[str, object] | None, proposal_mean_ms: float | None) -> dict[str, object]:
    out: dict[str, object] = {field: None for field in TIME_DIAG_WORKLOAD_FIELDS}
    if metric_row is None:
        out["candidate_entries_per_macro_step"] = None
        out["active_robot_decisions_per_macro_step"] = None
        out["proposal_ms_per_candidate_entry"] = None
        out["proposal_ms_per_active_robot_decision"] = None
        return out

    for field in TIME_DIAG_WORKLOAD_FIELDS:
        out[field] = safe_number(metric_row.get(field))

    num_robots = safe_number(metric_row.get("num_robots"))
    mcand = safe_number(metric_row.get("mcand"))
    noop = safe_number(metric_row.get("noop"))
    candidate_entries = (num_robots * mcand) if num_robots is not None and mcand is not None else None
    active_robot_decisions = (num_robots * (1.0 - noop)) if num_robots is not None and noop is not None else None
    out["candidate_entries_per_macro_step"] = candidate_entries
    out["active_robot_decisions_per_macro_step"] = active_robot_decisions
    out["proposal_ms_per_candidate_entry"] = _safe_ratio(proposal_mean_ms, candidate_entries)
    out["proposal_ms_per_active_robot_decision"] = _safe_ratio(proposal_mean_ms, active_robot_decisions)
    return out


def _augment_workload_with_timing_observed(
    *,
    workload: dict[str, object],
    group_rows: list[dict[str, object]],
    proposal_mean_ms: float | None,
) -> dict[str, object]:
    candidate_entries = safe_number(workload.get("candidate_entries_per_macro_step"))
    active_decisions = safe_number(workload.get("active_robot_decisions_per_macro_step"))

    if candidate_entries is None:
        candidate_entries = _pooled_mean_from_total_or_mean(
            group_rows,
            total_field="n_candidate_pairs_total",
            mean_field="n_candidate_pairs_mean",
        )
        workload["candidate_entries_per_macro_step"] = candidate_entries

    if active_decisions is None:
        active_decisions = _pooled_mean_from_total_or_mean(
            group_rows,
            total_field="n_proposals_total",
            mean_field="n_proposals_mean",
        )
        workload["active_robot_decisions_per_macro_step"] = active_decisions

    if safe_number(workload.get("proposal_ms_per_candidate_entry")) is None:
        workload["proposal_ms_per_candidate_entry"] = _safe_ratio(proposal_mean_ms, candidate_entries)
    if safe_number(workload.get("proposal_ms_per_active_robot_decision")) is None:
        workload["proposal_ms_per_active_robot_decision"] = _safe_ratio(proposal_mean_ms, active_decisions)
    return workload


def _build_proposal_resolver_diagnostics(
    *,
    canonical_rows_all: list[dict[str, object]],
    timing_groups_for_combo: list[dict[str, object]],
    joined_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    group_keys = {_timing_group_key_from_aggregated_group(group) for group in timing_groups_for_combo}
    canonical_rows = [row for row in canonical_rows_all if _timing_group_key_from_canonical_row(row) in group_keys]
    _ensure_one_to_one_row_assignment(canonical_rows)

    matched_metric_by_group: dict[tuple[str, str, str, str, str, str, str], dict[str, object]] = {}
    for row in joined_rows:
        matched_metric_by_group[_metrics_join_key_fields(row)] = row

    group_rows_map: dict[tuple[str, str, str, str, str, str, str], list[dict[str, object]]] = {}
    for row in canonical_rows:
        gk = _timing_group_key_from_canonical_row(row)
        if gk not in matched_metric_by_group:
            continue
        group_rows_map.setdefault(gk, []).append(row)

    run_rows_out: list[dict[str, object]] = []
    group_rows_out: list[dict[str, object]] = []

    for gk, rows in sorted(group_rows_map.items()):
        scenario, route, protocol, method, inference_mode, proposer, resolver = gk
        matched_metric_row = matched_metric_by_group.get(gk)

        proposal_total_sum = sum(float(safe_number(row.get("proposal_total_ms")) or 0.0) for row in rows)
        measured_sum = sum(float(safe_number(row.get("measured_decisions")) or 0.0) for row in rows)
        proposal_pooled, proposal_weighted, proposal_unweighted, proposal_std, proposal_min, proposal_max = _proposal_group_means_or_error(rows, group_key=gk)
        resolution_pooled = _pooled_mean_from_total(rows, total_field="resolution_total_ms")
        decision_total_pooled = _pooled_mean_from_total(rows, total_field="decision_total_ms")

        workload = _diagnostic_workload_values(matched_metric_row, proposal_pooled)
        workload = _augment_workload_with_timing_observed(
            workload=workload,
            group_rows=rows,
            proposal_mean_ms=proposal_pooled,
        )

        group_rows_out.append(
            {
                "scenario": scenario,
                "route": route,
                "protocol": protocol,
                "method": method,
                "inference_mode": inference_mode,
                "proposer": proposer,
                "resolver": resolver,
                "n_timing_runs": len(rows),
                "n_measured_decisions": int(round(measured_sum)),
                "seeds": _format_compact_list([str(row.get("seed_canonical") or "") for row in rows]),
                "episodes": _format_compact_list([str(row.get("episode_canonical") or "") for row in rows]),
                "source_files": _format_compact_list([str(row.get("source_timing_file") or "") for row in rows]),
                "proposal_total_ms": proposal_total_sum,
                "proposal_mean_pooled_ms": proposal_pooled,
                "proposal_mean_weighted_ms": proposal_weighted,
                "proposal_mean_unweighted_ms": proposal_unweighted,
                "proposal_mean_std_ms": proposal_std,
                "proposal_mean_min_ms": proposal_min,
                "proposal_mean_max_ms": proposal_max,
                "resolution_mean_pooled_ms": resolution_pooled,
                "decision_total_mean_pooled_ms": decision_total_pooled,
                **workload,
            }
        )

        for row in rows:
            run_rows_out.append(
                {
                    "source_file": row.get("source_timing_file", ""),
                    "scenario": scenario,
                    "route": route,
                    "protocol": protocol,
                    "method": method,
                    "inference_mode": inference_mode,
                    "proposer": proposer,
                    "resolver": resolver,
                    "seed": row.get("seed"),
                    "episode": row.get("episode"),
                    "timing_protocol": row.get("timing_protocol"),
                    "device": row.get("device"),
                    "host_name": row.get("host_name"),
                    "cpu_model": row.get("cpu_model"),
                    "torch_num_threads": row.get("torch_num_threads"),
                    "omp_num_threads": row.get("omp_num_threads"),
                    "mkl_num_threads": row.get("mkl_num_threads"),
                    "warmup_decisions": row.get("warmup_decisions"),
                    "measured_decisions": row.get("measured_decisions"),
                    "proposal_total_ms": row.get("proposal_total_ms"),
                    "proposal_mean_ms": row.get("proposal_mean_ms"),
                    "proposal_p50_ms": row.get("proposal_p50_ms"),
                    "proposal_p90_ms": row.get("proposal_p90_ms"),
                    "proposal_p95_ms": row.get("proposal_p95_ms"),
                    "proposal_max_ms": row.get("proposal_max_ms"),
                    "resolution_total_ms": row.get("resolution_total_ms"),
                    "resolution_mean_ms": row.get("resolution_mean_ms"),
                    "decision_total_mean_ms": row.get("decision_mean_ms"),
                }
            )

    matched_replicates_rows: list[dict[str, object]] = []
    matched_summary_rows: list[dict[str, object]] = []
    proposer_scope_map: dict[tuple[str, str, str, str, str, str], dict[str, list[dict[str, object]]]] = {}
    for gk, rows in group_rows_map.items():
        scenario, route, protocol, method, inference_mode, proposer, resolver = gk
        proposer_scope = (scenario, route, protocol, method, inference_mode, proposer)
        proposer_scope_map.setdefault(proposer_scope, {})[resolver] = rows

    for proposer_scope, resolver_rows in sorted(proposer_scope_map.items()):
        scenario, route, protocol, method, inference_mode, proposer = proposer_scope
        resolver_replicates: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
        for resolver, rows in resolver_rows.items():
            replicate_map: dict[tuple[str, str], dict[str, object]] = {}
            for row in rows:
                proposal_mean = safe_number(row.get("proposal_mean_ms"))
                if proposal_mean is None:
                    continue
                rep = (str(row.get("seed_canonical") or ""), str(row.get("episode_canonical") or ""))
                replicate_map[rep] = row
            resolver_replicates[resolver] = replicate_map

        nonempty_sets = [set(m.keys()) for m in resolver_replicates.values() if m]
        if not nonempty_sets:
            print(f"[warn] No common replicate set exists for proposer '{proposer}'", file=sys.stderr)
            continue
        common = set.intersection(*nonempty_sets)
        if len(common) < 2:
            print(
                f"[warn] Fewer than two matched replicates for proposer '{proposer}' ({len(common)} replicate(s))",
                file=sys.stderr,
            )
        if not common:
            print(f"[warn] No common replicate set exists for proposer '{proposer}'", file=sys.stderr)
            continue

        for resolver, replicate_map in resolver_replicates.items():
            values: list[float] = []
            for seed, episode in sorted(common):
                row = replicate_map.get((seed, episode))
                if row is None:
                    continue
                value = safe_number(row.get("proposal_mean_ms"))
                if value is None:
                    continue
                values.append(float(value))
                matched_replicates_rows.append(
                    {
                        "scenario": scenario,
                        "route": route,
                        "protocol": protocol,
                        "method": method,
                        "inference_mode": inference_mode,
                        "proposer": proposer,
                        "resolver": resolver,
                        "seed": seed,
                        "episode": episode,
                        "proposal_mean_ms": float(value),
                        "measured_decisions": row.get("measured_decisions"),
                        "source_file": row.get("source_timing_file", ""),
                    }
                )

            matched_summary_rows.append(
                {
                    "scenario": scenario,
                    "route": route,
                    "protocol": protocol,
                    "method": method,
                    "inference_mode": inference_mode,
                    "proposer": proposer,
                    "resolver": resolver,
                    "n_matched_replicates": len(values),
                    "matched_proposal_mean_ms": (sum(values) / float(len(values))) if values else None,
                    "matched_proposal_std_ms": _sample_std(values),
                    "matched_proposal_min_ms": min(values) if values else None,
                    "matched_proposal_max_ms": max(values) if values else None,
                }
            )

    stats = {
        "n_timing_rows": len(canonical_rows),
        "n_canonical_timing_groups": len(group_rows_map),
        "n_matched_metrics_rows": len(joined_rows),
        "n_unmatched_timing_groups": max(0, len(group_keys) - len(group_rows_map)),
        "n_unmatched_metrics_rows": max(0, len(joined_rows) - len(group_rows_map)),
    }
    return run_rows_out, group_rows_out, matched_replicates_rows, matched_summary_rows, stats


def _write_dict_rows_csv(output_path: Path, rows: list[dict[str, object]], header: list[str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})
    return output_path


def _plot_proposal_latency_by_proposer_and_resolver(group_rows: list[dict[str, object]], output_dir: Path) -> Path:
    resolvers = _ordered_resolvers([str(row.get("resolver") or "") for row in group_rows])
    proposers = sorted({str(row.get("proposer") or "") for row in group_rows})
    lookup = {(str(row.get("proposer")), str(row.get("resolver"))): row for row in group_rows}
    nrows, ncols = _subplots_layout(len(resolvers))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(max(5 * ncols, 8), 5 * nrows), squeeze=False, sharey=True)
    axis_grid = _visible_axis_grid(axes)
    for idx, resolver in enumerate(resolvers):
        axis = axis_grid[idx]
        xs = list(range(len(proposers)))
        vals: list[float] = []
        errs: list[float] = []
        labels_x: list[int] = []
        for pos, proposer in enumerate(proposers):
            row = lookup.get((proposer, resolver))
            if row is None:
                continue
            value = safe_number(row.get("proposal_mean_pooled_ms"))
            if value is None:
                continue
            vals.append(float(value))
            errs.append(float(safe_number(row.get("proposal_mean_std_ms")) or 0.0))
            labels_x.append(pos)
        axis.bar(labels_x, vals, yerr=errs, capsize=2.5, color="#54A24B", edgecolor="#2F3E4E", linewidth=0.6)
        axis.set_title(_display_label(resolver))
        axis.set_xticks(xs)
        axis.set_xticklabels([_display_label(p) for p in proposers], rotation=43, ha="right")
        axis.grid(axis="y", alpha=0.2, linestyle="-")
        axis.set_axisbelow(True)
        if idx % ncols == 0:
            axis.set_ylabel("Proposal latency per decision (ms)")
    for axis in axis_grid[len(resolvers):]:
        axis.set_visible(False)
    fig.suptitle("Observed proposal latency by proposer and resolver", fontsize=14)
    fig.text(0.01, 0.01, "Proposal latency is measured on trajectories generated by each proposer-resolver combination.", fontsize=8, alpha=0.85)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))
    out = output_dir / "proposal_latency_by_proposer_and_resolver.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_proposal_workload_panels(group_rows: list[dict[str, object]], output_dir: Path) -> Path:
    resolvers = _ordered_resolvers([str(row.get("resolver") or "") for row in group_rows])
    proposers = sorted({str(row.get("proposer") or "") for row in group_rows})
    resolver_colors = _series_color_map(resolvers, preferred=RESOLVER_COLOR_MAP)
    lookup = {(str(row.get("proposer")), str(row.get("resolver"))): row for row in group_rows}
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.5, 5.3), squeeze=False)
    fields = [
        ("candidate_entries_per_macro_step", "Candidate entries per macro step"),
        ("active_robot_decisions_per_macro_step", "Active robot decisions per macro step"),
    ]
    bar_width = GROUPED_BAR_WIDTH
    base_positions = list(range(len(proposers)))
    offsets = [(idx - (len(resolvers) - 1) / 2.0) * bar_width for idx in range(len(resolvers))]
    for axis, (field, ylabel) in zip(axes[0], fields):
        for idx, resolver in enumerate(resolvers):
            vals: list[float] = []
            xpos: list[float] = []
            for pos, proposer in enumerate(proposers):
                row = lookup.get((proposer, resolver))
                value = safe_number(row.get(field)) if row is not None else None
                if value is None:
                    continue
                vals.append(float(value))
                xpos.append(pos + offsets[idx])
            axis.bar(xpos, vals, width=bar_width, color=resolver_colors.get(resolver, "#4C78A8"), edgecolor="#2F3E4E", linewidth=0.6, label=_display_label(resolver))
        axis.set_xticks(base_positions)
        axis.set_xticklabels([_display_label(p) for p in proposers], rotation=43, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2, linestyle="-")
        axis.set_axisbelow(True)
    axes[0][0].set_title("Candidate workload")
    axes[0][1].set_title("Active-decision workload")
    axes[0][1].legend(title="resolver", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 0.97))
    out = output_dir / "proposal_workload_by_proposer_and_resolver.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_proposal_normalized_panels(group_rows: list[dict[str, object]], output_dir: Path) -> Path:
    resolvers = _ordered_resolvers([str(row.get("resolver") or "") for row in group_rows])
    proposers = sorted({str(row.get("proposer") or "") for row in group_rows})
    resolver_colors = _series_color_map(resolvers, preferred=RESOLVER_COLOR_MAP)
    lookup = {(str(row.get("proposer")), str(row.get("resolver"))): row for row in group_rows}
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.5, 5.3), squeeze=False)
    fields = [
        ("proposal_ms_per_candidate_entry", "Proposal ms per candidate entry (diagnostic ratio)"),
        ("proposal_ms_per_active_robot_decision", "Proposal ms per active robot decision (diagnostic ratio)"),
    ]
    bar_width = GROUPED_BAR_WIDTH
    base_positions = list(range(len(proposers)))
    offsets = [(idx - (len(resolvers) - 1) / 2.0) * bar_width for idx in range(len(resolvers))]
    for axis, (field, ylabel) in zip(axes[0], fields):
        for idx, resolver in enumerate(resolvers):
            vals: list[float] = []
            xpos: list[float] = []
            for pos, proposer in enumerate(proposers):
                row = lookup.get((proposer, resolver))
                value = safe_number(row.get(field)) if row is not None else None
                if value is None:
                    continue
                vals.append(float(value))
                xpos.append(pos + offsets[idx])
            axis.bar(xpos, vals, width=bar_width, color=resolver_colors.get(resolver, "#4C78A8"), edgecolor="#2F3E4E", linewidth=0.6, label=_display_label(resolver))
        axis.set_xticks(base_positions)
        axis.set_xticklabels([_display_label(p) for p in proposers], rotation=43, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2, linestyle="-")
        axis.set_axisbelow(True)
    axes[0][0].set_title("Workload-normalized proposal latency")
    axes[0][1].set_title("Workload-normalized proposal latency")
    axes[0][1].legend(title="resolver", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.text(0.01, 0.01, "Normalized ratios are descriptive diagnostics and do not represent algorithmic complexity estimates.", fontsize=8, alpha=0.85)
    fig.tight_layout(rect=(0.0, 0.03, 0.84, 0.97))
    out = output_dir / "proposal_latency_normalized_by_workload.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_proposal_vs_workload_scatter(group_rows: list[dict[str, object]], output_dir: Path) -> Path:
    resolver_labels = _ordered_resolvers([str(row.get("resolver") or "") for row in group_rows])
    resolver_colors = _series_color_map(resolver_labels, preferred=RESOLVER_COLOR_MAP)
    policy_markers = _policy_marker_map([str(row.get("proposer") or "") for row in group_rows])
    fig, axis = plt.subplots(figsize=(8.8, 5.8))
    corr_x: list[float] = []
    corr_y: list[float] = []
    for row in group_rows:
        x_value = safe_number(row.get("candidate_entries_per_macro_step"))
        y_value = safe_number(row.get("proposal_mean_pooled_ms"))
        if x_value is None or y_value is None:
            continue
        resolver = str(row.get("resolver") or "")
        proposer = str(row.get("proposer") or "")
        corr_x.append(float(x_value))
        corr_y.append(float(y_value))
        axis.scatter(
            [float(x_value)],
            [float(y_value)],
            marker=policy_markers.get(proposer, "o"),
            color=resolver_colors.get(resolver, "#4C78A8"),
            edgecolors="#2F3E4E",
            linewidths=0.6,
            s=52,
            alpha=0.95,
            zorder=3,
        )
        axis.annotate(_display_label(proposer), (float(x_value), float(y_value)), textcoords="offset points", xytext=(3, 3), fontsize=8)
    axis.set_xlabel("Mean candidate entries per macro decision")
    axis.set_ylabel("Measured proposal latency per decision (ms)")
    axis.set_title("Proposal latency vs candidate workload")
    axis.grid(axis="y", alpha=0.2, linestyle="-")
    axis.set_axisbelow(True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out = output_dir / "proposal_latency_vs_candidate_workload.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if len(corr_x) >= 3:
        pearson = _pearson_correlation(corr_x, corr_y)
        spearman = _spearman_correlation(corr_x, corr_y)
        print(f"[info] Pearson correlation (proposal latency vs candidate workload): {pearson}")
        print(f"[info] Spearman correlation (proposal latency vs candidate workload): {spearman}")
    return out


def _plot_matched_replicates(
    matched_rows: list[dict[str, object]],
    matched_summary_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path | None:
    proposers = sorted({str(row.get("proposer") or "") for row in matched_rows})
    if not proposers:
        return None
    nrows, ncols = _subplots_layout(len(proposers))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(max(5 * ncols, 8), 5 * nrows), squeeze=False, sharey=False)
    axis_grid = _visible_axis_grid(axes)
    summary_lookup = {
        (str(row.get("proposer") or ""), str(row.get("resolver") or "")): row
        for row in matched_summary_rows
    }
    for idx, proposer in enumerate(proposers):
        axis = axis_grid[idx]
        proposer_rows = [row for row in matched_rows if str(row.get("proposer") or "") == proposer]
        resolvers = _ordered_resolvers([str(row.get("resolver") or "") for row in proposer_rows])
        resolver_index = {resolver: i for i, resolver in enumerate(resolvers)}
        replicate_keys = sorted({(str(row.get("seed") or ""), str(row.get("episode") or "")) for row in proposer_rows})
        for rep in replicate_keys:
            xs: list[float] = []
            ys: list[float] = []
            for row in proposer_rows:
                if (str(row.get("seed") or ""), str(row.get("episode") or "")) != rep:
                    continue
                resolver = str(row.get("resolver") or "")
                value = safe_number(row.get("proposal_mean_ms"))
                if value is None:
                    continue
                xs.append(float(resolver_index[resolver]))
                ys.append(float(value))
            if xs and ys:
                axis.plot(xs, ys, color="#808080", alpha=0.4, linewidth=0.8)
                axis.scatter(xs, ys, color="#4C78A8", s=20, alpha=0.8)

        for resolver in resolvers:
            summary = summary_lookup.get((proposer, resolver))
            if summary is None:
                continue
            mean_v = safe_number(summary.get("matched_proposal_mean_ms"))
            std_v = safe_number(summary.get("matched_proposal_std_ms")) or 0.0
            if mean_v is None:
                continue
            x = float(resolver_index[resolver])
            axis.errorbar([x], [float(mean_v)], yerr=[float(std_v)], fmt="o", color="#C44E52", capsize=2.5, elinewidth=1.0)

        n_matched = len(replicate_keys)
        axis.set_title(f"{_display_label(proposer)} (n={n_matched})")
        axis.set_xticks(list(range(len(resolvers))))
        axis.set_xticklabels([_display_label(resolver) for resolver in resolvers], rotation=35, ha="right")
        axis.set_ylabel("Proposal latency (ms)")
        axis.grid(axis="y", alpha=0.2, linestyle="-")
        axis.set_axisbelow(True)

    for axis in axis_grid[len(proposers):]:
        axis.set_visible(False)
    fig.suptitle("Proposal latency across resolvers using matched replicates", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out = output_dir / "proposal_latency_matched_replicates.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_resolver_spread(group_rows: list[dict[str, object]], matched_summary_rows: list[dict[str, object]], output_dir: Path) -> Path:
    proposers = sorted({str(row.get("proposer") or "") for row in group_rows})
    grouped_by_proposer: dict[str, list[dict[str, object]]] = {proposer: [] for proposer in proposers}
    for row in group_rows:
        grouped_by_proposer[str(row.get("proposer") or "")].append(row)

    matched_by_proposer: dict[str, list[dict[str, object]]] = {proposer: [] for proposer in proposers}
    for row in matched_summary_rows:
        matched_by_proposer.setdefault(str(row.get("proposer") or ""), []).append(row)

    xs = list(range(len(proposers)))
    raw_ratios: list[float | None] = []
    matched_ratios: list[float | None] = []
    workload_ratios: list[float | None] = []
    for proposer in proposers:
        rows = grouped_by_proposer.get(proposer, [])
        raw_vals = [float(v) for v in [safe_number(row.get("proposal_mean_pooled_ms")) for row in rows] if v is not None and v > 0.0]
        candidate_vals = [float(v) for v in [safe_number(row.get("candidate_entries_per_macro_step")) for row in rows] if v is not None and v > 0.0]
        matched_vals = [float(v) for v in [safe_number(row.get("matched_proposal_mean_ms")) for row in matched_by_proposer.get(proposer, [])] if v is not None and v > 0.0]
        raw_ratios.append(_safe_ratio(max(raw_vals), min(raw_vals)) if raw_vals else None)
        matched_ratios.append(_safe_ratio(max(matched_vals), min(matched_vals)) if matched_vals else None)
        workload_ratios.append(_safe_ratio(max(candidate_vals), min(candidate_vals)) if candidate_vals else None)

    fig, axis = plt.subplots(figsize=(10.0, 5.8))
    bar_width = 0.22
    ratio_sets = [
        (raw_ratios, -bar_width, "raw proposal-latency max/min ratio", "#4C78A8"),
        (matched_ratios, 0.0, "matched proposal-latency max/min ratio", "#F58518"),
        (workload_ratios, bar_width, "candidate-workload max/min ratio", "#54A24B"),
    ]
    for values, offset, label, color in ratio_sets:
        xpos: list[float] = []
        ypos: list[float] = []
        for i, value in enumerate(values):
            if value is None:
                continue
            xpos.append(i + offset)
            ypos.append(float(value))
        axis.bar(xpos, ypos, width=bar_width, color=color, edgecolor="#2F3E4E", linewidth=0.6, label=label)
    axis.axhline(1.0, linestyle="--", linewidth=1.0, color="#111111")
    axis.set_xticks(xs)
    axis.set_xticklabels([_display_label(p) for p in proposers], rotation=35, ha="right")
    axis.set_ylabel("Spread ratio")
    axis.set_title("Proposal-latency and workload spread across resolvers")
    axis.grid(axis="y", alpha=0.2, linestyle="-")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(rect=(0.0, 0.0, 0.8, 0.97))
    out = output_dir / "proposal_latency_resolver_spread.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _print_time_diagnostics_summary(
    *,
    group_rows: list[dict[str, object]],
    matched_summary_rows: list[dict[str, object]],
    stats: dict[str, int],
) -> None:
    proposers = sorted({str(row.get("proposer") or "") for row in group_rows})
    print("[info] proposer | n_resolvers | raw_min_ms | raw_max_ms | raw max/min ratio | matched-replicate spread | candidate-workload spread")
    for proposer in proposers:
        rows = [row for row in group_rows if str(row.get("proposer") or "") == proposer]
        resolver_count = len({str(row.get("resolver") or "") for row in rows})
        raw_values = [float(v) for v in [safe_number(row.get("proposal_mean_pooled_ms")) for row in rows] if v is not None and v > 0.0]
        workload_values = [float(v) for v in [safe_number(row.get("candidate_entries_per_macro_step")) for row in rows] if v is not None and v > 0.0]
        matched_values = [
            float(v)
            for v in [
                safe_number(row.get("matched_proposal_mean_ms"))
                for row in matched_summary_rows
                if str(row.get("proposer") or "") == proposer
            ]
            if v is not None and v > 0.0
        ]
        raw_ratio = _safe_ratio(max(raw_values), min(raw_values)) if raw_values else None
        matched_ratio = _safe_ratio(max(matched_values), min(matched_values)) if matched_values else None
        workload_ratio = _safe_ratio(max(workload_values), min(workload_values)) if workload_values else None
        print(
            "[info] "
            f"{proposer} | {resolver_count} | "
            f"{(min(raw_values) if raw_values else 'na')} | {(max(raw_values) if raw_values else 'na')} | "
            f"{(raw_ratio if raw_ratio is not None else 'raw resolver spread')} | "
            f"{(matched_ratio if matched_ratio is not None else 'matched-replicate spread')} | "
            f"{(workload_ratio if workload_ratio is not None else 'insufficient workload data')}"
        )

    print(
        "[info] counts: "
        f"timing rows={stats.get('n_timing_rows', 0)}, "
        f"canonical timing groups={stats.get('n_canonical_timing_groups', 0)}, "
        f"matched metrics rows={stats.get('n_matched_metrics_rows', 0)}, "
        f"unmatched timing groups={stats.get('n_unmatched_timing_groups', 0)}, "
        f"unmatched metrics rows={stats.get('n_unmatched_metrics_rows', 0)}"
    )


def run_time_diagnostics(
    *,
    canonical_rows_all: list[dict[str, object]],
    timing_groups_for_combo: list[dict[str, object]],
    joined_rows: list[dict[str, object]],
    output_dir: Path,
    unmatched_timing_groups_count: int,
    unmatched_metrics_rows_count: int,
) -> list[Path]:
    run_rows, group_rows, matched_rows, matched_summary_rows, stats = _build_proposal_resolver_diagnostics(
        canonical_rows_all=canonical_rows_all,
        timing_groups_for_combo=timing_groups_for_combo,
        joined_rows=joined_rows,
    )
    if not group_rows:
        return []

    saved: list[Path] = []
    runs_header = [
        "source_file",
        "scenario",
        "route",
        "protocol",
        "method",
        "inference_mode",
        "proposer",
        "resolver",
        "seed",
        "episode",
        "timing_protocol",
        "device",
        "host_name",
        "cpu_model",
        "torch_num_threads",
        "omp_num_threads",
        "mkl_num_threads",
        "warmup_decisions",
        "measured_decisions",
        "proposal_total_ms",
        "proposal_mean_ms",
        "proposal_p50_ms",
        "proposal_p90_ms",
        "proposal_p95_ms",
        "proposal_max_ms",
        "resolution_total_ms",
        "resolution_mean_ms",
        "decision_total_mean_ms",
    ]
    groups_header = [
        "scenario",
        "route",
        "protocol",
        "method",
        "inference_mode",
        "proposer",
        "resolver",
        "n_timing_runs",
        "n_measured_decisions",
        "seeds",
        "episodes",
        "source_files",
        "proposal_total_ms",
        "proposal_mean_pooled_ms",
        "proposal_mean_weighted_ms",
        "proposal_mean_unweighted_ms",
        "proposal_mean_std_ms",
        "proposal_mean_min_ms",
        "proposal_mean_max_ms",
        "resolution_mean_pooled_ms",
        "decision_total_mean_pooled_ms",
    ] + TIME_DIAG_WORKLOAD_FIELDS + [
        "candidate_entries_per_macro_step",
        "active_robot_decisions_per_macro_step",
        "proposal_ms_per_candidate_entry",
        "proposal_ms_per_active_robot_decision",
    ]
    matched_rows_header = [
        "scenario",
        "route",
        "protocol",
        "method",
        "inference_mode",
        "proposer",
        "resolver",
        "seed",
        "episode",
        "proposal_mean_ms",
        "measured_decisions",
        "source_file",
    ]
    matched_summary_header = [
        "scenario",
        "route",
        "protocol",
        "method",
        "inference_mode",
        "proposer",
        "resolver",
        "n_matched_replicates",
        "matched_proposal_mean_ms",
        "matched_proposal_std_ms",
        "matched_proposal_min_ms",
        "matched_proposal_max_ms",
    ]
    saved.append(_write_dict_rows_csv(output_dir / TIME_DIAG_RUNS_CSV, run_rows, runs_header))
    saved.append(_write_dict_rows_csv(output_dir / TIME_DIAG_GROUPS_CSV, group_rows, groups_header))
    saved.append(_write_dict_rows_csv(output_dir / TIME_DIAG_MATCHED_REPLICATES_CSV, matched_rows, matched_rows_header))
    saved.append(_write_dict_rows_csv(output_dir / TIME_DIAG_MATCHED_SUMMARY_CSV, matched_summary_rows, matched_summary_header))

    saved.append(_plot_proposal_latency_by_proposer_and_resolver(group_rows, output_dir))
    saved.append(_plot_proposal_workload_panels(group_rows, output_dir))
    saved.append(_plot_proposal_normalized_panels(group_rows, output_dir))
    saved.append(_plot_proposal_vs_workload_scatter(group_rows, output_dir))
    matched_plot = _plot_matched_replicates(matched_rows, matched_summary_rows, output_dir)
    if matched_plot is not None:
        saved.append(matched_plot)
    else:
        print("[warn] No matched replicate comparison was available for plotting", file=sys.stderr)
    saved.append(_plot_resolver_spread(group_rows, matched_summary_rows, output_dir))

    stats["n_unmatched_timing_groups"] = int(unmatched_timing_groups_count)
    stats["n_unmatched_metrics_rows"] = int(unmatched_metrics_rows_count)

    _print_time_diagnostics_summary(
        group_rows=group_rows,
        matched_summary_rows=matched_summary_rows,
        stats=stats,
    )
    return saved


def main() -> int:
    args = parse_args()
    if float(args.work_x_padding) < 0.0:
        raise ValueError("--work-x-padding must be non-negative")

    rows, available_metrics = load_rows(args.csv_path)
    metrics = select_metrics(args.metrics, available_metrics)
    plot_types = resolve_plot_types(args)

    rl_quality_rows, rl_timing_rows = load_rl_time_overlay(args)
    if rl_quality_rows:
        rows = rows + rl_quality_rows

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

    aggregated_timing_groups_all: list[dict[str, object]] = []
    canonical_timing_rows_all: list[dict[str, object]] = []
    if "time_cmp" in plot_types:
        metrics_rows_for_context = [
            row
            for row in rows
            if _scenario_of_row(row) in set(scenarios)
        ]
        metrics_rows_for_context = _filtered_rows(
            metrics_rows_for_context,
            exclude_resolvers=exclude_resolvers,
            exclude_policies=exclude_policies,
        )

        known_scenarios = sorted({_scenario_of_row(row) for row in metrics_rows_for_context})
        allowed_routes = {
            _route_alias_from_text(str(row.get("route_construction") or "nearest"))
            for row in metrics_rows_for_context
        }
        allowed_protocols = {
            _protocol_alias_of_row(row)
            for row in metrics_rows_for_context
        }

        timing_files = _find_timing_files(
            timing_dir=args.timing_dir,
            timing_files=args.timing_files,
            csv_parent=args.csv_path.parent,
        )
        timing_rows = _load_timing_rows(timing_files)
        if rl_timing_rows:
            timing_rows = timing_rows + rl_timing_rows
        canonical_timing_rows = _canonicalize_timing_rows(
            timing_rows,
            known_scenarios=set(known_scenarios),
            allowed_routes=allowed_routes,
            allowed_protocols=allowed_protocols,
        )
        _detect_duplicate_timing_rows(canonical_timing_rows)
        _ensure_one_to_one_row_assignment(canonical_timing_rows)
        canonical_timing_rows_all = canonical_timing_rows
        aggregated_timing_groups_all = _aggregate_timing_groups(canonical_timing_rows)
        if not aggregated_timing_groups_all:
            print("[warn] No usable timing groups remained after canonicalization and validation", file=sys.stderr)

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

        if "time_cmp" in plot_types:
            resolver_combos = sorted({_resolver_combo_of_row(row) for row in scenario_rows})
            for route_alias, protocol_alias in resolver_combos:
                combo_rows = [
                    row for row in scenario_rows
                    if _resolver_combo_of_row(row) == (route_alias, protocol_alias)
                ]
                if not combo_rows:
                    continue

                timing_groups_for_combo = [
                    group
                    for group in aggregated_timing_groups_all
                    if str(group.get("scenario")) == scenario
                    and str(group.get("route")) == route_alias
                    and str(group.get("protocol")) == protocol_alias
                ]

                joined_rows, missing_metric_keys, unmatched_timing_keys = _join_metrics_with_timing(
                    combo_rows,
                    timing_groups_for_combo,
                )

                for key in sorted(missing_metric_keys):
                    print(f"[warn] Missing timing match for metrics key: {key}", file=sys.stderr)
                for key in sorted(unmatched_timing_keys):
                    print(f"[warn] Timing group did not match metrics row: {key}", file=sys.stderr)

                if not joined_rows:
                    print(
                        f"[warn] No matched timing points for scenario='{scenario}' combo='{route_alias}_{protocol_alias}'; skipping time plots",
                        file=sys.stderr,
                    )
                    continue

                _check_timing_metadata(
                    timing_groups_for_combo,
                    strict=bool(args.strict_time_metadata),
                    context_label=f"scenario={scenario}, combo={route_alias}_{protocol_alias}",
                )

                time_output_dir = _time_combo_output_dir(output_dir, scenario, route_alias, protocol_alias)
                saved_paths.extend(
                    plot_time_cmp(
                        joined_rows,
                        metrics,
                        time_output_dir,
                        pareto=bool(args.pareto),
                        annotate_time=bool(args.annotate_time),
                        time_linear=bool(args.time_linear),
                    )
                )
                saved_paths.extend(plot_time_component_comparisons(joined_rows, time_output_dir))

                if args.time_phase_stat in {"mean", "both"}:
                    saved_paths.extend(plot_time_phase_breakdown_mean(joined_rows, time_output_dir))
                if args.time_phase_stat in {"total", "both"}:
                    saved_paths.extend(plot_time_phase_totals(joined_rows, time_output_dir))

                saved_paths.append(
                    _write_time_cmp_audit_csv(
                        output_path=time_output_dir / TIME_CMP_AUDIT_CSV,
                        joined_rows=joined_rows,
                        metrics=metrics,
                        source_metrics_file=args.csv_path,
                    )
                )
                saved_paths.append(
                    _write_time_phase_audit_csv(
                        time_output_dir / TIME_PHASE_AUDIT_CSV,
                        timing_groups_for_combo,
                    )
                )

                if "time_diag" in plot_types:
                    saved_paths.extend(
                        run_time_diagnostics(
                            canonical_rows_all=canonical_timing_rows_all,
                            timing_groups_for_combo=timing_groups_for_combo,
                            joined_rows=joined_rows,
                            output_dir=time_output_dir,
                            unmatched_timing_groups_count=len(unmatched_timing_keys),
                            unmatched_metrics_rows_count=len(missing_metric_keys),
                        )
                    )

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