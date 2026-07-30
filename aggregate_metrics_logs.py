#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "rew",
    "wait",
    "travel",
    "comp",
    "mwt",
    "mtt",
    "pkr",
    "cmr",
    "vcmr",
    "invdr",
    "opvr", # only pickup violations
    "odvr", # only dropoff violations
    "pdvr", # pickup and dropoff violations
    "shared",
    "noop",
    "mcand",
    "cne_fr",
    "cne_mn",
    "dstep",
    "msd",
    "ovrlap",
    "cemn",
    "ect",
    "unop",
    "ncpr",
    "psur",
    "ctot",
    "crat",
    "catx",
]

METRIC_ALIASES = {
    "travel": "trav",
    "trav": "trav",
    "ect": "ecr",
    "ecr": "ecr",
}

DERIVED_METRICS = {"opvr", "odvr", "pdvr"}
SUMMARY_SUFFIX = "±std"
ROUNDING_EPS = 1e-12


def _scenario_alias_from_instance(instance: str) -> str:
    name = str(instance or "").strip().lower()
    if "rand_dest" in name:
        return "randdest"
    if "corridor_wave" in name:
        return "corridor_wave"
    if "asymmetric" in name:
        return "corridor_asymmetric"
    if "mixed" in name:
        return "corridor_mixed"
    if "noisy" in name:
        return "corridor_noisy"
    if "hard" in name:
        return "corridor_hard"
    if "wave" in name:
        return "wave"
    return "unknown"


@dataclass(frozen=True)
class MetricStats:
    mean: float
    std: float | str | None


@dataclass
class ParsedLog:
    path: Path
    metadata: dict[str, str]
    per_seed_rows: list[dict[str, Any]]
    summary_by_policy: dict[str, dict[str, MetricStats]]

    @property
    def resolver(self) -> str:
        return self.metadata.get("resolver", "unknown")

    @property
    def instance(self) -> str:
        return self.metadata.get("instance", "unknown")

    @property
    def num_robots(self) -> int | str:
        instance = str(self.instance or "")
        match = re.search(r"taxis(\d+)", instance, flags=re.IGNORECASE)
        if not match:
            return ""
        return int(match.group(1))

    @property
    def max_robot_capacity(self) -> int | str:
        metadata_value = str(self.metadata.get("max_robot_capacity", "")).strip()
        if metadata_value:
            match = re.search(r"(\d+)", metadata_value)
            if match:
                return int(match.group(1))

        source_candidates = [self.instance, self.path.name]
        for candidate in source_candidates:
            match = re.search(r"cap(\d+)", str(candidate or ""), flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return ""

    @property
    def scenario(self) -> str:
        value = self.metadata.get("scenario", "")
        text = str(value).strip().lower()
        if text:
            return text
        return _scenario_alias_from_instance(self.instance)

    @property
    def route_construction(self) -> str:
        value = self.metadata.get("route_construction", "nearest")
        text = str(value).strip()
        return text if text else "nearest"

    @property
    def admission_aware(self) -> str:
        value = self.metadata.get("admission_aware", "false")
        text = str(value).strip().lower()
        return text if text in {"true", "false"} else "false"

    @property
    def protocol(self) -> str:
        value = self.metadata.get("protocol", "")
        text = str(value).strip().lower()
        if text in {"forced", "admission"}:
            return text
        return "admission" if self.admission_aware == "true" else "forced"

    def policies(self) -> list[str]:
        names = {row["pol"] for row in self.per_seed_rows if "pol" in row}
        names.update(self.summary_by_policy.keys())
        return sorted(names)


class IncompleteMetricsLogError(ValueError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate metrics*.log files into a wide CSV with one row per policy/resolver. "
            "Summary statistics are used when available; otherwise means/stds are computed from per-seed rows."
        )
    )
    parser.add_argument("folder", type=Path, help="Folder containing metrics*.log files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <folder>/metrics_wide.csv)",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Exclude *_std columns from the output",
    )
    parser.add_argument(
        "--add-metrics",
        nargs="+",
        default=[],
        help="Additional metric names to append to the default list",
    )
    parser.add_argument(
        "--route-construction",
        type=str,
        default="nearest",
        help="Default route_construction to use when metadata is missing/empty (default: nearest)",
    )
    return parser.parse_args()


def _find_header_index(lines: list[str], *, require_token: str | None = None, require_marker: str | None = None) -> int | None:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.lower().startswith("pol"):
            continue
        if require_token and require_token not in stripped.split():
            continue
        if require_marker and require_marker not in stripped:
            continue
        return index
    return None


def _parse_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        for part in line.split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def _to_number(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if "/" in token:
        return token
    try:
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            return int(token)
        return float(token)
    except ValueError:
        return token


def _parse_per_seed_rows(lines: list[str], header_index: int, stop_index: int | None) -> list[dict[str, Any]]:
    header_tokens = lines[header_index].replace("|", " ").split()
    end_index = stop_index if stop_index is not None else len(lines)
    rows: list[dict[str, Any]] = []

    for line in lines[header_index + 1:end_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        if "|" not in stripped:
            continue
        tokens = stripped.replace("|", " ").split()
        if len(tokens) != len(header_tokens):
            continue
        row = {key: _to_number(value) for key, value in zip(header_tokens, tokens)}
        if not isinstance(row.get("pol"), str):
            continue
        rows.append(row)

    return rows


def _parse_summary_rows(lines: list[str], header_index: int) -> dict[str, dict[str, MetricStats]]:
    header_tokens = lines[header_index].replace("|", " ").split()
    metric_names = [token[:-len(SUMMARY_SUFFIX)] if token.endswith(SUMMARY_SUFFIX) else token for token in header_tokens]
    summary_by_policy: dict[str, dict[str, MetricStats]] = {}

    for line in lines[header_index + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("="):
            continue
        if stripped.lower().startswith("pol"):
            continue
        if stripped.endswith(":"):
            break
        if "|" not in stripped:
            continue
        tokens = stripped.replace("|", " ").split()
        if len(tokens) != len(metric_names):
            continue

        policy = tokens[0]
        policy_stats: dict[str, MetricStats] = {}
        for metric_name, token in zip(metric_names[1:], tokens[1:]):
            if "±" not in token:
                continue
            mean_text, std_text = token.split("±", 1)
            policy_stats[metric_name] = MetricStats(mean=float(mean_text), std=float(std_text))
        if policy_stats:
            summary_by_policy[policy] = policy_stats

    return summary_by_policy


def parse_metrics_log(path: Path, *, default_route_construction: str = "nearest") -> ParsedLog:
    lines = path.read_text(encoding="utf-8").splitlines()
    per_seed_header_index = _find_header_index(lines, require_token="seed")
    if per_seed_header_index is None:
        raise ValueError(f"Could not find per-seed header in {path}")

    summary_header_index = _find_header_index(lines, require_marker=SUMMARY_SUFFIX)
    if summary_header_index is None:
        raise IncompleteMetricsLogError(f"Skipping incomplete metrics log without summary statistics: {path.name}")

    metadata = _parse_metadata(lines[:per_seed_header_index])
    if not str(metadata.get("route_construction", "")).strip():
        metadata["route_construction"] = str(default_route_construction)
    per_seed_rows = _parse_per_seed_rows(lines, per_seed_header_index, summary_header_index)
    summary_by_policy = _parse_summary_rows(lines, summary_header_index)

    return ParsedLog(
        path=path,
        metadata=metadata,
        per_seed_rows=per_seed_rows,
        summary_by_policy=summary_by_policy,
    )


def _population_stats(values: list[float]) -> MetricStats:
    if not values:
        raise ValueError("Cannot compute statistics for an empty value list")
    mean_value = statistics.fmean(values)
    std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
    return MetricStats(mean=mean_value, std=std_value)


def _normalize_metric_name(metric_name: str) -> str:
    return METRIC_ALIASES.get(metric_name, metric_name)


def _clip_small_negative(value: float) -> float:
    if value < 0.0 and abs(value) <= ROUNDING_EPS:
        return 0.0
    return value


def _compute_derived_metric(row: dict[str, Any], metric_name: str) -> float:
    pkr = float(row["pkr"])
    pkvr = float(row["pkvr"])
    invdr = float(row["invdr"])
    ddvr = float(row["ddvr"])
    pickup_violation_total = pkr * pkvr

    if metric_name == "opvr":
        return _clip_small_negative(invdr - ddvr)
    if metric_name == "pdvr":
        return _clip_small_negative(pickup_violation_total - invdr + ddvr)
    if metric_name == "odvr":
        return _clip_small_negative(invdr - pickup_violation_total)
    raise KeyError(metric_name)


def _aggregate_per_seed_metric(rows: list[dict[str, Any]], metric_name: str) -> MetricStats:
    values: list[float] = []
    for row in rows:
        if metric_name in DERIVED_METRICS:
            required = {"pkr", "pkvr", "invdr", "ddvr"}
            if not required.issubset(row):
                continue
            values.append(_compute_derived_metric(row, metric_name))
            continue

        source_metric = _normalize_metric_name(metric_name)
        value = row.get(source_metric)
        if isinstance(value, (int, float)):
            values.append(float(value))

    if not values:
        raise KeyError(metric_name)
    return _population_stats(values)


def _summary_metric(summary_row: dict[str, MetricStats], metric_name: str, include_std: bool) -> MetricStats:
    if metric_name in DERIVED_METRICS:
        raise KeyError(metric_name)
    source_metric = _normalize_metric_name(metric_name)
    stats = summary_row[source_metric]
    if include_std:
        return stats
    return MetricStats(mean=stats.mean, std=None)


def _derived_from_summary(summary_row: dict[str, MetricStats], metric_name: str) -> MetricStats:
    required = {"pkr", "pkvr", "invdr", "ddvr"}
    if not required.issubset(summary_row):
        raise KeyError(metric_name)

    pkr = summary_row["pkr"].mean
    pkvr = summary_row["pkvr"].mean
    invdr = summary_row["invdr"].mean
    ddvr = summary_row["ddvr"].mean
    pickup_violation_total = pkr * pkvr

    if metric_name == "opvr":
        mean = _clip_small_negative(invdr - ddvr)
    elif metric_name == "pdvr":
        mean = _clip_small_negative(pickup_violation_total - invdr + ddvr)
    elif metric_name == "odvr":
        mean = _clip_small_negative(invdr - pickup_violation_total)
    else:
        raise KeyError(metric_name)

    return MetricStats(mean=mean, std="NA")


def _policy_rows(parsed_log: ParsedLog, policy: str) -> list[dict[str, Any]]:
    return [row for row in parsed_log.per_seed_rows if row.get("pol") == policy]


def resolve_metric(parsed_log: ParsedLog, policy: str, metric_name: str, include_std: bool) -> MetricStats:
    summary_row = parsed_log.summary_by_policy.get(policy)
    if summary_row is not None:
        try:
            return _summary_metric(summary_row, metric_name, include_std)
        except KeyError:
            if metric_name in DERIVED_METRICS:
                try:
                    return _derived_from_summary(summary_row, metric_name)
                except KeyError:
                    pass

    rows = _policy_rows(parsed_log, policy)
    if rows:
        stats = _aggregate_per_seed_metric(rows, metric_name)
        if include_std:
            return stats
        return MetricStats(mean=stats.mean, std=None)

    raise KeyError(f"Metric '{metric_name}' is unavailable for policy '{policy}' in {parsed_log.path.name}")


def available_metrics(parsed_logs: list[ParsedLog]) -> list[str]:
    metrics = set(DERIVED_METRICS)
    metrics.update(METRIC_ALIASES.keys())
    metrics.update(METRIC_ALIASES.values())
    for parsed_log in parsed_logs:
        for row in parsed_log.per_seed_rows:
            for key, value in row.items():
                if key == "pol":
                    continue
                if isinstance(value, (int, float)):
                    metrics.add(key)
        for summary_row in parsed_log.summary_by_policy.values():
            metrics.update(summary_row.keys())
    return sorted(metrics)


def build_output_rows(parsed_logs: list[ParsedLog], metric_names: list[str], include_std: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parsed_log in parsed_logs:
        for policy in parsed_log.policies():
            out_row: dict[str, Any] = {
                "source_file": parsed_log.path.name,
                "scenario": parsed_log.scenario,
                "instance": parsed_log.instance,
                "num_robots": parsed_log.num_robots,
                "max_robot_capacity": parsed_log.max_robot_capacity,
                "protocol": parsed_log.protocol,
                "resolver": parsed_log.resolver,
                "route_construction": parsed_log.route_construction,
                "admission_aware": parsed_log.admission_aware,
                "pol": policy,
            }
            for metric_name in metric_names:
                try:
                    stats = resolve_metric(parsed_log, policy, metric_name, include_std)
                    out_row[metric_name] = stats.mean
                    if include_std:
                        out_row[f"{metric_name}_std"] = stats.std
                except KeyError:
                    out_row[metric_name] = ""
                    if include_std:
                        out_row[f"{metric_name}_std"] = ""
            rows.append(out_row)
    return rows


def _deduplicate_metric_names(metrics: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for metric in metrics:
        normalized = metric.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def main() -> int:
    args = _parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Input folder does not exist: {folder}", file=sys.stderr)
        return 1

    output_path = args.output.expanduser().resolve() if args.output else folder / "metrics_wide.csv"
    include_std = not args.no_std
    metric_names = _deduplicate_metric_names(DEFAULT_METRICS + list(args.add_metrics))

    log_paths = sorted(folder.glob("metrics*.log"))
    if not log_paths:
        print(f"No metrics*.log files found in {folder}", file=sys.stderr)
        return 1

    default_route_construction = str(getattr(args, "route_construction", "nearest") or "nearest").strip() or "nearest"
    skipped_paths: list[Path] = []
    parsed_logs = []
    for path in log_paths:
        try:
            parsed_logs.append(parse_metrics_log(path, default_route_construction=default_route_construction))
        except IncompleteMetricsLogError:
            skipped_paths.append(path)

    if skipped_paths:
        print(
            "Skipped incomplete metrics log(s) without summary statistics: "
            + ", ".join(path.name for path in skipped_paths),
            file=sys.stderr,
        )

    if not parsed_logs:
        print("No complete metrics logs with summary statistics found in the input folder.", file=sys.stderr)
        return 1

    known_metrics = set(available_metrics(parsed_logs))
    unknown_metrics = [metric for metric in metric_names if metric not in known_metrics]
    if unknown_metrics:
        print(
            "Unknown metric(s): "
            + ", ".join(unknown_metrics)
            + "\nAvailable metrics: "
            + ", ".join(sorted(known_metrics)),
            file=sys.stderr,
        )
        return 1

    rows = build_output_rows(parsed_logs, metric_names, include_std)
    fieldnames = [
        "source_file",
        "scenario",
        "instance",
        "num_robots",
        "max_robot_capacity",
        "protocol",
        "resolver",
        "route_construction",
        "admission_aware",
        "pol",
    ]
    for metric_name in metric_names:
        fieldnames.append(metric_name)
        if include_std:
            fieldnames.append(f"{metric_name}_std")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        formatted_rows = [
            {key: f"{value:.2f}" if isinstance(value, float) else value for key, value in row.items()}
            for row in rows
        ]
        writer.writerows(formatted_rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())