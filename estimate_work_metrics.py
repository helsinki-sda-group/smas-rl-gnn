#!/usr/bin/env python
"""Estimate analytical work proxy columns for metrics_wide.csv.

Usage:
python aggregate_metrics_logs.py results_folder

python estimate_work_metrics.py \
  results_folder/metrics_wide.csv \
  --in-place

python plot_metrics_wide.py \
  results_folder/metrics_wide.csv \
  --work-cmp \
  --metrics rew \
  --scenario corridor_asymmetric \
  --pareto

These work values are an analytical operation-count proxy derived from
aggregated metrics, intended for quality-complexity comparisons rather than
runtime benchmarking.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

WORK_COLUMNS = [
    "work_model",
    "work_route_stops",
    "work_insertion_pairs",
    "work_candidate_scan",
    "work_active_proposals",
    "work_competition_factor",
    "work_proposer",
    "work_resolver",
    "work_total",
    "work_warning",
]

REQUIRED_BASE_COLUMNS = {"pol", "resolver"}
WORK_MODEL_NAME = "analytical_v1"

POLICY_ALIASES = {
    "closest": "pickup_distance",
    "pickup_distance": "pickup_distance",
    "pickup_deadline": "pickup_deadline",
    "pickup_deadline_distance": "pickup_deadline_distance",
    "pickupdistance": "pickup_distance",
    "pickupdeadline": "pickup_deadline",
    "pickupdeadlinedistance": "pickup_deadline_distance",
    "predicted_reward": "predicted_reward",
    "predicted_reward_joint": "predicted_reward_joint",
    "proposal_joint_competition": "proposal_joint_competition",
    "random": "random",
    "unique": "unique",
}

RESOLVER_ALIASES = {
    "capacity": "capacity",
    "closest_than_capacity": "closest_then_capacity",
    "closest_then_capacity": "closest_then_capacity",
    "hungarian": "hungarian",
    "predicted_reward": "predicted_reward",
    "predicted_reward_joint": "predicted_reward_joint",
    "random": "random",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append estimated computational work (analytical work proxy) columns to metrics_wide.csv."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to metrics_wide.csv")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (default: metrics_wide_work.csv)")
    parser.add_argument("--in-place", action="store_true", help="Rewrite the input CSV atomically")
    parser.add_argument("--joint-multiplier", type=float, default=1.5, help="Multiplier for joint predicted-reward variants")
    parser.add_argument("--route-stops", type=int, default=None, help="Override route_stops for all rows")
    parser.add_argument(
        "--default-num-robots",
        type=float,
        default=None,
        help="Fallback robot count for rows with missing num_robots",
    )
    parser.add_argument(
        "--unknown-mode",
        choices=["warn", "error"],
        default="warn",
        help="How to handle unknown policy/resolver names",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text


def parse_cell(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper() == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def insertion_pairs(route_stops: float) -> float:
    return ((route_stops + 1.0) * (route_stops + 2.0)) / 2.0


def row_identity(row: dict[str, str]) -> str:
    scenario = str(row.get("scenario", "")).strip() or "<unknown_scenario>"
    resolver = str(row.get("resolver", "")).strip() or "<unknown_resolver>"
    policy = str(row.get("pol", "")).strip() or "<unknown_policy>"
    source_file = str(row.get("source_file", "")).strip() or "<unknown_source_file>"
    return f"{scenario} / {resolver} / {policy} / {source_file}"


def canonical_policy_name(policy_name: str) -> str | None:
    normalized = normalize_name(policy_name)
    return POLICY_ALIASES.get(normalized)


def canonical_resolver_name(resolver_name: str) -> str | None:
    normalized = normalize_name(resolver_name)
    return RESOLVER_ALIASES.get(normalized)


def _parse_num_robots_from_instance(row: dict[str, str]) -> float | None:
    instance = str(row.get("instance", "") or "")
    match = re.search(r"taxis(\d+)", instance, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def proposer_work(
    *,
    policy_name: str,
    candidate_scan_work: float,
    insertion_pairs_value: float,
    competition_factor: float,
    joint_multiplier: float,
) -> float | None:
    policy = canonical_policy_name(policy_name)
    if policy is None:
        return None

    if policy in {"random", "unique", "pickup_distance", "pickup_deadline"}:
        return candidate_scan_work
    if policy == "pickup_deadline_distance":
        return 2.0 * candidate_scan_work
    if policy == "predicted_reward":
        return candidate_scan_work * insertion_pairs_value
    if policy == "predicted_reward_joint":
        return candidate_scan_work * insertion_pairs_value * joint_multiplier
    if policy == "proposal_joint_competition":
        return candidate_scan_work * insertion_pairs_value * competition_factor
    return None


def resolver_work(
    *,
    resolver_name: str,
    active_proposal_work: float,
    insertion_pairs_value: float,
    joint_multiplier: float,
    decisions_per_episode: float,
    num_robots: float,
    mean_candidates: float,
) -> float | None:
    resolver = canonical_resolver_name(resolver_name)
    if resolver is None:
        return None

    if resolver in {"random", "capacity"}:
        return active_proposal_work
    if resolver == "closest_then_capacity":
        return 2.0 * active_proposal_work
    if resolver == "predicted_reward":
        return active_proposal_work * insertion_pairs_value
    if resolver == "predicted_reward_joint":
        return active_proposal_work * insertion_pairs_value * joint_multiplier
    if resolver == "hungarian":
        estimated_tasks_per_step = max(num_robots * mean_candidates, 1.0)
        matrix_size = max(float(num_robots), estimated_tasks_per_step)
        matrix_scoring_work = decisions_per_episode * num_robots * estimated_tasks_per_step
        assignment_work = decisions_per_episode * (matrix_size ** 3)
        return matrix_scoring_work + assignment_work
    return None


def _write_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def _compute_route_stops(
    row: dict[str, str],
    *,
    route_stops_override: int | None,
) -> float:
    if route_stops_override is not None:
        return float(route_stops_override)

    max_capacity = parse_cell(row.get("max_robot_capacity"))
    if max_capacity is None:
        identity = row_identity(row)
        raise ValueError(
            "Missing max_robot_capacity and no --route-stops provided for row: "
            f"{identity}"
        )
    return 2.0 * max_capacity


def _ensure_required_columns(fieldnames: list[str], *, route_stops_override: int | None) -> None:
    missing = sorted(REQUIRED_BASE_COLUMNS.difference(fieldnames))
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}")
    if route_stops_override is None and "max_robot_capacity" not in fieldnames:
        raise ValueError(
            "Input CSV requires max_robot_capacity unless --route-stops is provided."
        )


def enrich_rows(
    rows: list[dict[str, str]],
    *,
    joint_multiplier: float,
    route_stops_override: int | None,
    default_num_robots: float | None,
    unknown_mode: str,
) -> tuple[list[dict[str, str]], int, int]:
    enriched_rows: list[dict[str, str]] = []
    enriched_count = 0
    skipped_unknown_count = 0

    warned_unknown_policies: set[str] = set()
    warned_unknown_resolvers: set[str] = set()

    for row in rows:
        warnings: list[str] = []
        optional_missing = False

        num_robots = parse_cell(row.get("num_robots"))
        if num_robots is None:
            num_robots = _parse_num_robots_from_instance(row)
            if num_robots is not None:
                optional_missing = True
                warnings.append("imputed_num_robots_from_instance")
        if num_robots is None and default_num_robots is not None:
            num_robots = float(default_num_robots)
            optional_missing = True
            warnings.append("imputed_num_robots_from_default")
        if num_robots is None:
            raise ValueError(
                "Missing required num_robots for row (provide --default-num-robots if needed): "
                f"{row_identity(row)}"
            )
        num_robots = max(num_robots, 0.0)

        mcand = parse_cell(row.get("mcand"))
        if mcand is None:
            raise ValueError(
                "Missing required mcand for row: "
                f"{row_identity(row)}"
            )

        msd = parse_cell(row.get("msd"))
        if msd is None:
            raise ValueError(
                "Missing required msd for row: "
                f"{row_identity(row)}"
            )

        noop = parse_cell(row.get("noop"))
        if noop is None:
            noop = 0.0
            optional_missing = True

        dstep = parse_cell(row.get("dstep"))
        if dstep is None:
            dstep = msd
            optional_missing = True

        ovrlap = parse_cell(row.get("ovrlap"))
        if ovrlap is None:
            ovrlap = 0.0
            optional_missing = True

        shared = parse_cell(row.get("shared"))
        if shared is None:
            shared = 0.0
            optional_missing = True

        cemn = parse_cell(row.get("cemn"))
        if cemn is None:
            cemn = 0.0
            optional_missing = True

        route_stops = _compute_route_stops(
            row,
            route_stops_override=route_stops_override,
        )
        insertion_pairs_value = insertion_pairs(route_stops)

        R = max(num_robots, 0.0)
        K = max(mcand, 0.0)
        M = max(msd, 0.0)
        D = max(dstep, 0.0)
        noop_fraction = clip(noop, 0.0, 1.0)

        candidate_scan_work = M * R * K
        active_proposal_work = M * R * (1.0 - noop_fraction)

        logged_competition = max(cemn, 0.0)
        fallback_competition = max(ovrlap, 0.0) * max(shared, 0.0)
        competition_factor = 1.0 + max(logged_competition, fallback_competition)

        policy_name = str(row.get("pol", "")).strip()
        resolver_name = str(row.get("resolver", "")).strip()
        policy_norm = normalize_name(policy_name)
        resolver_norm = normalize_name(resolver_name)

        proposer = proposer_work(
            policy_name=policy_name,
            candidate_scan_work=candidate_scan_work,
            insertion_pairs_value=insertion_pairs_value,
            competition_factor=competition_factor,
            joint_multiplier=joint_multiplier,
        )
        resolver = resolver_work(
            resolver_name=resolver_name,
            active_proposal_work=active_proposal_work,
            insertion_pairs_value=insertion_pairs_value,
            joint_multiplier=joint_multiplier,
            decisions_per_episode=D,
            num_robots=R,
            mean_candidates=K,
        )

        unknown_algorithm = False
        if proposer is None:
            unknown_algorithm = True
            warnings.append("unknown_policy")
            if policy_norm not in warned_unknown_policies:
                print(
                    f"[warn] Unknown policy name '{policy_name}' (normalized: {policy_norm})",
                    file=sys.stderr,
                )
                warned_unknown_policies.add(policy_norm)

        if resolver is None:
            unknown_algorithm = True
            warnings.append("unknown_resolver")
            if resolver_norm not in warned_unknown_resolvers:
                print(
                    f"[warn] Unknown resolver name '{resolver_name}' (normalized: {resolver_norm})",
                    file=sys.stderr,
                )
                warned_unknown_resolvers.add(resolver_norm)

        if unknown_algorithm and unknown_mode == "error":
            raise ValueError(
                "Unknown algorithm in row: "
                f"{row_identity(row)}"
            )

        if optional_missing:
            warnings.append("missing_optional_metrics")

        out_row = dict(row)
        if unknown_algorithm:
            skipped_unknown_count += 1
            out_row["work_model"] = ""
            out_row["work_route_stops"] = ""
            out_row["work_insertion_pairs"] = ""
            out_row["work_candidate_scan"] = ""
            out_row["work_active_proposals"] = ""
            out_row["work_competition_factor"] = ""
            out_row["work_proposer"] = ""
            out_row["work_resolver"] = ""
            out_row["work_total"] = ""
            out_row["work_warning"] = ";".join(dict.fromkeys(warnings))
            enriched_rows.append(out_row)
            continue

        enriched_count += 1
        total = float(proposer) + float(resolver)
        out_row["work_model"] = WORK_MODEL_NAME
        out_row["work_route_stops"] = _write_float(route_stops)
        out_row["work_insertion_pairs"] = _write_float(insertion_pairs_value)
        out_row["work_candidate_scan"] = _write_float(candidate_scan_work)
        out_row["work_active_proposals"] = _write_float(active_proposal_work)
        out_row["work_competition_factor"] = _write_float(competition_factor)
        out_row["work_proposer"] = _write_float(proposer)
        out_row["work_resolver"] = _write_float(resolver)
        out_row["work_total"] = _write_float(total)
        out_row["work_warning"] = ";".join(dict.fromkeys(warnings))
        enriched_rows.append(out_row)

    return enriched_rows, enriched_count, skipped_unknown_count


def _resolved_output_path(
    csv_path: Path,
    *,
    output: Path | None,
    in_place: bool,
) -> Path:
    if output is not None and in_place:
        raise ValueError("Cannot use --output together with --in-place")
    if in_place:
        return csv_path
    if output is not None:
        return output.expanduser().resolve()
    return csv_path.with_name("metrics_wide_work.csv")


def _prepare_fieldnames(existing_fieldnames: list[str]) -> list[str]:
    fieldnames = list(existing_fieldnames)
    for column in WORK_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    return fieldnames


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def _write_rows(path: Path, *, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_path = args.csv_path.expanduser().resolve()
    if input_path.is_dir():
        csv_path = input_path / "metrics_wide.csv"
    else:
        csv_path = input_path

    if not csv_path.exists():
        print(f"Input CSV does not exist: {csv_path}", file=sys.stderr)
        return 1
    if csv_path.is_dir():
        print(f"Input must be a CSV file, got directory: {csv_path}", file=sys.stderr)
        return 1

    try:
        output_path = _resolved_output_path(
            csv_path,
            output=args.output,
            in_place=args.in_place,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        existing_fieldnames, rows = _read_csv_rows(csv_path)
        _ensure_required_columns(existing_fieldnames, route_stops_override=args.route_stops)
        fieldnames = _prepare_fieldnames(existing_fieldnames)

        enriched_rows, enriched_count, skipped_unknown_count = enrich_rows(
            rows,
            joint_multiplier=float(args.joint_multiplier),
            route_stops_override=args.route_stops,
            default_num_robots=args.default_num_robots,
            unknown_mode=args.unknown_mode,
        )

        if args.in_place:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=str(csv_path.parent),
                prefix=f"{csv_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_handle:
                tmp_path = Path(tmp_handle.name)
            try:
                _write_rows(tmp_path, fieldnames=fieldnames, rows=enriched_rows)
                tmp_path.replace(csv_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
            written_path = csv_path
        else:
            _write_rows(output_path, fieldnames=fieldnames, rows=enriched_rows)
            written_path = output_path

        print(f"Enriched {enriched_count} rows")
        print(f"Skipped {skipped_unknown_count} rows with unknown algorithms")
        print(f"Wrote {written_path}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
