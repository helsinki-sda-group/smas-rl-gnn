#!/usr/bin/env python
"""Estimate reward and analytical computational work from evaluation_metrics.log.

This script:
1. Parses evaluation_metrics.log (seed-wise rows at different timesteps).
2. Aggregates metrics per timestep across seeds.
3. Selects the timestep with the highest mean reward.
4. Estimates computational work using the same base fields as baseline work estimation:
   mcand, msd, dstep, noop, ovrlap, shared (+ num_robots and max_robot_capacity).

By default, estimation targets inference-only actor/proposer cost for GNN proposer.
Use --training to include critic forward-pass cost as well.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from estimate_work_metrics import clip, insertion_pairs, normalize_name, resolver_work


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate reward/work at the best evaluation timestep (max mean reward)."
    )
    parser.add_argument(
        "log_path",
        type=Path,
        nargs="?",
        default=Path("evaluation_metrics.log"),
        help="Path to evaluation_metrics.log",
    )
    parser.add_argument(
        "--proposer",
        type=str,
        default="1hop",
        help="Proposer family (default: 1hop).",
    )
    parser.add_argument(
        "--resolver",
        type=str,
        default="closest",
        help="Resolver name or alias (e.g. closest, capacity, hungarian).",
    )
    parser.add_argument(
        "--num-robots",
        type=float,
        default=6.0,
        help="Robot count R used by baseline work formulas (default: 6).",
    )
    parser.add_argument(
        "--max-robot-capacity",
        type=float,
        default=2.0,
        help="Max robot capacity used to derive route_stops (default: 2).",
    )
    parser.add_argument(
        "--joint-multiplier",
        type=float,
        default=1.5,
        help="Joint multiplier used by resolver formulas (default: 1.5).",
    )
    parser.add_argument(
        "--noop-candidates",
        type=float,
        default=1.0,
        help="Extra noop candidate count for actor inference/proposal head (default: 1).",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="If set, include critic forward-pass work in proposer-side estimate.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for one-row summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top timesteps by mean reward to print (default: 5).",
    )
    return parser.parse_args()


def _to_float(text: str) -> float | None:
    value = str(text).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _split_columns_from_header(header_line: str) -> list[str]:
    columns: list[str] = []
    for segment in header_line.split("|"):
        parts = [part.strip().lower() for part in segment.strip().split() if part.strip()]
        columns.extend(parts)
    return columns


def parse_evaluation_log(log_path: Path) -> list[dict[str, Any]]:
    lines = [
        line.rstrip("\n")
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    header_idx = None
    header_columns: list[str] = []
    for idx, line in enumerate(lines):
        low = line.lower()
        if "|" in line and "pol" in low and "seed" in low and "rew" in low and "ts" in low:
            cols = _split_columns_from_header(line)
            if cols:
                header_idx = idx
                header_columns = cols
                break

    if header_idx is None:
        raise ValueError(f"Could not find header in {log_path}")

    rows: list[dict[str, Any]] = []
    for line in lines[header_idx + 1 :]:
        if line.lstrip().startswith("#"):
            continue
        parts: list[str] = []
        for segment in line.split("|"):
            parts.extend([part for part in segment.strip().split() if part])

        if len(parts) < len(header_columns):
            parts.extend([""] * (len(header_columns) - len(parts)))
        if len(parts) > len(header_columns):
            parts = parts[: len(header_columns)]

        row: dict[str, Any] = {}
        for col, raw in zip(header_columns, parts):
            value = raw.strip()
            if col in {"seed", "ts"}:
                try:
                    row[col] = int(value)
                except ValueError:
                    row[col] = None
                continue

            casted = _to_float(value)
            row[col] = casted if casted is not None else value

        if row.get("ts") is None:
            continue
        if _to_float(str(row.get("rew", ""))) is None:
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"No data rows parsed from {log_path}")

    return rows


def group_by_timestep(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    by_ts: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        ts = int(row["ts"])
        by_ts.setdefault(ts, []).append(row)

    metric_names = ["rew", "mcand", "msd", "dstep", "noop", "ovrlap", "shared"]

    grouped: list[dict[str, float]] = []
    for ts, ts_rows in by_ts.items():
        out: dict[str, float] = {"ts": float(ts), "n_seeds": float(len(ts_rows))}
        for metric in metric_names:
            values: list[float] = []
            for row in ts_rows:
                value = row.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            if not values:
                raise ValueError(f"Missing numeric metric '{metric}' for ts={ts}")
            mean_val = sum(values) / float(len(values))
            if len(values) > 1:
                var = sum((value - mean_val) ** 2 for value in values) / float(len(values))
                std_val = math.sqrt(var)
            else:
                std_val = 0.0
            out[f"{metric}_mean"] = mean_val
            out[f"{metric}_std"] = std_val
        grouped.append(out)

    grouped.sort(key=lambda row: row["ts"])
    return grouped


def _canonical_resolver_name(name: str) -> str:
    normalized = normalize_name(name)
    resolver_aliases = {
        "closest": "closest_then_capacity",
        "ctc": "closest_then_capacity",
        "closest_than_capacity": "closest_then_capacity",
    }
    return resolver_aliases.get(normalized, normalized)


def _proposer_mode(name: str) -> str:
    normalized = normalize_name(name)
    aliases = {
        "1hop": "1hop",
        "gnn_1hop": "1hop",
        "one_hop": "1hop",
        "2hop": "2hop",
        "gnn_2hop": "2hop",
        "two_hop": "2hop",
    }
    return aliases.get(normalized, normalized)


def proposer_actor_work(
    *,
    proposer: str,
    candidate_scan_work: float,
    insertion_pairs_value: float,
    competition_factor: float,
    joint_multiplier: float,
) -> float:
    mode = _proposer_mode(proposer)

    if mode == "1hop":
        return candidate_scan_work
    if mode == "2hop":
        # Proxy: approximately double neighborhood-processing compared with 1-hop.
        return 2.0 * candidate_scan_work

    # Fallback to baseline heuristic-like modes where names overlap.
    if mode in {"random", "unique", "pickup_distance", "pickup_deadline"}:
        return candidate_scan_work
    if mode == "pickup_deadline_distance":
        return 2.0 * candidate_scan_work
    if mode == "predicted_reward":
        return candidate_scan_work * insertion_pairs_value
    if mode == "predicted_reward_joint":
        return candidate_scan_work * insertion_pairs_value * joint_multiplier
    if mode in {"proposal_joint_competition", "predicted_reward_joint_competition"}:
        return candidate_scan_work * insertion_pairs_value * competition_factor

    raise ValueError(
        "Unsupported proposer mode: "
        f"{proposer}. Supported: 1hop, 2hop, and baseline-like aliases."
    )


def estimate_best_point(
    *,
    best_row: dict[str, float],
    proposer: str,
    resolver: str,
    num_robots: float,
    max_robot_capacity: float,
    joint_multiplier: float,
    noop_candidates: float,
    training: bool,
) -> dict[str, float | str]:
    R = max(float(num_robots), 0.0)
    M = max(float(best_row["msd_mean"]), 0.0)
    D = max(float(best_row["dstep_mean"]), 0.0)
    K = max(float(best_row["mcand_mean"]), 0.0)
    noop_fraction = clip(float(best_row["noop_mean"]), 0.0, 1.0)

    # In actor inference/training pass, include one explicit NOOP action head by default.
    K_actor = K + max(float(noop_candidates), 0.0)

    route_stops = 2.0 * max(float(max_robot_capacity), 0.0)
    insertion = insertion_pairs(route_stops)

    candidate_scan_actor = M * R * K_actor
    active_proposals = M * R * (1.0 - noop_fraction)

    ovrlap = max(float(best_row["ovrlap_mean"]), 0.0)
    shared = max(float(best_row["shared_mean"]), 0.0)
    competition_factor = 1.0 + (ovrlap * shared)

    actor_work = proposer_actor_work(
        proposer=proposer,
        candidate_scan_work=candidate_scan_actor,
        insertion_pairs_value=insertion,
        competition_factor=competition_factor,
        joint_multiplier=joint_multiplier,
    )
    critic_work = actor_work if training else 0.0
    proposer_total = actor_work + critic_work

    resolver_name = _canonical_resolver_name(resolver)
    resolver_estimate = resolver_work(
        resolver_name=resolver_name,
        active_proposal_work=active_proposals,
        insertion_pairs_value=insertion,
        joint_multiplier=joint_multiplier,
        decisions_per_episode=D,
        num_robots=R,
        mean_candidates=K,
    )
    if resolver_estimate is None:
        raise ValueError(
            "Unsupported resolver mode for baseline formula: "
            f"{resolver} (normalized: {resolver_name})"
        )

    total = proposer_total + float(resolver_estimate)

    return {
        "selected_ts": int(best_row["ts"]),
        "selected_n_seeds": int(best_row["n_seeds"]),
        "reward_mean": float(best_row["rew_mean"]),
        "reward_std": float(best_row["rew_std"]),
        "proposer": proposer,
        "resolver": resolver_name,
        "num_robots": R,
        "max_robot_capacity": max_robot_capacity,
        "joint_multiplier": joint_multiplier,
        "noop_candidates": noop_candidates,
        "training_mode": "true" if training else "false",
        "work_route_stops": route_stops,
        "work_insertion_pairs": insertion,
        "work_candidate_scan_actor": candidate_scan_actor,
        "work_active_proposals": active_proposals,
        "work_competition_factor": competition_factor,
        "work_actor": actor_work,
        "work_critic": critic_work,
        "work_proposer": proposer_total,
        "work_resolver": float(resolver_estimate),
        "work_total": total,
    }


def pick_best_timestep(grouped_rows: list[dict[str, float]]) -> dict[str, float]:
    # Max reward mean; tie-break by smaller timestep.
    return max(grouped_rows, key=lambda row: (row["rew_mean"], -row["ts"]))


def maybe_write_summary_csv(summary: dict[str, float | str], output_csv: Path | None) -> None:
    if output_csv is None:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def print_top_timesteps(grouped_rows: list[dict[str, float]], top_k: int) -> None:
    ranked = sorted(grouped_rows, key=lambda row: row["rew_mean"], reverse=True)
    limit = max(int(top_k), 1)
    print("Top timesteps by mean reward:")
    print("ts\tn_seeds\trew_mean\trew_std\tmcand\tmsd\tdstep\tnoop")
    for row in ranked[:limit]:
        print(
            f"{int(row['ts'])}\t{int(row['n_seeds'])}\t"
            f"{row['rew_mean']:.4f}\t{row['rew_std']:.4f}\t"
            f"{row['mcand_mean']:.4f}\t{row['msd_mean']:.4f}\t"
            f"{row['dstep_mean']:.4f}\t{row['noop_mean']:.4f}"
        )


def print_summary(summary: dict[str, float | str]) -> None:
    print("\nSelected evaluation point and estimated computational work:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


def main() -> int:
    args = parse_args()
    log_path = args.log_path.expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    rows = parse_evaluation_log(log_path)
    grouped = group_by_timestep(rows)
    best = pick_best_timestep(grouped)

    print_top_timesteps(grouped, args.top_k)

    summary = estimate_best_point(
        best_row=best,
        proposer=args.proposer,
        resolver=args.resolver,
        num_robots=float(args.num_robots),
        max_robot_capacity=float(args.max_robot_capacity),
        joint_multiplier=float(args.joint_multiplier),
        noop_candidates=float(args.noop_candidates),
        training=bool(args.training),
    )

    print_summary(summary)

    output_csv = args.output_csv.expanduser().resolve() if args.output_csv is not None else None
    maybe_write_summary_csv(summary, output_csv)
    if output_csv is not None:
        print(f"\nWrote summary CSV: {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
