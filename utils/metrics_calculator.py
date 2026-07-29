from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import json
import xml.etree.ElementTree as ET
import pandas as pd


POLICY_WIDTH = 26
REWARD_PARAM_HEADER_KEYS = [
    "w_comp",
    "w_wait",
    "w_deadline",
    "w_travel",
    "wait_cap",
    "travel_cap",
    "deadline_cap",
]
REWARD_PARAM_DEFAULTS = {
    "w_comp": 1.0,
    "w_wait": 1.5,
    "w_deadline": 10.0,
    "w_travel": 2.0,
    "wait_cap": 600.0,
    "travel_cap": 90.0,
    "deadline_cap": 600.0,
}


def extract_route_files_from_sumocfg(sumo_cfg_path: str) -> str:
    """Return SUMO <route-files> value from sumocfg, or 'unknown' on failure."""
    candidate_paths: List[str] = [str(sumo_cfg_path)]
    cfg_basename = os.path.basename(str(sumo_cfg_path).replace("\\", "/"))
    if cfg_basename:
        candidate_paths.extend([
            cfg_basename,
            os.path.join("configs", cfg_basename),
        ])

    seen = set()
    for path in candidate_paths:
        norm = os.path.normpath(path)
        if norm in seen or not os.path.exists(norm):
            continue
        seen.add(norm)
        try:
            root = ET.parse(norm).getroot()
            route_files_node = root.find("./input/route-files")
            if route_files_node is not None:
                value = str(route_files_node.attrib.get("value", "")).strip()
                if value:
                    return value
        except Exception:
            continue
    return "unknown"


def build_metrics_metadata_lines(
    *,
    sumo_cfg_path: str,
    conflict_resolution: str,
    route_construction: Optional[str] = None,
    reward_params: Optional[Dict],
    completion_mode: Optional[str] = None,
    reassignment_mode: Optional[str] = None,
) -> List[str]:
    """Build standardized metadata header lines for metrics log files."""
    reward_params_dict = dict(reward_params or {})
    instance = extract_route_files_from_sumocfg(sumo_cfg_path)

    line_parts: List[str] = [
        f"instance={instance}",
        f"resolver={str(conflict_resolution)}",
    ]
    if route_construction is not None:
        line_parts.append(f"route_construction={str(route_construction)}")
    if completion_mode is not None:
        line_parts.append(f"completion_mode={str(completion_mode)}")
    if reassignment_mode is not None:
        line_parts.append(f"reassignment_mode={str(reassignment_mode)}")
    if "reward_type" in reward_params_dict:
        line_parts.append(f"reward_type={reward_params_dict.get('reward_type')}")
    line_parts.extend(
        f"{key}={reward_params_dict.get(key, REWARD_PARAM_DEFAULTS[key])}"
        for key in REWARD_PARAM_HEADER_KEYS
    )
    return [", ".join(line_parts)]


def _sum_int_column(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return int(vals.sum())


def _count_pipe_items(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len([part for part in text.split("|") if str(part).strip()])


def _warn_metrics_validation(episode_dir: str, message: str, counters: Dict[str, int]) -> None:
    print(f"Warning: [coordination] {message} in {episode_dir} counters={counters}")


@dataclass
class EpisodeMetrics:
    ts: int = 0  # timesteps (from model filename)
    overlap_rate: float = 0.0
    mean_shared_tasks_per_step: float = 0.0
    policy: str = ""
    seed: int = 0
    reward_sum: float = 0.0
    capacity_sum: float = 0.0
    step_sum: float = 0.0
    deadline_sum: float = 0.0
    wait_sum: float = 0.0
    travel_sum: float = 0.0
    completion_sum: float = 0.0
    nonserved_sum: float = 0.0

    total_tasks: int = 0
    picked_up_tasks: int = 0
    pickup_rate: float = 0.0

    obsolete_tasks: int = 0
    obsolete_rate: float = 0.0

    pickup_violated: int = 0
    pickup_violated_rate: float = 0.0

    mean_wait_time: float = 0.0
    completed_tasks: int = 0
    completion_rate: float = 0.0

    assigned_never_picked: int = 0
    assigned_never_picked_rate: float = 0.0

    mean_travel_time_completed: float = 0.0
    picked_not_completed: int = 0
    picked_not_completed_rate: float = 0.0

    noop_fraction: float = 0.0
    overload_assignment_fraction: float = 0.0
    mean_candidates_per_taxi: float = 0.0
    cand_nonempty_frac: float = 0.0
    cand_mean_nonempty: float = 0.0
    decision_steps: int = 0
    macro_reward_mean: float = 0.0
    macro_steps_done: int = 0

    # robot-level proposal/coordination counters (summed from coordination.csv)
    robot_decisions: int = 0
    empty_candidate_decisions: int = 0
    nonempty_candidate_decisions: int = 0
    unforced_noops: int = 0
    real_proposals: int = 0
    unique_proposals: int = 0
    conflicting_proposals: int = 0
    distinct_proposed_tasks: int = 0
    conflict_buckets: int = 0
    survived_proposals: int = 0
    rejected_proposals: int = 0
    final_assignments: int = 0
    off_proposal_assignments: int = 0

    # robot-level proposal/coordination rates
    empty_candidate_rate: float = 0.0
    unforced_noop_rate: float = 0.0
    nonconflicting_proposal_rate: float = 0.0
    proposal_survival_rate: float = 0.0
    off_proposal_assignment_rate: float = 0.0
    best_owner_margin: float = 0.0
    ego_best_owner_rate: float = 0.0
    competition_suppression_rate: float = 0.0
    competitors_evaluated_mean: float = 0.0
    competition_tie_rate: float = 0.0
    competition_single_owner_rate: float = 0.0

    # conflict-level metrics (from conflicts.csv + coordination.csv)
    conflicts_total: int = 0
    conflict_ratio: float = 0.0
    conflict_avg_taxis_per_conflict: float = 0.0

    # new: deadline / valid-completion breakdown
    dropoff_event_count: int = 0
    dropoff_event_rate: float = 0.0
    valid_completed_tasks: int = 0
    valid_completion_rate: float = 0.0
    invalid_dropoff_count: int = 0
    invalid_dropoff_rate: float = 0.0
    dropoff_deadline_violated: int = 0
    dropoff_deadline_violation_rate: float = 0.0


def compute_episode_metrics_from_logs(
    episode_dir: str,
    episode_info: Dict,
    policy: str,
    seed: int,
    num_robots: Optional[int] = None,
) -> EpisodeMetrics:
    """
    Compute comprehensive metrics from CSV log files.

    Args:
        episode_dir: path to episode directory containing task_lifecycle.csv, taxi_events.csv, rewards_macro.csv
        episode_info: info dict from environment (for reward_sum if available)
        policy: policy name
        seed: seed used for this run
        num_robots: number of robots/taxis (required for noop_fraction)
    """
    metrics = EpisodeMetrics(policy=policy, seed=seed)

    rewards_macro_path = os.path.join(episode_dir, "rewards_macro.csv")
    if os.path.exists(rewards_macro_path):
        try:
            df_rewards = pd.read_csv(rewards_macro_path)
            metrics.reward_sum = float(df_rewards["reward"].sum())
            metrics.capacity_sum = float(df_rewards["capacity_avg"].sum())
            metrics.step_sum = float(df_rewards["step_avg"].sum())
            if "deadline_avg" in df_rewards.columns:
                metrics.deadline_sum = float(df_rewards["deadline_avg"].sum())
            elif "missed_deadline_avg" in df_rewards.columns:
                metrics.deadline_sum = float(df_rewards["missed_deadline_avg"].sum())
            else:
                metrics.deadline_sum = 0.0

            metrics.wait_sum = float(df_rewards["wait_avg"].sum()) if "wait_avg" in df_rewards.columns else 0.0
            metrics.travel_sum = float(df_rewards["travel_avg"].sum()) if "travel_avg" in df_rewards.columns else 0.0
            metrics.completion_sum = float(df_rewards["completion_avg"].sum())
            metrics.nonserved_sum = float(df_rewards["nonserved_avg"].sum())
        except Exception as e:
            print(f"Warning: Could not read rewards_macro.csv: {e}")
    else:
        metrics.reward_sum = episode_info.get("episode_reward", 0.0)

    coordination_path = os.path.join(episode_dir, "coordination.csv")
    if os.path.exists(coordination_path):
        try:
            df_coord = pd.read_csv(coordination_path)
            if not df_coord.empty:
                metrics.robot_decisions = _sum_int_column(df_coord, "robot_decisions")
                metrics.empty_candidate_decisions = _sum_int_column(df_coord, "empty_candidate_decisions")
                metrics.nonempty_candidate_decisions = _sum_int_column(df_coord, "nonempty_candidate_decisions")
                metrics.unforced_noops = _sum_int_column(df_coord, "unforced_noops")
                metrics.real_proposals = _sum_int_column(df_coord, "real_proposals")
                metrics.unique_proposals = _sum_int_column(df_coord, "unique_proposals")
                metrics.conflicting_proposals = _sum_int_column(df_coord, "conflicting_proposals")
                metrics.distinct_proposed_tasks = _sum_int_column(df_coord, "distinct_proposed_tasks")
                metrics.conflict_buckets = _sum_int_column(df_coord, "conflict_buckets")
                metrics.survived_proposals = _sum_int_column(df_coord, "survived_proposals")
                metrics.rejected_proposals = _sum_int_column(df_coord, "rejected_proposals")
                metrics.final_assignments = _sum_int_column(df_coord, "final_assignments")
                metrics.off_proposal_assignments = _sum_int_column(df_coord, "off_proposal_assignments")

                n = metrics.robot_decisions
                e = metrics.empty_candidate_decisions
                c = metrics.nonempty_candidate_decisions
                u = metrics.unforced_noops
                p = metrics.real_proposals
                q1 = metrics.unique_proposals
                s = metrics.survived_proposals
                f = metrics.final_assignments
                o = metrics.off_proposal_assignments

                metrics.empty_candidate_rate = (e / n) if n > 0 else 0.0
                metrics.unforced_noop_rate = (u / c) if c > 0 else 0.0
                metrics.nonconflicting_proposal_rate = (q1 / p) if p > 0 else 0.0
                metrics.proposal_survival_rate = (s / p) if p > 0 else 0.0
                metrics.off_proposal_assignment_rate = (o / f) if f > 0 else 0.0

                if "best_owner_margin" in df_coord.columns:
                    vals = pd.to_numeric(df_coord["best_owner_margin"], errors="coerce").fillna(0.0)
                    metrics.best_owner_margin = float(vals.mean()) if len(vals) > 0 else 0.0
                if "ego_best_owner_rate" in df_coord.columns:
                    vals = pd.to_numeric(df_coord["ego_best_owner_rate"], errors="coerce").fillna(0.0)
                    metrics.ego_best_owner_rate = float(vals.mean()) if len(vals) > 0 else 0.0
                if "competition_suppression_rate" in df_coord.columns:
                    vals = pd.to_numeric(df_coord["competition_suppression_rate"], errors="coerce").fillna(0.0)
                    metrics.competition_suppression_rate = float(vals.mean()) if len(vals) > 0 else 0.0
                if "competitors_evaluated_mean" in df_coord.columns:
                    vals = pd.to_numeric(df_coord["competitors_evaluated_mean"], errors="coerce").fillna(0.0)
                    metrics.competitors_evaluated_mean = float(vals.mean()) if len(vals) > 0 else 0.0
                if "competition_tie_rate" in df_coord.columns:
                    vals = pd.to_numeric(df_coord["competition_tie_rate"], errors="coerce").fillna(0.0)
                    metrics.competition_tie_rate = float(vals.mean()) if len(vals) > 0 else 0.0
                if "competition_single_owner_rate" in df_coord.columns:
                    vals = pd.to_numeric(df_coord["competition_single_owner_rate"], errors="coerce").fillna(0.0)
                    metrics.competition_single_owner_rate = float(vals.mean()) if len(vals) > 0 else 0.0

                counters = {
                    "robot_decisions": int(metrics.robot_decisions),
                    "empty_candidate_decisions": int(metrics.empty_candidate_decisions),
                    "nonempty_candidate_decisions": int(metrics.nonempty_candidate_decisions),
                    "unforced_noops": int(metrics.unforced_noops),
                    "real_proposals": int(metrics.real_proposals),
                    "unique_proposals": int(metrics.unique_proposals),
                    "conflicting_proposals": int(metrics.conflicting_proposals),
                    "distinct_proposed_tasks": int(metrics.distinct_proposed_tasks),
                    "conflict_buckets": int(metrics.conflict_buckets),
                    "survived_proposals": int(metrics.survived_proposals),
                    "rejected_proposals": int(metrics.rejected_proposals),
                    "final_assignments": int(metrics.final_assignments),
                    "off_proposal_assignments": int(metrics.off_proposal_assignments),
                }

                if (metrics.empty_candidate_decisions + metrics.nonempty_candidate_decisions) != metrics.robot_decisions:
                    _warn_metrics_validation(
                        episode_dir,
                        "empty_candidate_decisions + nonempty_candidate_decisions != robot_decisions",
                        counters,
                    )
                if metrics.unforced_noops > metrics.nonempty_candidate_decisions:
                    _warn_metrics_validation(episode_dir, "unforced_noops > nonempty_candidate_decisions", counters)
                if (metrics.real_proposals + metrics.unforced_noops) != metrics.nonempty_candidate_decisions:
                    _warn_metrics_validation(
                        episode_dir,
                        "real_proposals + unforced_noops != nonempty_candidate_decisions",
                        counters,
                    )
                if (metrics.unique_proposals + metrics.conflicting_proposals) != metrics.real_proposals:
                    _warn_metrics_validation(
                        episode_dir,
                        "unique_proposals + conflicting_proposals != real_proposals",
                        counters,
                    )
                if metrics.survived_proposals > metrics.real_proposals:
                    _warn_metrics_validation(episode_dir, "survived_proposals > real_proposals", counters)
                if metrics.rejected_proposals != (metrics.real_proposals - metrics.survived_proposals):
                    _warn_metrics_validation(
                        episode_dir,
                        "rejected_proposals != real_proposals - survived_proposals",
                        counters,
                    )
                if metrics.robot_decisions > 0 and (
                    metrics.survived_proposals != (metrics.unique_proposals + metrics.conflict_buckets)
                ):
                    _warn_metrics_validation(
                        episode_dir,
                        "survived_proposals != unique_proposals + conflict_buckets",
                        counters,
                    )
                if metrics.off_proposal_assignments > 0:
                    _warn_metrics_validation(
                        episode_dir,
                        "off_proposal_assignments is nonzero (diagnostic invariant)",
                        counters,
                    )
        except Exception as e:
            print(f"Warning: Could not read coordination.csv: {e}")

    conflicts_path = os.path.join(episode_dir, "conflicts.csv")
    if os.path.exists(conflicts_path):
        try:
            df_conflicts = pd.read_csv(conflicts_path)
            if not df_conflicts.empty:
                metrics.conflicts_total = int(len(df_conflicts))
                if "taxi_candidates" in df_conflicts.columns:
                    taxi_counts = df_conflicts["taxi_candidates"].apply(_count_pipe_items).astype(float)
                    if len(taxi_counts) > 0:
                        metrics.conflict_avg_taxis_per_conflict = float(taxi_counts.mean())
        except Exception as e:
            print(f"Warning: Could not read conflicts.csv: {e}")

    # Keep definition aligned with readme/05_conflicts_log.md: conflicts_total / tasks_total.
    # Here tasks_total is reconstructed from coordination logs via distinct_proposed_tasks.
    if metrics.distinct_proposed_tasks > 0:
        metrics.conflict_ratio = float(metrics.conflicts_total) / float(metrics.distinct_proposed_tasks)
    else:
        metrics.conflict_ratio = 0.0

    task_lifecycle_path = os.path.join(episode_dir, "task_lifecycle.csv")
    if not os.path.exists(task_lifecycle_path):
        return metrics

    try:
        df_lifecycle = pd.read_csv(task_lifecycle_path)
    except Exception as e:
        print(f"Warning: Could not read task_lifecycle.csv: {e}")
        return metrics

    if df_lifecycle.empty:
        return metrics

    df_lifecycle["actual_pickup_time"] = pd.to_numeric(df_lifecycle["actual_pickup_time"], errors="coerce")
    df_lifecycle["actual_dropoff_time"] = pd.to_numeric(df_lifecycle["actual_dropoff_time"], errors="coerce")
    df_lifecycle["reservation_time"] = pd.to_numeric(df_lifecycle["reservation_time"], errors="coerce")
    df_lifecycle["pickup_deadline"] = pd.to_numeric(df_lifecycle["pickup_deadline"], errors="coerce")
    df_lifecycle["actual_waiting_time"] = pd.to_numeric(df_lifecycle["actual_waiting_time"], errors="coerce")
    df_lifecycle["actual_travel_time"] = pd.to_numeric(df_lifecycle["actual_travel_time"], errors="coerce")
    df_lifecycle["was_obsolete"] = df_lifecycle["was_obsolete"].astype(str).str.lower() == "true"
    df_lifecycle["assigned_taxi"] = df_lifecycle["assigned_taxi"].fillna("")

    # Remove duplicate rows (by task_id) to avoid counting same task multiple times
    # Keep the last occurrence in case task was logged multiple times with updates
    df_lifecycle = df_lifecycle.drop_duplicates(subset=["task_id"], keep="last")

    total_tasks = len(df_lifecycle)
    metrics.total_tasks = total_tasks

    picked_up_mask = df_lifecycle["actual_pickup_time"].notna()
    picked_up = picked_up_mask.sum()
    metrics.picked_up_tasks = int(picked_up)
    metrics.pickup_rate = picked_up / total_tasks if total_tasks > 0 else 0.0

    obsolete = df_lifecycle["was_obsolete"].sum()
    metrics.obsolete_tasks = int(obsolete)
    metrics.obsolete_rate = obsolete / total_tasks if total_tasks > 0 else 0.0

    if picked_up > 0:
        picked_df = df_lifecycle[picked_up_mask].copy()
        violated = (picked_df["actual_pickup_time"] > picked_df["pickup_deadline"]).sum()
        metrics.pickup_violated = int(violated)
        metrics.pickup_violated_rate = violated / picked_up
    else:
        metrics.pickup_violated = 0
        metrics.pickup_violated_rate = 0.0

    if picked_up > 0:
        wait_times = df_lifecycle[picked_up_mask]["actual_waiting_time"].dropna()
        metrics.mean_wait_time = float(wait_times.mean()) if len(wait_times) > 0 else 0.0
    else:
        metrics.mean_wait_time = 0.0

    completed_mask = df_lifecycle["actual_dropoff_time"].notna()
    completed = completed_mask.sum()
    metrics.completed_tasks = int(completed)
    metrics.completion_rate = completed / total_tasks if total_tasks > 0 else 0.0

    if completed > 0:
        travel_times = df_lifecycle[completed_mask]["actual_travel_time"].dropna()
        metrics.mean_travel_time_completed = float(travel_times.mean()) if len(travel_times) > 0 else 0.0
    else:
        metrics.mean_travel_time_completed = 0.0

    assigned_mask = df_lifecycle["assigned_taxi"].str.len() > 0
    assigned_but_not_picked = (assigned_mask & ~picked_up_mask).sum()
    metrics.assigned_never_picked = int(assigned_but_not_picked)
    metrics.assigned_never_picked_rate = assigned_but_not_picked / total_tasks if total_tasks > 0 else 0.0

    picked_not_completed = (picked_up_mask & ~completed_mask).sum()
    metrics.picked_not_completed = int(picked_not_completed)
    metrics.picked_not_completed_rate = picked_not_completed / picked_up if picked_up > 0 else 0.0

    # valid-completion and deadline breakdown
    has_pdl = "pickup_deadline" in df_lifecycle.columns
    has_ddl = "dropoff_deadline" in df_lifecycle.columns
    if has_ddl:
        df_lifecycle["dropoff_deadline"] = pd.to_numeric(df_lifecycle["dropoff_deadline"], errors="coerce")
    if has_pdl:
        pass  # already coerced above

    dropoff_ev = int(completed_mask.sum())
    metrics.dropoff_event_count = dropoff_ev
    metrics.dropoff_event_rate = dropoff_ev / total_tasks if total_tasks > 0 else 0.0

    if dropoff_ev > 0 and has_pdl and has_ddl:
        dropped_df = df_lifecycle[completed_mask].copy()
        pickup_ok = (
            dropped_df["actual_pickup_time"].notna() &
            dropped_df["pickup_deadline"].notna() &
            (dropped_df["actual_pickup_time"] <= dropped_df["pickup_deadline"])
        )
        dropoff_ok = (
            dropped_df["actual_dropoff_time"].notna() &
            dropped_df["dropoff_deadline"].notna() &
            (dropped_df["actual_dropoff_time"] <= dropped_df["dropoff_deadline"])
        )
        valid_mask = pickup_ok & dropoff_ok
        valid_count = int(valid_mask.sum())
        metrics.valid_completed_tasks = valid_count
        metrics.valid_completion_rate = valid_count / total_tasks if total_tasks > 0 else 0.0
        metrics.invalid_dropoff_count = dropoff_ev - valid_count
        metrics.invalid_dropoff_rate = metrics.invalid_dropoff_count / total_tasks if total_tasks > 0 else 0.0

        ddl_violated = int(
            (dropped_df["actual_dropoff_time"].notna() &
             dropped_df["dropoff_deadline"].notna() &
             (dropped_df["actual_dropoff_time"] > dropped_df["dropoff_deadline"])).sum()
        )
        metrics.dropoff_deadline_violated = ddl_violated
        metrics.dropoff_deadline_violation_rate = ddl_violated / total_tasks if total_tasks > 0 else 0.0
    else:
        metrics.valid_completed_tasks = 0
        metrics.valid_completion_rate = 0.0
        metrics.invalid_dropoff_count = 0
        metrics.invalid_dropoff_rate = 0.0
        metrics.dropoff_deadline_violated = 0
        metrics.dropoff_deadline_violation_rate = 0.0

    debug_path = os.path.join(episode_dir, "debug.csv")
    noop_count = 0
    total_steps = 0
    if os.path.exists(debug_path) and num_robots:
        try:
            df_debug = pd.read_csv(debug_path)
            apply_input = df_debug[df_debug["tag"] == "apply-input"]
            if len(apply_input) > 0:
                total_steps = len(apply_input)
                for _, row in apply_input.iterrows():
                    try:
                        data = json.loads(row["data"])
                        assignments = data.get("assignments_raw", [])
                        noop_count += sum(1 for a in assignments if a is None)
                    except Exception:
                        pass
            metrics.noop_fraction = noop_count / (total_steps * num_robots) if total_steps > 0 else 0.0
        except Exception as e:
            print(f"Warning: Could not compute noop_fraction from debug.csv: {e}")

    overload_count = 0
    if os.path.exists(debug_path):
        try:
            df_debug = pd.read_csv(debug_path)
            apply_input_rows = df_debug[df_debug["tag"] == "apply-input"].values
            apply_winners_rows = df_debug[df_debug["tag"] == "apply-winners"].values

            for inp_row, winner_row in zip(apply_input_rows, apply_winners_rows):
                try:
                    inp_data = json.loads(inp_row[2])
                    winner_data = json.loads(winner_row[2])

                    winners = winner_data.get("winners", {})
                    cand_counts = inp_data.get("cand_counts", [])

                    if winners and cand_counts:
                        overload_count += 1
                except Exception:
                    pass

            metrics.overload_assignment_fraction = overload_count / total_steps if total_steps > 0 else 0.0
        except Exception as e:
            print(f"Warning: Could not compute overload_assignment_fraction: {e}")

    candidates_path = os.path.join(episode_dir, "candidates.csv")
    if os.path.exists(candidates_path):
        try:
            df_candidates = pd.read_csv(candidates_path)
            if len(df_candidates) > 0:
                candidate_counts = df_candidates["cand_res_ids"].apply(
                    lambda x: len(str(x).split("|")) if pd.notna(x) and str(x).strip() else 0
                )
                metrics.mean_candidates_per_taxi = float(candidate_counts.mean()) if len(candidate_counts) > 0 else 0.0

                nonempty_mask = candidate_counts > 0
                metrics.cand_nonempty_frac = float(nonempty_mask.sum() / len(candidate_counts)) if len(candidate_counts) > 0 else 0.0
                nonempty_counts = candidate_counts[nonempty_mask]
                metrics.cand_mean_nonempty = float(nonempty_counts.mean()) if len(nonempty_counts) > 0 else 0.0

                df_candidates = df_candidates.assign(_cand_count=candidate_counts)
                decision_steps = (
                    df_candidates.groupby("time", group_keys=False)
                    .apply(lambda g: (g["_cand_count"] > 0).any(), include_groups=False)
                    .sum()
                )
                metrics.decision_steps = int(decision_steps)

                # --- Compute overlap_rate and mean_shared_tasks_per_step ---
                overlap_steps = 0
                shared_tasks_counts = []
                grouped = df_candidates.groupby("time")
                for _, group in grouped:
                    # Build task-to-taxi mapping
                    task_to_taxis = {}
                    taxi_col = "taxi_id" if "taxi_id" in group.columns else "taxi"
                    for taxi_id, cand_str in zip(group[taxi_col], group["cand_res_ids"]):
                        if pd.isna(cand_str) or not str(cand_str).strip():
                            continue
                        for task_id in str(cand_str).split("|"):
                            if task_id not in task_to_taxis:
                                task_to_taxis[task_id] = set()
                            task_to_taxis[task_id].add(taxi_id)
                    shared_tasks = [t for t, taxis in task_to_taxis.items() if len(taxis) >= 2]
                    if shared_tasks:
                        overlap_steps += 1
                    shared_tasks_counts.append(len(shared_tasks))
                n_steps = len(shared_tasks_counts)
                metrics.overlap_rate = overlap_steps / n_steps if n_steps > 0 else 0.0
                metrics.mean_shared_tasks_per_step = float(sum(shared_tasks_counts)) / n_steps if n_steps > 0 else 0.0
            else:
                metrics.mean_candidates_per_taxi = 0.0
                metrics.cand_nonempty_frac = 0.0
                metrics.cand_mean_nonempty = 0.0
                metrics.decision_steps = 0
                metrics.overlap_rate = 0.0
                metrics.mean_shared_tasks_per_step = 0.0
        except Exception as e:
            print(f"Warning: Could not compute mean_candidates_per_taxi or overlap metrics: {e}")

    if os.path.exists(rewards_macro_path):
        try:
            df_rewards = pd.read_csv(rewards_macro_path)
            if len(df_rewards) > 0:
                metrics.macro_reward_mean = float(df_rewards["reward"].mean())
                metrics.macro_steps_done = len(df_rewards)
        except Exception as e:
            print(f"Warning: Could not compute macro_reward_mean and macro_steps_done: {e}")

    return metrics


def metrics_to_string(metrics: EpisodeMetrics) -> str:
    identity_block = f"{metrics.policy:<{POLICY_WIDTH}} {metrics.seed:>4} {metrics.ts:>8}"
    reward_block = (
        f" {metrics.reward_sum:>8.2f} {metrics.capacity_sum:>8.2f} {metrics.step_sum:>8.2f}"
        f" {metrics.deadline_sum:>8.2f} {metrics.wait_sum:>8.2f} {metrics.travel_sum:>8.2f}"
        f" {metrics.completion_sum:>8.2f} {metrics.nonserved_sum:>8.2f}"
    )
    picked_total = f"{metrics.picked_up_tasks:>2}/{metrics.total_tasks:<2}"
    violated_picked = f"{metrics.pickup_violated:>2}/{metrics.picked_up_tasks:<2}"
    completed_total = f"{metrics.completed_tasks:>2}/{metrics.total_tasks:<2}"
    assigned_total = f"{metrics.assigned_never_picked:>2}/{metrics.total_tasks:<2}"
    picked_not_completed = f"{metrics.picked_not_completed:>2}/{metrics.picked_up_tasks:<2}"
    ridepool_block = (
        f"{picked_total:>5} {metrics.pickup_rate:>6.2f}"
        f" {metrics.obsolete_tasks:>2} {metrics.obsolete_rate:>6.2f}"
        f" {violated_picked:>5} {metrics.pickup_violated_rate:>6.2f}"
        f" {metrics.mean_wait_time:>7.2f}"
        f" {completed_total:>5} {metrics.completion_rate:>6.2f}"
        f" {assigned_total:>5} {metrics.assigned_never_picked_rate:>6.2f}"
        f" {metrics.mean_travel_time_completed:>7.2f}"
        f" {picked_not_completed:>5} {metrics.picked_not_completed_rate:>6.2f}"
    )
    candidate_block = (
        f" {metrics.noop_fraction:>6.3f} {metrics.overload_assignment_fraction:>6.3f}"
        f" {metrics.mean_candidates_per_taxi:>6.2f} {metrics.cand_nonempty_frac:>6.3f}"
        f" {metrics.cand_mean_nonempty:>6.2f} {metrics.decision_steps:>6}"
        f" {metrics.macro_reward_mean:>8.3f} {metrics.macro_steps_done:>6}"
        f" {metrics.overlap_rate:>8.3f} {metrics.mean_shared_tasks_per_step:>8.3f}"
    )
    coordination_block = (
        f" {metrics.empty_candidate_rate:>6.3f} {metrics.unforced_noop_rate:>6.3f}"
        f" {metrics.nonconflicting_proposal_rate:>6.3f} {metrics.proposal_survival_rate:>6.3f}"
        f" {metrics.off_proposal_assignment_rate:>6.3f}"
        f" {metrics.best_owner_margin:>6.3f} {metrics.ego_best_owner_rate:>6.3f}"
        f" {metrics.competition_suppression_rate:>6.3f} {metrics.competitors_evaluated_mean:>6.3f}"
        f" {metrics.competition_tie_rate:>6.3f} {metrics.competition_single_owner_rate:>6.3f}"
    )
    conflict_block = (
        f" {metrics.conflicts_total:>6d} {metrics.conflict_ratio:>6.3f}"
        f" {metrics.conflict_avg_taxis_per_conflict:>6.3f}"
    )
    return (
        f"{identity_block} |{reward_block} | {ridepool_block} |{candidate_block} |{coordination_block} |{conflict_block}"
    )


def get_metrics_header() -> str:
    identity_header = f"{'pol':<{POLICY_WIDTH}} {'seed':>4} {'ts':>8}"
    reward_header = (
        f" {'rew':>8} {'cap':>8} {'step':>8} {'dln':>8} {'wait':>8} {'trav':>8} {'comp':>8} {'nsv':>8}"
    )
    ridepool_header = (
        f"{'pku':>5} {'pkr':>6} {'obs':>2} {'obsr':>6} {'pkv':>5} {'pkvr':>6}"
        f" {'mwt':>7} {'cmp':>5} {'cmr':>6} {'anp':>5} {'anpr':>6} {'mtt':>7} {'pnc':>5} {'pncr':>6}"
    )
    candidate_header = (
        f" {'noop':>6} {'overld':>6} {'mcand':>6} {'cne_fr':>6} {'cne_mn':>6} {'dstep':>6}"
        f" {'macmr':>8} {'msd':>6} {'ovrlap':>8} {'shared':>8}"
    )
    coordination_header = (
        f" {'ecr':>6} {'unop':>6} {'ncpr':>6} {'psur':>6} {'offpr':>6}"
        f" {'bomr':>6} {'ebor':>6} {'cssr':>6} {'cemn':>6} {'ctrr':>6} {'csor':>6}"
    )
    conflict_header = f" {'ctot':>6} {'crat':>6} {'catx':>6}"
    return f"{identity_header} |{reward_header} |{ridepool_header} |{candidate_header} |{coordination_header} |{conflict_header}"


def ensure_metrics_log(path: str, overwrite: bool = False, metadata_lines: Optional[List[str]] = None) -> None:
    if overwrite or not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            for line in list(metadata_lines or []):
                f.write(line.rstrip("\n") + "\n")
            f.write(get_metrics_header() + "\n")


def append_metrics_log(path: str, metrics: EpisodeMetrics) -> None:
    ensure_metrics_log(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(metrics_to_string(metrics) + "\n")


def compute_metrics_summary(metrics_list: List[EpisodeMetrics]) -> tuple[EpisodeMetrics, EpisodeMetrics]:
    """
    Compute mean and std of all numeric fields across a list of metrics.
    
    Returns:
        tuple of (mean_metrics, std_metrics)
    """
    import numpy as np
    from dataclasses import fields
    
    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")
    
    # Get all numeric fields (exclude policy and seed)
    numeric_fields = [f.name for f in fields(EpisodeMetrics) 
                     if f.name not in ['policy', 'seed']]
    
    # Compute mean and std for each field
    mean_values = {}
    std_values = {}
    
    for field_name in numeric_fields:
        values = [getattr(m, field_name) for m in metrics_list]
        mean_values[field_name] = float(np.mean(values))
        std_values[field_name] = float(np.std(values))
    
    # Create metrics objects with special policy labels
    mean_metrics = EpisodeMetrics(policy="MEAN", seed=0, **mean_values)
    std_metrics = EpisodeMetrics(policy="STD", seed=0, **std_values)
    
    return mean_metrics, std_metrics


def append_metrics_summary(path: str, metrics_list: List[EpisodeMetrics]) -> None:
    """Compute and append mean/std summary rows to metrics log."""
    mean_metrics, std_metrics = compute_metrics_summary(metrics_list)
    with open(path, "a", encoding="utf-8") as f:
        f.write(metrics_to_string(mean_metrics) + "\n")
        f.write(metrics_to_string(std_metrics) + "\n")
