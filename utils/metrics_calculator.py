from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import json
import pandas as pd


@dataclass
class EpisodeMetrics:
    policy: str = ""
    seed: int = 0
    reward_sum: float = 0.0
    capacity_sum: float = 0.0
    step_sum: float = 0.0
    missed_deadline_sum: float = 0.0
    wait_sum: float = 0.0
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
            metrics.missed_deadline_sum = float(df_rewards["missed_deadline_avg"].sum())
            metrics.wait_sum = float(df_rewards["wait_avg"].sum())
            metrics.completion_sum = float(df_rewards["completion_avg"].sum())
            metrics.nonserved_sum = float(df_rewards["nonserved_avg"].sum())
        except Exception as e:
            print(f"Warning: Could not read rewards_macro.csv: {e}")
    else:
        metrics.reward_sum = episode_info.get("episode_reward", 0.0)

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
                    df_candidates.groupby("time")
                    .apply(lambda g: (g["_cand_count"] > 0).any())
                    .sum()
                )
                metrics.decision_steps = int(decision_steps)
            else:
                metrics.mean_candidates_per_taxi = 0.0
                metrics.cand_nonempty_frac = 0.0
                metrics.cand_mean_nonempty = 0.0
                metrics.decision_steps = 0
        except Exception as e:
            print(f"Warning: Could not compute mean_candidates_per_taxi: {e}")

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
    return (
        f"{metrics.policy:<10} {metrics.seed:>4} | "
        f"{metrics.reward_sum:>8.2f} {metrics.capacity_sum:>8.2f} {metrics.step_sum:>8.2f}"
        f" {metrics.missed_deadline_sum:>8.2f} {metrics.wait_sum:>8.2f}"
        f" {metrics.completion_sum:>8.2f} {metrics.nonserved_sum:>8.2f} | "
        f"{metrics.picked_up_tasks:>2}/{metrics.total_tasks:<2} {metrics.pickup_rate:>6.2f}"
        f" {metrics.obsolete_tasks:>2} {metrics.obsolete_rate:>6.2f}"
        f" {metrics.pickup_violated:>2}/{metrics.picked_up_tasks:<2} {metrics.pickup_violated_rate:>6.2f}"
        f" {metrics.mean_wait_time:>7.2f}"
        f" {metrics.completed_tasks:>2}/{metrics.total_tasks:<2} {metrics.completion_rate:>6.2f}"
        f" {metrics.assigned_never_picked:>2}/{metrics.total_tasks:<2} {metrics.assigned_never_picked_rate:>6.2f}"
        f" {metrics.mean_travel_time_completed:>7.2f}"
        f" {metrics.picked_not_completed:>2}/{metrics.picked_up_tasks:<2} {metrics.picked_not_completed_rate:>6.2f} | "
        f" {metrics.noop_fraction:>6.3f} {metrics.overload_assignment_fraction:>6.3f}"
        f" {metrics.mean_candidates_per_taxi:>6.2f} {metrics.cand_nonempty_frac:>6.3f}"
        f" {metrics.cand_mean_nonempty:>6.2f} {metrics.decision_steps:>6}"
        f" {metrics.macro_reward_mean:>8.3f} {metrics.macro_steps_done:>6}"
    )


def get_metrics_header() -> str:
    return (
        "pol        seed |      rew      cap     step      mdl     wait     comp      nsv |   pku    pkr obs  obsr   pkv   pkvr    mwt   cmp    cmr   anp   anpr     mtt   pnc    pncr |   noop  overld  mcand  cne_fr cne_mn   dstep    macmr    msd"
    )


def ensure_metrics_log(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(get_metrics_header() + "\n")


def append_metrics_log(path: str, metrics: EpisodeMetrics) -> None:
    ensure_metrics_log(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(metrics_to_string(metrics) + "\n")
