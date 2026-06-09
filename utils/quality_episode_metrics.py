"""
Episode-level quality diagnostics for ride-pooling RL experiments.

Main entry point: compute_quality_episode_metrics()
All metrics are written to quality_episode_metrics.csv in run_dir.
Optional per-task / per-decision event rows can also be collected.

Enabled by logging.extended_quality_metrics: true in the run config.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Optional

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dist_stats(values: list, prefix: str) -> dict:
    """Return mean/std/p50/p90/p95/count stats for a list of floats."""
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_count": 0,
        }
    arr = np.array(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_count": len(arr),
    }


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def _read_csv_safe(path: str, required_cols: Optional[list] = None) -> "Optional[pd.DataFrame]":
    """Read a CSV from path, returning None on any error."""
    if not _HAS_PANDAS:
        return None
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        if required_cols:
            missing = set(required_cols) - set(df.columns)
            if missing:
                warnings.warn(f"quality_episode_metrics: {path} missing columns {missing}")
                return None
        return df
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"quality_episode_metrics: failed to read {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# section computations
# ---------------------------------------------------------------------------

def _compute_reward_section(context: dict) -> dict:
    """Section 1: aggregate reward stats (from rew_accum in context)."""
    ra = context.get("rew_accum", {})
    row: dict = {}

    def _get(key: str, default: float = 0.0) -> float:
        return float(ra.get(key, default))

    # completion (mode-dependent: pickup/dropoff/valid_dropoff)
    row["rew_completion_sum"] = _get("completion_event_sum")
    row["rew_completion_count"] = int(_get("completion_event_count"))
    row["rew_completion_mean"] = _safe_ratio(row["rew_completion_sum"], row["rew_completion_count"])

    # dropoff event tracking (all modes — diagnostic breakdown)
    row["rew_dropoff_event_sum"] = _get("dropoff_event_sum")
    row["rew_dropoff_event_count"] = int(_get("dropoff_event_count"))
    row["rew_dropoff_event_mean"] = _safe_ratio(row["rew_dropoff_event_sum"], row["rew_dropoff_event_count"])

    row["rew_valid_dropoff_sum"] = _get("valid_dropoff_sum")
    row["rew_valid_dropoff_count"] = int(_get("valid_dropoff_count"))
    row["rew_valid_dropoff_mean"] = _safe_ratio(row["rew_valid_dropoff_sum"], row["rew_valid_dropoff_count"])

    row["rew_invalid_dropoff_sum"] = _get("invalid_dropoff_sum")
    row["rew_invalid_dropoff_count"] = int(_get("invalid_dropoff_count"))
    row["rew_invalid_dropoff_mean"] = _safe_ratio(row["rew_invalid_dropoff_sum"], row["rew_invalid_dropoff_count"])

    # wait
    row["rew_wait_event_pickup_sum"] = _get("wait_event_pickup_sum")
    row["rew_wait_event_pickup_count"] = int(_get("wait_event_pickup_count"))
    row["rew_wait_event_pickup_mean"] = _safe_ratio(row["rew_wait_event_pickup_sum"], row["rew_wait_event_pickup_count"])

    row["rew_wait_obsolete_pickup_sum"] = _get("wait_obsolete_pickup_sum")
    row["rew_wait_obsolete_pickup_count"] = int(_get("wait_obsolete_pickup_count"))

    row["rew_wait_terminal_never_picked_sum"] = _get("wait_terminal_never_picked_sum")
    row["rew_wait_terminal_never_picked_count"] = int(_get("wait_terminal_never_picked_count"))

    row["rew_wait_unattributed_sum"] = _get("wait_unattributed_sum")

    # travel (wait_travel mode only)
    row["rew_travel_event_dropoff_sum"] = _get("travel_event_dropoff_sum")
    row["rew_travel_event_dropoff_count"] = int(_get("travel_event_dropoff_count"))
    row["rew_travel_event_dropoff_mean"] = _safe_ratio(row["rew_travel_event_dropoff_sum"], row["rew_travel_event_dropoff_count"])

    row["rew_travel_terminal_picked_not_dropped_sum"] = _get("travel_terminal_picked_not_dropped_sum")
    row["rew_travel_terminal_picked_not_dropped_count"] = int(_get("travel_terminal_picked_not_dropped_count"))

    # deadline (deadline mode only)
    row["rew_deadline_pickup_lateness_sum"] = _get("deadline_pickup_lateness_sum")
    row["rew_deadline_pickup_lateness_count"] = int(_get("deadline_pickup_lateness_count"))
    row["rew_deadline_pickup_lateness_mean"] = _safe_ratio(row["rew_deadline_pickup_lateness_sum"], row["rew_deadline_pickup_lateness_count"])

    row["rew_deadline_dropoff_lateness_sum"] = _get("deadline_dropoff_lateness_sum")
    row["rew_deadline_dropoff_lateness_count"] = int(_get("deadline_dropoff_lateness_count"))
    row["rew_deadline_dropoff_lateness_mean"] = _safe_ratio(row["rew_deadline_dropoff_lateness_sum"], row["rew_deadline_dropoff_lateness_count"])

    # obsolete
    row["rew_obsolete_total_sum"] = _get("obsolete_total_sum")
    row["rew_obsolete_total_count"] = int(_get("obsolete_total_count"))

    # terminal penalties
    row["rew_terminal_total_sum"] = _get("terminal_total_sum")
    row["rew_terminal_total_count"] = int(_get("terminal_total_count"))

    return row


def _compute_task_section(episode_dir: str, context: dict) -> tuple[dict, list[dict]]:
    """Section 2: task-level quality metrics from task_lifecycle.csv."""
    row: dict = {}
    task_events: list[dict] = []

    if not _HAS_PANDAS:
        return row, task_events

    lc_path = os.path.join(episode_dir, "task_lifecycle.csv")
    df = _read_csv_safe(lc_path, required_cols=["task_id", "was_obsolete"])
    if df is None or len(df) == 0:
        row.update({
            "task_total_count": 0,
            "task_completed_count": 0,
            "task_completed_rate": 0.0,
            "task_obsolete_count": 0,
            "task_obsolete_rate": 0.0,
            "task_never_picked_count": 0,
            "task_never_picked_rate": 0.0,
            "task_picked_not_dropped_count": 0,
            "task_picked_not_dropped_rate": 0.0,
            "task_obs_dropoff_count": 0,
            "task_obs_dropoff_rate": 0.0,
            "task_dropoff_event_count": 0,
            "task_dropoff_event_rate": 0.0,
            "task_valid_completed_count": 0,
            "task_valid_completed_rate": 0.0,
            "task_invalid_dropoff_count": 0,
            "task_invalid_dropoff_rate": 0.0,
            "task_pickup_deadline_violation_count": 0,
            "task_pickup_deadline_violation_rate": 0.0,
            "task_dropoff_deadline_violation_count": 0,
            "task_dropoff_deadline_violation_rate": 0.0,
            "task_obsolete_pickup_count": 0,
            "task_obsolete_pickup_rate": 0.0,
            "task_obsolete_dropoff_count": 0,
            "task_obsolete_dropoff_rate": 0.0,
        })
        return row, task_events

    n_total = len(df)
    row["task_total_count"] = n_total

    # completed = actual_dropoff_time is not null and not obsolete
    has_dropoff = "actual_dropoff_time" in df.columns
    has_pickup = "actual_pickup_time" in df.columns

    if has_dropoff:
        completed_mask = df["actual_dropoff_time"].notna() & (~df["was_obsolete"].astype(bool))
        row["task_completed_count"] = int(completed_mask.sum())
    else:
        row["task_completed_count"] = 0
    row["task_completed_rate"] = _safe_ratio(row["task_completed_count"], n_total)

    obsolete_mask = df["was_obsolete"].astype(bool)
    row["task_obsolete_count"] = int(obsolete_mask.sum())
    row["task_obsolete_rate"] = _safe_ratio(row["task_obsolete_count"], n_total)

    if has_pickup:
        never_picked_mask = df["actual_pickup_time"].isna()
        row["task_never_picked_count"] = int(never_picked_mask.sum())
        row["task_never_picked_rate"] = _safe_ratio(row["task_never_picked_count"], n_total)

        picked_not_dropped_count = 0
        obs_dropoff_count = 0
        if has_dropoff and "dropoff_deadline" in df.columns:
            for _, r in df.iterrows():
                picked_up = pd.notna(r["actual_pickup_time"])
                dropped_off = pd.notna(r["actual_dropoff_time"])
                has_dd = pd.notna(r.get("dropoff_deadline"))
                if picked_up and not dropped_off:
                    picked_not_dropped_count += 1
                # was_obsolete_dropoff: picked up but did not complete before dropoff_deadline
                if picked_up and has_dd:
                    if not dropped_off or float(r["actual_dropoff_time"]) > float(r["dropoff_deadline"]):
                        obs_dropoff_count += 1
        row["task_picked_not_dropped_count"] = picked_not_dropped_count
        row["task_picked_not_dropped_rate"] = _safe_ratio(picked_not_dropped_count, n_total)
        row["task_obs_dropoff_count"] = obs_dropoff_count
        row["task_obs_dropoff_rate"] = _safe_ratio(obs_dropoff_count, n_total)
    else:
        row["task_never_picked_count"] = 0
        row["task_never_picked_rate"] = 0.0
        row["task_picked_not_dropped_count"] = 0
        row["task_picked_not_dropped_rate"] = 0.0
        row["task_obs_dropoff_count"] = 0
        row["task_obs_dropoff_rate"] = 0.0

    # --- new deadline/validity metrics ---
    has_pdl = "pickup_deadline" in df.columns
    has_ddl = "dropoff_deadline" in df.columns

    dropoff_event_count = 0
    valid_completed_count = 0
    invalid_dropoff_count = 0
    pickup_deadline_violation_count = 0
    dropoff_deadline_violation_count = 0
    obsolete_pickup_count = 0
    obsolete_dropoff_count = 0

    if has_dropoff and has_pickup:
        for _, r in df.iterrows():
            ptime = r.get("actual_pickup_time") if has_pickup else None
            dtime = r.get("actual_dropoff_time") if has_dropoff else None
            pdl = r.get("pickup_deadline") if has_pdl else None
            ddl = r.get("dropoff_deadline") if has_ddl else None

            has_ptime = pd.notna(ptime)
            has_dtime = pd.notna(dtime)
            has_pdl_val = has_pdl and pd.notna(pdl)
            has_ddl_val = has_ddl and pd.notna(ddl)

            if has_dtime:
                dropoff_event_count += 1
                # valid: both deadlines met
                pickup_ok = has_ptime and has_pdl_val and float(ptime) <= float(pdl)
                dropoff_ok = has_ddl_val and float(dtime) <= float(ddl)
                if pickup_ok and dropoff_ok:
                    valid_completed_count += 1
                else:
                    invalid_dropoff_count += 1
                # dropoff deadline violation
                if has_dtime and has_ddl_val and float(dtime) > float(ddl):
                    dropoff_deadline_violation_count += 1

            if has_ptime:
                # pickup deadline violation
                if has_pdl_val and float(ptime) > float(pdl):
                    pickup_deadline_violation_count += 1
                # obsolete_pickup: was marked obsolete (missed pickup window)
                if bool(r.get("was_obsolete", False)):
                    obsolete_pickup_count += 1

            # obsolete_dropoff: dropped off but deadline was missed (or was already obsolete)
            if has_dtime:
                is_obs = bool(r.get("was_obsolete", False))
                ddl_missed = has_ddl_val and float(dtime) > float(ddl)
                if is_obs or ddl_missed:
                    obsolete_dropoff_count += 1

    row["task_dropoff_event_count"] = dropoff_event_count
    row["task_dropoff_event_rate"] = _safe_ratio(dropoff_event_count, n_total)
    row["task_valid_completed_count"] = valid_completed_count
    row["task_valid_completed_rate"] = _safe_ratio(valid_completed_count, n_total)
    row["task_invalid_dropoff_count"] = invalid_dropoff_count
    row["task_invalid_dropoff_rate"] = _safe_ratio(invalid_dropoff_count, n_total)
    row["task_pickup_deadline_violation_count"] = pickup_deadline_violation_count
    row["task_pickup_deadline_violation_rate"] = _safe_ratio(pickup_deadline_violation_count, n_total)
    row["task_dropoff_deadline_violation_count"] = dropoff_deadline_violation_count
    row["task_dropoff_deadline_violation_rate"] = _safe_ratio(dropoff_deadline_violation_count, n_total)
    row["task_obsolete_pickup_count"] = obsolete_pickup_count
    row["task_obsolete_pickup_rate"] = _safe_ratio(obsolete_pickup_count, n_total)
    row["task_obsolete_dropoff_count"] = obsolete_dropoff_count
    row["task_obsolete_dropoff_rate"] = _safe_ratio(obsolete_dropoff_count, n_total)

    # wait time distribution
    if has_pickup and "reservation_time" in df.columns:
        wait_col = "actual_waiting_time" if "actual_waiting_time" in df.columns else None
        if wait_col:
            wait_vals = df[wait_col].dropna().tolist()
        else:
            wait_vals = []
            for _, r in df.iterrows():
                if pd.notna(r["actual_pickup_time"]) and pd.notna(r.get("reservation_time")):
                    wait_vals.append(float(r["actual_pickup_time"]) - float(r["reservation_time"]))
        row.update(_dist_stats(wait_vals, "task_wait_time"))
    else:
        row.update(_dist_stats([], "task_wait_time"))

    # travel time distribution
    if has_pickup and has_dropoff:
        travel_col = "actual_travel_time" if "actual_travel_time" in df.columns else None
        if travel_col:
            travel_vals = df[travel_col].dropna().tolist()
        else:
            travel_vals = []
            for _, r in df.iterrows():
                if pd.notna(r["actual_pickup_time"]) and pd.notna(r["actual_dropoff_time"]):
                    travel_vals.append(float(r["actual_dropoff_time"]) - float(r["actual_pickup_time"]))
        row.update(_dist_stats(travel_vals, "task_travel_time"))
    else:
        row.update(_dist_stats([], "task_travel_time"))

    # estimated vs actual travel time ratio
    if has_dropoff and "estimated_travel_time" in df.columns:
        ratios = []
        for _, r in df.iterrows():
            est = r.get("estimated_travel_time")
            actual_travel = r.get("actual_travel_time") if "actual_travel_time" in df.columns else None
            if pd.notna(est) and pd.notna(actual_travel) and float(est) > 0:
                ratios.append(float(actual_travel) / float(est))
        row.update(_dist_stats(ratios, "task_travel_ratio"))
    else:
        row.update(_dist_stats([], "task_travel_ratio"))

    # pickup deadline lateness distribution (deadline mode)
    if has_pickup and "pickup_deadline" in df.columns:
        lateness_vals = []
        for _, r in df.iterrows():
            if pd.notna(r["actual_pickup_time"]) and pd.notna(r.get("pickup_deadline")):
                lat = float(r["actual_pickup_time"]) - float(r["pickup_deadline"])
                lateness_vals.append(lat)
        row.update(_dist_stats(lateness_vals, "task_pickup_lateness"))
    else:
        row.update(_dist_stats([], "task_pickup_lateness"))

    # task-level event rows
    task_events = []
    if df is not None and len(df) > 0:
        for _, r in df.iterrows():
            event: dict = {
                "task_id": r.get("task_id", ""),
                "was_obsolete": bool(r.get("was_obsolete", False)),
                "reservation_time": r.get("reservation_time"),
                "actual_pickup_time": r.get("actual_pickup_time"),
                "actual_dropoff_time": r.get("actual_dropoff_time"),
                "actual_waiting_time": r.get("actual_waiting_time"),
                "actual_travel_time": r.get("actual_travel_time"),
            }
            task_events.append(event)

    return row, task_events


def _compute_pooling_section(episode_dir: str, context: dict) -> dict:
    """Section 3: pooling metrics from robot_occupancy.csv."""
    row: dict = {}
    if not _HAS_PANDAS:
        return row

    occ_path = os.path.join(episode_dir, "robot_occupancy.csv")
    df = _read_csv_safe(occ_path, required_cols=["onboard_count"])
    if df is None or len(df) == 0:
        row.update({
            "pool_mean_onboard": 0.0,
            "pool_max_onboard": 0.0,
            "pool_frac_multi_pax": 0.0,
            "pool_frac_empty": 0.0,
            "pool_steps_total": 0,
        })
        return row

    counts = df["onboard_count"].astype(float)
    total = len(counts)
    row["pool_mean_onboard"] = float(counts.mean())
    row["pool_max_onboard"] = float(counts.max())
    row["pool_frac_multi_pax"] = float((counts >= 2).sum()) / total if total > 0 else 0.0
    row["pool_frac_empty"] = float((counts == 0).sum()) / total if total > 0 else 0.0
    row["pool_steps_total"] = total

    # per-robot breakdown
    if "robot_id" in df.columns:
        robot_means = df.groupby("robot_id")["onboard_count"].mean()
        row.update(_dist_stats(robot_means.tolist(), "pool_robot_mean_onboard"))

    return row


def _compute_decision_section(episode_dir: str) -> tuple[dict, list[dict]]:
    """Section 4: decision / NOOP metrics from decisions.csv."""
    row: dict = {}
    decision_events: list[dict] = []
    if not _HAS_PANDAS:
        return row, decision_events

    dec_path = os.path.join(episode_dir, "decisions.csv")
    df = _read_csv_safe(dec_path, required_cols=["is_noop", "num_candidates"])
    if df is None or len(df) == 0:
        row.update({
            "dec_total": 0,
            "dec_noop_count": 0,
            "dec_noop_rate": 0.0,
            "dec_noop_no_candidates_count": 0,
            "dec_noop_with_candidates_count": 0,
            "dec_mean_candidates": 0.0,
        })
        return row, decision_events

    n_total = len(df)
    noop_mask = df["is_noop"].astype(bool)
    n_noop = int(noop_mask.sum())
    row["dec_total"] = n_total
    row["dec_noop_count"] = n_noop
    row["dec_noop_rate"] = _safe_ratio(n_noop, n_total)

    # NOOP with zero candidates = forced NOOP (no feasible tasks)
    noop_no_cand_mask = noop_mask & (df["num_candidates"].astype(int) == 0)
    noop_with_cand_mask = noop_mask & (df["num_candidates"].astype(int) > 0)
    row["dec_noop_no_candidates_count"] = int(noop_no_cand_mask.sum())
    row["dec_noop_with_candidates_count"] = int(noop_with_cand_mask.sum())
    row["dec_mean_candidates"] = float(df["num_candidates"].mean())

    # ETA-rank and feasibility margin metrics: NOT computable with current data
    # (ETAs per candidate not stored). Set to 0.0 to document unavailability.
    row["dec_choice_eta_rank_mean"] = 0.0         # unavailable — see quality_episode_metrics.md
    row["dec_choice_pickup_margin_mean"] = 0.0    # unavailable — see quality_episode_metrics.md

    decision_events = df.to_dict("records")
    return row, decision_events


def _compute_candidate_section(episode_dir: str) -> dict:
    """Section 5: candidate availability from candidates.csv."""
    row: dict = {}
    if not _HAS_PANDAS:
        return row

    cand_path = os.path.join(episode_dir, "candidates.csv")
    df = _read_csv_safe(cand_path)
    if df is None or len(df) == 0:
        row["cand_mean_slots"] = 0.0
        row["cand_zero_slots_rate"] = 0.0
        return row

    if "cand_slots" in df.columns:
        def _count_slots(value: Any) -> int:
            if pd.isna(value):
                return 0
            text = str(value).strip()
            if not text:
                return 0
            if "|" in text:
                return len([part for part in text.split("|") if str(part).strip()])
            try:
                return int(float(text))
            except Exception:
                return 1

        slot_counts = df["cand_slots"].apply(_count_slots).astype(float)
        row["cand_mean_slots"] = float(slot_counts.mean())
        row["cand_zero_slots_rate"] = float((slot_counts == 0).sum()) / len(slot_counts) if len(slot_counts) > 0 else 0.0
    else:
        row["cand_mean_slots"] = 0.0
        row["cand_zero_slots_rate"] = 0.0

    return row


def _compute_conflict_section(conflict_stats: dict) -> dict:
    """Section 6: conflict quality metrics from conflict_stats dict."""
    row: dict = {}
    cs = conflict_stats or {}

    def _g(k: str, default: float = 0.0) -> float:
        return float(cs.get(k, default))

    row["conf_total"] = int(_g("conflicts_total"))
    row["conf_tasks_total"] = int(_g("tasks_total"))
    row["conf_winner_pickup"] = int(_g("winner_pickup"))
    row["conf_winner_margin"] = int(_g("winner_margin"))
    row["conf_winner_raw_logit"] = int(_g("winner_raw_logit"))
    row["conf_policy_matches_resolver"] = int(_g("policy_action_matches_resolver"))
    row["conf_resolver_override"] = int(_g("resolver_override"))
    row["conf_margin_win_count"] = int(_g("margin_win_count"))
    row["conf_margin_lose_count"] = int(_g("margin_lose_count"))

    if row["conf_total"] > 0:
        row["conf_winner_pickup_rate"] = _safe_ratio(row["conf_winner_pickup"], row["conf_total"])
        row["conf_resolver_override_rate"] = _safe_ratio(row["conf_resolver_override"], row["conf_total"])
        row["conf_policy_matches_resolver_rate"] = _safe_ratio(row["conf_policy_matches_resolver"], row["conf_total"])
    else:
        row["conf_winner_pickup_rate"] = 0.0
        row["conf_resolver_override_rate"] = 0.0
        row["conf_policy_matches_resolver_rate"] = 0.0

    return row


def _compute_macro_reward_section(episode_dir: str) -> dict:
    """Read rewards_macro.csv to get top-level aggregated reward stats."""
    row: dict = {}
    if not _HAS_PANDAS:
        return row

    macro_path = os.path.join(episode_dir, "rewards_macro.csv")
    df = _read_csv_safe(macro_path)
    if df is None or len(df) == 0:
        return row

    if "reward" in df.columns:
        rewards = df["reward"].dropna().astype(float)
        row["macro_reward_mean"] = float(rewards.mean())
        row["macro_reward_std"] = float(rewards.std()) if len(rewards) > 1 else 0.0
        row["macro_reward_sum"] = float(rewards.sum())
        row["macro_steps"] = len(rewards)
    return row


# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------

def compute_quality_episode_metrics(
    episode_dir: str,
    context: dict,
    conflict_stats: dict,
    config_id: str,
    run_id: str,
    ts: int,
    episode: int,
    include_task_level: bool = False,
    include_decision_level: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    """Compute all episode quality metrics.

    Parameters
    ----------
    episode_dir : str
        Path to the episode directory (contains task_lifecycle.csv, etc.)
    context : dict
        Output of adapter.get_episode_quality_context()
    conflict_stats : dict
        Output of logger.get_episode_conflict_stats()
    config_id : str
        Config identifier (cfg.prefix in train.py)
    run_id : str
        Run identifier
    ts : int
        Training timestep at end of episode
    episode : int
        Episode index
    include_task_level : bool
        Whether to collect per-task event rows
    include_decision_level : bool
        Whether to collect per-decision event rows

    Returns
    -------
    flat_row : dict
        Single flat dict for quality_episode_metrics.csv
    task_events : list[dict]
        Per-task rows (empty when include_task_level=False)
    decision_events : list[dict]
        Per-decision rows (empty when include_decision_level=False)
    """
    flat_row: dict[str, Any] = {
        "config_id": config_id,
        "run_id": run_id,
        "ts": int(ts),
        "episode": int(episode),
        "reward_type": context.get("reward_type", ""),
        "completion_mode": context.get("completion_mode", ""),
        "num_robots": context.get("num_robots", 0),
        "max_robot_capacity": context.get("max_robot_capacity", 0),
    }

    # reward subcomponents
    flat_row.update(_compute_reward_section(context))

    # macro reward
    flat_row.update(_compute_macro_reward_section(episode_dir))

    # task lifecycle
    task_section, task_events_raw = _compute_task_section(episode_dir, context)
    flat_row.update(task_section)

    # pooling
    flat_row.update(_compute_pooling_section(episode_dir, context))

    # decision / NOOP
    dec_section, decision_events_raw = _compute_decision_section(episode_dir)
    flat_row.update(dec_section)

    # candidates
    flat_row.update(_compute_candidate_section(episode_dir))

    # conflict stats
    flat_row.update(_compute_conflict_section(conflict_stats))

    task_events = task_events_raw if include_task_level else []
    decision_events = decision_events_raw if include_decision_level else []

    return flat_row, task_events, decision_events
