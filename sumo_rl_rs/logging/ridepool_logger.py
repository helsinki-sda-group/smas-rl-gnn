# sumo_rl_rs/logging/ridepool_logger.py
from __future__ import annotations
import os, csv, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Any
import json

from pathlib import Path
import shutil, datetime as dt

@dataclass
class RidepoolLogConfig:
    out_dir: str = "runs"
    run_name: Optional[str] = None       # if None, we use timestamp
    episode_index: int = 0               # will increment per episode
    flush_every: int = 200                 # flush to disk every N writes
    console_debug: bool = False          # print some lines to console
    erase_run_dir_on_start: bool = False         # if True: delete the whole run directory at construction
    erase_episode_dir_on_start: bool = False     # if True: delete an episode dir when starting it
    csv_postfix: Optional[str] = None    # postfix for all CSV filenames (e.g., "random1"). If None, no postfix added
    log_conflict_metrics: bool = False    # write run-level conflicts.log with episode summaries
    overwrite_conflicts_log_on_start: bool = False  # if True: delete root conflicts.log at logger init
    prune_episode_dir_after_metrics: bool = False  # if True: callback removes episode_* dir after appending run-level metrics
    # Extended quality diagnostics (Section 3 of quality_episode_metrics spec)
    extended_quality_metrics: bool = False           # compute and persist episode-level quality metrics
    extended_quality_plots: bool = False             # generate quality metric plots (used by callback/plotter)
    extended_quality_include_task_level: bool = False    # write per-task rows to task_quality_events.csv
    extended_quality_include_decision_level: bool = False  # write per-decision rows to decision_quality_events.csv

class RidepoolLogger:
    """
    One logger to rule them all:
      - dispatch.csv          : one row per taxi per step with prev/base/seq and raw currentCustomers
      - conflicts.csv         : one row per reservation when conflict happened
      - candidates.csv        : one row per taxi per step (candidate slots, reservation ids)
      - rewards.csv           : one row per taxi per step with reward decomposition
      - fleet_counts.csv      : one row per step with counters (idle, en_route, occupied, pickup_occupied)
      - episode_totals.csv    : summary per episode
      - task_lifecycle.csv    : comprehensive task tracking 
      - taxi_events.csv       : taxi events by step 
      - plots/*.png           : a few basic plots
    """
    def __init__(self, cfg: RidepoolLogConfig):
        self.cfg = cfg
        stamp = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(cfg.out_dir, stamp)

        if cfg.erase_run_dir_on_start and os.path.isdir(self.run_dir):
            if cfg.console_debug:
                print(f"[RidepoolLogger] Removing existing run dir: {self.run_dir}")
            shutil.rmtree(self.run_dir, ignore_errors=True)

        os.makedirs(self.run_dir, exist_ok=True)

        self.ep_dir: Optional[str] = None
        self.last_ep_dir: Optional[str] = None
        self._counters = {"writes": 0}
        self._files: Dict[str, Any] = {}
        self._writers: Dict[str, csv.DictWriter] = {}
        self._csv_postfix = cfg.csv_postfix or ""
        # Map logical filenames to actual filenames (for backward compatibility with postfix)
        self._fname_map: Dict[str, str] = {}
        # step-local timeseries for plotting
        self._ts: Dict[str, List[float]] = {
            "idle": [], "en_route": [], "occupied": [], "pickup_occupied": [],
            "sum_reward": [], "pickups": [], "dropoffs": []
        }
        self._conflicts_log_path = os.path.abspath("conflicts.log")
        if bool(getattr(self.cfg, "log_conflict_metrics", False)) and bool(getattr(self.cfg, "overwrite_conflicts_log_on_start", False)):
            try:
                if os.path.exists(self._conflicts_log_path):
                    os.remove(self._conflicts_log_path)
            except Exception:
                pass
        self._last_conflict_stats: Dict[str, float] = {}
        self._reset_conflict_episode_stats()

    def _reset_conflict_episode_stats(self) -> None:
        self._conflict_stats: Dict[str, float] = {
            "tasks_total": 0.0,
            "conflicts_total": 0.0,
            "winner_pickup": 0.0,
            "winner_margin": 0.0,
            "winner_raw_logit": 0.0,
            "margin_matches_pickup": 0.0,
            "raw_logit_matches_pickup": 0.0,
            "policy_action_matches_resolver": 0.0,
            "resolver_override": 0.0,
            "margin_win_sum": 0.0,
            "margin_win_count": 0.0,
            "margin_lose_sum": 0.0,
            "margin_lose_count": 0.0,
            "margin_gap_sum": 0.0,
            "margin_gap_count": 0.0,
        }

    def _append_conflicts_summary(self, episode_idx: int) -> None:
        if not bool(getattr(self.cfg, "log_conflict_metrics", False)):
            return
        tasks_total = float(self._conflict_stats["tasks_total"])
        conflicts_total = float(self._conflict_stats["conflicts_total"])
        winner_pickup = float(self._conflict_stats["winner_pickup"])
        winner_margin = float(self._conflict_stats["winner_margin"])
        winner_raw_logit = float(self._conflict_stats["winner_raw_logit"])
        margin_matches_pickup = float(self._conflict_stats["margin_matches_pickup"])
        raw_logit_matches_pickup = float(self._conflict_stats["raw_logit_matches_pickup"])
        policy_action_matches_resolver = float(self._conflict_stats["policy_action_matches_resolver"])
        resolver_override = float(self._conflict_stats["resolver_override"])

        margin_win_count = float(self._conflict_stats["margin_win_count"])
        margin_lose_count = float(self._conflict_stats["margin_lose_count"])
        margin_gap_count = float(self._conflict_stats["margin_gap_count"])

        conflict_ratio = (conflicts_total / tasks_total) if tasks_total > 0 else 0.0
        resolver_override_rate = (resolver_override / conflicts_total) if conflicts_total > 0 else 0.0
        p_resolver_matches_pickup = (winner_pickup / conflicts_total) if conflicts_total > 0 else 0.0
        p_resolver_matches_margin = (winner_margin / conflicts_total) if conflicts_total > 0 else 0.0
        p_resolver_matches_raw_logit = (winner_raw_logit / conflicts_total) if conflicts_total > 0 else 0.0
        p_margin_matches_pickup = (margin_matches_pickup / conflicts_total) if conflicts_total > 0 else 0.0
        p_raw_logit_matches_pickup = (raw_logit_matches_pickup / conflicts_total) if conflicts_total > 0 else 0.0
        p_policy_action_matches_resolver = (policy_action_matches_resolver / conflicts_total) if conflicts_total > 0 else 0.0
        avg_margin_win = (self._conflict_stats["margin_win_sum"] / margin_win_count) if margin_win_count > 0 else 0.0
        avg_margin_lose = (self._conflict_stats["margin_lose_sum"] / margin_lose_count) if margin_lose_count > 0 else 0.0
        avg_margin_gap = (self._conflict_stats["margin_gap_sum"] / margin_gap_count) if margin_gap_count > 0 else 0.0

        file_exists = os.path.exists(self._conflicts_log_path)
        with open(self._conflicts_log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if (not file_exists) or os.path.getsize(self._conflicts_log_path) == 0:
                writer.writerow([
                    "episode",
                    "conflicts_total",
                    "tasks_total",
                    "conflict_ratio",
                    "winner_pickup",
                    "winner_margin",
                    "p_resolver_matches_pickup",
                    "p_resolver_matches_margin",
                    "p_resolver_matches_raw_logit",
                    "p_margin_matches_pickup",
                    "p_raw_logit_matches_pickup",
                    "p_policy_action_matches_resolver",
                    "resolver_override",
                    "resolver_override_rate",
                    "avg_margin_win",
                    "avg_margin_lose",
                    "avg_margin_gap",
                ])
            writer.writerow([
                int(episode_idx),
                f"{conflicts_total:.2f}",
                f"{tasks_total:.2f}",
                f"{conflict_ratio:.2f}",
                f"{winner_pickup:.2f}",
                f"{winner_margin:.2f}",
                f"{p_resolver_matches_pickup:.4f}",
                f"{p_resolver_matches_margin:.4f}",
                f"{p_resolver_matches_raw_logit:.4f}",
                f"{p_margin_matches_pickup:.4f}",
                f"{p_raw_logit_matches_pickup:.4f}",
                f"{p_policy_action_matches_resolver:.4f}",
                f"{resolver_override:.2f}",
                f"{resolver_override_rate:.2f}",
                f"{avg_margin_win:.2f}",
                f"{avg_margin_lose:.2f}",
                f"{avg_margin_gap:.2f}",
            ])

    # ---------- lifecycle ----------
    def set_csv_postfix(self, postfix: str) -> None:
        """Set the postfix for all CSV filenames. If empty string, no postfix is added."""
        self._csv_postfix = postfix
        self._fname_map.clear()  # Clear the mapping cache when postfix changes

    def _get_csv_filename(self, base_name: str) -> str:
        """
        Get the actual CSV filename with postfix applied.
        base_name should be without .csv extension (e.g., 'dispatch')
        Returns the postfixed filename (e.g., 'dispatch_random1.csv' or 'dispatch.csv')
        """
        if base_name in self._fname_map:
            return self._fname_map[base_name]
        
        if self._csv_postfix:
            fname = f"{base_name}_{self._csv_postfix}.csv"
        else:
            fname = f"{base_name}.csv"
        
        self._fname_map[base_name] = fname
        return fname

    def start_episode(self, episode_idx: Optional[int] = None):
        if episode_idx is not None:
            self.cfg.episode_index = int(episode_idx)
        ep_name = f"episode_{self.cfg.episode_index:04d}"
        self.ep_dir = os.path.join(self.run_dir, ep_name)

        if self.cfg.erase_episode_dir_on_start and os.path.isdir(self.ep_dir):
            if self.cfg.console_debug:
                print(f"[RidepoolLogger] Removing existing episode dir: {self.ep_dir}")
            shutil.rmtree(self.ep_dir, ignore_errors=True)

        os.makedirs(self.ep_dir, exist_ok=True)

        self._open_csv(self._get_csv_filename("dispatch"), ["time","taxi","prev_seq","base_ids","seq","seq_pd","raw_currentCustomers","notes"])
        self._open_csv(
            self._get_csv_filename("conflicts"),
            [
                "time",
                "res_id",
                "taxi_candidates",
                "remaining_caps",
                "distances",
                "winner",
                "win_label",
                "win_label_high_logit",
            ],
        )
        self._open_csv(self._get_csv_filename("candidates"), ["time","taxi","cand_slots","cand_res_ids","cand_persons","cand_pd_seq"])
        self._open_csv(self._get_csv_filename("rewards"), ["time","taxi","reward","capacity","step","deadline","wait","travel","completion", "nonserved"])
        self._open_csv(self._get_csv_filename("fleet_counts"), ["time","idle","en_route","occupied","pickup_occupied"])
        self._open_csv(self._get_csv_filename("episode_totals"), ["episode","sum_reward","n_pickups","n_dropoffs","duration"])
        self._open_csv(self._get_csv_filename("rewards_macro"), ["macro_steps","reward","capacity_avg","step_avg","deadline_avg", "wait_avg", "travel_avg", "completion_avg", "nonserved_avg"])

        self._open_csv(self._get_csv_filename("task_lifecycle"), [
            "task_id", "reservation_time", "pickup_deadline", "estimated_travel_time",
            "dropoff_deadline", "actual_pickup_time", "actual_dropoff_time",
            "assigned_step", "assigned_taxi", "pickup_step", "pickup_taxi",
            "dropoff_step", "dropoff_taxi", "was_obsolete",
            "actual_waiting_time", "actual_travel_time"
        ])
        self._open_csv(self._get_csv_filename("taxi_events"), ["step", "taxi", "event_type", "task_id"])

        # Extended quality diagnostics: open additional per-episode CSVs when enabled
        if bool(getattr(self.cfg, "extended_quality_metrics", False)):
            self._open_csv(self._get_csv_filename("robot_occupancy"), [
                "time", "robot_id", "onboard_count", "customer_ids", "shadow_plan_length"
            ])
            self._open_csv(self._get_csv_filename("decisions"), [
                "time", "robot_id", "selected_task_id", "num_candidates", "is_noop"
            ])
        # reset timeseries
        for k in self._ts:
            self._ts[k].clear()
        self._reset_conflict_episode_stats()

    def end_episode(self, sum_reward: float, n_pickups: int, n_dropoffs: int, duration: float):
        self.last_ep_dir = self.ep_dir
        self._last_conflict_stats = dict(self._conflict_stats)
        self._append_conflicts_summary(self.cfg.episode_index)
        self._write(self._get_csv_filename("episode_totals"), dict(
            episode=self.cfg.episode_index,
            sum_reward=float(sum_reward),
            n_pickups=int(n_pickups),
            n_dropoffs=int(n_dropoffs),
            duration=float(duration),
        ))
        self._close_all()
        # plots
        self._plot_ts("fleet_counts.png", [
            ("idle","Idle"), ("en_route","En-route to PU"),
            ("occupied","Occupied DO"), ("pickup_occupied","Onboard+PU")
        ], ylabel="Count")
        self._plot_ts("rewards_sum.png", [("sum_reward","Sum reward")], ylabel="Reward")
        self._plot_ts("completions.png", [("pickups","Pickups"), ("dropoffs","Dropoffs")], ylabel="Count")
        self.cfg.episode_index += 1

    def log_conflict_task_count(self, tasks_total: int) -> None:
        if not bool(getattr(self.cfg, "log_conflict_metrics", False)):
            return
        self._conflict_stats["tasks_total"] += float(max(0, int(tasks_total)))

    def log_conflict_metrics_event(
        self,
        *,
        winner: str,
        pickup_winners: Sequence[str],
        margin_winners: Sequence[str],
        raw_logit_winners: Sequence[str],
        policy_action_robot: Optional[str],
        winner_margin: Optional[float],
        loser_margins: Sequence[float],
    ) -> None:
        if not bool(getattr(self.cfg, "log_conflict_metrics", False)):
            return

        winner = str(winner)
        pickup_set = {str(x) for x in pickup_winners}
        margin_set = {str(x) for x in margin_winners}
        raw_logit_set = {str(x) for x in raw_logit_winners}
        policy_action_robot = str(policy_action_robot) if policy_action_robot is not None else None

        self._conflict_stats["conflicts_total"] += 1.0

        if winner in pickup_set:
            self._conflict_stats["winner_pickup"] += 1.0
        if winner in margin_set:
            self._conflict_stats["winner_margin"] += 1.0
        if winner in raw_logit_set:
            self._conflict_stats["winner_raw_logit"] += 1.0

        if pickup_set and margin_set and (pickup_set & margin_set):
            self._conflict_stats["margin_matches_pickup"] += 1.0
        if pickup_set and raw_logit_set and (pickup_set & raw_logit_set):
            self._conflict_stats["raw_logit_matches_pickup"] += 1.0
        if policy_action_robot is not None and winner == policy_action_robot:
            self._conflict_stats["policy_action_matches_resolver"] += 1.0

        if pickup_set and margin_set:
            pickup_ref = sorted(pickup_set)[0]
            margin_ref = sorted(margin_set)[0]
            if margin_ref != pickup_ref:
                self._conflict_stats["resolver_override"] += 1.0

        if winner_margin is not None:
            try:
                win_m = float(winner_margin)
            except Exception:
                win_m = float("nan")
            if win_m == win_m:
                self._conflict_stats["margin_win_sum"] += win_m
                self._conflict_stats["margin_win_count"] += 1.0

                for lose_m in loser_margins:
                    try:
                        lose = float(lose_m)
                    except Exception:
                        continue
                    if lose != lose:
                        continue
                    self._conflict_stats["margin_lose_sum"] += lose
                    self._conflict_stats["margin_lose_count"] += 1.0
                    self._conflict_stats["margin_gap_sum"] += (win_m - lose)
                    self._conflict_stats["margin_gap_count"] += 1.0

    def close(self):
        self._close_all()

    # ---------- csv helpers ----------

    def _ensure_csv(self, fname: str, header: List[str]) -> None:
        # auto-start episode if not started
        if self.ep_dir is None:
            self.start_episode(self.cfg.episode_index)
        if fname in self._writers:
            return
        self._open_csv(fname, header)

    def _open_csv(self, fname: str, header: List[str]) -> None:
        """
        Open CSV and ensure it has the exact header we want.
        If an older file exists with a different header, rotate it to .old
        and create a fresh file with the new header.
        """
        assert self.ep_dir is not None, "ep_dir must be set (call start_episode first or use lazy start)"
        fpath = os.path.join(self.ep_dir, fname)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        # If a file exists with a different header, rotate it
        needs_new = True
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            with open(fpath, "r", newline="") as fr:
                reader = csv.reader(fr)
                existing = next(reader, None)
            if existing == header:
                # exact header matches; we can append
                needs_new = False
            else:
                base, ext = os.path.splitext(fpath)
                rotated = f"{base}.old{int(time.time())}{ext}"
                os.replace(fpath, rotated)

        mode = "a" if not needs_new else "w"
        fh = open(fpath, mode, newline="")
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        if needs_new:
            writer.writeheader()

        self._files[fname] = fh
        self._writers[fname] = writer

    def _write(self, fname: str, row: Dict[str, Any]) -> None:
        self._writers[fname].writerow(row)
        self._counters["writes"] += 1
        if (self._counters["writes"] % max(1, self.cfg.flush_every)) == 0:
            self._files[fname].flush()

    def _close_all(self):
        for f in self._files.values():
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        self._files.clear()
        self._writers.clear()

    # ---------- public log API ----------
    def log_dispatch(self, t, taxi, prev_seq, base_ids, seq, raw_currentCustomers, seq_pd="", notes=""):
        """
        prev_seq, base_ids, seq are lists of reservation ids (strings).
        seq_pd: a single string like 'p1:PU|p1:DO|p2:PU|p2:DO|...'
        raw_currentCustomers: raw TraCI string (person ids, usually "p7 p12 ...")
        """
        fname = self._get_csv_filename("dispatch")
        self._ensure_csv(fname,
                        ["time","taxi","prev_seq","base_ids","seq","seq_pd","raw_currentCustomers","notes"])
        self._write(fname, dict(
            time=float(t),
            taxi=str(taxi),
            prev_seq="|".join(map(str, prev_seq)),
            base_ids="|".join(map(str, base_ids)),
            seq="|".join(map(str, seq)),
            seq_pd=str(seq_pd),
            raw_currentCustomers=str(raw_currentCustomers),
            notes=str(notes),
        ))

    def log_conflict(
        self,
        t: float,
        res_id: str,
        taxi_candidates: Sequence[str],
        remaining_caps: Sequence[int],
        distances: Sequence[float],
        winner: str,
        win_label: int = 0,
        win_label_high_logit: int = 0,
    ):
        fname = self._get_csv_filename("conflicts")
        self._ensure_csv(
            fname,
            [
                "time",
                "res_id",
                "taxi_candidates",
                "remaining_caps",
                "distances",
                "winner",
                "win_label",
                "win_label_high_logit",
            ],
        )
        self._write(fname, dict(
            time=float(t), res_id=str(res_id),
            taxi_candidates="|".join(map(str, taxi_candidates)),
            remaining_caps="|".join(map(str, remaining_caps)),
            distances="|".join(map(str, distances)),
            winner=str(winner),
            win_label=int(win_label),
            win_label_high_logit=int(win_label_high_logit),
        ))

    def log_candidates(self, t: float, taxi: str, cand_slots, cand_res_ids, cand_persons, cand_pd_seq):
        """
        cand_slots:   list[int]
        cand_res_ids: list[str]
        cand_persons: list[str]  (per candidate: 'p1+p2' or '')
        cand_pd_seq:  list[str]  (per candidate: 'p1:PU+p2:PU+p1:DO+p2:DO' or '')
        """
        fname = self._get_csv_filename("candidates")
        self._ensure_csv(fname,
                        ["time","taxi","cand_slots","cand_res_ids","cand_persons","cand_pd_seq"])
        self._write(fname, dict(
            time=float(t),
            taxi=str(taxi),
            cand_slots="|".join(map(str, cand_slots)),
            cand_res_ids="|".join(map(str, cand_res_ids)),
            cand_persons="|".join(cand_persons),
            cand_pd_seq="|".join(cand_pd_seq),
        ))

    def log_rewards(self, t: float, taxi: str, reward: float, terms: Dict[str, float]):
        fname = self._get_csv_filename("rewards")
        self._ensure_csv(fname, ["time","taxi","reward","capacity","step","deadline","wait","travel","completion", "nonserved"])

        terms_round = {k: round(float(v),2) for k,v in terms.items()}

        self._write(fname, dict(
            time=float(t), taxi=str(taxi), reward=round(float(reward),2),
            capacity=terms_round["capacity"],
            step=terms_round["step"],
            deadline=terms_round.get("deadline", 0.0),
            wait=terms_round.get("wait", 0.0),
            travel=terms_round.get("travel", 0.0),
            completion=terms_round["completion"],
            nonserved=terms_round["nonserved"],
        ))

    def log_macro_step(self, info):
        fname = self._get_csv_filename("rewards_macro")
        self._ensure_csv(fname, 
                 ["macro_steps","reward","capacity_avg","step_avg","deadline_avg", "wait_avg", "travel_avg", "completion_avg", "nonserved_avg"])

        deadline_avg = info.get("macro_deadline", info.get("macro_missed_deadline", info.get("macro_abandoned")))
        travel_avg = info.get("macro_travel", 0.0)
        self._write(fname, dict(
                macro_steps = info["macro_steps"],
                reward = info["macro_reward"],
                capacity_avg = info["macro_capacity"],
                step_avg = info["macro_step"],
            deadline_avg = deadline_avg,
            wait_avg = info["macro_wait"],
            travel_avg = travel_avg,
                completion_avg = info["macro_completion"],
                nonserved_avg = info["macro_nonserved"],
        ))

    def log_fleet_counts(self, t: float, idle: int, en_route: int, occupied: int, pickup_occupied: int):
        fname = self._get_csv_filename("fleet_counts")
        self._ensure_csv(fname, ["time","idle","en_route","occupied","pickup_occupied"])
        self._write(fname, dict(
            time=float(t), idle=int(idle), en_route=int(en_route),
            occupied=int(occupied), pickup_occupied=int(pickup_occupied)
        ))
        self._ts["idle"].append(idle)
        self._ts["en_route"].append(en_route)
        self._ts["occupied"].append(occupied)
        self._ts["pickup_occupied"].append(pickup_occupied)

    def log_ts_reward(self, sum_reward: float, pickups: int, dropoffs: int):
        self._ts["sum_reward"].append(sum_reward)
        self._ts["pickups"].append(pickups)
        self._ts["dropoffs"].append(dropoffs)

    def log_debug(self, tnow: float, tag: str, payload: dict) -> None:
        """
        Generic debugging hook: writes a row to debug.csv
        """
        try:
            fname = self._get_csv_filename("debug")
            self._ensure_csv(fname, ["time", "tag", "data"])
            self._write(fname, dict(
                time=float(tnow),
                tag=str(tag),
                data=json.dumps(payload, ensure_ascii=False)
            ))
        except Exception as e:
            print(f"[logger] log_debug failed: {e}")

# ---------- Task lifecycle tracking ----------
    def log_task_lifecycle(self, task_id: str, **kwargs):
        """
        Log comprehensive task lifecycle information.
        
        Expected fields:
        - reservation_time: when task was created
        - pickup_deadline: deadline for pickup
        - estimated_travel_time: estimated time from pickup to dropoff
        - dropoff_deadline: deadline for dropoff
        - actual_pickup_time: when actually picked up (None if not picked up)
        - actual_dropoff_time: when actually dropped off (None if not dropped off)
        - assigned_step: step when first assigned (None if never assigned)
        - assigned_taxi: taxi that was assigned (None if never assigned)
        - pickup_step: step when picked up (None if not picked up)
        - pickup_taxi: taxi that picked up (None if not picked up)
        - dropoff_step: step when dropped off (None if not dropped off)
        - dropoff_taxi: taxi that dropped off (None if not dropped off)
        - was_obsolete: True if task became obsolete before pickup
        - actual_waiting_time: time from reservation to pickup (None if not picked up)
        - actual_travel_time: time from pickup to dropoff (None if not dropped off)
        """
        header = [
            "task_id", "reservation_time", "pickup_deadline", "estimated_travel_time",
            "dropoff_deadline", "actual_pickup_time", "actual_dropoff_time",
            "assigned_step", "assigned_taxi", "pickup_step", "pickup_taxi",
            "dropoff_step", "dropoff_taxi", "was_obsolete",
            "actual_waiting_time", "actual_travel_time"
        ]
        fname = self._get_csv_filename("task_lifecycle")
        self._ensure_csv(fname, header)
        
        row = {"task_id": str(task_id)}
        row.update(kwargs)
        self._write(fname, row)

    def log_taxi_event(self, step: int, taxi: str, event_type: str, task_id: str):
        """
        Log taxi events: assigned, picked_up, dropped_off
        
        Args:
            step: simulation step number
            taxi: taxi id (e.g., 't0')
            event_type: 'assigned', 'picked_up', 'dropped_off'
            task_id: task/passenger id (e.g., 'p0', '0')
        """
        fname = self._get_csv_filename("taxi_events")
        header = ["step", "taxi", "event_type", "task_id"]
        self._ensure_csv(fname, header)
        
        self._write(fname, dict(
            step=int(step),
            taxi=str(taxi),
            event_type=str(event_type),
            task_id=str(task_id)
        ))

    def log_robot_occupancy(
        self,
        t: float,
        robot_id: str,
        onboard_count: int,
        customer_ids: str,
        shadow_plan_length: int,
    ) -> None:
        """Log per-robot per-step occupancy. Written only when extended_quality_metrics=True."""
        fname = self._get_csv_filename("robot_occupancy")
        self._ensure_csv(fname, ["time", "robot_id", "onboard_count", "customer_ids", "shadow_plan_length"])
        self._write(fname, dict(
            time=float(t),
            robot_id=str(robot_id),
            onboard_count=int(onboard_count),
            customer_ids=str(customer_ids),
            shadow_plan_length=int(shadow_plan_length),
        ))

    def log_decision(
        self,
        t: float,
        robot_id: str,
        selected_task_id: str,
        num_candidates: int,
        is_noop: bool,
    ) -> None:
        """Log per-robot decision (before conflict resolution). Written only when extended_quality_metrics=True."""
        fname = self._get_csv_filename("decisions")
        self._ensure_csv(fname, ["time", "robot_id", "selected_task_id", "num_candidates", "is_noop"])
        self._write(fname, dict(
            time=float(t),
            robot_id=str(robot_id),
            selected_task_id=str(selected_task_id),
            num_candidates=int(num_candidates),
            is_noop=int(bool(is_noop)),
        ))

    def get_episode_conflict_stats(self) -> dict:
        """Return a copy of the current episode's conflict stats.
        Must be called BEFORE start_episode() resets them.
        """
        if self._last_conflict_stats:
            return dict(self._last_conflict_stats)
        return dict(self._conflict_stats)

    def get_last_episode_conflict_stats(self) -> dict:
        """Return the most recently closed episode's conflict stats."""
        return self.get_episode_conflict_stats()

    # ---------- plotting ----------
    def _plot_ts(self, png_name: str, series: List[tuple], ylabel: str = ""):
        # Plotting disabled
        return
