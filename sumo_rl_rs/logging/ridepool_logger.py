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
        self._open_csv(self._get_csv_filename("conflicts"), ["time","res_id","taxi_candidates","remaining_caps","distances","winner"])
        self._open_csv(self._get_csv_filename("candidates"), ["time","taxi","cand_slots","cand_res_ids","cand_persons","cand_pd_seq"])
        self._open_csv(self._get_csv_filename("rewards"), ["time","taxi","reward","capacity","step","missed_deadline","wait_at_pickups","completion", "nonserved"])
        self._open_csv(self._get_csv_filename("fleet_counts"), ["time","idle","en_route","occupied","pickup_occupied"])
        self._open_csv(self._get_csv_filename("episode_totals"), ["episode","sum_reward","n_pickups","n_dropoffs","duration"])
        self._open_csv(self._get_csv_filename("rewards_macro"), ["macro_steps","reward","capacity_avg","step_avg","missed_deadline_avg", "wait_avg", "completion_avg", "nonserved_avg"])

        self._open_csv(self._get_csv_filename("task_lifecycle"), [
            "task_id", "reservation_time", "pickup_deadline", "estimated_travel_time",
            "dropoff_deadline", "actual_pickup_time", "actual_dropoff_time",
            "assigned_step", "assigned_taxi", "pickup_step", "pickup_taxi",
            "dropoff_step", "dropoff_taxi", "was_obsolete",
            "actual_waiting_time", "actual_travel_time"
        ])
        self._open_csv(self._get_csv_filename("taxi_events"), ["step", "taxi", "event_type", "task_id"])
        # reset timeseries
        for k in self._ts:
            self._ts[k].clear()

    def end_episode(self, sum_reward: float, n_pickups: int, n_dropoffs: int, duration: float):
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

    def log_conflict(self, t: float, res_id: str, taxi_candidates: Sequence[str],
                     remaining_caps: Sequence[int], distances: Sequence[float], winner: str):
        fname = self._get_csv_filename("conflicts")
        self._ensure_csv(fname, ["time","res_id","taxi_candidates","remaining_caps","distances", "winner"])
        self._write(fname, dict(
            time=float(t), res_id=str(res_id),
            taxi_candidates="|".join(map(str, taxi_candidates)),
            remaining_caps="|".join(map(str, remaining_caps)),
            distances="|".join(map(str, distances)),
            winner=str(winner),
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
        self._ensure_csv(fname, ["time","taxi","reward","capacity","step","missed_deadline","wait_at_pickups","completion", "nonserved"])

        terms_round = {k: round(float(v),2) for k,v in terms.items()}

        self._write(fname, dict(
            time=float(t), taxi=str(taxi), reward=round(float(reward),2),
            capacity=terms_round["capacity"],
            step=terms_round["step"],
            missed_deadline=terms_round["missed_deadline"],
            wait_at_pickups=terms_round["wait_at_pickups"],
            completion=terms_round["completion"],
            nonserved=terms_round["nonserved"],
        ))

    def log_macro_step(self, info):
        fname = self._get_csv_filename("rewards_macro")
        self._ensure_csv(fname, 
                         ["macro_steps","reward","capacity_avg","step_avg","missed_deadline_avg", "wait_avg", "completion_avg", "nonserved_avg"])

        missed_deadline_avg = info.get("macro_missed_deadline", info.get("macro_abandoned"))
        self._write(fname, dict(
                macro_steps = info["macro_steps"],
                reward = info["macro_reward"],
                capacity_avg = info["macro_capacity"],
                step_avg = info["macro_step"],
                missed_deadline_avg = missed_deadline_avg,
                wait_avg = info["macro_wait"],
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

    # ---------- plotting ----------
    def _plot_ts(self, png_name: str, series: List[tuple], ylabel: str = ""):
        # Plotting disabled
        return
