# sumo_rl_rs/logging/ridepool_logger.py
from __future__ import annotations
import os, csv, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Any
import matplotlib.pyplot as plt
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
    # NEW: clean-up switches
    erase_run_dir_on_start: bool = False         # if True: delete the whole run directory at construction
    erase_episode_dir_on_start: bool = False     # if True: delete an episode dir when starting it

class RidepoolLogger:
    """
    One logger to rule them all:
      - dispatch.csv          : one row per taxi per step with prev/base/seq and raw currentCustomers
      - conflicts.csv         : one row per reservation when conflict happened
      - candidates.csv        : one row per taxi per step (candidate slots, reservation ids)
      - rewards.csv           : one row per taxi per step with reward decomposition
      - fleet_counts.csv      : one row per step with counters (idle, en_route, occupied, pickup_occupied)
      - episode_totals.csv    : summary per episode
      - plots/*.png           : a few basic plots
    """
    def __init__(self, cfg: RidepoolLogConfig):
        self.cfg = cfg
        stamp = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(cfg.out_dir, stamp)

        # --- NEW: optionally nuke the whole run directory on start ---
        if cfg.erase_run_dir_on_start and os.path.isdir(self.run_dir):
            if cfg.console_debug:
                print(f"[RidepoolLogger] Removing existing run dir: {self.run_dir}")
            shutil.rmtree(self.run_dir, ignore_errors=True)

        os.makedirs(self.run_dir, exist_ok=True)

        self.ep_dir: Optional[str] = None
        self._counters = {"writes": 0}
        self._files: Dict[str, Any] = {}
        self._writers: Dict[str, csv.DictWriter] = {}
        # step-local timeseries for plotting
        self._ts: Dict[str, List[float]] = {
            "idle": [], "en_route": [], "occupied": [], "pickup_occupied": [],
            "sum_reward": [], "pickups": [], "dropoffs": []
        }

    # ---------- lifecycle ----------
    def start_episode(self, episode_idx: Optional[int] = None):
        if episode_idx is not None:
            self.cfg.episode_index = int(episode_idx)
        ep_name = f"episode_{self.cfg.episode_index:04d}"
        self.ep_dir = os.path.join(self.run_dir, ep_name)

        # --- NEW: optionally nuke the episode dir on start ---
        if self.cfg.erase_episode_dir_on_start and os.path.isdir(self.ep_dir):
            if self.cfg.console_debug:
                print(f"[RidepoolLogger] Removing existing episode dir: {self.ep_dir}")
            shutil.rmtree(self.ep_dir, ignore_errors=True)

        os.makedirs(self.ep_dir, exist_ok=True)

        self._open_csv("dispatch.csv", ["time","taxi","prev_seq","base_ids","seq","seq_pd","raw_currentCustomers","notes"])
        self._open_csv("conflicts.csv", ["time","res_id","taxi_candidates","remaining_caps","winner"])
        self._open_csv("candidates.csv", ["time","taxi","cand_slots","cand_res_ids","cand_persons","cand_pd_seq"])
        self._open_csv("rewards.csv", ["time","taxi","reward","capacity","step","abandoned","wait_at_pickups","completion", "nonserved"])
        self._open_csv("fleet_counts.csv", ["time","idle","en_route","occupied","pickup_occupied"])
        self._open_csv("episode_totals.csv", ["episode","sum_reward","n_pickups","n_dropoffs","duration"])
        # reset timeseries
        for k in self._ts:
            self._ts[k].clear()

    def end_episode(self, sum_reward: float, n_pickups: int, n_dropoffs: int, duration: float):
        self._write("episode_totals.csv", dict(
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
        self._ensure_csv("dispatch.csv",
                        ["time","taxi","prev_seq","base_ids","seq","seq_pd","raw_currentCustomers","notes"])
        self._write("dispatch.csv", dict(
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
                     remaining_caps: Sequence[int], winner: str):
        self._ensure_csv("conflicts.csv", ["time","res_id","taxi_candidates","remaining_caps","winner"])
        self._write("conflicts.csv", dict(
            time=float(t), res_id=str(res_id),
            taxi_candidates="|".join(map(str, taxi_candidates)),
            remaining_caps="|".join(map(str, remaining_caps)),
            winner=str(winner),
        ))

    def log_candidates(self, t: float, taxi: str, cand_slots, cand_res_ids, cand_persons, cand_pd_seq):
        """
        cand_slots:   list[int]
        cand_res_ids: list[str]
        cand_persons: list[str]  (per candidate: 'p1+p2' or '')
        cand_pd_seq:  list[str]  (per candidate: 'p1:PU+p2:PU+p1:DO+p2:DO' or '')
        """
        self._ensure_csv("candidates.csv",
                        ["time","taxi","cand_slots","cand_res_ids","cand_persons","cand_pd_seq"])
        self._write("candidates.csv", dict(
            time=float(t),
            taxi=str(taxi),
            cand_slots="|".join(map(str, cand_slots)),
            cand_res_ids="|".join(map(str, cand_res_ids)),
            cand_persons="|".join(cand_persons),
            cand_pd_seq="|".join(cand_pd_seq),
        ))

    def log_rewards(self, t: float, taxi: str, reward: float, terms: Dict[str, float]):
        self._ensure_csv("rewards.csv", ["time","taxi","reward","capacity","step","abandoned","wait_at_pickups","completion", "nonserved"])
        self._write("rewards.csv", dict(
            time=float(t), taxi=str(taxi), reward=float(reward),
            capacity=float(terms.get("capacity", 0.0)),
            step=float(terms.get("step", 0.0)),
            abandoned=float(terms.get("abandoned", 0.0)),
            wait_at_pickups=float(terms.get("wait_at_pickups", 0.0)),
            completion=float(terms.get("completion", 0.0)),
            nonserved=float(terms.get("nonserved", 0.0)),
        ))

    def log_fleet_counts(self, t: float, idle: int, en_route: int, occupied: int, pickup_occupied: int):
        self._ensure_csv("fleet_counts.csv", ["time","idle","en_route","occupied","pickup_occupied"])
        self._write("fleet_counts.csv", dict(
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
            self._ensure_csv("debug.csv", ["time", "tag", "data"])
            self._write("debug.csv", dict(
                time=float(tnow),
                tag=str(tag),
                data=json.dumps(payload, ensure_ascii=False)
            ))
        except Exception as e:
            print(f"[logger] log_debug failed: {e}")


    # ---------- plotting ----------
    def _plot_ts(self, png_name: str, series: List[tuple], ylabel: str = ""):
        if not self._ts or not any(self._ts.get(k, []) for k, _ in series):
            return
        plt.figure(figsize=(10, 5))
        for key, label in series:
            if self._ts.get(key):
                xs = list(range(len(self._ts[key])))
                ys = self._ts[key]
                plt.plot(xs, ys, label=label)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(png_name.replace("_", " ")[:-4])
        plt.grid(True)
        plt.legend()
        ep_dir = self.ep_dir
        if ep_dir is None:
            return
        os.makedirs(os.path.join(ep_dir, "plots"), exist_ok=True)
        plt.savefig(os.path.join(ep_dir, "plots", png_name))
        plt.close()
