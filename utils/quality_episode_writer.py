"""
Persistent CSV writer for episode-level quality metrics.

Appends flat rows to quality_episode_metrics.csv in the run directory.
Optionally also appends to task_quality_events.csv and decision_quality_events.csv.

All files live in run_dir (not episode_dir) so they survive episode directory pruning.
"""
from __future__ import annotations

import csv
import os
import numbers
from typing import Any


class QualityEpisodeWriter:
    """Appends episode quality rows to persistent CSVs in run_dir."""

    METRICS_FILE = "quality_episode_metrics.csv"
    TASK_EVENTS_FILE = "task_quality_events.csv"
    DECISION_EVENTS_FILE = "decision_quality_events.csv"
    GROUP_FILES = {
        "reward": "quality_episode_reward_metrics.csv",
        "task": "quality_episode_task_metrics.csv",
        "pool": "quality_episode_pool_metrics.csv",
        "decision": "quality_episode_decision_metrics.csv",
        "candidate": "quality_episode_candidate_metrics.csv",
        "conflict": "quality_episode_conflict_metrics.csv",
    }
    META_FIELDS = [
        "config_id", "run_id", "ts", "episode",
        "reward_type", "completion_mode", "num_robots", "max_robot_capacity",
    ]

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def append_episode(
        self,
        flat_row: dict[str, Any],
        task_events: list[dict],
        decision_events: list[dict],
    ) -> None:
        """Append one episode's data to the persistent CSVs.

        Parameters
        ----------
        flat_row : dict
            Single flat metrics dict (output of compute_quality_episode_metrics[0]).
        task_events : list[dict]
            Per-task rows. Written to task_quality_events.csv if non-empty.
        decision_events : list[dict]
            Per-decision rows. Written to decision_quality_events.csv if non-empty.
        """
        if flat_row:
            rounded_row = self._round_row(flat_row)
            self._append_csv(self.METRICS_FILE, rounded_row)
            self._append_grouped_metrics(rounded_row)

        # Always ensure separate files exist, even if this episode has no events.
        task_default_fields = [
            "config_id", "run_id", "ts", "episode", "task_id", "was_obsolete",
            "reservation_time", "actual_pickup_time", "actual_dropoff_time",
            "actual_waiting_time", "actual_travel_time",
        ]
        decision_default_fields = [
            "config_id", "run_id", "ts", "episode", "time", "robot_id",
            "selected_task_id", "num_candidates", "is_noop",
        ]
        self._ensure_file(self.TASK_EVENTS_FILE, task_default_fields)
        self._ensure_file(self.DECISION_EVENTS_FILE, decision_default_fields)

        if task_events:
            episode_meta = {k: flat_row.get(k) for k in ("config_id", "run_id", "ts", "episode")}
            for evt in task_events:
                self._append_csv(self.TASK_EVENTS_FILE, self._round_row({**episode_meta, **evt}))

        if decision_events:
            episode_meta = {k: flat_row.get(k) for k in ("config_id", "run_id", "ts", "episode")}
            for evt in decision_events:
                self._append_csv(self.DECISION_EVENTS_FILE, self._round_row({**episode_meta, **evt}))

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _append_csv(self, filename: str, row: dict[str, Any]) -> None:
        """Append a single dict as a CSV row, writing header when file is new."""
        path = os.path.join(self.run_dir, filename)
        is_new = not os.path.isfile(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()), extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def _ensure_file(self, filename: str, fieldnames: list[str]) -> None:
        """Create CSV with header if it does not exist yet."""
        path = os.path.join(self.run_dir, filename)
        if os.path.isfile(path):
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

    def _round_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Round all non-integer numeric values to two decimals."""
        out: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, numbers.Integral):
                out[k] = int(v)
            elif isinstance(v, numbers.Real):
                out[k] = round(float(v), 2)
            else:
                out[k] = v
        return out

    def _append_grouped_metrics(self, row: dict[str, Any]) -> None:
        """Append grouped per-episode aggregate rows split by metric family."""
        grouped_rows = {
            "reward": self._group_row(row, lambda key: key.startswith("rew_") or key.startswith("macro_")),
            "task": self._group_row(row, lambda key: key.startswith("task_")),
            "pool": self._group_row(row, lambda key: key.startswith("pool_")),
            "decision": self._group_row(row, lambda key: key.startswith("dec_")),
            "candidate": self._group_row(row, lambda key: key.startswith("cand_")),
            "conflict": self._group_row(row, lambda key: key.startswith("conf_")),
        }
        for group_name, grouped_row in grouped_rows.items():
            self._append_csv(self.GROUP_FILES[group_name], grouped_row)

    def _group_row(self, row: dict[str, Any], predicate) -> dict[str, Any]:
        """Return a grouped row with shared meta fields plus matching metrics."""
        grouped = {field: row.get(field) for field in self.META_FIELDS if field in row}
        for key, value in row.items():
            if key in grouped:
                continue
            if predicate(key):
                grouped[key] = value
        return grouped
