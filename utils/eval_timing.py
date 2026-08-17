from __future__ import annotations

import csv
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


TIMING_PROTOCOL = "shared_outer_v1"
TIMING_INFERENCE_MODE_ACTOR_CRITIC = "actor_critic"
TIMING_EDGE_COUNT_CONVENTION = "pre_encoder_no_self_loops"

STEP_COLUMNS: List[str] = [
    "method",
    "policy",
    "proposer",
    "resolver",
    "protocol",
    "inference_mode",
    "seed",
    "episode",
    "decision_idx",
    "sim_time_s",
    "device",
    "warmup",
    "n_robots",
    "n_tasks",
    "n_candidate_pairs",
    "n_nonempty_robots",
    "n_proposals",
    "n_bid_tasks",
    "n_conflicting_tasks",
    "n_conflicting_task_proposals",
    "n_actor_graphs",
    "n_actor_nonempty_graphs",
    "n_actor_nodes",
    "n_actor_edges",
    "n_actor_candidates",
    "env_pre_controller_ns",
    "pre_step_sync_ns",
    "proposal_ns",
    "resolution_ns",
    "commit_dispatch_ns",
    "simulation_ns",
    "post_step_logging_ns",
    "decision_total_ns",
    "other_ns",
    "gnn_obs_build_ns",
    "candidate_filter_ns",
    "gnn_tensor_prepare_ns",
    "gnn_graph_prepare_ns",
    "gnn_policy_total_ns",
    "gnn_actor_ns",
    "gnn_critic_ns",
    "gnn_action_ns",
    "gnn_action_mapping_ns",
    "gnn_actor_amortized_robot_ns",
    "gnn_actor_path_accounted_amortized_robot_ns",
    "gnn_actor_only_proposal_est_amortized_robot_ns",
]


def now_ns() -> int:
    import time

    return time.perf_counter_ns()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _percentiles_ns_to_ms(values_ns: np.ndarray) -> Dict[str, float]:
    if values_ns.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    as_float = values_ns.astype(np.float64)
    return {
        "p50": float(np.percentile(as_float, 50)) / 1_000_000.0,
        "p90": float(np.percentile(as_float, 90)) / 1_000_000.0,
        "p95": float(np.percentile(as_float, 95)) / 1_000_000.0,
        "max": float(np.max(as_float)) / 1_000_000.0,
    }


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def collect_run_metadata(
    *,
    scenario: str,
    policy: str,
    proposer: str,
    resolver: str,
    protocol: str,
    inference_mode: str,
    seed: int,
    n_robots: int,
    device: str,
    internal_gnn_timing: bool,
) -> Dict[str, Any]:
    torch_version = ""
    pyg_version = ""
    sb3_version = ""
    torch_threads = 0
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_threads = int(torch.get_num_threads())
    except Exception:
        pass
    try:
        import torch_geometric  # type: ignore

        pyg_version = str(torch_geometric.__version__)
    except Exception:
        pass
    try:
        import stable_baselines3 as sb3

        sb3_version = str(sb3.__version__)
    except Exception:
        pass

    cpu_model = ""
    try:
        cpu_model = platform.processor() or ""
    except Exception:
        cpu_model = ""

    return {
        "scenario": str(scenario),
        "policy": str(policy),
        "proposer": str(proposer),
        "resolver": str(resolver),
        "protocol": str(protocol),
        "inference_mode": str(inference_mode),
        "seed": int(seed),
        "n_robots": int(n_robots),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch_version,
        "torch_geometric_version": pyg_version,
        "stable_baselines3_version": sb3_version,
        "device": str(device),
        "torch_num_threads": int(torch_threads),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "host_name": socket.gethostname(),
        "cpu_model": cpu_model,
        "git_commit": _git_commit_hash(),
        "internal_gnn_timing": int(bool(internal_gnn_timing)),
        "actor_edge_count_convention": TIMING_EDGE_COUNT_CONVENTION,
        "actor_graphs_include_empty_candidates": 1,
        "timing_protocol": TIMING_PROTOCOL,
    }


@dataclass
class TimingConfig:
    enabled: bool = False
    internal_gnn: bool = True
    warmup_episodes: int = 0


class TimingRunCollector:
    def __init__(self, *, config: TimingConfig, static_meta: Dict[str, Any]):
        self.config = config
        self.static_meta = dict(static_meta)
        self.rows: List[Dict[str, Any]] = []

    @staticmethod
    def _compute_derived(row: Dict[str, Any]) -> None:
        n_graphs = max(0, _safe_int(row.get("n_actor_graphs", 0)))

        proposal_ns = max(0, _safe_int(row.get("proposal_ns", 0)))
        gnn_critic_ns = max(0, _safe_int(row.get("gnn_critic_ns", 0)))
        gnn_actor_ns = max(0, _safe_int(row.get("gnn_actor_ns", 0)))
        gnn_obs_build_ns = max(0, _safe_int(row.get("gnn_obs_build_ns", 0)))
        gnn_tensor_prepare_ns = max(0, _safe_int(row.get("gnn_tensor_prepare_ns", 0)))
        gnn_graph_prepare_ns = max(0, _safe_int(row.get("gnn_graph_prepare_ns", 0)))
        gnn_action_ns = max(0, _safe_int(row.get("gnn_action_ns", 0)))
        gnn_action_mapping_ns = max(0, _safe_int(row.get("gnn_action_mapping_ns", 0)))

        if n_graphs <= 0:
            row["gnn_actor_amortized_robot_ns"] = 0
            row["gnn_actor_path_accounted_amortized_robot_ns"] = 0
            row["gnn_actor_only_proposal_est_amortized_robot_ns"] = 0
            return

        row["gnn_actor_amortized_robot_ns"] = int(round(gnn_actor_ns / n_graphs))

        actor_path_accounted_ns = (
            gnn_obs_build_ns
            + gnn_tensor_prepare_ns
            + gnn_graph_prepare_ns
            + gnn_actor_ns
            + gnn_action_ns
            + gnn_action_mapping_ns
        )
        row["gnn_actor_path_accounted_amortized_robot_ns"] = int(
            round(actor_path_accounted_ns / n_graphs)
        )

        actor_only_proposal_est_ns = max(0, proposal_ns - gnn_critic_ns)
        row["gnn_actor_only_proposal_est_amortized_robot_ns"] = int(
            round(actor_only_proposal_est_ns / n_graphs)
        )

    def add_step(self, row: Dict[str, Any]) -> None:
        if not self.config.enabled:
            return

        merged = dict(self.static_meta)
        merged.update(dict(row))

        # Normalize required fields
        for key in STEP_COLUMNS:
            if key not in merged:
                merged[key] = 0

        merged["warmup"] = int(_safe_int(merged.get("warmup", 0)) != 0)
        merged["seed"] = _safe_int(merged.get("seed", 0))
        merged["episode"] = _safe_int(merged.get("episode", 0))
        merged["decision_idx"] = _safe_int(merged.get("decision_idx", 0))
        merged["sim_time_s"] = _safe_float(merged.get("sim_time_s", 0.0))

        for ns_key in [
            "env_pre_controller_ns",
            "pre_step_sync_ns",
            "proposal_ns",
            "resolution_ns",
            "commit_dispatch_ns",
            "simulation_ns",
            "post_step_logging_ns",
            "decision_total_ns",
            "other_ns",
            "gnn_obs_build_ns",
            "gnn_tensor_prepare_ns",
            "gnn_graph_prepare_ns",
            "gnn_policy_total_ns",
            "gnn_actor_ns",
            "gnn_critic_ns",
            "gnn_action_ns",
            "gnn_action_mapping_ns",
        ]:
            merged[ns_key] = max(0, _safe_int(merged.get(ns_key, 0)))

        if "other_ns" not in row:
            merged["other_ns"] = max(
                0,
                _safe_int(merged.get("decision_total_ns", 0))
                - _safe_int(merged.get("env_pre_controller_ns", 0))
                - _safe_int(merged.get("pre_step_sync_ns", 0))
                - _safe_int(merged.get("proposal_ns", 0))
                - _safe_int(merged.get("resolution_ns", 0))
                - _safe_int(merged.get("commit_dispatch_ns", 0))
                - _safe_int(merged.get("simulation_ns", 0))
                - _safe_int(merged.get("post_step_logging_ns", 0))
            )

        self._compute_derived(merged)
        self.rows.append(merged)

    def write_steps_csv(self, run_dir: str) -> str:
        if not self.config.enabled:
            return ""

        path = Path(run_dir) / "timing_steps.csv"
        return self.write_steps_csv_path(str(path))

    def write_steps_csv_path(self, csv_path: str) -> str:
        if not self.config.enabled:
            return ""

        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STEP_COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, 0) for k in STEP_COLUMNS})
        return str(path)

    def _group_key(self, row: Dict[str, Any]) -> tuple:
        return (
            str(row.get("method", "")),
            str(row.get("policy", "")),
            str(row.get("proposer", "")),
            str(row.get("resolver", "")),
            str(row.get("protocol", "")),
            str(row.get("inference_mode", "")),
            _safe_int(row.get("seed", 0)),
            _safe_int(row.get("episode", 0)),
            str(row.get("device", "")),
        )

    def _group_rows(self) -> Dict[tuple, List[Dict[str, Any]]]:
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for row in self.rows:
            key = self._group_key(row)
            groups.setdefault(key, []).append(row)
        return groups

    def write_summary_csv(self, run_dir: str) -> str:
        if not self.config.enabled:
            return ""

        path = Path(run_dir) / "timing_summary.csv"
        return self.write_summary_csv_path(str(path))

    def write_summary_csv_path(self, csv_path: str) -> str:
        if not self.config.enabled:
            return ""

        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        groups = self._group_rows()

        summary_rows: List[Dict[str, Any]] = []
        for key, rows in groups.items():
            non_warmup = [r for r in rows if _safe_int(r.get("warmup", 0)) == 0]
            warmup_count = len(rows) - len(non_warmup)
            measured = non_warmup

            def ns_array(col: str) -> np.ndarray:
                return np.asarray([_safe_int(r.get(col, 0)) for r in measured], dtype=np.int64)

            prop = ns_array("proposal_ns")
            res = ns_array("resolution_ns")
            commit = ns_array("commit_dispatch_ns")
            sim = ns_array("simulation_ns")
            env_pre = ns_array("env_pre_controller_ns")
            pre_sync = ns_array("pre_step_sync_ns")
            post_log = ns_array("post_step_logging_ns")
            dec = ns_array("decision_total_ns")
            other = ns_array("other_ns")

            row: Dict[str, Any] = {
                "method": key[0],
                "policy": key[1],
                "proposer": key[2],
                "resolver": key[3],
                "protocol": key[4],
                "inference_mode": key[5],
                "seed": key[6],
                "episode": key[7],
                "device": key[8],
                "timing_protocol": TIMING_PROTOCOL,
                "warmup_decisions": int(warmup_count),
                "measured_decisions": int(len(measured)),
            }

            def add_time_stats(prefix: str, values_ns: np.ndarray) -> None:
                total_ns = int(values_ns.sum()) if values_ns.size > 0 else 0
                mean_ns = float(values_ns.mean()) if values_ns.size > 0 else 0.0
                q = _percentiles_ns_to_ms(values_ns)
                row[f"{prefix}_total_ms"] = total_ns / 1_000_000.0
                row[f"{prefix}_mean_ms"] = mean_ns / 1_000_000.0
                row[f"{prefix}_p50_ms"] = q["p50"]
                row[f"{prefix}_p90_ms"] = q["p90"]
                row[f"{prefix}_p95_ms"] = q["p95"]
                row[f"{prefix}_max_ms"] = q["max"]

            add_time_stats("env_pre_controller", env_pre)
            add_time_stats("pre_step_sync", pre_sync)
            add_time_stats("proposal", prop)
            add_time_stats("resolution", res)
            add_time_stats("commit_dispatch", commit)
            add_time_stats("simulation", sim)
            add_time_stats("post_step_logging", post_log)
            add_time_stats("decision", dec)
            add_time_stats("other", other)

            for col in [
                "gnn_obs_build_ns",
                "candidate_filter_ns",
                "gnn_tensor_prepare_ns",
                "gnn_graph_prepare_ns",
                "gnn_policy_total_ns",
                "gnn_actor_ns",
                "gnn_critic_ns",
                "gnn_action_ns",
                "gnn_action_mapping_ns",
            ]:
                vals = ns_array(col)
                row[f"{col[:-3]}_total_ms"] = (int(vals.sum()) / 1_000_000.0) if vals.size > 0 else 0.0
                row[f"{col[:-3]}_mean_ms"] = (float(vals.mean()) / 1_000_000.0) if vals.size > 0 else 0.0

            for count_col in [
                "n_candidate_pairs",
                "n_proposals",
                "n_conflicting_tasks",
                "n_conflicting_task_proposals",
                "n_bid_tasks",
                "n_actor_graphs",
                "n_actor_nonempty_graphs",
                "n_actor_nodes",
                "n_actor_edges",
                "n_actor_candidates",
            ]:
                total = int(sum(_safe_int(r.get(count_col, 0)) for r in measured))
                row[f"{count_col}_total"] = total
                row[f"{count_col}_mean"] = (total / len(measured)) if measured else 0.0

            total_actor_graphs = int(sum(_safe_int(r.get("n_actor_graphs", 0)) for r in measured))
            total_actor_ns = int(sum(_safe_int(r.get("gnn_actor_ns", 0)) for r in measured))

            actor_path_accounted_total_ns = int(
                sum(
                    _safe_int(r.get("gnn_obs_build_ns", 0))
                    + _safe_int(r.get("gnn_tensor_prepare_ns", 0))
                    + _safe_int(r.get("gnn_graph_prepare_ns", 0))
                    + _safe_int(r.get("gnn_actor_ns", 0))
                    + _safe_int(r.get("gnn_action_ns", 0))
                    + _safe_int(r.get("gnn_action_mapping_ns", 0))
                    for r in measured
                )
            )
            actor_only_proposal_est_total_ns = int(
                sum(
                    max(0, _safe_int(r.get("proposal_ns", 0)) - _safe_int(r.get("gnn_critic_ns", 0)))
                    for r in measured
                )
            )

            if total_actor_graphs > 0:
                row["gnn_actor_amortized_robot_ms"] = total_actor_ns / total_actor_graphs / 1_000_000.0
                row["gnn_actor_path_accounted_amortized_robot_ms"] = (
                    actor_path_accounted_total_ns / total_actor_graphs / 1_000_000.0
                )
                row["gnn_actor_only_proposal_est_amortized_robot_ms"] = (
                    actor_only_proposal_est_total_ns / total_actor_graphs / 1_000_000.0
                )
                row["mean_actor_nodes_per_graph"] = (
                    row["n_actor_nodes_total"] / max(1, total_actor_graphs)
                )
                row["mean_actor_edges_per_graph"] = (
                    row["n_actor_edges_total"] / max(1, total_actor_graphs)
                )
                row["mean_actor_candidates_per_graph"] = (
                    row["n_actor_candidates_total"] / max(1, total_actor_graphs)
                )
            else:
                row["gnn_actor_amortized_robot_ms"] = 0.0
                row["gnn_actor_path_accounted_amortized_robot_ms"] = 0.0
                row["gnn_actor_only_proposal_est_amortized_robot_ms"] = 0.0
                row["mean_actor_nodes_per_graph"] = 0.0
                row["mean_actor_edges_per_graph"] = 0.0
                row["mean_actor_candidates_per_graph"] = 0.0

            per_decision_actor = np.asarray(
                [_safe_int(r.get("gnn_actor_amortized_robot_ns", 0)) for r in measured], dtype=np.int64
            )
            per_decision_accounted = np.asarray(
                [_safe_int(r.get("gnn_actor_path_accounted_amortized_robot_ns", 0)) for r in measured],
                dtype=np.int64,
            )
            per_decision_broad = np.asarray(
                [_safe_int(r.get("gnn_actor_only_proposal_est_amortized_robot_ns", 0)) for r in measured],
                dtype=np.int64,
            )

            for prefix, vals in [
                ("gnn_actor_amortized_robot", per_decision_actor),
                ("gnn_actor_path_accounted_amortized_robot", per_decision_accounted),
                ("gnn_actor_only_proposal_est_amortized_robot", per_decision_broad),
            ]:
                q = _percentiles_ns_to_ms(vals)
                row[f"{prefix}_p50_ms"] = q["p50"]
                row[f"{prefix}_p90_ms"] = q["p90"]
                row[f"{prefix}_p95_ms"] = q["p95"]
                row[f"{prefix}_max_ms"] = q["max"]

            summary_rows.append(row)

        # Include static metadata in each summary row for reproducibility.
        meta_fields = [
            "scenario",
            "python_version",
            "torch_version",
            "torch_geometric_version",
            "stable_baselines3_version",
            "torch_num_threads",
            "omp_num_threads",
            "mkl_num_threads",
            "host_name",
            "cpu_model",
            "git_commit",
            "internal_gnn_timing",
            "actor_edge_count_convention",
            "actor_graphs_include_empty_candidates",
        ]
        for row in summary_rows:
            for field in meta_fields:
                row[field] = self.static_meta.get(field, "")

        # Stable header with deterministic ordering.
        leading = [
            "method",
            "policy",
            "proposer",
            "resolver",
            "protocol",
            "inference_mode",
            "seed",
            "episode",
            "device",
            "timing_protocol",
            "warmup_decisions",
            "measured_decisions",
        ]
        ordered_tail: List[str] = []
        for row in summary_rows:
            for key_name in row.keys():
                if key_name not in leading and key_name not in ordered_tail:
                    ordered_tail.append(key_name)
        header = leading + ordered_tail

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k, "") for k in header})

        return str(path)


def timing_config_from_opt(opt_obj: Any) -> TimingConfig:
    timing_cfg = getattr(opt_obj, "timing", None)
    enabled = False
    internal_gnn = True
    warmup_episodes = 0
    if timing_cfg is not None:
        enabled = bool(getattr(timing_cfg, "enabled", False))
        internal_gnn = bool(getattr(timing_cfg, "internal_gnn", True))
        warmup_episodes = int(getattr(timing_cfg, "warmup_episodes", 0) or 0)
    return TimingConfig(enabled=enabled, internal_gnn=internal_gnn, warmup_episodes=warmup_episodes)
