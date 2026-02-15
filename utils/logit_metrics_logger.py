"""
Utilities for logging and analyzing logit dynamics during evaluation.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class LogitStepMetrics:
    """Metrics for a single decision step's logits."""
    step: int = 0
    best_cand_logit: float = 0.0
    noop_logit: float = 0.0
    margin: float = 0.0  # best_cand - noop
    top1_top2_margin: float = 0.0  # top1 - top2 (among candidates)
    num_valid_candidates: int = 0


@dataclass
class LogitEpisodeMetrics:
    """Aggregated logit metrics for an entire episode."""
    policy: str = ""  # deterministic or stochastic
    seed: int = 0
    ts: int = 0  # training timestep
    
    # Mean values across all decision steps
    mean_best_cand_logit: float = 0.0
    mean_noop_logit: float = 0.0
    mean_margin: float = 0.0
    mean_top1_top2_margin: float = 0.0
    
    # Std dev values
    std_best_cand_logit: float = 0.0
    std_noop_logit: float = 0.0
    std_margin: float = 0.0
    std_top1_top2_margin: float = 0.0
    
    # Additional statistics
    mean_num_valid_candidates: float = 0.0
    num_decision_steps: int = 0
    
    # Raw step data (not logged, used for aggregation)
    step_metrics: List[LogitStepMetrics] = field(default_factory=list)


def compute_logit_step_metrics(logits: np.ndarray, mask: np.ndarray, noop_logit_value: float) -> LogitStepMetrics:
    """
    Compute logit metrics for a single decision step.
    
    Args:
        logits: Logits for all robots and candidates [R, K_max+1], where last column is noop
        mask: Boolean mask [R, K_max+1] indicating valid candidates
        noop_logit_value: The scalar noop logit value used
        
    Returns:
        LogitStepMetrics for this step
    """
    metrics = LogitStepMetrics()
    
    # Extract candidate logits (excluding noop column)
    cand_logits = logits[:, :-1]  # [R, K_max]
    cand_mask = mask[:, :-1]  # [R, K_max]
    
    # Find best candidate logit across all robots
    valid_cand_logits = cand_logits[cand_mask]
    if len(valid_cand_logits) > 0:
        metrics.best_cand_logit = float(np.max(valid_cand_logits))
        metrics.num_valid_candidates = len(valid_cand_logits)
        
        # Compute top1-top2 margin
        sorted_logits = np.sort(valid_cand_logits)[::-1]  # descending
        if len(sorted_logits) >= 2:
            metrics.top1_top2_margin = float(sorted_logits[0] - sorted_logits[1])
        else:
            metrics.top1_top2_margin = 0.0
    else:
        metrics.best_cand_logit = float('-inf')
        metrics.num_valid_candidates = 0
        metrics.top1_top2_margin = 0.0
    
    metrics.noop_logit = noop_logit_value
    metrics.margin = metrics.best_cand_logit - metrics.noop_logit
    
    return metrics


def aggregate_episode_logit_metrics(step_metrics: List[LogitStepMetrics], 
                                     policy: str, seed: int, ts: int) -> LogitEpisodeMetrics:
    """
    Aggregate step-level logit metrics into episode-level statistics.
    
    Args:
        step_metrics: List of LogitStepMetrics for each decision step
        policy: Policy type (deterministic or stochastic)
        seed: Random seed used
        ts: Training timestep of the model
        
    Returns:
        LogitEpisodeMetrics with aggregated statistics
    """
    ep_metrics = LogitEpisodeMetrics(policy=policy, seed=seed, ts=ts)
    ep_metrics.step_metrics = step_metrics
    ep_metrics.num_decision_steps = len(step_metrics)
    
    if len(step_metrics) == 0:
        return ep_metrics
    
    # Extract arrays for each metric
    best_cand_logits = np.array([m.best_cand_logit for m in step_metrics if np.isfinite(m.best_cand_logit)])
    noop_logits = np.array([m.noop_logit for m in step_metrics])
    margins = np.array([m.margin for m in step_metrics if np.isfinite(m.margin)])
    top1_top2_margins = np.array([m.top1_top2_margin for m in step_metrics])
    num_valid_cands = np.array([m.num_valid_candidates for m in step_metrics])
    
    # Compute means
    ep_metrics.mean_best_cand_logit = float(np.mean(best_cand_logits)) if len(best_cand_logits) > 0 else 0.0
    ep_metrics.mean_noop_logit = float(np.mean(noop_logits)) if len(noop_logits) > 0 else 0.0
    ep_metrics.mean_margin = float(np.mean(margins)) if len(margins) > 0 else 0.0
    ep_metrics.mean_top1_top2_margin = float(np.mean(top1_top2_margins)) if len(top1_top2_margins) > 0 else 0.0
    ep_metrics.mean_num_valid_candidates = float(np.mean(num_valid_cands)) if len(num_valid_cands) > 0 else 0.0
    
    # Compute std devs
    ep_metrics.std_best_cand_logit = float(np.std(best_cand_logits)) if len(best_cand_logits) > 0 else 0.0
    ep_metrics.std_noop_logit = float(np.std(noop_logits)) if len(noop_logits) > 0 else 0.0
    ep_metrics.std_margin = float(np.std(margins)) if len(margins) > 0 else 0.0
    ep_metrics.std_top1_top2_margin = float(np.std(top1_top2_margins)) if len(top1_top2_margins) > 0 else 0.0
    
    return ep_metrics


def get_logit_metrics_header() -> str:
    """Return the header line for logit_metrics.log."""
    return (
        "pol        seed      ts | "
        "best_cand  noop    margin  top1-2  | "
        "std_bcand  std_noop  std_marg  std_t12 | "
        "ncands  nsteps"
    )


def logit_metrics_to_string(metrics: LogitEpisodeMetrics) -> str:
    """Convert LogitEpisodeMetrics to a formatted string for logging."""
    return (
        f"{metrics.policy:<10} {metrics.seed:>4} {metrics.ts:>8} | "
        f"{metrics.mean_best_cand_logit:>9.4f} {metrics.mean_noop_logit:>6.4f} "
        f"{metrics.mean_margin:>9.4f} {metrics.mean_top1_top2_margin:>7.4f} | "
        f"{metrics.std_best_cand_logit:>9.4f} {metrics.std_noop_logit:>9.4f} "
        f"{metrics.std_margin:>9.4f} {metrics.std_top1_top2_margin:>8.4f} | "
        f"{metrics.mean_num_valid_candidates:>6.2f} {metrics.num_decision_steps:>7}"
    )


def ensure_logit_metrics_log(log_path: str) -> None:
    """Ensure the logit metrics log file exists with proper header."""
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write(get_logit_metrics_header() + "\n")


def append_logit_metrics_log(log_path: str, metrics: LogitEpisodeMetrics) -> None:
    """Append a logit metrics entry to the log file."""
    ensure_logit_metrics_log(log_path)
    with open(log_path, 'a') as f:
        f.write(logit_metrics_to_string(metrics) + "\n")
