# scripts/rp_logger_callback.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from utils.metrics_calculator import (
    compute_episode_metrics_from_logs,
    append_metrics_log,
    ensure_metrics_log,
)

class RPLoggerCallback(BaseCallback):
    def __init__(self, rp_logger, controller, verbose: int = 0, metrics_log_path: str | None = None,
                 num_robots: int | None = None, seed: int = 0, reset_fn = None):
        super().__init__(verbose)
        self.rp_logger = rp_logger                # RidepoolLogger
        self.controller = controller              # RLControllerAdapter
        self.metrics_log_path = metrics_log_path
        self.num_robots = num_robots
        self.seed = seed
        self.reset_fn = reset_fn  # Optional: RotatingSeedResetFn for dynamic seeds
        self.ep_idx = 0
        self.sum_reward = 0.0
        self.steps_in_ep = 0

    def _on_training_start(self) -> None:
        if self.metrics_log_path:
            ensure_metrics_log(self.metrics_log_path)

    def _on_step(self) -> bool:
        # rewards is shape (n_envs,), we assume n_envs=1 unless you set otherwise
        rews = self.locals.get("rewards", None)
        if rews is not None:
            self.sum_reward += float(np.sum(rews))
        self.steps_in_ep += 1

        # detect episode end from dones
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", [])
        if dones is not None and any(dones):
            # (Optional) pull episode length/reward from Monitor, if present
            ep_len = self.steps_in_ep
            ep_reward = self.sum_reward
            if infos:
                for info in infos:
                    epinfo = info.get("episode")
                    if epinfo:
                        ep_len = int(epinfo.get("l", ep_len))
                        ep_reward = float(epinfo.get("r", ep_reward))
                        break

            # Controller handles episode close + CSV flush. We only append metrics.
            if self.metrics_log_path:
                episode_dir = getattr(self.rp_logger, "last_ep_dir", None) or self.rp_logger.ep_dir
                info_for_metrics = infos[0] if infos else {}
                
                # Get current seed from reset_fn if available
                current_seed = self.seed
                if self.reset_fn and hasattr(self.reset_fn, 'get_current_seed'):
                    current_seed = self.reset_fn.get_current_seed()
                
                metrics = compute_episode_metrics_from_logs(
                    episode_dir=episode_dir,
                    episode_info=info_for_metrics,
                    policy=str(self.ep_idx),
                    seed=current_seed,
                    num_robots=self.num_robots,
                )
                append_metrics_log(self.metrics_log_path, metrics)
            self.ep_idx += 1
            self.sum_reward = 0.0
            self.steps_in_ep = 0
        return True

    def _on_training_end(self) -> None:
        try:
            self.rp_logger.close()
        except Exception:
            pass
