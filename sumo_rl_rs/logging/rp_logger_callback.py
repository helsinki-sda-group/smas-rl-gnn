# scripts/rp_logger_callback.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from utils.metrics_calculator import (
    compute_episode_metrics_from_logs,
    append_metrics_log,
    ensure_metrics_log,
)

class RPLoggerCallback(BaseCallback):
    def __init__(self, rp_logger, controller, verbose: int = 0, metrics_log_path: str | None = None,
                 num_robots: int | None = None, seed: int = 0, reset_fn = None, save_model_dir: str | None = None):
        super().__init__(verbose)
        self.rp_logger = rp_logger                # RidepoolLogger
        self.controller = controller              # RLControllerAdapter
        self.metrics_log_path = metrics_log_path
        self.num_robots = num_robots
        self.seed = seed
        self.reset_fn = reset_fn  # Optional: RotatingSeedResetFn for dynamic seeds
        self.save_model_dir = save_model_dir      # Directory to save models after each rollout
        self.ep_idx = 0
        self.sum_reward = 0.0
        self.steps_in_ep = 0

    def _on_training_start(self) -> None:
        if self.metrics_log_path:
            ensure_metrics_log(self.metrics_log_path, overwrite=True)
        
        # Create save directory if specified
        if self.save_model_dir:
            os.makedirs(self.save_model_dir, exist_ok=True)
    
    def _on_rollout_end(self) -> None:
        """Called after each rollout collection phase."""
        if self.save_model_dir:
            # Save model with episode and timestep information
            model_filename = f"model_episode{self.ep_idx}_ts{self.num_timesteps}.zip"
            model_path = os.path.join(self.save_model_dir, model_filename)
            self.model.save(model_path)
            # Log noop_logit value
            try:
                noop_logit = self.model.policy.noop_logit.item()
                log_path = os.path.join(self.save_model_dir, "noop_logit.log")
                with open(log_path, "a") as log_file:
                    log_file.write(f"episode={self.ep_idx}, ts={self.num_timesteps}, noop_logit={noop_logit}\n")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[WARN] Could not log noop_logit: {e}")
            if self.verbose > 0:
                print(f"Model saved to {model_path}")
        return True

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
                # Set ts (timesteps) field for training log
                if hasattr(metrics, '__dict__'):
                    metrics.ts = getattr(self, 'num_timesteps', 0)
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
