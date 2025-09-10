# scripts/rp_logger_callback.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RPLoggerCallback(BaseCallback):
    def __init__(self, rp_logger, controller, verbose: int = 0):
        super().__init__(verbose)
        self.rp_logger = rp_logger                # RidepoolLogger
        self.controller = controller              # RLControllerAdapter
        self.ep_idx = 0
        self.sum_reward = 0.0
        self.steps_in_ep = 0

    def _on_training_start(self) -> None:
        # open the first episode log folder
        self.rp_logger.start_episode(self.ep_idx)

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

            # The controller already logs per-step details internally.
            # Here we close the episode and roll to a new one.
            self.rp_logger.end_episode(
                sum_reward=ep_reward,
                n_pickups=0,   # optional: fill from your own counters if you keep them
                n_dropoffs=0,  # optional: fill from your own counters if you keep them
                duration=ep_len,
            )
            self.ep_idx += 1
            self.sum_reward = 0.0
            self.steps_in_ep = 0
            self.rp_logger.start_episode(self.ep_idx)
        return True

    def _on_training_end(self) -> None:
        # close the last episode if training ended without a terminal done
        try:
            self.rp_logger.end_episode(
                sum_reward=self.sum_reward,
                n_pickups=0, n_dropoffs=0, duration=self.steps_in_ep
            )
        except Exception:
            pass
        self.rp_logger.close()
