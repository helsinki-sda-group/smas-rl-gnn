# scripts/rp_logger_callback.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import numpy as np
import os
import shutil
from stable_baselines3.common.callbacks import BaseCallback
from utils.metrics_calculator import (
    compute_episode_metrics_from_logs,
    append_metrics_log,
    ensure_metrics_log,
)
from utils.logit_metrics_logger import (
    compute_logit_step_metrics,
    aggregate_episode_logit_metrics,
    append_logit_metrics_log,
    ensure_logit_metrics_log,
)

class RPLoggerCallback(BaseCallback):
    def __init__(self, rp_logger, controller, verbose: int = 0, metrics_log_path: str | None = None,
                 num_robots: int | None = None, seed: int = 0, reset_fn = None, save_model_dir: str | None = None,
                 logit_metrics_log_path: str | None = None, continue_training: bool = False,
                 config_id: str = ""):
        super().__init__(verbose)
        self.rp_logger = rp_logger                # RidepoolLogger
        self.controller = controller              # RLControllerAdapter
        self.metrics_log_path = metrics_log_path
        self.logit_metrics_log_path = logit_metrics_log_path
        self.num_robots = num_robots
        self.seed = seed
        self.reset_fn = reset_fn  # Optional: RotatingSeedResetFn for dynamic seeds
        self.save_model_dir = save_model_dir      # Directory to save models after each rollout
        self.continue_training = continue_training
        self.config_id = config_id
        self.ep_idx = 0
        self.sum_reward = 0.0
        self.steps_in_ep = 0
        self.logit_step_metrics = []
        self._edge_sum = np.zeros((3,), dtype=np.float64)
        self._edge_sumsq = np.zeros((3,), dtype=np.float64)
        self._edge_min = np.full((3,), np.inf, dtype=np.float64)
        self._edge_max = np.full((3,), -np.inf, dtype=np.float64)
        self._edge_count = 0
        self._edge_logged = False
        self.comp_norms_log_path = os.path.abspath(os.path.join(os.getcwd(), "comp_norms.log"))

    def _on_training_start(self) -> None:
        if self.metrics_log_path:
            ensure_metrics_log(self.metrics_log_path, overwrite=not self.continue_training)
        if self.logit_metrics_log_path:
            ensure_logit_metrics_log(self.logit_metrics_log_path, overwrite=not self.continue_training)

        if not self.continue_training and os.path.exists(self.comp_norms_log_path):
            try:
                os.remove(self.comp_norms_log_path)
            except Exception:
                pass
        
        # Create save directory if specified
        if self.save_model_dir:
            os.makedirs(self.save_model_dir, exist_ok=True)
    
    def _on_rollout_end(self) -> None:
        """Called after each rollout collection phase."""
        if not self._edge_logged and self._edge_count > 0:
            mean = self._edge_sum / float(self._edge_count)
            var = (self._edge_sumsq / float(self._edge_count)) - np.square(mean)
            std = np.sqrt(np.maximum(var, 0.0))
            min_v = self._edge_min
            max_v = self._edge_max
            print(
                "[EdgeAttr rollout] count={} mean={} std={} min={} max={}".format(
                    int(self._edge_count),
                    np.round(mean, 6).tolist(),
                    np.round(std, 6).tolist(),
                    np.round(min_v, 6).tolist(),
                    np.round(max_v, 6).tolist(),
                )
            )
            self._edge_logged = True
        self._edge_sum[:] = 0.0
        self._edge_sumsq[:] = 0.0
        self._edge_min[:] = np.inf
        self._edge_max[:] = -np.inf
        self._edge_count = 0
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

        try:
            gnn_ac = getattr(self.model.policy, "gnn_ac", None)
            stats = gnn_ac.pop_comp_norm_stats() if gnn_ac is not None else None
            if stats is not None:
                file_exists = os.path.exists(self.comp_norms_log_path)
                with open(self.comp_norms_log_path, "a") as f:
                    if not file_exists:
                        f.write(
                            "ts,ep,norm_h,norm_z,p_has_comp,logit_base,logit_comp,logit_ind,"
                            "bias_base,norm_w_h,norm_w_c,norm_w_s,norm_w_d,attn_entropy,max_attn,"
                            "ratio_comp_base,ratio_comp_gap,norm_u,std_comp,mean_num_comp,max_num_comp,"
                            "mean_score,std_score,count\n"
                        )
                    f.write(
                        f"{int(self.num_timesteps)},{int(self.ep_idx)},"
                        f"{stats['norm_h']:.3f},{stats['norm_z']:.3f},{stats['p_has_comp']:.3f},"
                        f"{stats['logit_base']:.3f},{stats['logit_comp']:.3f},{stats['logit_ind']:.3f},"
                        f"{stats['bias_base']:.3f},{stats['norm_w_h']:.3f},{stats['norm_w_c']:.3f},"
                        f"{stats['norm_w_s']:.3f},{stats['norm_w_d']:.3f},{stats['attn_entropy']:.3f},"
                        f"{stats['max_attn']:.3f},{stats['ratio_comp_base']:.3f},{stats['ratio_comp_gap']:.3f},"
                        f"{stats['norm_u']:.3f},{stats['std_comp']:.3f},{stats['mean_num_comp']:.3f},"
                        f"{stats['max_num_comp']:.3f},{stats['mean_score']:.3f},{stats['std_score']:.3f},"
                        f"{int(stats['count'])}\n"
                    )
        except Exception as e:
            if self.verbose > 0:
                print(f"[WARN] Could not log comp norms: {e}")
        return True

    def _on_step(self) -> bool:
        # rewards is shape (n_envs,), we assume n_envs=1 unless you set otherwise
        rews = self.locals.get("rewards", None)
        if rews is not None:
            self.sum_reward += float(np.sum(rews))
        self.steps_in_ep += 1

        if self.logit_metrics_log_path:
            obs = self.locals.get("new_obs", None)
            if obs is None:
                obs = self.locals.get("obs", None)
            if obs is not None:
                try:
                    import torch as th
                    with th.no_grad():
                        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                        _ = self.model.policy.extract_features(
                            obs_tensor,
                            features_extractor=self.model.policy.features_extractor,
                        )
                        obs_dict_b = self.model.policy.features_extractor.last_obs
                        logits_k, _ = self.model.policy._build_batch_outputs(obs_dict_b)
                        mask_k = obs_dict_b["cand_mask"]
                        logits, mask = self.model.policy._append_noop(logits_k, mask_k)

                        logits_np = logits.squeeze(0).detach().cpu().numpy()
                        mask_np = mask.squeeze(0).detach().cpu().numpy()
                        noop_logit_value = float(self.model.policy.noop_logit.item())

                        step_metrics = compute_logit_step_metrics(logits_np, mask_np, noop_logit_value)
                        step_metrics.step = self.steps_in_ep - 1
                        self.logit_step_metrics.append(step_metrics)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"[WARN] Could not capture logit metrics: {e}")
        if not self._edge_logged:
            obs = self.locals.get("new_obs", None)
            if obs is None:
                obs = self.locals.get("obs", None)
            if obs is not None and isinstance(obs, dict):
                edge_attr = obs.get("edge_attr", None)
                if edge_attr is not None:
                    try:
                        if hasattr(edge_attr, "detach"):
                            edge_attr = edge_attr.detach().cpu().numpy()
                        ea = np.asarray(edge_attr)
                        if ea.ndim >= 2:
                            slice_ea = ea[..., 0:3].reshape(-1, 3)
                            if slice_ea.size > 0:
                                self._edge_sum += slice_ea.sum(axis=0)
                                self._edge_sumsq += np.square(slice_ea).sum(axis=0)
                                self._edge_min = np.minimum(self._edge_min, slice_ea.min(axis=0))
                                self._edge_max = np.maximum(self._edge_max, slice_ea.max(axis=0))
                                self._edge_count += slice_ea.shape[0]
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"[WARN] Could not compute edge_attr stats: {e}")

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
                metrics.ts = getattr(self, 'num_timesteps', 0)
                append_metrics_log(self.metrics_log_path, metrics)
            if self.logit_metrics_log_path:
                current_seed = self.seed
                if self.reset_fn and hasattr(self.reset_fn, 'get_current_seed'):
                    current_seed = self.reset_fn.get_current_seed()
                logit_metrics = aggregate_episode_logit_metrics(
                    self.logit_step_metrics,
                    policy="train",
                    seed=current_seed,
                    ts=getattr(self, 'num_timesteps', 0),
                )
                append_logit_metrics_log(self.logit_metrics_log_path, logit_metrics)

            # Extended quality diagnostics — must run BEFORE pruning episode dir
            if bool(getattr(self.rp_logger.cfg, "extended_quality_metrics", False)):
                try:
                    from utils.quality_episode_metrics import compute_quality_episode_metrics
                    from utils.quality_episode_writer import QualityEpisodeWriter
                    _episode_dir = getattr(self.rp_logger, "last_ep_dir", None) or self.rp_logger.ep_dir
                    _ep_context = self.controller.get_last_episode_quality_context()
                    _cf_stats = self.rp_logger.get_last_episode_conflict_stats()
                    _config_id = getattr(self, "config_id", "") or str(getattr(self.rp_logger.cfg, "run_name", "") or "")
                    _run_id = str(getattr(self.rp_logger.cfg, "run_name", "") or "")
                    # Always collect separate event-level files for diagnostics.
                    _include_task = True
                    _include_dec = True
                    _flat_row, _task_evts, _dec_evts = compute_quality_episode_metrics(
                        episode_dir=_episode_dir,
                        context=_ep_context,
                        conflict_stats=_cf_stats,
                        config_id=_config_id,
                        run_id=_run_id,
                        ts=getattr(self, "num_timesteps", 0),
                        episode=self.ep_idx,
                        include_task_level=_include_task,
                        include_decision_level=_include_dec,
                    )
                    # Write quality outputs to the current working directory (job dir on Mahti)
                    # so they are co-located with conflicts.log and training_metrics*.log.
                    _writer = QualityEpisodeWriter(os.path.abspath(os.getcwd()))
                    _writer.append_episode(_flat_row, _task_evts, _dec_evts)
                except Exception as _qe:
                    try:
                        _err_path = os.path.join(os.path.abspath(os.getcwd()), "quality_episode_metrics_errors.log")
                        with open(_err_path, "a", encoding="utf-8") as _fh:
                            _fh.write(
                                f"episode={int(self.ep_idx)} ts={int(getattr(self, 'num_timesteps', 0))} "
                                f"episode_dir={_episode_dir!s} error={type(_qe).__name__}: {_qe}\n"
                            )
                    except Exception:
                        pass
                    if self.verbose > 0:
                        print(f"[WARN] quality_episode_metrics failed: {_qe}")

            # Optional inode-saving mode for HPC: remove per-episode folder only
            # after run-level metrics have been appended.
            if bool(getattr(self.rp_logger.cfg, "prune_episode_dir_after_metrics", False)):
                episode_dir = getattr(self.rp_logger, "last_ep_dir", None) or self.rp_logger.ep_dir
                if episode_dir and os.path.isdir(episode_dir):
                    try:
                        shutil.rmtree(episode_dir, ignore_errors=True)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"[WARN] Could not remove episode dir {episode_dir}: {e}")
            self.ep_idx += 1
            self.sum_reward = 0.0
            self.steps_in_ep = 0
            self.logit_step_metrics = []
        return True

    def _on_training_end(self) -> None:
        try:
            self.rp_logger.close()
        except Exception:
            pass
