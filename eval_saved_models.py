#!/usr/bin/env python
"""
Evaluate all saved models from rp_gnn_debug/saved_models on train and eval seeds.

Features:
- Loads all model_episodeX_tsY.zip files from specified directory
- Evaluates on both TRAIN_SEEDS and EVAL_SEEDS
- Records detailed metrics (episode count, training step, seed, attempt, value)
- Generates plots similar to plot_training_results.py
- Creates separate results folder with logs and plots

Usage:
    python eval_saved_models.py --eval-runs 3 --seeds train eval --model-dir runs/rp_gnn_debug/saved_models --output-dir eval_results
"""

import os
import sys
import argparse
import glob
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch as th

from stable_baselines3 import PPO
from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter 
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from utils.sumo_bootstrap import start_sumo, _imports, _build_args
from utils.feature_fns import make_feature_fn
from utils.metrics_calculator import compute_episode_metrics_from_logs
from utils.logit_metrics_logger import (
    compute_logit_step_metrics,
    aggregate_episode_logit_metrics,
    append_logit_metrics_log,
    LogitStepMetrics
)


class FixedSeedResetFn:
    """Reset function that uses a fixed seed for reproducibility."""
    def __init__(self, sumocfg_path: str, use_gui: bool, seed: int, sumo_port: int | None = None):
        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.seed = seed
        self.sumo_port = sumo_port
    
    def __call__(self) -> None:
        extra_args = ["--seed", str(self.seed), "--device.taxi.dispatch-algorithm", "traci"]
        traci_module, checkBinary = _imports()
        args = _build_args(self.sumocfg_path, extra_args)
        
        if traci_module.isLoaded():
            traci_module.load(args)
        else:
            binary = checkBinary("sumo-gui" if self.use_gui else "sumo")
            traci_module.start([binary, *args], port=self.sumo_port)


class Tee(object):
    """Redirect stdout to both console and file."""
    def __init__(self, filename):
        self.file = open(filename, "w", encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def parse_model_filename(filename):
    """Parse model filename to extract episode and timestep."""
    pattern = r'model_episode(\d+)_ts(\d+)\.zip'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def extract_episode_metrics(episode_dir, seed, attempt):
    """Extract metrics from episode logs."""
    from utils.metrics_calculator import EpisodeMetrics
    try:
        metrics = compute_episode_metrics_from_logs(
            episode_dir=episode_dir,
            episode_info={},
            policy="eval",
            seed=seed,
            num_robots=5,
        )
        if isinstance(metrics, EpisodeMetrics):
            return metrics
        else:
            return EpisodeMetrics(policy="eval", seed=seed)
    except Exception as e:
        print(f"Warning: Could not extract metrics: {e}")
        return EpisodeMetrics(policy="eval", seed=seed)


def evaluate_model(model_path, episode_idx, ts_idx, seed, attempt, config, port_base):
    """Evaluate a single model on one seed with fresh SUMO instance."""
    try:
        # Load model
        model = PPO.load(model_path)
        
        # Setup logger with fresh instance
        rp_logger = RidepoolLogger(
            RidepoolLogConfig(
                out_dir=config['eval_run_dir'],
                run_name=f"model_ep{episode_idx}_ts{ts_idx}_seed{seed}_att{attempt}",
                erase_run_dir_on_start=True,
                erase_episode_dir_on_start=True,
                console_debug=False
            )
        )
        
        # Start fresh SUMO instance with unique port
        port = port_base + attempt
        extra_args = ["--seed", str(seed), "--device.taxi.dispatch-algorithm", "traci"]
        traci = start_sumo(config['sumo_cfg'], use_gui=config['use_gui'], extra_args=extra_args, remote_port=port)
        
        # Setup reset function
        reset_fn = FixedSeedResetFn(
            config['sumo_cfg'],
            use_gui=config['use_gui'],
            seed=seed,
            sumo_port=port,
        )
        
        # Create controller with fresh SUMO
        controller = RLControllerAdapter(
            sumo=traci,
            reset_fn=reset_fn,
            k_max=config['k_max'],
            vicinity_m=config['vicinity_m'],
            completion_mode="dropoff",
            max_steps=config['max_steps'],
            min_episode_steps=config['min_episode_steps'],
            serve_to_empty=True,
            require_seen_reservation=True,
            max_wait_delay_s=config['max_wait_delay_s'],
            max_travel_delay_s=config['max_travel_delay_s'],
            max_robot_capacity=config['max_robot_capacity'],
            logger=rp_logger,
        )
        
        feature_fn = make_feature_fn(controller, use_xy_pickup=bool(config.get('use_xy_pickup', False)))
        
        # Create environment
        env = RidepoolRTEnv(
            controller,
            R=config['R'], K_max=config['k_max'], N_max=config['N_max'],
            E_max=config['E_max'], F=config['F'], G=0,
            feature_fn=feature_fn,
            global_stats_fn=None,
            decision_dt=config['decision_dt'],
        )
        
        # Run single evaluation episode (one per fresh SUMO instance)
        obs, info = env.reset()
        ep_reward = 0.0
        done = False
        step_idx = 0
        
        # Collect logit metrics for each decision step
        logit_step_metrics = []
        
        while not done:
            #model.policy.noop_logit.data.fill_(-1.0)
            
            # Capture logits before prediction
            try:
                with th.no_grad():
                    obs_tensor, _ = model.policy.obs_to_tensor(obs)
                    _ = model.policy.extract_features(obs_tensor, features_extractor=model.policy.features_extractor)
                    obs_dict_b = model.policy.features_extractor.last_obs
                    logits_k, _ = model.policy._build_batch_outputs(obs_dict_b)
                    mask_k = obs_dict_b["cand_mask"]
                    logits, mask = model.policy._append_noop(logits_k, mask_k)
                    
                    # Extract logits and mask as numpy arrays
                    logits_np = logits.squeeze(0).detach().cpu().numpy()  # [R, K_max+1]
                    mask_np = mask.squeeze(0).detach().cpu().numpy()  # [R, K_max+1]
                    noop_logit_value = float(model.policy.noop_logit.item())
                    
                    # Compute step logit metrics
                    step_metrics = compute_logit_step_metrics(logits_np, mask_np, noop_logit_value)
                    step_metrics.step = step_idx
                    logit_step_metrics.append(step_metrics)
                    
                    if config.get('print_steps', False):
                        logits_masked = logits.masked_fill(~mask, -1e9)
                        logits_masked_np = logits_masked.squeeze(0).detach().cpu().numpy()
                        print(f"[STEP {step_idx}] obs={obs}")
                        print(f"[STEP {step_idx}] logits={logits_masked_np}")
                        print(f"[STEP {step_idx}] logit_metrics: best_cand={step_metrics.best_cand_logit:.4f}, "
                              f"noop={step_metrics.noop_logit:.4f}, margin={step_metrics.margin:.4f}")
            except Exception as e:
                print(f"[STEP {step_idx}] logit capture error: {e}")
            
            action, _states = model.predict(obs, deterministic=config.get('deterministic', False))
            
            if config.get('print_steps', False):
                print(f"[STEP {step_idx}] action={action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step_idx += 1

        # **FIX**: Flush logger files to ensure all data is written before extracting metrics
        if hasattr(rp_logger, '_files'):
            for f in rp_logger._files.values():
                try:
                    f.flush()
                except Exception:
                    pass
        
        # Extract detailed metrics
        from utils.metrics_calculator import EpisodeMetrics
        ep_dir = getattr(rp_logger, 'last_ep_dir', None) or rp_logger.ep_dir
        episode_metrics = None
        if ep_dir and os.path.exists(ep_dir):
            try:
                rewards_file = os.path.join(ep_dir, "rewards_macro.csv")
                if os.path.exists(rewards_file):
                    metrics = extract_episode_metrics(ep_dir, seed, attempt)
                    if isinstance(metrics, EpisodeMetrics):
                        episode_metrics = metrics
            except Exception as e:
                print(f"Warning: Could not extract episode metrics: {e}")
        if episode_metrics is None:
            episode_metrics = EpisodeMetrics(policy="eval", seed=seed, reward_sum=ep_reward)
        # Inject ts (timesteps) field into EpisodeMetrics for logging
        if hasattr(episode_metrics, '__dict__'):
            episode_metrics.ts = ts_idx
        elif isinstance(episode_metrics, dict):
            episode_metrics['ts'] = ts_idx
        
        # Aggregate logit metrics
        policy_type = "deterministic" if config.get('deterministic', False) else "stochastic"
        logit_episode_metrics = aggregate_episode_logit_metrics(logit_step_metrics, policy_type, seed, ts_idx)
        
        env.close()
        rp_logger.close()
        return {
            'reward': ep_reward,
            'episode_metrics': episode_metrics,
            'logit_metrics': logit_episode_metrics
        }
    
    except Exception as e:
        print(f"ERROR evaluating model at {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def ma(data, window):
    """Moving average preserving array length."""
    data = np.array(data, dtype=float)
    window = min(window, len(data))
    result = np.convolve(data, np.ones(window)/window, mode='same')
    half_window = window // 2
    for i in range(half_window):
        result[i] = np.mean(data[:i+1])
        result[-(i+1)] = np.mean(data[-(i+1):])
    return result


def plot_evaluation_results(results_df, output_dir, ma_window=10):
    """Generate plots from evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    # Group by timestep, averaging across all seeds and attempts
    grouped = results_df.groupby(['ts_idx'])['reward'].agg(['mean', 'std', 'count']).reset_index()
    
    ts = grouped['ts_idx'].values
    mean_rew = grouped['mean'].values
    std_rew = grouped['std'].values
    counts = grouped['count'].values
    
    # Plot mean with error bars
    ax.errorbar(ts, mean_rew, yerr=std_rew/np.sqrt(counts), fmt='o-', alpha=0.6, 
                label='Mean Reward (SEM)', color='#3498db', capsize=5, markersize=6)
    
    # Plot moving average
    if len(mean_rew) > 1:
        ma_rew = ma(mean_rew, ma_window)
        ax.plot(ts, ma_rew, 'r-', lw=2.5, alpha=0.7, label=f'Moving Average (w={ma_window})')
    
    ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Evaluation Reward', fontsize=11, fontweight='bold')
    ax.set_title('Model Performance vs Training Steps', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_vs_timesteps.png'), dpi=150, bbox_inches='tight')
    print(f"[OK] Saved reward_vs_timesteps.png")
    plt.close()
    
    # Plot 2: Mean reward by seed (averaged over attempts and timesteps)
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')
    
    seed_data = results_df.groupby('seed')['reward'].agg(['mean', 'std', 'count']).reset_index()
    seeds = seed_data['seed'].values
    means = seed_data['mean'].values
    sems = seed_data['std'].values / np.sqrt(seed_data['count'].values)
    
    bars = ax.bar(range(len(seeds)), means, yerr=sems, capsize=5, alpha=0.7, color='#27ae60', edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'{int(s)}' for s in seeds], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Seed', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
    ax.set_title('Mean Reward by Seed (Averaged over All Models and Attempts)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_by_seed.png'), dpi=150, bbox_inches='tight')
    print(f"[OK] Saved reward_by_seed.png")
    plt.close()
    
    # Plot 3: Per-seed plots (one plot per seed showing evolution with training steps)
    seeds_unique = sorted(results_df['seed'].unique())
    for seed in seeds_unique:
        seed_data = results_df[results_df['seed'] == seed].sort_values('ts_idx')
        
        if len(seed_data) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#fafafa')
        ax.set_facecolor('#fafafa')
        
        # Group by timestep for this seed
        ts_grouped = seed_data.groupby('ts_idx')['reward'].agg(['mean', 'std', 'count']).reset_index()
        ts = ts_grouped['ts_idx'].values
        means = ts_grouped['mean'].values
        sems = ts_grouped['std'].values / np.sqrt(ts_grouped['count'].values)
        
        ax.errorbar(ts, means, yerr=sems, fmt='o-', alpha=0.6, 
                    label='Mean Reward', color='#3498db', capsize=5, markersize=6)
        
        # Moving average
        if len(means) > 1:
            ma_rew = ma(means, ma_window)
            ax.plot(ts, ma_rew, 'r-', lw=2.5, alpha=0.7, label=f'Moving Average (w={ma_window})')
        
        ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('Evaluation Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'Model Performance for Seed {int(seed)}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'reward_seed{int(seed)}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] Saved {len(seeds_unique)} per-seed plots")
    
    # Plot 4: Component breakdown (if available in metrics)
    try:
        # Aggregate component metrics across all evaluations, sorted by timesteps
        eval_sorted = results_df.sort_values('ts_idx').reset_index(drop=True)
        all_rewards = []
        all_cap = []
        all_step = []
        all_mdl = []
        all_wait = []
        all_comp = []
        
        for idx, row in eval_sorted.iterrows():
            metrics_dict = row.get('metrics_dict', {})
            if metrics_dict:
                all_rewards.append(metrics_dict.get('rew', np.nan))
                all_cap.append(metrics_dict.get('cap', np.nan))
                all_step.append(metrics_dict.get('step', np.nan))
                all_mdl.append(metrics_dict.get('mdl', np.nan))
                all_wait.append(metrics_dict.get('wait', np.nan))
                all_comp.append(metrics_dict.get('comp', np.nan))
        
        if any(all_rewards) and len(all_rewards) > 0:
            fig = plt.figure(figsize=(14, 8), facecolor='#fafafa')
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
            
            eps = np.arange(len(all_rewards))
            
            # Total reward over evaluations
            ax = fig.add_subplot(gs[0, 0])
            ax.set_facecolor('#fafafa')
            ax.plot(eps, all_rewards, 'o-', alpha=0.7, color='#2c3e50', markersize=4)
            ax.fill_between(eps, all_rewards, alpha=0.3, color='#2c3e50')
            ax.set_xlabel('Evaluation Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Total Reward', fontsize=10, fontweight='bold')
            ax.set_title('Total Reward Over All Evaluations', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Capacity utilization
            ax = fig.add_subplot(gs[0, 1])
            ax.set_facecolor('#fafafa')
            ax.plot(eps, all_cap, 'o-', alpha=0.7, color='#e74c3c', markersize=4)
            ax.fill_between(eps, all_cap, alpha=0.3, color='#e74c3c')
            ax.set_xlabel('Evaluation Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Capacity Reward', fontsize=10, fontweight='bold')
            ax.set_title('Capacity Utilization', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Component breakdown
            ax = fig.add_subplot(gs[1, :])
            ax.set_facecolor('#fafafa')
            baseline = np.zeros(len(eps))
            ax.fill_between(eps, baseline, baseline + np.array(all_mdl),
                            color='#e74c3c', alpha=0.4, label='Middleware')
            ax.fill_between(eps, baseline + np.array(all_mdl), 
                            baseline + np.array(all_mdl) + np.array(all_wait),
                            color='#f39c12', alpha=0.4, label='Wait')
            ax.fill_between(eps, baseline + np.array(all_mdl) + np.array(all_wait),
                            baseline + np.array(all_mdl) + np.array(all_wait) + np.array(all_comp),
                            color='#27ae60', alpha=0.4, label='Completion')
            
            ax.set_xlabel('Evaluation Index', fontsize=10, fontweight='bold')
            ax.set_ylabel('Reward', fontsize=10, fontweight='bold')
            ax.set_title('Reward Component Breakdown', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(alpha=0.25, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'reward_components.png'), dpi=150, bbox_inches='tight')
            print(f"[OK] Saved reward_components.png")
            plt.close()
    except Exception as e:
        print(f"[WARN] Could not generate component breakdown plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate all saved models')
    parser.add_argument('--config', type=str, default='configs/rp_gnn.yaml',
                        help='Path to config YAML')
    parser.add_argument('--eval-runs', type=int, default=3,
                        help='Number of evaluation runs per seed (default: 3)')
    parser.add_argument('--ma-window', type=int, default=10,
                        help='Moving average window size (default: 10)')
    parser.add_argument('--model-sample', type=float, default=1.0,
                        help='Fraction of models to evaluate (default: 1.0 = all). 0.1 = every 10th model')
    parser.add_argument('--seeds', dest='seed_set', type=str, default='both', choices=['train', 'eval', 'both'],
                        help='Which seeds to evaluate on (default: both)')
    parser.add_argument('--model-dir', type=str, default='runs/rp_gnn_debug/saved_models',
                        help='Directory containing model.zip files')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                        help='Output directory for results')
    parser.add_argument('--sumoport', type=int, default=None,
                        help='Base SUMO remote port (default: 8900)')
    parser.add_argument('--gui', action='store_true', help='Enable SUMO GUI (default: disabled)')
    parser.add_argument('--sorted', action='store_true',
                        help='Sort candidates by pickup distance (default: randomized)')
    parser.add_argument('--print-steps', action='store_true',
                        help='Print observation, logits, and action for each env step')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic=True for model.predict')
    from utils.config import Config
    cfg = Config(parser)
    opt = cfg.opt
    args = opt
    
    # Seeds
    TRAIN_SEEDS = list(opt.seeds.train)
    EVAL_SEEDS = list(opt.seeds.eval)
    
    if args.seed_set == 'train':
        seeds_to_eval = TRAIN_SEEDS
    elif args.seed_set == 'eval':
        seeds_to_eval = EVAL_SEEDS
    else:
        seeds_to_eval = TRAIN_SEEDS + EVAL_SEEDS
    
    output_dir = str(getattr(args, "output_dir", "eval_results"))
    model_dir = str(getattr(args, "model_dir", "runs/rp_gnn_debug/saved_models"))

    feature_dim = int(opt.features.base_dim)
    if bool(opt.features.use_xy_pickup):
        feature_dim += 2

    # Configuration
    config = {
        'sumo_cfg': opt.env.sumo_cfg,
        'use_gui': bool(opt.env.use_gui) or bool(getattr(args, "gui", False)),
        'R': int(opt.env.R),
        'k_max': int(opt.env.K_max),
        'N_max': int(opt.env.N_max),
        'E_max': int(opt.env.E_max),
        'F': feature_dim,
            'use_xy_pickup': bool(opt.features.use_xy_pickup),
        'vicinity_m': float(opt.env.vicinity_m),
        'max_steps': int(opt.env.max_steps),
        'max_wait_delay_s': float(opt.env.max_wait_delay_s),
        'max_travel_delay_s': float(opt.env.max_travel_delay_s),
        'max_robot_capacity': int(opt.env.max_robot_capacity),
        'decision_dt': int(opt.env.decision_dt),
        'min_episode_steps': int(opt.env.min_episode_steps),
        'eval_runs': int(getattr(args, "eval_runs", 3)),
        'eval_run_dir': os.path.join(output_dir, 'evaluation_runs'),
        'print_steps': bool(getattr(args, "print_steps", False)),
        'deterministic': bool(getattr(args, "deterministic", False)),
        'sorted_candidates': bool(getattr(args, "sorted", False)) or bool(opt.env.sorted_candidates),
    }
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(output_dir, f'evaluation_{timestamp}')
    os.makedirs(config['eval_run_dir'], exist_ok=True)
    os.makedirs(output_base, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_base, 'evaluation.log')
    tee = Tee(log_file)
    
    print("=" * 80)
    print(f"EVALUATING SAVED MODELS")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_base}")
    print(f"Seeds: {args.seed_set} ({len(seeds_to_eval)} seeds)")
    print(f"Eval runs per seed: {getattr(args, 'eval_runs', 3)}")
    print(f"Moving average window: {getattr(args, 'ma_window', 10)}")
    print(f"Model sample rate: {getattr(args, 'model_sample', 1.0)*100:.1f}%")
    print()
    
    # Find all models and sort by timestep (not by filename)
    model_files = glob.glob(os.path.join(model_dir, 'model_episode*.zip'))
    
    # Parse and sort by timestep value
    model_info = []
    for mf in model_files:
        filename = os.path.basename(mf)
        episode_idx, ts_idx = parse_model_filename(filename)
        if episode_idx is not None:
            model_info.append((ts_idx, mf))
    
    # Sort by timestep (ascending order)
    model_info.sort(key=lambda x: x[0])
    model_files = [mf for _, mf in model_info]
    
    # Sample models if requested
    model_sample = float(getattr(args, "model_sample", 1.0))
    if model_sample < 1.0:
        sample_size = max(1, int(len(model_files) * model_sample))
        sample_indices = np.linspace(0, len(model_files) - 1, sample_size, dtype=int)
        model_files = [model_files[i] for i in sample_indices]
        print(f"Sampling {len(model_files)} models ({model_sample*100:.1f}% of {len(model_info)} total)")
    else:
        print(f"Found {len(model_files)} model files (sorted by timestep)")
    print()
    
    if not model_files:
        print(f"ERROR: No models found in {model_dir}")
        tee.close()
        return
    
    # Evaluate all models
    results = []
    metrics_file = os.path.join(output_base, f'evaluation_metrics.log')
    logit_metrics_file = os.path.join(output_base, f'logit_metrics.log')
    from utils.metrics_calculator import ensure_metrics_log
    ensure_metrics_log(metrics_file, overwrite=True)
    from utils.logit_metrics_logger import ensure_logit_metrics_log
    ensure_logit_metrics_log(logit_metrics_file)
    
    eval_runs = int(getattr(args, "eval_runs", 3))
    total_evals = len(model_files) * len(seeds_to_eval) * eval_runs
    current_eval = 0
    port_base = getattr(args, "sumoport", None) if getattr(args, "sumoport", None) is not None else 8900
    
    for model_idx, model_path in enumerate(model_files):
        filename = os.path.basename(model_path)
        episode_idx, ts_idx = parse_model_filename(filename)
        
        if episode_idx is None:
            print(f"[SKIP] Skipping {filename} (could not parse)")
            continue
        
        print(f"\n[Model {model_idx+1}/{len(model_files)}] {filename}")
        
        for seed_idx, seed in enumerate(seeds_to_eval):
            for attempt in range(eval_runs):
                current_eval += 1
                pct = 100.0 * current_eval / total_evals
                print(f"  [{current_eval}/{total_evals} ({pct:.1f}%)] Seed {seed}, Attempt {attempt+1}/{eval_runs}...", end=' ')
                
                result = evaluate_model(model_path, episode_idx, ts_idx, seed, attempt, config, port_base)
                if result:
                    results.append({
                        'episode': episode_idx,
                        'ts_idx': ts_idx,
                        'seed': seed,
                        'attempt': attempt,
                        'reward': result['reward'],
                        'episode_metrics': result['episode_metrics'],
                        'logit_metrics': result.get('logit_metrics', None),
                    })
                    from utils.metrics_calculator import append_metrics_log
                    append_metrics_log(metrics_file, result['episode_metrics'])
                    
                    # Log logit metrics if available
                    if 'logit_metrics' in result and result['logit_metrics'] is not None:
                        append_logit_metrics_log(logit_metrics_file, result['logit_metrics'])
                    
                    print(f"[OK] reward={result['reward']:.4f}")
                else:
                    print(f"[FAIL]")
    
    print(f"\n{'='*80}")
    print(f"[OK] Evaluation complete. Processed {len(results)} evaluations")
    
    # Create results DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        csv_file = os.path.join(output_base, 'evaluation_results.csv')
        results_df.to_csv(csv_file, index=False)
        print(f"[OK] Saved results to {csv_file}")
        
        # Generate plots
        print("\nGenerating plots...")
        plot_evaluation_results(results_df, output_base, ma_window=int(getattr(args, "ma_window", 10)))
    
    print(f"\n[OK] All results saved to {output_base}")
    tee.close()


if __name__ == '__main__':
    main()
