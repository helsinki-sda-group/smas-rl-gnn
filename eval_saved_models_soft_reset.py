#!/usr/bin/env python
"""
Evaluate all saved models using soft reset (like training).

This script uses the same reset mechanism as training:
- Single persistent SUMO process
- Soft reset via traci.load() between episodes
- RandomSeedResetFn that samples from seed pool
- Consistent RNG initialization

Features:
- Evaluates all model_episodeX_tsY.zip files
- Can test on train seeds, eval seeds, or both
- Multiple evaluation runs per seed
- Records detailed metrics
- Generates plots

Usage:
    python eval_saved_models_soft_reset.py --eval-runs 3 --seeds both
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

from stable_baselines3 import PPO
from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter 
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from utils.sumo_bootstrap import start_sumo, _imports, _build_args
from utils.feature_fns import make_feature_fn
from utils.metrics_calculator import compute_episode_metrics_from_logs


class RandomSeedResetFn:
    """Reset function that samples from seed pool (like training)."""
    def __init__(self, sumocfg_path: str, use_gui: bool, seeds: list, random_seed: int = 42):
        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.seeds = list(seeds)
        self.rng = np.random.RandomState(random_seed)
        self.current_seed = self.seeds[0]
    
    def __call__(self) -> None:
        # Sample a seed from the pool
        self.current_seed = int(self.rng.choice(self.seeds))
        extra_args = ["--seed", str(self.current_seed), "--device.taxi.dispatch-algorithm", "traci"]
        
        traci, checkBinary = _imports()
        args = _build_args(self.sumocfg_path, extra_args)
        
        if traci.isLoaded():
            # Soft reset - reload in the same process
            traci.load(args)
        else:
            # First start
            binary = checkBinary("sumo-gui" if self.use_gui else "sumo")
            traci.start([binary, *args])
    
    def get_current_seed(self) -> int:
        """Get the seed used for the current episode."""
        return self.current_seed


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
    try:
        metrics = compute_episode_metrics_from_logs(
            episode_dir=episode_dir,
            episode_info={},
            policy="eval",
            seed=seed,
            num_robots=5,
        )
        # Convert EpisodeMetrics dataclass to dict
        if hasattr(metrics, '__dataclass_fields__'):
            return {
                'rew': metrics.reward_sum,
                'cap': metrics.capacity_sum,
                'step': metrics.step_sum,
                'mdl': metrics.missed_deadline_sum,
                'wait': metrics.wait_sum,
                'comp': metrics.completion_sum,
                'nsv': metrics.nonserved_sum,
            }
        elif hasattr(metrics, '__dict__'):
            d = metrics.__dict__
            return {
                'rew': d.get('reward_sum', 0),
                'cap': d.get('capacity_sum', 0),
                'step': d.get('step_sum', 0),
                'mdl': d.get('missed_deadline_sum', 0),
                'wait': d.get('wait_sum', 0),
                'comp': d.get('completion_sum', 0),
                'nsv': d.get('nonserved_sum', 0),
            }
        elif isinstance(metrics, dict):
            return metrics
        else:
            return {}
    except Exception as e:
        print(f"Warning: Could not extract metrics: {e}")
        return {}


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
    
    # Plot 1: Mean evaluation reward vs training steps (with moving average)
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
    ax.set_title('Model Performance vs Training Steps (Soft Reset)', fontsize=12, fontweight='bold')
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
        ax.set_title(f'Model Performance for Seed {int(seed)} (Soft Reset)', fontsize=12, fontweight='bold')
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
    parser = argparse.ArgumentParser(description='Evaluate all saved models with soft reset (like training)')
    parser.add_argument('--config', type=str, default='configs/rp_gnn.yaml',
                        help='Path to config YAML')
    parser.add_argument('--eval-runs', type=int, default=3,
                        help='Number of evaluation runs per seed (default: 3)')
    parser.add_argument('--ma-window', type=int, default=10,
                        help='Moving average window size (default: 10)')
    parser.add_argument('--model-sample', type=float, default=1.0,
                        help='Fraction of models to evaluate (default: 1.0 = all)')
    parser.add_argument('--seeds', type=str, default='both', choices=['train', 'eval', 'both'],
                        help='Which seeds to evaluate on (default: both)')
    parser.add_argument('--seed-sampler-rng', type=int, default=42,
                        help='Random seed for the seed sampler RNG (default: 42)')
    parser.add_argument('--model-dir', type=str, default='runs/rp_gnn_debug/saved_models',
                        help='Directory containing model.zip files')
    parser.add_argument('--output-dir', type=str, default='eval_results_soft_reset',
                        help='Output directory for results')
    parser.add_argument('--gui', action='store_true', help='Enable SUMO GUI (default: disabled)')
    parser.add_argument('--print-steps', action='store_true',
                        help='Print observation, logits, and action for each env step')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic=True for model.predict')
    parser.add_argument('--sorted', action='store_true',
                        help='Sort candidates by pickup distance (default: randomized)')
    from utils.config import Config
    cfg = Config(parser)
    opt = cfg.opt
    args = opt
    
    # Seeds
    TRAIN_SEEDS = list(opt.seeds.train)
    EVAL_SEEDS = list(opt.seeds.eval)
    
    if args.seeds == 'train':
        seeds_to_eval = TRAIN_SEEDS
    elif args.seeds == 'eval':
        seeds_to_eval = EVAL_SEEDS
    else:
        seeds_to_eval = TRAIN_SEEDS + EVAL_SEEDS
    
    # Configuration (matching training setup)
    config = {
        'sumo_cfg': opt.env.sumo_cfg,
        'use_gui': bool(opt.env.use_gui) or bool(args.gui),
        'R': int(opt.env.R),
        'k_max': int(opt.env.K_max),
        'N_max': int(opt.env.N_max),
        'E_max': int(opt.env.E_max),
        'F': int(opt.env.F),
        'vicinity_m': float(opt.env.vicinity_m),
        'max_steps': int(opt.env.max_steps),
        'max_wait_delay_s': float(opt.env.max_wait_delay_s),
        'max_travel_delay_s': float(opt.env.max_travel_delay_s),
        'max_robot_capacity': int(opt.env.max_robot_capacity),
        'decision_dt': int(opt.env.decision_dt),
        'min_episode_steps': int(opt.env.min_episode_steps),
        'print_steps': bool(args.print_steps),
        'deterministic': bool(args.deterministic),
        'sorted_candidates': bool(args.sorted) or bool(opt.env.sorted_candidates),
    }
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(args.output_dir, f'evaluation_{timestamp}')
    eval_run_dir = os.path.join(output_base, 'runs')
    os.makedirs(eval_run_dir, exist_ok=True)
    os.makedirs(output_base, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_base, 'evaluation.log')
    tee = Tee(log_file)
    
    print("=" * 80)
    print(f"EVALUATING SAVED MODELS (SOFT RESET - TRAINING-LIKE)")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {output_base}")
    print(f"Seeds: {args.seeds} ({len(seeds_to_eval)} seeds)")
    print(f"Eval runs per seed: {args.eval_runs}")
    print(f"Moving average window: {args.ma_window}")
    print(f"Model sample rate: {args.model_sample*100:.1f}%")
    print(f"Seed sampler RNG: {args.seed_sampler_rng}")
    print(f"Reset mode: SOFT RESET (traci.load)")
    print()
    
    # Find all models and sort by timestep
    model_files = glob.glob(os.path.join(args.model_dir, 'model_episode*.zip'))
    
    # Parse and sort by timestep value
    model_info = []
    for mf in model_files:
        filename = os.path.basename(mf)
        episode_idx, ts_idx = parse_model_filename(filename)
        if episode_idx is not None:
            model_info.append((ts_idx, episode_idx, mf))
    
    # Sort by timestep (ascending order)
    model_info.sort(key=lambda x: x[0])
    
    # Sample models if requested
    if args.model_sample < 1.0:
        sample_size = max(1, int(len(model_info) * args.model_sample))
        sample_indices = np.linspace(0, len(model_info) - 1, sample_size, dtype=int)
        model_info = [model_info[i] for i in sample_indices]
        print(f"Sampling {len(model_info)} models ({args.model_sample*100:.1f}% of total)")
    else:
        print(f"Found {len(model_info)} model files (sorted by timestep)")
    print()
    
    if not model_info:
        print(f"ERROR: No models found in {args.model_dir}")
        tee.close()
        return
    
    # ============================================================================
    # Initialize SUMO and environment ONCE (like training)
    # ============================================================================
    print("Initializing persistent SUMO process and environment...")
    
    # Create reset function that samples from seed pool
    reset_fn = RandomSeedResetFn(
        config['sumo_cfg'],
        use_gui=args.gui,
        seeds=seeds_to_eval,
        random_seed=args.seed_sampler_rng
    )
    
    # Start initial SUMO instance
    initial_seed = seeds_to_eval[0]
    traci = start_sumo(
        config['sumo_cfg'],
        use_gui=args.gui,
        extra_args=["--seed", str(initial_seed), "--device.taxi.dispatch-algorithm", "traci"]
    )
    
    # Setup logger
    rp_logger = RidepoolLogger(
        RidepoolLogConfig(
            out_dir=eval_run_dir,
            run_name="eval_soft_reset",
            erase_run_dir_on_start=True,
            erase_episode_dir_on_start=True,
            console_debug=False
        )
    )
    
    # Create controller with soft reset function
    controller = RLControllerAdapter(
        sumo=traci,
        reset_fn=reset_fn,
        k_max=config['k_max'],
        vicinity_m=config['vicinity_m'],
        sorted_candidates=config.get('sorted_candidates', False),
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
    
    feature_fn = make_feature_fn(controller)
    
    # Create environment
    env = RidepoolRTEnv(
        controller,
        R=config['R'], K_max=config['k_max'], N_max=config['N_max'],
        E_max=config['E_max'], F=config['F'], G=0,
        feature_fn=feature_fn,
        global_stats_fn=None,
        decision_dt=config['decision_dt'],
    )
    
    print("[OK] Environment initialized with soft reset")
    print()
    
    # ============================================================================
    # Evaluate all models
    # ============================================================================
    results = []
    metrics_file = os.path.join(output_base, 'evaluation_metrics.log')
    
    with open(metrics_file, 'w', encoding='utf-8') as mf:
        # Write header
        mf.write("episode | timesteps | seed | attempt | reward | reward_cap | reward_step | reward_mdl | reward_wait | reward_comp | nsv\n")
    
    total_episodes = len(model_info) * args.eval_runs
    episode_counter = 0
    
    for model_idx, (ts_idx, episode_idx, model_path) in enumerate(model_info):
        filename = os.path.basename(model_path)
        
        print(f"\n[Model {model_idx+1}/{len(model_info)}] {filename}")
        
        # Load model
        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            continue
        
        # Run multiple evaluation episodes for this model
        for run_idx in range(args.eval_runs):
            episode_counter += 1
            pct = 100.0 * episode_counter / total_episodes
            
            try:
                # Reset environment (will sample a seed from the pool)
                obs, info = env.reset()
                current_seed = reset_fn.get_current_seed()
                
                print(f"  [{episode_counter}/{total_episodes} ({pct:.1f}%)] Run {run_idx+1}/{args.eval_runs}, Seed {current_seed}...", end=' ')
                
                # Run episode
                ep_reward = 0.0
                done = False
                step_idx = 0
                
                while not done:
                    action, _states = model.predict(obs, deterministic=config.get('deterministic', False))
                    if config.get('print_steps', False):
                        try:
                            with th.no_grad():
                                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                                _ = model.policy.extract_features(obs_tensor, features_extractor=model.policy.features_extractor)
                                obs_dict_b = model.policy.features_extractor.last_obs
                                logits_k, _ = model.policy._build_batch_outputs(obs_dict_b)
                                mask_k = obs_dict_b["cand_mask"]
                                logits, mask = model.policy._append_noop(logits_k, mask_k)
                                logits = logits.masked_fill(~mask, -1e9)
                                logits_np = logits.squeeze(0).detach().cpu().numpy()
                            print(f"[STEP {step_idx}] obs={obs}")
                            print(f"[STEP {step_idx}] logits={logits_np}")
                            print(f"[STEP {step_idx}] action={action}")
                        except Exception as e:
                            print(f"[STEP {step_idx}] print error: {e}")
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
                ep_dir = getattr(rp_logger, 'last_ep_dir', None) or rp_logger.ep_dir
                metrics_dict = {}
                
                if ep_dir and os.path.exists(ep_dir):
                    try:
                        rewards_file = os.path.join(ep_dir, "rewards_macro.csv")
                        if os.path.exists(rewards_file):
                            metrics = extract_episode_metrics(ep_dir, current_seed, run_idx)
                            if metrics:
                                metrics_dict = metrics
                    except Exception as e:
                        pass
                
                # Fallback if metrics not extracted
                if not metrics_dict:
                    metrics_dict = {'rew': ep_reward, 'cap': 0, 'step': 0, 'mdl': 0, 'wait': 0, 'comp': 0, 'nsv': 0}
                
                # Store result
                results.append({
                    'episode': episode_idx,
                    'ts_idx': ts_idx,
                    'seed': current_seed,
                    'attempt': run_idx,
                    'reward': ep_reward,
                    'metrics_dict': metrics_dict,
                })
                
                # Write to metrics file
                with open(metrics_file, 'a', encoding='utf-8') as mf:
                    mf.write(f"{episode_idx} | {ts_idx} | {current_seed} | {run_idx} | {ep_reward:.4f} | " +
                            f"{metrics_dict.get('cap', 0):.4f} | {metrics_dict.get('step', 0):.4f} | {metrics_dict.get('mdl', 0):.4f} | " +
                            f"{metrics_dict.get('wait', 0):.4f} | {metrics_dict.get('comp', 0):.4f} | {metrics_dict.get('nsv', 0):.4f}\n")
                
                print(f"[OK] reward={ep_reward:.4f}")
                
            except Exception as e:
                print(f"[FAIL] {e}")
                import traceback
                traceback.print_exc()
    
    # Close environment
    env.close()
    rp_logger.close()
    
    print(f"\n{'='*80}")
    print(f"[OK] Evaluation complete. Processed {len(results)} evaluations")
    
    # Create results DataFrame and generate plots
    if results:
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        csv_file = os.path.join(output_base, 'evaluation_results.csv')
        results_df.to_csv(csv_file, index=False)
        print(f"[OK] Saved results to {csv_file}")
        
        # Generate plots
        print("\nGenerating plots...")
        plot_evaluation_results(results_df, output_base, ma_window=args.ma_window)
    
    print(f"\n[OK] All results saved to {output_base}")
    tee.close()


if __name__ == '__main__':
    main()
