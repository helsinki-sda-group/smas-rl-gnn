#!/usr/bin/env python
"""
Script to train a model for 5 steps with seed=100, then evaluate it 5 times with seed=100.
All training and evaluation logs are written to disk.
"""

import os
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

from rt_gnn_rl.policy.sb3_gnn_policy import RTGNNPolicy
from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter 
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from sumo_rl_rs.logging.rp_logger_callback import RPLoggerCallback
from utils.sumo_bootstrap import start_sumo, _imports, _build_args
from utils.feature_fns import make_feature_fn
import numpy as np


class FixedSeedResetFn:
    """Reset function that uses a fixed seed for reproducibility."""
    def __init__(self, sumocfg_path: str, use_gui: bool, seed: int, traci_connection):
        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.seed = seed
        self.traci_conn = traci_connection  # Store the labeled connection object
    
    def __call__(self) -> None:
        extra_args = ["--seed", str(self.seed), "--device.taxi.dispatch-algorithm", "traci"]
        
        # Import here to avoid circular dependencies
        traci_module, checkBinary = _imports()
        args = _build_args(self.sumocfg_path, extra_args)
        
        # Use the specific labeled connection, not the global module
        self.traci_conn.load(args)


class Tee(object):
    """Redirect stdout to both console and file."""
    def __init__(self, filename):
        self.file = open(filename, "w")
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


def log_message(msg, log_file=None):
    """Print message and optionally write to log file."""
    print(msg)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg + '\n')


def main():
    # Configuration
    SEED = 100
    TRAIN_STEPS = 5
    EVAL_RUNS = 5
    
    SUMO_CFG = "configs/small_net.sumocfg"
    USE_GUI = False
    R = 5           # number of robots (taxis)
    K_max = 3       # candidates per robot
    N_max = 16      # max nodes per ego-graph
    E_max = 64      # max edges per ego-graph
    F = 11          # node feature dimension
    G = 0           # global stats dim

    VICINITY_M = 2000.0
    MAX_STEPS = 1200
    MAX_WAIT_DELAY_S = 240.0
    MAX_TRAVEL_DELAY_S = 900.0
    MAX_ROBOT_CAPACITY = 2
    DECISION_DT = 60  # Same for train and eval
    MIN_EPISODE_STEPS = 100

    # Output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_file = f"train_seed100_{timestamp}.txt"
    eval_log_file = f"eval_seed100_{timestamp}.txt"
    model_path = f"ppo_seed100_{timestamp}.zip"
    
    # Redirect stdout to training log file
    tee = Tee(train_log_file)
    
    print("="*80)
    print(f"TRAINING PHASE - Seed {SEED}, {TRAIN_STEPS} steps")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Training log: {train_log_file}")
    print(f"Evaluation log: {eval_log_file}")
    print(f"Model will be saved to: {model_path}")
    print()

    # Start SUMO with seed=100
    print(f"Starting SUMO with seed={SEED}...")
    traci = start_sumo(
        SUMO_CFG, 
        use_gui=USE_GUI,
        extra_args=["--seed", str(SEED), "--device.taxi.dispatch-algorithm", "traci"],
        label="train", 
        port=8813
    )

    # Create reset function with fixed seed
    reset_fn = FixedSeedResetFn(
        SUMO_CFG, 
        use_gui=USE_GUI,
        seed=SEED,
        traci_connection=traci
    )

    # Setup logger
    print("Setting up logger...")
    rp_logger = RidepoolLogger(
        RidepoolLogConfig(
            out_dir="runs",
            run_name=f"seed100_train_{timestamp}",
            erase_run_dir_on_start=True,
            erase_episode_dir_on_start=True,
            console_debug=True
        )
    )

    # Create controller
    print("Creating RL controller...")
    controller = RLControllerAdapter(
        sumo=traci,
        reset_fn=reset_fn,
        k_max=K_max,
        vicinity_m=VICINITY_M,
        completion_mode="dropoff",
        max_steps=MAX_STEPS,
        min_episode_steps=MIN_EPISODE_STEPS,
        serve_to_empty=True,
        require_seen_reservation=True,
        max_wait_delay_s=MAX_WAIT_DELAY_S,
        max_travel_delay_s=MAX_TRAVEL_DELAY_S,
        max_robot_capacity=MAX_ROBOT_CAPACITY,
        logger=rp_logger,
    )
    feature_fn = make_feature_fn(controller)

    # Create Gym environment
    print("Creating environment...")
    env = RidepoolRTEnv(
        controller,
        R=R, K_max=K_max, N_max=N_max, E_max=E_max,
        F=F, G=G,
        feature_fn=feature_fn,
        global_stats_fn=None, 
        decision_dt=DECISION_DT,  
    )
    monitor_csv = f"monitor_seed100_{timestamp}.csv"
    env = Monitor(env, filename=monitor_csv, info_keywords=("episode_reward",))

    # Create PPO model
    print("Creating PPO model...")
    policy_kwargs = dict(in_dim=F, hidden=128, k_max=K_max)
    model = PPO(
        RTGNNPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        clip_range_vf=None,
        vf_coef=0.35,
        ent_coef=0.003,
        gae_lambda=0.95,
        n_epochs=5,
        verbose=1,
        seed=SEED  # Set seed for reproducibility
    )

    # Setup callback for logging
    metrics_log_path = f"training_metrics_seed100_{timestamp}.log"
    callback = RPLoggerCallback(
        rp_logger,
        controller,
        metrics_log_path=metrics_log_path,
        num_robots=R,
        reset_fn=None,
    )

    # Train model
    print("\n" + "="*80)
    print(f"Starting training for {TRAIN_STEPS} timesteps...")
    print(f"Environment config: decision_dt={DECISION_DT}, min_episode_steps={MIN_EPISODE_STEPS}")
    print(f"Controller config: max_steps={MAX_STEPS}, serve_to_empty=True, require_seen_reservation=True")
    print("="*80 + "\n")
    
    model.learn(total_timesteps=TRAIN_STEPS, callback=callback, log_interval=1)
    
    # Save model
    print("\n" + "="*80)
    print(f"Training complete. Saving model to {model_path}...")
    print("="*80 + "\n")
    model.save(model_path)
    
    print(f"Model saved successfully.")
    print(f"Training metrics saved to: {metrics_log_path}")
    print(f"Monitor CSV saved to: {monitor_csv}")
    
    # Close training environment
    env.close()
    tee.close()
    
    print(f"\nTraining logs written to: {train_log_file}")
    
    # =========================================================================
    # EVALUATION PHASE
    # =========================================================================
    
    # Open evaluation log file
    eval_tee = Tee(eval_log_file)
    
    print("\n" + "="*80)
    print(f"EVALUATION PHASE - {EVAL_RUNS} runs with seed {SEED}")
    print("="*80)
    print(f"Loading model from: {model_path}")
    print()
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Run evaluation multiple times
    eval_results = []
    
    for eval_run in range(EVAL_RUNS):
        print("\n" + "-"*80)
        print(f"Evaluation Run {eval_run + 1}/{EVAL_RUNS} (Seed: {SEED})")
        print("-"*80 + "\n")
        
        # Start SUMO with seed=100
        traci_eval = start_sumo(
            SUMO_CFG, 
            use_gui=USE_GUI,
            extra_args=["--seed", str(SEED), "--device.taxi.dispatch-algorithm", "traci"],
            label=f"eval_{eval_run}", 
            port=8814 + eval_run
        )
        
        # Create reset function for evaluation
        reset_fn_eval = FixedSeedResetFn(
            SUMO_CFG, 
            use_gui=USE_GUI,
            seed=SEED,
            traci_connection=traci_eval
        )
        
        # Setup logger for this evaluation
        rp_logger_eval = RidepoolLogger(
            RidepoolLogConfig(
                out_dir="runs",
                run_name=f"seed100_eval_{timestamp}_run{eval_run}",
                erase_run_dir_on_start=True,
                erase_episode_dir_on_start=True,
                console_debug=True
            )
        )
        
        # Create controller for evaluation
        controller_eval = RLControllerAdapter(
            sumo=traci_eval,
            reset_fn=reset_fn_eval,
            k_max=K_max,
            vicinity_m=VICINITY_M,
            completion_mode="dropoff",
            max_steps=MAX_STEPS,
            min_episode_steps=MIN_EPISODE_STEPS,
            serve_to_empty=True,
            require_seen_reservation=True,
            max_wait_delay_s=MAX_WAIT_DELAY_S,
            max_travel_delay_s=MAX_TRAVEL_DELAY_S,
            max_robot_capacity=MAX_ROBOT_CAPACITY,
            logger=rp_logger_eval,
        )
        feature_fn_eval = make_feature_fn(controller_eval)
        
        # Create environment for evaluation
        env_eval = RidepoolRTEnv(
            controller_eval,
            R=R, K_max=K_max, N_max=N_max, E_max=E_max,
            F=F, G=G,
            feature_fn=feature_fn_eval,
            global_stats_fn=None, 
            decision_dt=DECISION_DT,  
        )
        
        # Run one episode
        print(f"Resetting environment...")
        obs, info = env_eval.reset()
        print(f"Environment reset complete. Initial action_mask shape: {info.get('action_mask', 'N/A').shape if hasattr(info.get('action_mask'), 'shape') else 'N/A'}")
        
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env_eval.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            print(f"  Step {step_count}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}, macro_steps={info.get('macro_steps', 'N/A')}")
        
        # Get final metrics
        final_info = info
        
        print(f"\nEvaluation Run {eval_run + 1} Results:")
        print(f"  Episode reward: {episode_reward:.4f}")
        print(f"  Steps: {step_count}")
        print(f"  Final macro_steps: {final_info.get('macro_steps', 'N/A')}")
        print(f"  Steps done: {final_info.get('steps_done', 'N/A')}")
        print(f"  Episode info keys: {list(final_info.keys())}")
        
        eval_results.append({
            'run': eval_run + 1,
            'reward': episode_reward,
            'steps': step_count,
            'info': final_info
        })
        
        # Close evaluation environment
        env_eval.close()
    
    # Summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Number of evaluation runs: {EVAL_RUNS}")
    print(f"Seed: {SEED}")
    print()
    
    rewards = [r['reward'] for r in eval_results]
    steps = [r['steps'] for r in eval_results]
    
    print(f"Reward statistics:")
    print(f"  Mean:   {np.mean(rewards):.4f}")
    print(f"  Std:    {np.std(rewards):.4f}")
    print(f"  Min:    {np.min(rewards):.4f}")
    print(f"  Max:    {np.max(rewards):.4f}")
    print()
    print(f"Steps statistics:")
    print(f"  Mean:   {np.mean(steps):.1f}")
    print(f"  Std:    {np.std(steps):.1f}")
    print(f"  Min:    {np.min(steps)}")
    print(f"  Max:    {np.max(steps)}")
    print()
    
    print("Individual results:")
    for result in eval_results:
        print(f"  Run {result['run']}: reward={result['reward']:.4f}, steps={result['steps']}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Training log: {train_log_file}")
    print(f"Evaluation log: {eval_log_file}")
    print(f"Model: {model_path}")
    
    eval_tee.close()


if __name__ == "__main__":
    main()
