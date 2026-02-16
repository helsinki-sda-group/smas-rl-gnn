#!/usr/bin/env python
"""
Load a trained model from evaluation checkpoint and run evaluation with seed=100.
Runs 5 evaluation episodes.
"""

import os
import sys
from datetime import datetime
from stable_baselines3 import PPO
import numpy as np

from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter 
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from utils.sumo_bootstrap import start_sumo, _imports, _build_args
from utils.feature_fns import make_feature_fn


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


def main():
    # Configuration
    SEED = 300
    EVAL_RUNS = 5
    MODEL_PATH = "ppo_seed100_20260204_234359.zip" # "runs/evaluation/evaluation_0_ts0_ep0/model.zip"
    
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
    DECISION_DT = 60
    MIN_EPISODE_STEPS = 100

    # Output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log_file = f"eval_checkpoint_seed100_{timestamp}.txt"
    
    # Redirect stdout to log file
    tee = Tee(eval_log_file)
    
    print("="*80)
    print(f"EVALUATION FROM CHECKPOINT - Seed {SEED}, {EVAL_RUNS} runs")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Evaluation log: {eval_log_file}")
    print(f"Model path: {MODEL_PATH}")
    print()

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        tee.close()
        return
    
    # Load the trained model
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")
    print()
    
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
                run_name=f"checkpoint_eval_seed100_{timestamp}_run{eval_run}",
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
    print(f"Model: {MODEL_PATH}")
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
    print(f"Evaluation log: {eval_log_file}")
    print(f"Model: {MODEL_PATH}")
    
    tee.close()


if __name__ == "__main__":
    main()
