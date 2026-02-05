#!/usr/bin/env python
"""
Calculate rewards from SUMO environment using non-deterministic actions.
Run multiple episodes with seed 100 to see reward distribution.
"""

import os
import sys
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rt_gnn_rl.policy.sb3_gnn_policy import RTGNNPolicy
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
        self.traci_conn = traci_connection
    
    def __call__(self) -> None:
        extra_args = ["--seed", str(self.seed), "--device.taxi.dispatch-algorithm", "traci"]
        traci_module, checkBinary = _imports()
        args = _build_args(self.sumocfg_path, extra_args)
        self.traci_conn.load(args)


def main():
    # Configuration
    SEED = 100
    NUM_EPISODES = 5  # Run 5 episodes with non-deterministic actions
    
    SUMO_CFG = "configs/small_net.sumocfg"
    USE_GUI = False
    R = 5           # number of robots (taxis)
    K_max = 3       # candidates per robot
    N_max = 16      # max nodes per ego-graph
    E_max = 64      # max edges per ego-graph
    F = 9           # node feature dimension
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
    log_file = f"rewards_nondet_seed100_{timestamp}.txt"
    
    print("="*80)
    print(f"REWARD CALCULATION - Non-deterministic Actions, Seed {SEED}")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Log file: {log_file}")
    print(f"Number of episodes: {NUM_EPISODES}")
    print()

    # Start SUMO with seed=100
    print(f"Starting SUMO with seed={SEED}...")
    traci = start_sumo(
        SUMO_CFG, 
        use_gui=USE_GUI,
        extra_args=["--seed", str(SEED), "--device.taxi.dispatch-algorithm", "traci"],
        label="reward_test", 
        port=8816
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
            run_name=f"rewards_nondet_seed100_{timestamp}",
            erase_run_dir_on_start=True,
            erase_episode_dir_on_start=True,
            console_debug=False
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
    env = Monitor(env, filename=f"monitor_rewards_nondet_{timestamp}.csv", info_keywords=("episode_reward",))

    # Create PPO model WITHOUT training (random weights)
    print("Creating PPO model (NO TRAINING - random weights)...")
    policy_kwargs = dict(in_dim=F, hidden=128, k_max=K_max)
    model = PPO(
        RTGNNPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        clip_range_vf=None,
        vf_coef=0.35,
        ent_coef=0.003,
        gae_lambda=0.95,
        n_epochs=5,
        verbose=0,
        seed=SEED
    )
    
    print("Model created with random weights (no training)")
    print()

    # Run multiple episodes with non-deterministic actions
    print("="*80)
    print(f"RUNNING {NUM_EPISODES} EPISODES WITH NON-DETERMINISTIC ACTIONS")
    print("="*80)
    print()
    
    episode_rewards = []
    episode_steps = []
    episode_details = []
    
    for ep in range(NUM_EPISODES):
        print(f"\nEpisode {ep+1}/{NUM_EPISODES}")
        print("-"*80)
        
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        step_rewards = []
        
        while not done:
            # Use NON-DETERMINISTIC prediction
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            step_rewards.append(reward)
            
            if step_count <= 5 or step_count % 5 == 0:  # Print first 5 steps and every 5th step after
                print(f"  Step {step_count}: action={action}, reward={reward:.4f}, macro_steps={info.get('macro_steps')}")
        
        print(f"\nEpisode {ep+1} Summary:")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Steps: {step_count}")
        print(f"  Step rewards: min={np.min(step_rewards):.4f}, max={np.max(step_rewards):.4f}, mean={np.mean(step_rewards):.4f}")
        
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        episode_details.append({
            'episode': ep + 1,
            'reward': episode_reward,
            'steps': step_count,
            'step_rewards': step_rewards
        })
    
    # Calculate statistics
    print("\n" + "="*80)
    print("STATISTICS ACROSS ALL EPISODES")
    print("="*80)
    print()
    
    print(f"Total episodes: {NUM_EPISODES}")
    print()
    
    print("Reward Statistics:")
    print(f"  Mean:   {np.mean(episode_rewards):.4f}")
    print(f"  Std:    {np.std(episode_rewards):.4f}")
    print(f"  Min:    {np.min(episode_rewards):.4f}")
    print(f"  Max:    {np.max(episode_rewards):.4f}")
    print(f"  Median: {np.median(episode_rewards):.4f}")
    print()
    
    print("Episode Steps Statistics:")
    print(f"  Mean:   {np.mean(episode_steps):.1f}")
    print(f"  Std:    {np.std(episode_steps):.1f}")
    print(f"  Min:    {np.min(episode_steps)}")
    print(f"  Max:    {np.max(episode_steps)}")
    print(f"  Median: {np.median(episode_steps):.1f}")
    print()
    
    print("Individual Episode Results:")
    for detail in episode_details:
        print(f"  Episode {detail['episode']:2d}: reward={detail['reward']:8.4f}, steps={detail['steps']:3d}")
    print()
    
    # Save results to file
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"REWARD CALCULATION - Non-deterministic Actions, Seed {SEED}\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of episodes: {NUM_EPISODES}\n")
        f.write(f"Configuration: decision_dt={DECISION_DT}, min_episode_steps={MIN_EPISODE_STEPS}\n")
        f.write(f"Model: Untrained (random weights)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("STATISTICS ACROSS ALL EPISODES\n")
        f.write("="*80 + "\n")
        f.write(f"\nTotal episodes: {NUM_EPISODES}\n")
        f.write("\n")
        
        f.write("Reward Statistics:\n")
        f.write(f"  Mean:   {np.mean(episode_rewards):.4f}\n")
        f.write(f"  Std:    {np.std(episode_rewards):.4f}\n")
        f.write(f"  Min:    {np.min(episode_rewards):.4f}\n")
        f.write(f"  Max:    {np.max(episode_rewards):.4f}\n")
        f.write(f"  Median: {np.median(episode_rewards):.4f}\n")
        f.write("\n")
        
        f.write("Episode Steps Statistics:\n")
        f.write(f"  Mean:   {np.mean(episode_steps):.1f}\n")
        f.write(f"  Std:    {np.std(episode_steps):.1f}\n")
        f.write(f"  Min:    {np.min(episode_steps)}\n")
        f.write(f"  Max:    {np.max(episode_steps)}\n")
        f.write(f"  Median: {np.median(episode_steps):.1f}\n")
        f.write("\n")
        
        f.write("Individual Episode Results:\n")
        for detail in episode_details:
            f.write(f"  Episode {detail['episode']:2d}: reward={detail['reward']:8.4f}, steps={detail['steps']:3d}\n")
        f.write("\n")
    
    print(f"Results saved to: {log_file}")
    
    # =========================================================================
    # TRAIN MODEL FOR ONE BATCH (n_steps=128)
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING MODEL FOR ONE BATCH (n_steps=128)")
    print("="*80)
    print()
    
    # Train for one batch (n_steps=128 means it will collect 128 steps per batch)
    print("Starting training...")
    model.learn(total_timesteps=128, log_interval=1)
    print("\nTraining complete!")
    print("Model RNG state has been advanced by the training process")
    print()
    
    # =========================================================================
    # SAVE AND LOAD MODEL AFTER TRAINING
    # =========================================================================
    print("\n" + "="*80)
    print("SAVING MODEL AFTER TRAINING")
    print("="*80)
    
    model_path = f"model_trained_seed100_{timestamp}.zip"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    print("NOTE: After save, we will close env and load model")
    print()
    
    # Close the training environment
    env.close()
    print("Training environment closed.")
    
    # # =========================================================================
    # # LOAD MODEL FROM DISK (THIS WILL RESET RNG STATE)
    # # =========================================================================
    print("\n" + "="*80)
    print("LOADING MODEL FROM DISK")
    print("="*80)
    
    model = PPO.load(model_path)
    print(f"Model loaded from: {model_path}")
    print("WARNING: PPO.load() resets the RNG state to initial seed")
    print("         This is the key issue in train_and_eval_seed100.py!")
    print()
    
    # =========================================================================
    # EVALUATE TRAINED MODEL (5 RUNS, NON-DETERMINISTIC)
    # =========================================================================
    print("\n" + "="*80)
    print("EVALUATING TRAINED MODEL - 5 RUNS (NON-DETERMINISTIC)")
    print("CREATING NEW ENVIRONMENT FOR EACH RUN")
    print("="*80)
    print()
    
    NUM_EVAL_RUNS = 5
    eval_rewards = []
    eval_steps = []
    eval_details = []
    
    for eval_run in range(NUM_EVAL_RUNS):
        print(f"\nEvaluation Run {eval_run+1}/{NUM_EVAL_RUNS}")
        print("-"*80)
        
        # Create NEW SUMO instance for this evaluation run
        print(f"Starting NEW SUMO instance with seed={SEED}...")
        traci_eval = start_sumo(
            SUMO_CFG, 
            use_gui=USE_GUI,
            extra_args=["--seed", str(SEED), "--device.taxi.dispatch-algorithm", "traci"],
            label=f"eval_{eval_run}", 
            port=8820 + eval_run
        )
        
        # Create NEW reset function for this evaluation
        reset_fn_eval = FixedSeedResetFn(
            SUMO_CFG, 
            use_gui=USE_GUI,
            seed=SEED,
            traci_connection=traci_eval
        )
        
        # Setup NEW logger for this evaluation
        rp_logger_eval = RidepoolLogger(
            RidepoolLogConfig(
                out_dir="runs",
                run_name=f"rewards_nondet_eval_{timestamp}_run{eval_run}",
                erase_run_dir_on_start=True,
                erase_episode_dir_on_start=True,
                console_debug=False
            )
        )
        
        # Create NEW controller for this evaluation
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
        
        # Create NEW environment for this evaluation
        env_eval = RidepoolRTEnv(
            controller_eval,
            R=R, K_max=K_max, N_max=N_max, E_max=E_max,
            F=F, G=G,
            feature_fn=feature_fn_eval,
            global_stats_fn=None, 
            decision_dt=DECISION_DT,  
        )
        env_eval = Monitor(env_eval, filename=f"monitor_eval_{timestamp}_run{eval_run}.csv", info_keywords=("episode_reward",))
        
        # Run one episode
        obs, info = env_eval.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        step_rewards = []
        
        while not done:
            # Use NON-DETERMINISTIC prediction (stochastic)
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env_eval.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            step_rewards.append(reward)
            
            if step_count <= 5 or step_count % 5 == 0:
                print(f"  Step {step_count}: action={action}, reward={reward:.4f}, macro_steps={info.get('macro_steps')}")
        
        print(f"\nEvaluation Run {eval_run+1} Summary:")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Steps: {step_count}")
        print(f"  Step rewards: min={np.min(step_rewards):.4f}, max={np.max(step_rewards):.4f}, mean={np.mean(step_rewards):.4f}")
        
        eval_rewards.append(episode_reward)
        eval_steps.append(step_count)
        eval_details.append({
            'run': eval_run + 1,
            'reward': episode_reward,
            'steps': step_count,
            'step_rewards': step_rewards
        })
        
        # Close this evaluation environment
        env_eval.close()
    
    # Calculate evaluation statistics
    print("\n" + "="*80)
    print("EVALUATION STATISTICS (UNTRAINED MODEL)")
    print("="*80)
    print()
    
    print(f"Total evaluation runs: {NUM_EVAL_RUNS}")
    print()
    
    print("Reward Statistics:")
    print(f"  Mean:   {np.mean(eval_rewards):.4f}")
    print(f"  Std:    {np.std(eval_rewards):.4f}")
    print(f"  Min:    {np.min(eval_rewards):.4f}")
    print(f"  Max:    {np.max(eval_rewards):.4f}")
    print(f"  Median: {np.median(eval_rewards):.4f}")
    print()
    
    print("Episode Steps Statistics:")
    print(f"  Mean:   {np.mean(eval_steps):.1f}")
    print(f"  Std:    {np.std(eval_steps):.1f}")
    print(f"  Min:    {np.min(eval_steps)}")
    print(f"  Max:    {np.max(eval_steps)}")
    print(f"  Median: {np.median(eval_steps):.1f}")
    print()
    
    print("Individual Evaluation Results:")
    for detail in eval_details:
        print(f"  Run {detail['run']:2d}: reward={detail['reward']:8.4f}, steps={detail['steps']:3d}")
    print()
    
    # Comparison across two sets of runs (both untrained)
    print("="*80)
    print("COMPARISON: UNTRAINED (FIRST 5) vs TRAINED (NEXT 5)")
    print("="*80)
    print()
    print(f"First 5 episodes (untrained model, no save/load):")
    print(f"  Mean reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"  Mean steps:  {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
    print()
    print(f"Next 5 episodes (after training 1 batch with n_steps=128, fresh environment each time):")
    print(f"  Mean reward: {np.mean(eval_rewards):.4f} ± {np.std(eval_rewards):.4f}")
    print(f"  Mean steps:  {np.mean(eval_steps):.1f} ± {np.std(eval_steps):.1f}")
    print()
    print(f"Training Effect (NO save/load - RNG continues):")
    print(f"  Reward change: {np.mean(eval_rewards) - np.mean(episode_rewards):+.4f} ({((np.mean(eval_rewards) / np.mean(episode_rewards) - 1) * 100):+.1f}%)")
    print(f"  Steps change:  {np.mean(eval_steps) - np.mean(episode_steps):+.1f}")
    print()
    
    # Append evaluation results to log file
    with open(log_file, 'a') as f:
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("EVALUATION STATISTICS (UNTRAINED MODEL)\n")
        f.write("="*80 + "\n")
        f.write(f"\nTotal evaluation runs: {NUM_EVAL_RUNS}\n")
        f.write("\n")
        
        f.write("Reward Statistics:\n")
        f.write(f"  Mean:   {np.mean(eval_rewards):.4f}\n")
        f.write(f"  Std:    {np.std(eval_rewards):.4f}\n")
        f.write(f"  Min:    {np.min(eval_rewards):.4f}\n")
        f.write(f"  Max:    {np.max(eval_rewards):.4f}\n")
        f.write(f"  Median: {np.median(eval_rewards):.4f}\n")
        f.write("\n")
        
        f.write("Episode Steps Statistics:\n")
        f.write(f"  Mean:   {np.mean(eval_steps):.1f}\n")
        f.write(f"  Std:    {np.std(eval_steps):.1f}\n")
        f.write(f"  Min:    {np.min(eval_steps)}\n")
        f.write(f"  Max:    {np.max(eval_steps)}\n")
        f.write(f"  Median: {np.median(eval_steps):.1f}\n")
        f.write("\n")
        
        f.write("Individual Evaluation Results:\n")
        for detail in eval_details:
            f.write(f"  Run {detail['run']:2d}: reward={detail['reward']:8.4f}, steps={detail['steps']:3d}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("COMPARISON: UNTRAINED (FIRST 5) vs TRAINED (NEXT 5)\n")
        f.write("="*80 + "\n")
        f.write(f"\nFirst 5 episodes (untrained model, no save/load):\n")
        f.write(f"  Mean reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}\n")
        f.write(f"  Mean steps:  {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}\n")
        f.write("\n")
        f.write(f"Next 5 episodes (after training 1 batch with n_steps=128, fresh environment each time):\n")
        f.write(f"  Mean reward: {np.mean(eval_rewards):.4f} ± {np.std(eval_rewards):.4f}\n")
        f.write(f"  Mean steps:  {np.mean(eval_steps):.1f} ± {np.std(eval_steps):.1f}\n")
        f.write("\n")
        f.write(f"Training Effect (NO save/load - RNG continues):\n")
        f.write(f"  Reward change: {np.mean(eval_rewards) - np.mean(episode_rewards):+.4f} ({((np.mean(eval_rewards) / np.mean(episode_rewards) - 1) * 100):+.1f}%)\n")
        f.write(f"  Steps change:  {np.mean(eval_steps) - np.mean(episode_steps):+.1f}\n")
        f.write("\n")
    
    print(f"Updated results saved to: {log_file}")


if __name__ == "__main__":
    main()
