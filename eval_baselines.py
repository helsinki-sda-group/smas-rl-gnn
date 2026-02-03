import numpy as np
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List
from dataclasses import dataclass, field, asdict
import pandas as pd
import os

from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from utils.sumo_bootstrap import start_sumo, make_reset_fn
from utils.feature_fns import make_feature_fn

# Metrics dataclass
@dataclass
class EpisodeMetrics:
    policy: str = ""
    seed: int = 0
    reward_sum: float = 0.0
    capacity_sum: float = 0.0
    step_sum: float = 0.0
    missed_deadline_sum: float = 0.0
    wait_sum: float = 0.0
    completion_sum: float = 0.0
    nonserved_sum: float = 0.0
    
    total_tasks: int = 0
    picked_up_tasks: int = 0
    pickup_rate: float = 0.0
    
    obsolete_tasks: int = 0
    obsolete_rate: float = 0.0
    
    pickup_violated: int = 0
    pickup_violated_rate: float = 0.0
    
    mean_wait_time: float = 0.0
    completed_tasks: int = 0
    completion_rate: float = 0.0
    
    assigned_never_picked: int = 0
    assigned_never_picked_rate: float = 0.0
    
    mean_travel_time_completed: float = 0.0
    picked_not_completed: int = 0
    picked_not_completed_rate: float = 0.0
    
    # New training metrics
    noop_fraction: float = 0.0
    overload_assignment_fraction: float = 0.0
    mean_candidates_per_taxi: float = 0.0
    cand_nonempty_frac: float = 0.0
    cand_mean_nonempty: float = 0.0
    decision_steps: int = 0
    macro_reward_mean: float = 0.0
    macro_steps_done: int = 0

# 1) SUMO/controller setup (example; adapt to your config)
SUMO_CFG = "configs/small_net.sumocfg"
USE_GUI = False
R = 5           # number of robots (taxis) expected. # should match to taxis.rou.xml
K_max = 3        # candidates per robot
N_max = 16        # max nodes per ego-graph (robot + tasks in its neighborhood)
E_max = 64        # max edges per ego-graph
F = 9            # node feature dimension (robot node and task node should have the same dimensionality, padding is applied)
G = 0             # global stats dim 

VICINITY_M = 2000.0
MAX_STEPS = 1200
MAX_WAIT_DELAY_S = 240.0
MAX_TRAVEL_DELAY_S = 900.0
MAX_ROBOT_CAPACITY = 2

NUM_SEEDS = 3  # Number of seeds to use (first N from SEEDS list)
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
POLICIES = ["random", "greedy", "unique"]

def compute_episode_metrics(episode_dir: str, episode_info: Dict, policy: str, seed: int) -> EpisodeMetrics:
    """
    Compute comprehensive metrics from CSV log files.
    
    Args:
        episode_dir: path to episode directory containing task_lifecycle.csv, taxi_events.csv, rewards_macro.csv
        episode_info: info dict from environment (for reward_sum if available)
        policy: policy name
        seed: seed used for this run
    """
    metrics = EpisodeMetrics(policy=policy, seed=seed)
    
    # Try to get reward from rewards_macro.csv
    rewards_macro_path = os.path.join(episode_dir, "rewards_macro.csv")
    if os.path.exists(rewards_macro_path):
        try:
            df_rewards = pd.read_csv(rewards_macro_path)
            metrics.reward_sum = float(df_rewards["reward"].sum())
            metrics.capacity_sum = float(df_rewards["capacity_avg"].sum())
            metrics.step_sum = float(df_rewards["step_avg"].sum())
            metrics.missed_deadline_sum = float(df_rewards["missed_deadline_avg"].sum())
            metrics.wait_sum = float(df_rewards["wait_avg"].sum())
            metrics.completion_sum = float(df_rewards["completion_avg"].sum())
            metrics.nonserved_sum = float(df_rewards["nonserved_avg"].sum())
        except Exception as e:
            print(f"Warning: Could not read rewards_macro.csv: {e}")
    else:
        # Fallback to episode_info
        metrics.reward_sum = episode_info.get("episode_reward", 0.0)
    
    # Read task lifecycle
    task_lifecycle_path = os.path.join(episode_dir, "task_lifecycle.csv")
    if not os.path.exists(task_lifecycle_path):
        return metrics
    
    try:
        df_lifecycle = pd.read_csv(task_lifecycle_path)
    except Exception as e:
        print(f"Warning: Could not read task_lifecycle.csv: {e}")
        return metrics
    
    if df_lifecycle.empty:
        return metrics
    
    # Convert relevant columns to numeric
    df_lifecycle['actual_pickup_time'] = pd.to_numeric(df_lifecycle['actual_pickup_time'], errors='coerce')
    df_lifecycle['actual_dropoff_time'] = pd.to_numeric(df_lifecycle['actual_dropoff_time'], errors='coerce')
    df_lifecycle['reservation_time'] = pd.to_numeric(df_lifecycle['reservation_time'], errors='coerce')
    df_lifecycle['pickup_deadline'] = pd.to_numeric(df_lifecycle['pickup_deadline'], errors='coerce')
    df_lifecycle['actual_waiting_time'] = pd.to_numeric(df_lifecycle['actual_waiting_time'], errors='coerce')
    df_lifecycle['actual_travel_time'] = pd.to_numeric(df_lifecycle['actual_travel_time'], errors='coerce')
    df_lifecycle['was_obsolete'] = df_lifecycle['was_obsolete'].astype(str).str.lower() == 'true'
    df_lifecycle['assigned_taxi'] = df_lifecycle['assigned_taxi'].fillna('')
    
    total_tasks = len(df_lifecycle)
    metrics.total_tasks = total_tasks
    
    # Pickup status
    picked_up_mask = df_lifecycle['actual_pickup_time'].notna()
    picked_up = picked_up_mask.sum()
    metrics.picked_up_tasks = int(picked_up)
    metrics.pickup_rate = picked_up / total_tasks if total_tasks > 0 else 0.0
    
    # Obsolete tasks
    obsolete = df_lifecycle['was_obsolete'].sum()
    metrics.obsolete_tasks = int(obsolete)
    metrics.obsolete_rate = obsolete / total_tasks if total_tasks > 0 else 0.0
    
    # Pickup deadline violations (only among picked up tasks)
    if picked_up > 0:
        picked_df = df_lifecycle[picked_up_mask].copy()
        violated = (picked_df['actual_pickup_time'] > picked_df['pickup_deadline']).sum()
        metrics.pickup_violated = int(violated)
        metrics.pickup_violated_rate = violated / picked_up
    else:
        metrics.pickup_violated = 0
        metrics.pickup_violated_rate = 0.0
    
    # Wait times (for picked up tasks)
    if picked_up > 0:
        wait_times = df_lifecycle[picked_up_mask]['actual_waiting_time'].dropna()
        metrics.mean_wait_time = float(wait_times.mean()) if len(wait_times) > 0 else 0.0
    else:
        metrics.mean_wait_time = 0.0
    
    # Completion (picked up and dropped off)
    completed_mask = df_lifecycle['actual_dropoff_time'].notna()
    completed = completed_mask.sum()
    metrics.completed_tasks = int(completed)
    metrics.completion_rate = completed / total_tasks if total_tasks > 0 else 0.0
    
    # Travel times (for completed tasks)
    if completed > 0:
        travel_times = df_lifecycle[completed_mask]['actual_travel_time'].dropna()
        metrics.mean_travel_time_completed = float(travel_times.mean()) if len(travel_times) > 0 else 0.0
    else:
        metrics.mean_travel_time_completed = 0.0
    
    # Assigned but never picked (assigned_taxi is not empty but actual_pickup_time is NaN)
    assigned_mask = df_lifecycle['assigned_taxi'].str.len() > 0
    assigned_but_not_picked = (assigned_mask & ~picked_up_mask).sum()
    metrics.assigned_never_picked = int(assigned_but_not_picked)
    metrics.assigned_never_picked_rate = assigned_but_not_picked / total_tasks if total_tasks > 0 else 0.0
    
    # Picked but not completed
    picked_not_completed = (picked_up_mask & ~completed_mask).sum()
    metrics.picked_not_completed = int(picked_not_completed)
    metrics.picked_not_completed_rate = picked_not_completed / picked_up if picked_up > 0 else 0.0
    
    # NEW METRICS: Compute training-related metrics from debug.csv, candidates.csv, and rewards_macro.csv
    
    # 1) NO-OP FRACTION: Count assignments that are null (NOOP actions)
    debug_path = os.path.join(episode_dir, "debug.csv")
    noop_count = 0
    total_steps = 0
    if os.path.exists(debug_path):
        try:
            df_debug = pd.read_csv(debug_path)
            # Filter for apply-input entries which show the assignments for each step
            apply_input = df_debug[df_debug['tag'] == 'apply-input']
            if len(apply_input) > 0:
                total_steps = len(apply_input)
                for idx, row in apply_input.iterrows():
                    try:
                        import json
                        data = json.loads(row['data'])
                        assignments = data.get('assignments_raw', [])
                        # Count how many are None (NOOP)
                        noop_count += sum(1 for a in assignments if a is None)
                    except Exception:
                        pass
            metrics.noop_fraction = noop_count / (total_steps * R) if total_steps > 0 else 0.0
        except Exception as e:
            print(f"Warning: Could not compute noop_fraction from debug.csv: {e}")
    
    # 2) OVERLOAD-ASSIGNMENT FRACTION: Steps where taxi has unpicked assignment AND gets new assignment
    #    This requires tracking unpicked count per step - we'll estimate from candidates count
    overload_count = 0
    if os.path.exists(debug_path):
        try:
            df_debug = pd.read_csv(debug_path)
            apply_input_rows = df_debug[df_debug['tag'] == 'apply-input'].values
            apply_winners_rows = df_debug[df_debug['tag'] == 'apply-winners'].values
            
            # Match apply-input with apply-winners to see when assignments were made
            for i, (inp_row, winner_row) in enumerate(zip(apply_input_rows, apply_winners_rows)):
                try:
                    import json
                    inp_data = json.loads(inp_row[2])
                    winner_data = json.loads(winner_row[2])
                    
                    winners = winner_data.get('winners', {})
                    cand_counts = inp_data.get('cand_counts', [])
                    
                    # Count steps where there's both an unpicked assignment (cand_count > 0)
                    # and a new assignment was made (winners is not empty)
                    if winners and cand_counts:
                        # At least one new assignment and at least one taxi with candidates
                        overload_count += 1
                except Exception:
                    pass
            
            metrics.overload_assignment_fraction = overload_count / total_steps if total_steps > 0 else 0.0
        except Exception as e:
            print(f"Warning: Could not compute overload_assignment_fraction: {e}")
    
    # 3) MEAN CANDIDATES PER TAXI: Average candidates per taxi across all steps
    #    cand_res_ids is pipe-separated list, count pipes+1 to get number of candidates
    candidates_path = os.path.join(episode_dir, "candidates.csv")
    if os.path.exists(candidates_path):
        try:
            df_candidates = pd.read_csv(candidates_path)
            if len(df_candidates) > 0:
                # For each row, count the number of candidates by counting pipe-separated values
                # Must check pd.notna BEFORE converting to string (astype converts NaN to 'nan' string)
                candidate_counts = df_candidates['cand_res_ids'].apply(
                    lambda x: len(str(x).split('|')) if pd.notna(x) and str(x).strip() else 0
                )
                metrics.mean_candidates_per_taxi = float(candidate_counts.mean()) if len(candidate_counts) > 0 else 0.0
                
                # NEW: cand_nonempty_frac and cand_mean_nonempty
                nonempty_mask = candidate_counts > 0
                metrics.cand_nonempty_frac = float(nonempty_mask.sum() / len(candidate_counts)) if len(candidate_counts) > 0 else 0.0
                nonempty_counts = candidate_counts[nonempty_mask]
                metrics.cand_mean_nonempty = float(nonempty_counts.mean()) if len(nonempty_counts) > 0 else 0.0
                # decision_steps: macro-steps where at least one taxi has non-empty candidates
                df_candidates = df_candidates.assign(_cand_count=candidate_counts)
                decision_steps = (
                    df_candidates.groupby("time")
                    .apply(lambda g: (g["_cand_count"] > 0).any())
                    .sum()
                )
                metrics.decision_steps = int(decision_steps)
            else:
                metrics.mean_candidates_per_taxi = 0.0
                metrics.cand_nonempty_frac = 0.0
                metrics.cand_mean_nonempty = 0.0
                metrics.decision_steps = 0
        except Exception as e:
            print(f"Warning: Could not compute mean_candidates_per_taxi: {e}")
    
    # 4) MACRO REWARD MEAN: Mean reward per macro step
    if os.path.exists(rewards_macro_path):
        try:
            df_rewards = pd.read_csv(rewards_macro_path)
            if len(df_rewards) > 0:
                metrics.macro_reward_mean = float(df_rewards["reward"].mean())
                metrics.macro_steps_done = len(df_rewards)
        except Exception as e:
            print(f"Warning: Could not compute macro_reward_mean and macro_steps_done: {e}")
    
    return metrics

def metrics_to_string(metrics: EpisodeMetrics) -> str:
    """Format metrics as tab-separated string with fixed-width columns."""
    return (
        f"{metrics.policy:<10} {metrics.seed:>4} | "
        f"{metrics.reward_sum:>8.2f} {metrics.capacity_sum:>8.2f} {metrics.step_sum:>8.2f}"
        f" {metrics.missed_deadline_sum:>8.2f} {metrics.wait_sum:>8.2f}"
        f" {metrics.completion_sum:>8.2f} {metrics.nonserved_sum:>8.2f} | "
        f"{metrics.picked_up_tasks:>2}/{metrics.total_tasks:<2} {metrics.pickup_rate:>6.2f}"
        f" {metrics.obsolete_tasks:>2} {metrics.obsolete_rate:>6.2f}"
        f" {metrics.pickup_violated:>2}/{metrics.picked_up_tasks:<2} {metrics.pickup_violated_rate:>6.2f}"
        f" {metrics.mean_wait_time:>7.2f}"
        f" {metrics.completed_tasks:>2}/{metrics.total_tasks:<2} {metrics.completion_rate:>6.2f}"
        f" {metrics.assigned_never_picked:>2}/{metrics.total_tasks:<2} {metrics.assigned_never_picked_rate:>6.2f}"
        f" {metrics.mean_travel_time_completed:>7.2f}"
        f" {metrics.picked_not_completed:>2}/{metrics.picked_up_tasks:<2} {metrics.picked_not_completed_rate:>6.2f} | "
        f" {metrics.noop_fraction:>6.3f} {metrics.overload_assignment_fraction:>6.3f}"
        f" {metrics.mean_candidates_per_taxi:>6.2f} {metrics.cand_nonempty_frac:>6.3f}"
        f" {metrics.cand_mean_nonempty:>6.2f} {metrics.decision_steps:>6}"
        f" {metrics.macro_reward_mean:>8.3f} {metrics.macro_steps_done:>6}"
    )

def get_metrics_header() -> str:
    """Get header for metrics log with fixed-width alignment."""
    return (
        "pol        seed |      rew      cap     step      mdl     wait     comp      nsv |   pku    pkr obs  obsr   pkv   pkvr    mwt   cmp    cmr   anp   anpr     mtt   pnc    pncr |   noop  overld  mcand  cne_fr cne_mn   dstep    macmr    msd"
    )

# Initialize metrics log file
metrics_log_path = (
    f"metrics_v{int(VICINITY_M)}_ms{MAX_STEPS}_mwd{int(MAX_WAIT_DELAY_S)}_"
    f"mtd{int(MAX_TRAVEL_DELAY_S)}_cap{MAX_ROBOT_CAPACITY}.log"
)
with open(metrics_log_path, "w", encoding="utf-8") as f:
    f.write(f"vicinity_m={VICINITY_M}, max_steps={MAX_STEPS}, max_wait_delay_s={MAX_WAIT_DELAY_S}, "
            f"max_travel_delay_s={MAX_TRAVEL_DELAY_S}, max_robot_capacity={MAX_ROBOT_CAPACITY}\n")
    f.write(get_metrics_header() + "\n")

all_metrics_by_policy: Dict[str, List[EpisodeMetrics]] = {p: [] for p in POLICIES}

# Run evaluations for first NUM_SEEDS seeds
for seed in SEEDS[:NUM_SEEDS]:
    print(f"\n{'='*80}")
    print(f"Starting seed {seed}")
    print(f"{'='*80}")
    
    traci = start_sumo(SUMO_CFG, use_gui=False,
                       extra_args=[f"--seed", str(seed), "--device.taxi.dispatch-algorithm", "traci"])

    rp_logger = RidepoolLogger(
        RidepoolLogConfig(
            out_dir="runs",
            run_name=f"rp_eval_seed{seed}",
            erase_run_dir_on_start=True,
            erase_episode_dir_on_start=True,
            console_debug=False
        )
    )

    controller = RLControllerAdapter(
        sumo=traci,
        reset_fn=make_reset_fn(SUMO_CFG, use_gui=False,
                               extra_args=[f"--seed", str(seed), "--device.taxi.dispatch-algorithm", "traci"]),
        k_max=K_max,
        vicinity_m=VICINITY_M,
        completion_mode="dropoff",
        max_steps=MAX_STEPS,
        min_episode_steps=100,
        serve_to_empty=True,
        require_seen_reservation=True,
        max_wait_delay_s=MAX_WAIT_DELAY_S,
        max_travel_delay_s=MAX_TRAVEL_DELAY_S,
        max_robot_capacity=MAX_ROBOT_CAPACITY,
        logger=rp_logger,
        respect_sumo_end=True,
    )
    feature_fn = make_feature_fn(controller)

    env = RidepoolRTEnv(
        controller,
        R=R, K_max=K_max, N_max=N_max, E_max=E_max,
        F=F, G=0,
        feature_fn=feature_fn,
        global_stats_fn=None,
        decision_dt=60,
    )

    NOOP = K_max

    def greedy_nearest_action(action_mask: np.ndarray) -> np.ndarray:
        a = np.full((R,), NOOP, dtype=np.int64)
        for r in range(R):
            if action_mask[r,0] == 1:
                a[r] = 0
            else:
                a[r] = NOOP
        return a

    _rnd = np.random.default_rng(seed)

    def random_valid_action(action_mask: np.ndarray) -> np.ndarray:
        a = np.full((R,), NOOP, dtype=np.int64)
        for r in range(R):
            allowed = np.flatnonzero(action_mask[r] == 1)
            if allowed.size > 0:
                a[r] = int(_rnd.choice(allowed))
            else:
                a[r] = NOOP
        return a

    def greedy_unique_action(action_mask: np.ndarray) -> np.ndarray:
        env0 = env.unwrapped
        cand_ids = getattr(env0, "_last_cand_task_ids", None)
        if cand_ids is None:
            return greedy_nearest_action(action_mask)

        chosen = set()
        a = np.full((action_mask.shape[0],), NOOP, dtype=np.int64)

        for r in range(action_mask.shape[0]):
            for k in range(K_max):
                if action_mask[r, k] != 1:
                    continue
                task_id = int(cand_ids[r][k])
                if task_id < 0:
                    continue
                if task_id in chosen:
                    continue
                chosen.add(task_id)
                a[r] = k
                break

        return a

    # Run one episode per policy for this seed
    for policy_name in POLICIES:
        print(f"\n  Policy: {policy_name}")
        
        obs, info = env.reset()
        done = False
        trunc = False

        while not (done or trunc):
            mask = info.get("action_mask", env.unwrapped.action_mask())

            if policy_name == "greedy":
                action = greedy_nearest_action(mask)
            elif policy_name == "random":
                action = random_valid_action(mask)
            elif policy_name == "unique":
                action = greedy_unique_action(mask)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")
            
            obs, reward, done, trunc, info = env.step(action)

        # Get episode directory from logger
        episode_dir = rp_logger.ep_dir
        
        # Compute metrics from CSV files
        metrics = compute_episode_metrics(episode_dir, info, policy_name, seed)
        all_metrics_by_policy[policy_name].append(metrics)
        
        # Log to file
        with open(metrics_log_path, "a", encoding="utf-8") as f:
            f.write(metrics_to_string(metrics) + "\n")
        
        print(f"    Reward: {metrics.reward_sum:.2f}")
        print(f"    Pickups: {metrics.picked_up_tasks}/{metrics.total_tasks} ({metrics.pickup_rate:.1%})")
        print(f"    Completed: {metrics.completed_tasks}/{metrics.total_tasks} ({metrics.completion_rate:.1%})")
        print(f"    Pickup violations: {metrics.pickup_violated_rate:.1%}")

    with open(metrics_log_path, "a", encoding="utf-8") as f:
        f.write("-" * 215 + "\n")

    traci.close()

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS BY POLICY")
print(f"{'='*80}")

summary_path = metrics_log_path
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("\n\n# SUMMARY STATISTICS\n")
    f.write("pol           rew±std   |     cap±std       step±std     mdl±std       wait±std      comp±std       nsv±std   |     pkr±std       obsr±std      pkvr±std      mwt±std       cmr±std       anpr±std      pncr±std |  noop±std   overld±std   mcand±std  cne_fr±std cne_mn±std  dstep±std    macmr±std     msd±std\n")
    
    for policy_name in POLICIES:
        metrics_list = all_metrics_by_policy[policy_name]
        if not metrics_list:
            continue
        
        rewards = [m.reward_sum for m in metrics_list]
        caps = [m.capacity_sum for m in metrics_list]
        steps = [m.step_sum for m in metrics_list]
        mdls = [m.missed_deadline_sum for m in metrics_list]
        waits = [m.wait_sum for m in metrics_list]
        comps = [m.completion_sum for m in metrics_list]
        nsvs = [m.nonserved_sum for m in metrics_list]
        pickup_rates = [m.pickup_rate for m in metrics_list]
        obsolete_rates = [m.obsolete_rate for m in metrics_list]
        pickup_violated_rates = [m.pickup_violated_rate for m in metrics_list]
        wait_times = [m.mean_wait_time for m in metrics_list]
        completion_rates = [m.completion_rate for m in metrics_list]
        assigned_never_picked_rates = [m.assigned_never_picked_rate for m in metrics_list]
        picked_not_completed_rates = [m.picked_not_completed_rate for m in metrics_list]
        noop_fractions = [m.noop_fraction for m in metrics_list]
        overload_fractions = [m.overload_assignment_fraction for m in metrics_list]
        mean_candidates = [m.mean_candidates_per_taxi for m in metrics_list]
        cand_nonempty_fracs = [m.cand_nonempty_frac for m in metrics_list]
        cand_mean_nonemptys = [m.cand_mean_nonempty for m in metrics_list]
        decision_steps_list = [m.decision_steps for m in metrics_list]
        macro_rewards = [m.macro_reward_mean for m in metrics_list]
        macro_steps = [m.macro_steps_done for m in metrics_list]

        summary_line = (
            f"{policy_name:<10}"
            f" {np.mean(rewards):>6.2f}±{np.std(rewards):<5.2f} | "
            f" {np.mean(caps):>6.2f}±{np.std(caps):<5.2f}"
            f"  {np.mean(steps):>6.2f}±{np.std(steps):<5.2f}"
            f"  {np.mean(mdls):>6.2f}±{np.std(mdls):<5.2f}"
            f"  {np.mean(waits):>6.2f}±{np.std(waits):<5.2f}"
            f"  {np.mean(comps):>6.2f}±{np.std(comps):<5.2f}"
            f"  {np.mean(nsvs):>6.2f}±{np.std(nsvs):<5.2f} | "
            f" {np.mean(pickup_rates):>6.2f}±{np.std(pickup_rates):<5.2f}"
            f"  {np.mean(obsolete_rates):>6.2f}±{np.std(obsolete_rates):<5.2f}"
            f"  {np.mean(pickup_violated_rates):>6.2f}±{np.std(pickup_violated_rates):<5.2f}"
            f"  {np.mean(wait_times):>6.2f}±{np.std(wait_times):<5.2f}"
            f"  {np.mean(completion_rates):>6.2f}±{np.std(completion_rates):<5.2f}"
            f"  {np.mean(assigned_never_picked_rates):>6.2f}±{np.std(assigned_never_picked_rates):<5.2f}"
            f"  {np.mean(picked_not_completed_rates):>6.2f}±{np.std(picked_not_completed_rates):<5.2f} | "
            f" {np.mean(noop_fractions):>6.3f}±{np.std(noop_fractions):<5.3f}"
            f"  {np.mean(overload_fractions):>6.3f}±{np.std(overload_fractions):<5.3f}"
            f"  {np.mean(mean_candidates):>6.2f}±{np.std(mean_candidates):<5.2f}"
            f"  {np.mean(cand_nonempty_fracs):>6.3f}±{np.std(cand_nonempty_fracs):<5.3f}"
            f"  {np.mean(cand_mean_nonemptys):>6.2f}±{np.std(cand_mean_nonemptys):<5.2f}"
            f"  {np.mean(decision_steps_list):>6.1f}±{np.std(decision_steps_list):<5.1f}"
            f"  {np.mean(macro_rewards):>6.3f}±{np.std(macro_rewards):<5.3f}"
            f"  {np.mean(macro_steps):>6.1f}±{np.std(macro_steps):<5.1f}"
        )
        
        f.write(summary_line + "\n")
        print(f"\n{policy_name}:")
        print(f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Pickup Rate: {np.mean(pickup_rates):.1%} ± {np.std(pickup_rates):.1%}")
        print(f"  Obsolete Rate: {np.mean(obsolete_rates):.1%} ± {np.std(obsolete_rates):.1%}")
        print(f"  Pickup Violated Rate: {np.mean(pickup_violated_rates):.1%} ± {np.std(pickup_violated_rates):.1%}")
        print(f"  Mean Wait Time: {np.mean(wait_times):.2f} ± {np.std(wait_times):.2f}")
        print(f"  Completion Rate: {np.mean(completion_rates):.1%} ± {np.std(completion_rates):.1%}")
        print(f"  Assigned-Not-Picked Rate: {np.mean(assigned_never_picked_rates):.1%} ± {np.std(assigned_never_picked_rates):.1%}")
        print(f"  Picked-Not-Completed Rate: {np.mean(picked_not_completed_rates):.1%} ± {np.std(picked_not_completed_rates):.1%}")
        print(f"  NOOP Fraction: {np.mean(noop_fractions):.3f} ± {np.std(noop_fractions):.3f}")
        print(f"  Overload Assignment Fraction: {np.mean(overload_fractions):.3f} ± {np.std(overload_fractions):.3f}")
        print(f"  Mean Candidates per Taxi: {np.mean(mean_candidates):.2f} ± {np.std(mean_candidates):.2f}")
        print(f"  Cand Nonempty Fraction: {np.mean(cand_nonempty_fracs):.3f} ± {np.std(cand_nonempty_fracs):.3f}")
        print(f"  Cand Mean (Nonempty): {np.mean(cand_mean_nonemptys):.2f} ± {np.std(cand_mean_nonemptys):.2f}")
        print(f"  Decision Steps: {np.mean(decision_steps_list):.1f} ± {np.std(decision_steps_list):.1f}")
        print(f"  Macro Reward Mean: {np.mean(macro_rewards):.3f} ± {np.std(macro_rewards):.3f}")
        print(f"  Macro Steps Done: {np.mean(macro_steps):.1f} ± {np.std(macro_steps):.1f}")

    f.write("\n# METRIC LEGEND\n")
    f.write("short\tfull\n")
    legend_rows = [
        ("pol", "policy"),
        ("seed", "random seed"),
        ("rew", "reward_sum"),
        ("cap", "capacity_sum"),
        ("step", "step_sum"),
        ("mdl", "missed_deadline_sum"),
        ("wait", "wait_sum"),
        ("comp", "completion_sum"),
        ("nsv", "nonserved_sum"),
        ("pku", "pickups (picked/total)"),
        ("pkr", "pickup_rate"),
        ("obs", "obsolete"),
        ("obsr", "obsolete_rate"),
        ("pkv", "pickup_violated (violated/picked)"),
        ("pkvr", "pickup_violated_rate"),
        ("mwt", "mean_wait_time"),
        ("cmp", "completed (completed/total)"),
        ("cmr", "completion_rate"),
        ("anp", "assigned_never_picked (assigned_not_picked/total)"),
        ("anpr", "assigned_never_picked_rate"),
        ("mtt", "mean_travel_time"),
        ("pnc", "picked_not_completed (picked_not_completed/picked)"),
        ("pncr", "picked_not_completed_rate"),
        ("noop", "noop_fraction"),
        ("overld", "overload_assignment_fraction"),
        ("mcand", "mean_candidates_per_taxi"),
        ("cne_fr", "cand_nonempty_frac"),
        ("cne_mn", "cand_mean_nonempty"),
        ("dstep", "decision_steps (steps with any nonempty candidates)"),
        ("macmr", "macro_reward_mean"),
        ("msd", "macro_steps_done"),
    ]
    for short, full in legend_rows:
        f.write(f"{short}\t{full}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("SUMMARY STATISTICS BY POLICY\n")
    f.write("=" * 80 + "\n\n")
    for policy_name in POLICIES:
        metrics_list = all_metrics_by_policy[policy_name]
        if not metrics_list:
            continue

        rewards = [m.reward_sum for m in metrics_list]
        pickup_rates = [m.pickup_rate for m in metrics_list]
        obsolete_rates = [m.obsolete_rate for m in metrics_list]
        pickup_violated_rates = [m.pickup_violated_rate for m in metrics_list]
        wait_times = [m.mean_wait_time for m in metrics_list]
        completion_rates = [m.completion_rate for m in metrics_list]
        assigned_never_picked_rates = [m.assigned_never_picked_rate for m in metrics_list]
        picked_not_completed_rates = [m.picked_not_completed_rate for m in metrics_list]
        noop_fractions = [m.noop_fraction for m in metrics_list]
        overload_fractions = [m.overload_assignment_fraction for m in metrics_list]
        mean_candidates = [m.mean_candidates_per_taxi for m in metrics_list]
        cand_nonempty_fracs = [m.cand_nonempty_frac for m in metrics_list]
        cand_mean_nonemptys = [m.cand_mean_nonempty for m in metrics_list]
        decision_steps_list = [m.decision_steps for m in metrics_list]
        macro_rewards = [m.macro_reward_mean for m in metrics_list]
        macro_steps = [m.macro_steps_done for m in metrics_list]

        f.write(f"{policy_name}:\n")
        f.write(f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
        f.write(f"  Pickup Rate: {np.mean(pickup_rates)*100:.1f}% ± {np.std(pickup_rates)*100:.1f}%\n")
        f.write(f"  Obsolete Rate: {np.mean(obsolete_rates)*100:.1f}% ± {np.std(obsolete_rates)*100:.1f}%\n")
        f.write(f"  Pickup Violated Rate: {np.mean(pickup_violated_rates)*100:.1f}% ± {np.std(pickup_violated_rates)*100:.1f}%\n")
        f.write(f"  Mean Wait Time: {np.mean(wait_times):.2f} ± {np.std(wait_times):.2f}\n")
        f.write(f"  Completion Rate: {np.mean(completion_rates)*100:.1f}% ± {np.std(completion_rates)*100:.1f}%\n")
        f.write(f"  Assigned-Not-Picked Rate: {np.mean(assigned_never_picked_rates)*100:.1f}% ± {np.std(assigned_never_picked_rates)*100:.1f}%\n")
        f.write(f"  Picked-Not-Completed Rate: {np.mean(picked_not_completed_rates)*100:.1f}% ± {np.std(picked_not_completed_rates)*100:.1f}%\n")
        f.write(f"  NOOP Fraction: {np.mean(noop_fractions):.3f} ± {np.std(noop_fractions):.3f}\n")
        f.write(f"  Overload Assignment Fraction: {np.mean(overload_fractions):.3f} ± {np.std(overload_fractions):.3f}\n")
        f.write(f"  Mean Candidates per Taxi: {np.mean(mean_candidates):.2f} ± {np.std(mean_candidates):.2f}\n")
        f.write(f"  Cand Nonempty Fraction: {np.mean(cand_nonempty_fracs):.3f} ± {np.std(cand_nonempty_fracs):.3f}\n")
        f.write(f"  Cand Mean (Nonempty): {np.mean(cand_mean_nonemptys):.2f} ± {np.std(cand_mean_nonemptys):.2f}\n")
        f.write(f"  Decision Steps: {np.mean(decision_steps_list):.1f} ± {np.std(decision_steps_list):.1f}\n")
        f.write(f"  Macro Reward Mean: {np.mean(macro_rewards):.3f} ± {np.std(macro_rewards):.3f}\n")
        f.write(f"  Macro Steps Done: {np.mean(macro_steps):.1f} ± {np.std(macro_steps):.1f}\n\n")

print(f"\nMetrics saved to {metrics_log_path}")

 

