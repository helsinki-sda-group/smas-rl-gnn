import numpy as np
import argparse
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List
import pandas as pd
import os

from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from utils.sumo_bootstrap import start_sumo, make_reset_fn
from utils.feature_fns import make_feature_fn, compute_feature_dim, expand_edge_features
from utils.metrics_calculator import (
    EpisodeMetrics,
    POLICY_WIDTH,
    compute_episode_metrics_from_logs,
    metrics_to_string,
    get_metrics_header,
    build_metrics_metadata_lines,
)


SUPPORTED_BASELINE_POLICIES = {
    "random",
    "unique",
    "greedy",
    "pickup_distance",
    "pickup_deadline",
    "pickup_deadline_distance",
    "predicted_reward",
    "predicted_reward_joint",
}

# Shared slot-0 candidate baseline family.
SLOT0_SORTING_POLICY_MAP = {
    "greedy": "pickup_distance",
    "pickup_distance": "pickup_distance",
    "pickup_deadline": "pickup_deadline",
    "pickup_deadline_distance": "pickup_deadline_distance",
    "predicted_reward": "predicted_reward",
    "predicted_reward_joint": "predicted_reward_joint",
}

parser = argparse.ArgumentParser(description="Evaluate baseline policies")
parser.add_argument("--config", type=str, default="configs/rp_gnn.yaml", help="Path to config YAML")
parser.add_argument("--sumoport", type=int, default=None, help="SUMO remote port (default: SUMO default)")
parser.add_argument(
    "--candidates-sorting",
    type=str,
    default=None,
    choices=["pickup_distance", "pickup_deadline", "pickup_deadline_distance", "randomized", "predicted_reward", "predicted_reward_joint"],
    help=(
        "Candidate sorting mode: pickup_distance | pickup_deadline | "
        "pickup_deadline_distance | randomized | predicted_reward | predicted_reward_joint"
    ),
)
parser.add_argument(
    "--policies",
    nargs="+",
    default=None,
    choices=sorted(SUPPORTED_BASELINE_POLICIES),
    help="Optional override for baseline policies list",
)
parser.add_argument(
    "--sorted",
    action="store_true",
    help="DEPRECATED alias for --candidates-sorting=pickup_distance",
)
from utils.config import Config
cfg = Config(parser)
opt = cfg.opt
SUMO_PORT = opt.sumoport


def resolve_candidates_sorting(opt_obj) -> str:
    cli_mode = getattr(opt_obj, "candidates_sorting", None)
    if cli_mode not in (None, ""):
        return str(cli_mode)
    if bool(getattr(opt_obj, "sorted", False)):
        return "pickup_distance"
    env_mode = getattr(opt_obj.env, "candidates_sorting", None)
    if env_mode not in (None, ""):
        return str(env_mode)
    legacy_sorted = getattr(opt_obj.env, "sorted_candidates", None)
    if legacy_sorted is not None:
        return "pickup_distance" if bool(legacy_sorted) else "randomized"
    return "pickup_distance"


def resolve_policies(opt_obj) -> List[str]:
    cli_policies = getattr(opt_obj, "policies", None)
    if cli_policies not in (None, ""):
        policies = [str(p).strip().lower() for p in list(cli_policies)]
    else:
        policies = [str(p).strip().lower() for p in list(getattr(opt_obj.baselines, "policies", []))]

    invalid = sorted({p for p in policies if p not in SUPPORTED_BASELINE_POLICIES})
    if invalid:
        allowed = ", ".join(sorted(SUPPORTED_BASELINE_POLICIES))
        raise ValueError(f"Unsupported baseline policy name(s): {invalid}. Allowed: {allowed}")
    return policies


def policy_candidates_sorting(policy_name: str, default_sorting: str) -> str:
    """Map policy name to controller candidates_sorting mode."""
    return SLOT0_SORTING_POLICY_MAP.get(policy_name, default_sorting)

# 1) SUMO/controller setup (example; adapt to your config)
SUMO_CFG = opt.env.sumo_cfg
USE_GUI = bool(opt.env.use_gui)
R = int(opt.env.R)
K_max = int(opt.env.K_max)
N_max = int(opt.env.N_max)
E_max = int(opt.env.E_max)
features_cfg = getattr(opt, "features", {})
use_reservation_time = bool(features_cfg.get("use_reservation_time", False))
use_xy_pickup = bool(opt.features.use_xy_pickup)
use_node_type = bool(getattr(opt.features, "use_node_type", False))
use_ego_robot = bool(getattr(opt.features, "use_ego_robot", False))
use_edge_rt = bool(getattr(opt.features, "use_edge_rt", False))
robot_commitment = str(getattr(opt.features, "robot_commitment", "none"))
route_slots_k = int(getattr(opt.features, "route_slots_k", 2))
edge_features = expand_edge_features(
    list(getattr(opt.features, "edge_features", [])),
    robot_commitment=robot_commitment,
    route_slots_k=route_slots_k,
)

F = compute_feature_dim(
    use_xy_pickup=use_xy_pickup,
    use_node_type=use_node_type,
    use_edge_rt=use_edge_rt,
    use_ego_robot=use_ego_robot,
    robot_commitment=robot_commitment,
    route_slots_k=route_slots_k,
)
edge_feat_dim = len(edge_features) if use_edge_rt else 0
G = int(opt.env.G)

VICINITY_M = float(opt.env.vicinity_m)
MAX_STEPS = int(opt.env.max_steps)
MAX_WAIT_DELAY_S = float(opt.env.max_wait_delay_s)
MAX_TRAVEL_DELAY_S = float(opt.env.max_travel_delay_s)
MAX_ROBOT_CAPACITY = int(opt.env.max_robot_capacity)
CONFLICT_RESOLUTION = str(getattr(opt.env, "conflict_resolution", "closest_then_capacity"))
reward_params = dict(getattr(opt.env, "reward_params", {}) or {})
COMPLETION_MODE = str(getattr(opt.env, "completion_mode", "dropoff"))
CANDIDATES_SORTING = resolve_candidates_sorting(opt)

NUM_SEEDS = int(opt.baselines.num_seeds)
SEEDS = list(opt.seeds.eval)
#SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                  # 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
POLICIES = resolve_policies(opt)


# Initialize metrics log file
metrics_log_path = (
    f"metrics_v{int(VICINITY_M)}_ms{MAX_STEPS}_mwd{int(MAX_WAIT_DELAY_S)}_"
    f"mtd{int(MAX_TRAVEL_DELAY_S)}_cap{MAX_ROBOT_CAPACITY}.log"
)
with open(metrics_log_path, "w", encoding="utf-8") as f:
    f.write(f"vicinity_m={VICINITY_M}, max_steps={MAX_STEPS}, max_wait_delay_s={MAX_WAIT_DELAY_S}, "
            f"max_travel_delay_s={MAX_TRAVEL_DELAY_S}, max_robot_capacity={MAX_ROBOT_CAPACITY}\n")
    for line in build_metrics_metadata_lines(
        sumo_cfg_path=SUMO_CFG,
        conflict_resolution=CONFLICT_RESOLUTION,
        route_construction=str(getattr(opt.env, "route_construction", "nearest")),
        reward_params=reward_params,
        completion_mode=COMPLETION_MODE,
        reassignment_mode=str(getattr(opt.env, "reassignment_mode", "locked_until_pickup")),
    ):
        f.write(line + "\n")
    f.write(get_metrics_header() + "\n")

all_metrics_by_policy: Dict[str, List[EpisodeMetrics]] = {p: [] for p in POLICIES}

# Run evaluations for first NUM_SEEDS seeds
for seed in SEEDS[:NUM_SEEDS]:
    print(f"\n{'='*80}")
    print(f"Starting seed {seed}")
    print(f"{'='*80}")
    
    extra_args = [f"--seed", str(seed), "--device.taxi.dispatch-algorithm", "traci"]
    traci = start_sumo(SUMO_CFG, use_gui=USE_GUI, extra_args=extra_args, remote_port=SUMO_PORT)

    # Policy loop moved here for per-policy logger/env
    for policy_name in POLICIES:
        policy_sorting = policy_candidates_sorting(policy_name, CANDIDATES_SORTING)
        rp_logger = RidepoolLogger(
            RidepoolLogConfig(
                out_dir="runs",
                run_name=f"rp_eval_seed{seed}_{policy_name}",
                erase_run_dir_on_start=True,
                erase_episode_dir_on_start=True,
                console_debug=False,
                log_conflict_metrics=bool(getattr(opt.logging, "log_conflict_metrics", False)),
            )
        )

        try:
            controller = RLControllerAdapter(
                sumo=traci,
                reset_fn=make_reset_fn(
                    SUMO_CFG,
                    use_gui=USE_GUI,
                    extra_args=extra_args,
                    remote_port=SUMO_PORT,
                ),
                k_max=K_max,
                vicinity_m=VICINITY_M,
                candidates_sorting=policy_sorting,
                sorted_candidates=bool(getattr(opt, "sorted", False)),
                completion_mode=COMPLETION_MODE,
                reassignment_mode=str(getattr(opt.env, "reassignment_mode", "locked_until_pickup")),
                max_steps=MAX_STEPS,
                min_episode_steps=100,
                serve_to_empty=True,
                require_seen_reservation=True,
                max_wait_delay_s=MAX_WAIT_DELAY_S,
                max_travel_delay_s=MAX_TRAVEL_DELAY_S,
                max_robot_capacity=MAX_ROBOT_CAPACITY,
                logger=rp_logger,
                respect_sumo_end=True,
                conflict_resolution=CONFLICT_RESOLUTION,
                route_construction=str(getattr(opt.env, "route_construction", "nearest")),
                route_exhaustive_max_stops=int(getattr(opt.env, "route_exhaustive_max_stops", 8)),
                route_construction_debug=bool(getattr(opt.env, "route_construction_debug", False)),
                reward_params=reward_params,
            )
        except NotImplementedError as e:
            print(f"\n  Policy: {policy_name} (sorting={policy_sorting})")
            print(f"    SKIP: {e}")
            with open(metrics_log_path, "a", encoding="utf-8") as f:
                f.write(f"# SKIP policy={policy_name} seed={seed} reason={e}\n")
            continue
        feature_fn = make_feature_fn(
            controller,
            use_xy_pickup=use_xy_pickup,
            use_reservation_time=use_reservation_time,
            normalize_features=bool(getattr(opt.features, "normalize_features", False)),
            use_node_type=use_node_type,
            use_edge_rt=use_edge_rt,
            edge_features=edge_features,
            use_ego_robot=use_ego_robot,
            robot_commitment=robot_commitment,
            route_slots_k=route_slots_k,
        )

        env = RidepoolRTEnv(
            controller,
            R=R, K_max=K_max, N_max=N_max, E_max=E_max,
            F=F, G=0,
            feature_fn=feature_fn,
            global_stats_fn=None,
            decision_dt=int(opt.env.decision_dt),
            two_hop=bool(getattr(opt.env, "two_hop", False)),
            two_hop_directed=bool(getattr(opt.env, "two_hop_directed", False)),
            normalize_features=bool(getattr(opt.features, "normalize_features", False)),
            use_edge_rt=use_edge_rt,
            edge_feat_dim=edge_feat_dim,
            edge_features=edge_features,
        )

        # ...existing code for NOOP, action functions, and episode run...

        # The rest of the per-policy loop (from 'def greedy_nearest_action' to metrics extraction and logging)
        NOOP = K_max

        def slot0_candidate_action(action_mask: np.ndarray) -> np.ndarray:
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
                return slot0_candidate_action(action_mask)

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

        print(f"\n  Policy: {policy_name} (sorting={policy_sorting})")
        obs, info = env.reset()
        done = False
        trunc = False

        while not (done or trunc):
            mask = info.get("action_mask", env.unwrapped.action_mask())

            if policy_name in SLOT0_SORTING_POLICY_MAP:
                action = slot0_candidate_action(mask)
            elif policy_name == "random":
                action = random_valid_action(mask)
            elif policy_name == "unique":
                action = greedy_unique_action(mask)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            obs, reward, done, trunc, info = env.step(action)

        # Get episode directory from logger
        episode_dir = rp_logger.ep_dir
        print(f"[DEBUG] Using episode_dir: {episode_dir}")

        # **FIX**: Flush logger files to ensure all data is written before extracting metrics
        if hasattr(rp_logger, '_files'):
            for f in rp_logger._files.values():
                try:
                    f.flush()
                except Exception:
                    pass

        # Compute metrics from CSV files
        metrics = compute_episode_metrics_from_logs(
            episode_dir,
            info,
            policy_name,
            seed,
            num_robots=R,
        )
        print(f"[DEBUG] Computed reward_sum: {metrics.reward_sum}, completion_sum: {metrics.completion_sum}")
        all_metrics_by_policy[policy_name].append(metrics)

        # Log to file
        with open(metrics_log_path, "a", encoding="utf-8") as f:
            f.write(metrics_to_string(metrics) + "\n")
            f.flush()

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
    summary_header = (
        f"{'pol':<{POLICY_WIDTH}} rew±std   |     cap±std       step±std     dln±std       wait±std      trav±std      comp±std       nsv±std   |"
        f"     pkr±std       obsr±std      pkvr±std      mwt±std       cmr±std       anpr±std      pncr±std |"
        f"  noop±std   overld±std   mcand±std  cne_fr±std cne_mn±std  dstep±std    macmr±std     msd±std  |"
        f" drp_ev±std  vcmr±std    invdr±std   ddvr±std |"
        f"   ctot±std    crat±std    catx±std"
    )
    f.write(summary_header + "\n")
    
    for policy_name in POLICIES:
        metrics_list = all_metrics_by_policy[policy_name]
        if not metrics_list:
            continue
        
        rewards = [m.reward_sum for m in metrics_list]
        caps = [m.capacity_sum for m in metrics_list]
        steps = [m.step_sum for m in metrics_list]
        dlvs = [m.deadline_sum for m in metrics_list]
        waits = [m.wait_sum for m in metrics_list]
        travs = [m.travel_sum for m in metrics_list]
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
        dropoff_ev_rates = [m.dropoff_event_count / max(1, m.total_tasks) for m in metrics_list]
        valid_completion_rates = [m.valid_completion_rate for m in metrics_list]
        invalid_dropoff_rates = [m.invalid_dropoff_rate for m in metrics_list]
        ddv_rates = [m.dropoff_deadline_violation_rate for m in metrics_list]
        conflict_totals = [m.conflicts_total for m in metrics_list]
        conflict_ratios = [m.conflict_ratio for m in metrics_list]
        conflict_avg_taxis = [m.conflict_avg_taxis_per_conflict for m in metrics_list]

        summary_line = (
            f"{policy_name:<{POLICY_WIDTH}}"
            f" {np.mean(rewards):>6.2f}±{np.std(rewards):<5.2f} | "
            f" {np.mean(caps):>6.2f}±{np.std(caps):<5.2f}"
            f"  {np.mean(steps):>6.2f}±{np.std(steps):<5.2f}"
            f"  {np.mean(dlvs):>6.2f}±{np.std(dlvs):<5.2f}"
            f"  {np.mean(waits):>6.2f}±{np.std(waits):<5.2f}"
            f"  {np.mean(travs):>6.2f}±{np.std(travs):<5.2f}"
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
            f" |  {np.mean(dropoff_ev_rates):>6.3f}±{np.std(dropoff_ev_rates):<5.3f}"
            f"  {np.mean(valid_completion_rates):>6.3f}±{np.std(valid_completion_rates):<5.3f}"
            f"  {np.mean(invalid_dropoff_rates):>6.3f}±{np.std(invalid_dropoff_rates):<5.3f}"
            f"  {np.mean(ddv_rates):>6.3f}±{np.std(ddv_rates):<5.3f}"
            f" |  {np.mean(conflict_totals):>6.2f}±{np.std(conflict_totals):<5.2f}"
            f"  {np.mean(conflict_ratios):>6.3f}±{np.std(conflict_ratios):<5.3f}"
            f"  {np.mean(conflict_avg_taxis):>6.3f}±{np.std(conflict_avg_taxis):<5.3f}"
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
        print(f"  Dropoff Event Rate: {np.mean(dropoff_ev_rates):.3f} ± {np.std(dropoff_ev_rates):.3f}")
        print(f"  Valid Completion Rate: {np.mean(valid_completion_rates):.3f} ± {np.std(valid_completion_rates):.3f}")
        print(f"  Invalid Dropoff Rate: {np.mean(invalid_dropoff_rates):.3f} ± {np.std(invalid_dropoff_rates):.3f}")
        print(f"  Dropoff Deadline Violation Rate: {np.mean(ddv_rates):.3f} ± {np.std(ddv_rates):.3f}")
        print(f"  Conflicts Total: {np.mean(conflict_totals):.2f} ± {np.std(conflict_totals):.2f}")
        print(f"  Conflict Ratio: {np.mean(conflict_ratios):.3f} ± {np.std(conflict_ratios):.3f}")
        print(f"  Avg Taxis per Conflict: {np.mean(conflict_avg_taxis):.3f} ± {np.std(conflict_avg_taxis):.3f}")

    f.write("\n# METRIC LEGEND\n")
    f.write("short\tfull\n")
    legend_rows = [
        ("pol", "policy"),
        ("seed", "random seed"),
        ("rew", "reward_sum"),
        ("cap", "capacity_sum"),
        ("step", "step_sum"),
        ("dln", "deadline_sum"),
        ("wait", "wait_sum"),
        ("trav", "travel_sum"),
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
        ("drp_ev", "dropoff_event_rate (all dropoffs / total tasks)"),
        ("vcmr", "valid_completion_rate (pickup+dropoff before deadlines / total tasks)"),
        ("invdr", "invalid_dropoff_rate (dropoffs failing validity / total tasks)"),
        ("ddvr", "dropoff_deadline_violation_rate (dropped after dropoff_deadline / total tasks)"),
        ("ctot", "conflicts_total (rows in conflicts.csv)"),
        ("crat", "conflict_ratio (conflicts_total / tasks_total as in 05_conflicts.log)"),
        ("catx", "conflict_avg_taxis_per_conflict (mean taxi_candidates count in conflicts.csv)"),
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

 

