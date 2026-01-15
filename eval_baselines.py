import numpy as np
from stable_baselines3.common.monitor import Monitor

from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from utils.sumo_bootstrap import start_sumo, make_reset_fn
from utils.feature_fns import make_feature_fn

# 1) SUMO/controller setup (example; adapt to your config)
SUMO_CFG = "configs/small_net.sumocfg"
USE_GUI = False
R = 5           # number of robots (taxis) expected. # should match to taxis.rou.xml
K_max = 3        # candidates per robot
N_max = 16        # max nodes per ego-graph (robot + tasks in its neighborhood)
E_max = 64        # max edges per ego-graph
F = 9            # node feature dimension (robot node and task node should have the same dimensionality, padding is applied)
G = 0             # global stats dim 

traci = start_sumo(SUMO_CFG, use_gui=False,
                   extra_args=["--seed", "42", "--device.taxi.dispatch-algorithm", "traci"])

rp_logger = RidepoolLogger(
    RidepoolLogConfig(
        out_dir="runs",
        run_name="rp_gnn_debug",        # run dir will be: runs/rp_gnn_debug
        erase_run_dir_on_start=True,    # <-- nukes runs/rp_gnn_debug at startup
        erase_episode_dir_on_start=True,# optional: also clear episode_XXXX on start
        console_debug=True
    )
)

controller = RLControllerAdapter(
    sumo=traci,
    reset_fn=make_reset_fn(SUMO_CFG, use_gui=False,
                           extra_args=["--seed", "42", "--device.taxi.dispatch-algorithm", "traci"]),
    k_max=K_max,
    vicinity_m=1500.0,      # vicinity in meters
    completion_mode="dropoff", # task is marked as completed at dropoff
    max_steps=1000,
    min_episode_steps = 100,
    serve_to_empty=True,    # end only when nothing left to do
    require_seen_reservation=True, # don't allow done until we've seen at least one reservation
    max_wait_delay_s=240.0,     # allowed waiting time until pickup 
    max_travel_delay_s=900.0,  # no explicit penalty for that now (!)
    max_robot_capacity=2, # should match to taxis.rou.xml
    logger=rp_logger,
)
feature_fn = make_feature_fn(controller)

# not implemented yet, will raise error for G > 0
def global_stats_fn(world_state):
    # Return np.ndarray(G,), e.g. mean wait, fleet utilization, time-of-day …
    ...

# 3) Gym env
env = RidepoolRTEnv(
    controller,
    R=R, K_max=K_max, N_max=N_max, E_max=E_max,
    F=F, G=0,
    feature_fn=feature_fn,
    global_stats_fn=None, 
    decision_dt=60,  
)
env = Monitor(env, filename="monitor_eval.csv", info_keywords=("episode_reward",))

NOOP = K_max

def greedy_nearest_action(action_mask: np.ndarray) -> np.ndarray:
    """
    Pick slot 0 (nearest) if candidate exists. Otherwise NOOP.
    This works because the candidates are sorted in the decending distance order in a controller.
    
    :param action_mask: shape [R, K_max+1]. 1 if the candidate exists.
    :type action_mask: np.ndarray
    :return: action
    :rtype: ndarray
    """
    a = np.full((R,), NOOP, dtype=np.int64)
    for r in range(R):
        # if there is a candidate available (0-th is the nearest candidate)
        if action_mask[r,0] == 1:
            a[r] = 0
        else:
            a[r] = NOOP
    return a

_rnd = np.random.default_rng(0)

def random_valid_action(action_mask: np.ndarray) -> np.ndarray:
    """
    Uniform random over allowed actions per robot (including noop).
    
    :param action_mask: shape [R, K_max+1]. 1 if the candidate exists.
    :type action_mask: np.ndarray
    :return: action
    :rtype: ndarray
    """

    a = np.full((R,), NOOP, dtype=np.int64)
    for r in range(R):
        allowed = np.flatnonzero(action_mask[r] == 1)
        if allowed.size > 0:
            a[r] = int(_rnd.choice(allowed))
        else:
            a[r] = NOOP

    return a

def greedy_unique_action(action_mask: np.ndarray, info: dict, K_max: int) -> np.ndarray:
    """
    Greedy baseline that avoids collisions:
      - taxi picks the nearest candidate slot that corresponds to a reservation
        not already chosen by earlier taxis
      - if nothing available -> noop
    Requires access to env's last candidate ids per taxi.
    """
    env0 = env.unwrapped  # base env
    cand_ids = getattr(env0, "_last_cand_task_ids", None)  # shape [R, K_max], ints or -1
    if cand_ids is None:
        # fallback: just nearest valid slot
        return greedy_nearest_action(action_mask)

    NOOP = K_max
    chosen = set()
    a = np.full((action_mask.shape[0],), NOOP, dtype=np.int64)

    for r in range(action_mask.shape[0]):
        # try slots in order: 0..K_max-1
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


def run_policy(policy_name: str, n_episodes: int = 50):
    ep_returns = []
    ep_dropoffs =  []
    ep_pickups = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        trunc = False
        ret = 0.0

        # to be implemented
        dropoffs = 0
        pickups = 0

        while not (done or trunc):
            mask = info.get("action_mask", env.unwrapped.action_mask())

            if policy_name == "greedy":
                action = greedy_nearest_action(mask)
            elif policy_name == "random":
                action = random_valid_action(mask)
            elif policy_name == "noop":
                action = np.full((R, ), NOOP, dtype = np.int64)
            elif policy_name == "greedy_unique":
                action = greedy_unique_action(mask, info, K_max)
            else:
                raise ValueError(policy_name)
            
            obs, reward, done, trunc, info = env.step(action)
            # to implement: extend info with pick-ups / drop-offs
            ret += float(reward)

        ep_returns.append(ret)
        ep_dropoffs.append(dropoffs)
        ep_pickups.append(pickups)

        print(f"{policy_name}: mean return {np.mean(ep_returns):.3f} ± {np.std(ep_returns):.3f}")
        # print(f"{policy_name}: mean dropoffs {np.mean(ep_dropoffs):.2f}, pickups {np.mean(ep_pickups):.2f}")

#run_policy("noop", n_episodes=1)
run_policy("random", n_episodes=3)
run_policy("greedy", n_episodes=1)   
run_policy("greedy_unique", n_episodes=1)    
 

