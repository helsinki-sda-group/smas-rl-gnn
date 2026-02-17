# scripts/train_gnn_ppo.py
import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rt_gnn_rl.policy.sb3_gnn_policy import RTGNNPolicy
from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter 
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from sumo_rl_rs.logging.rp_logger_callback import RPLoggerCallback
from utils.sumo_bootstrap import start_sumo, make_reset_fn
import numpy as np
from utils.feature_fns import make_feature_fn

# to write PPO output to txt file
import sys

parser = argparse.ArgumentParser(description="Train GNN PPO with SUMO")
parser.add_argument("--sumoport", type=int, default=None, help="SUMO remote port (default: SUMO default)")
args = parser.parse_args()
SUMO_PORT = args.sumoport

class Tee(object):
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

Tee("train_output.txt")

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

# Training seeds - different from evaluation seeds [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
TRAIN_SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
               1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

# Initial seed for first episode
SEED = TRAIN_SEEDS[0]

# Create a class to randomly sample seeds from pool
class RandomSeedResetFn:
    """Reset function that randomly samples from training seeds pool on each episode."""
    def __init__(self, sumocfg_path: str, use_gui: bool, seeds: list[int], random_seed: int = 42, sumo_port: int | None = None):
        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.seeds = seeds
        self.rng = np.random.RandomState(random_seed)  # Separate RNG for seed selection
        self.current_seed = seeds[0]
        self.sumo_port = sumo_port
    
    def __call__(self) -> None:
        # Randomly sample a seed from the pool
        self.current_seed = self.rng.choice(self.seeds)
        extra_args = ["--seed", str(self.current_seed), "--device.taxi.dispatch-algorithm", "traci"]
        
        # Import here to avoid circular dependencies
        from utils.sumo_bootstrap import _imports, _build_args
        traci, checkBinary = _imports()
        args = _build_args(self.sumocfg_path, extra_args)
        
        if traci.isLoaded():
            # reload in the same process/port
            traci.load(args)
        else:
            binary = checkBinary("sumo-gui" if self.use_gui else "sumo")
            traci.start([binary, *args], port=self.sumo_port)
    
    def get_current_seed(self) -> int:
        """Get the seed that was used for the current episode."""
        return self.current_seed

# Create rotating reset function
reset_fn = RandomSeedResetFn(SUMO_CFG, use_gui=False, seeds=TRAIN_SEEDS, random_seed=42, sumo_port=SUMO_PORT)

extra_args = ["--seed", str(SEED), "--device.taxi.dispatch-algorithm", "traci"]
traci = start_sumo(SUMO_CFG, use_gui=False, extra_args=extra_args, remote_port=SUMO_PORT)

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
    reset_fn=reset_fn,  # Use rotating seed reset function
    k_max=K_max,
    vicinity_m=VICINITY_M,      # vicinity in meters
    completion_mode="dropoff", # task is marked as completed at dropoff
    max_steps=MAX_STEPS,
    min_episode_steps = 100,
    serve_to_empty=True,    # end only when nothing left to do
    require_seen_reservation=True, # don't allow done until we've seen at least one reservation
    max_wait_delay_s=MAX_WAIT_DELAY_S,     # allowed waiting time until pickup
    max_travel_delay_s=MAX_TRAVEL_DELAY_S,  # no explicit penalty for that now (!)
    max_robot_capacity=MAX_ROBOT_CAPACITY, # should match to taxis.rou.xml
    logger= rp_logger,
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
env = Monitor(env, filename="monitor.csv", info_keywords=("episode_reward",))

# 4) SB3 PPO with custom GNN policy
policy_kwargs = dict(in_dim=F, hidden=128, k_max=K_max)
model = PPO(

    RTGNNPolicy,
    env,
    policy_kwargs=policy_kwargs,
    n_steps=256,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99, # was 0.95
    clip_range=0.2,
    clip_range_vf=None,
    vf_coef=0.35,
    ent_coef=0.003, # was 0.03
    gae_lambda=0.95, # was 0.9
    n_epochs=5,
    verbose=1
)


print("Initial noop_logit:", model.policy.noop_logit.item())
model.policy.noop_logit.data.fill_(-1.0)
print("Forced noop_logit:", model.policy.noop_logit.item())

model.save("init_model/model_episode0_ts0.zip")

metrics_log_path = (
    f"training_metrics_v{int(VICINITY_M)}_ms{MAX_STEPS}_mwd{int(MAX_WAIT_DELAY_S)}_"
    f"mtd{int(MAX_TRAVEL_DELAY_S)}_cap{MAX_ROBOT_CAPACITY}.log"
)
logit_metrics_log_path = (
    f"training_logit_metrics_v{int(VICINITY_M)}_ms{MAX_STEPS}_mwd{int(MAX_WAIT_DELAY_S)}_"
    f"mtd{int(MAX_TRAVEL_DELAY_S)}_cap{MAX_ROBOT_CAPACITY}.log"
)

# Directory to save models after each rollout
model_save_dir = "runs/rp_gnn_debug/!saved_models"

callback = RPLoggerCallback(
    rp_logger,
    controller,
    metrics_log_path=metrics_log_path,
    logit_metrics_log_path=logit_metrics_log_path,
    num_robots=R,
    reset_fn=reset_fn,  # Pass reset_fn to get current seed
    save_model_dir=model_save_dir,  # Enable model saving after each rollout
)

model.learn(total_timesteps=50_000, callback=callback)
model.save("ppo_rp_gnn.zip")
