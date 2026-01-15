# scripts/train_gnn_ppo.py
import os
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
    max_wait_delay_s=600.0,     # allowed waiting time until pickup 
    max_travel_delay_s=900.0,  # no explicit penalty for that now (!)
    max_robot_capacity=2, # should match to taxis.rou.xml
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
    n_steps=128,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.95,
    clip_range=0.2,
    ent_coef=0.03,
    gae_lambda=0.90,
    n_epochs=10,
    verbose=1
)

callback = RPLoggerCallback(rp_logger, controller)

model.learn(total_timesteps=1_000_000, callback=callback)
model.save("ppo_rp_gnn.zip")
