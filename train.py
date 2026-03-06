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
from utils.feature_fns import make_feature_fn, compute_feature_dim

# to write PPO output to txt file
import sys

parser = argparse.ArgumentParser(description="Train GNN PPO with SUMO")
parser.add_argument("--config", type=str, default="configs/rp_gnn.yaml", help="Path to config YAML")
parser.add_argument("--sumoport", type=int, default=None, help="SUMO remote port (default: SUMO default)")
parser.add_argument("--sorted", action="store_true", help="Sort candidates by pickup distance (default: randomized)")
parser.add_argument("--continue-training", action="store_true", help="Continue training from latest saved model")
from utils.config import Config
cfg = Config(parser)
opt = cfg.opt
SUMO_PORT = opt.sumoport

class Tee(object):
    def __init__(self, filename, mode: str = "w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

continue_training = bool(getattr(opt, "continue_training", False) or bool(getattr(opt, "continue_training", False)))
Tee("train_output.txt", mode="a" if continue_training else "w")

# 1) SUMO/controller setup (example; adapt to your config)
SUMO_CFG = opt.env.sumo_cfg
USE_GUI = bool(opt.env.use_gui)
R = int(opt.env.R)
K_max = int(opt.env.K_max)
N_max = int(opt.env.N_max)
E_max = int(opt.env.E_max)
use_xy_pickup = bool(opt.features.use_xy_pickup)
use_node_type = bool(getattr(opt.features, "use_node_type", False))
use_ego_robot = bool(getattr(opt.features, "use_ego_robot", False))
use_edge_rt = bool(getattr(opt.features, "use_edge_rt", False))
edge_features = list(getattr(opt.features, "edge_features", []))
robot_commitment = str(getattr(opt.features, "robot_commitment", "none"))
route_slots_k = int(getattr(opt.features, "route_slots_k", 2))

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

# Training seeds - different from evaluation seeds [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
TRAIN_SEEDS = list(opt.seeds.train)

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
reset_fn = RandomSeedResetFn(SUMO_CFG, use_gui=USE_GUI, seeds=TRAIN_SEEDS, random_seed=42, sumo_port=SUMO_PORT)

extra_args = ["--seed", str(SEED), "--device.taxi.dispatch-algorithm", "traci"]
traci = start_sumo(SUMO_CFG, use_gui=USE_GUI, extra_args=extra_args, remote_port=SUMO_PORT)

rp_logger = RidepoolLogger(
    RidepoolLogConfig(
        out_dir=str(opt.logging.out_dir),
        run_name=str(opt.logging.run_name),        # run dir will be: runs/rp_gnn_debug
        erase_run_dir_on_start=not continue_training,
        erase_episode_dir_on_start=not continue_training,
        console_debug=True,
    )
)

controller = RLControllerAdapter(
    sumo=traci,
    reset_fn=reset_fn,  # Use rotating seed reset function
    k_max=K_max,
    vicinity_m=VICINITY_M,      # vicinity in meters
    sorted_candidates=bool(opt.sorted),
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
feature_fn = make_feature_fn(
    controller,
    use_xy_pickup=use_xy_pickup,
    normalize_features=bool(getattr(opt.features, "normalize_features", False)),
    use_node_type=use_node_type,
    use_edge_rt=use_edge_rt,
    edge_features=edge_features,
    use_ego_robot=use_ego_robot,
    robot_commitment=robot_commitment,
    route_slots_k=route_slots_k,
)

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
    decision_dt=int(opt.env.decision_dt),
    two_hop=bool(getattr(opt.env, "two_hop", False)),
    normalize_features=bool(getattr(opt.features, "normalize_features", False)),
    use_edge_rt=use_edge_rt,
    edge_feat_dim=edge_feat_dim,
    edge_features=edge_features,
)
env = Monitor(env, filename="monitor.csv", info_keywords=("episode_reward",))

# 4) SB3 PPO with custom GNN policy
gnn_layers = int(getattr(opt.ppo.policy_kwargs, "gnn_layers", 2))
gnn_layers_two_hop = int(getattr(opt.ppo.policy_kwargs, "gnn_layers_two_hop", gnn_layers))
chosen_layers = gnn_layers_two_hop if bool(getattr(opt.env, "two_hop", False)) else gnn_layers

policy_kwargs = dict(
    in_dim=F,
    hidden=int(opt.ppo.policy_kwargs.hidden),
    k_max=int(opt.ppo.policy_kwargs.k_max),
    logit_temperature=float(opt.ppo.policy_kwargs.logit_temperature),
    noop_init=float(opt.ppo.policy_kwargs.noop_init),
    freeze_noop_logit=bool(getattr(opt.ppo.policy_kwargs, "freeze_noop_logit", False)),
    edge_dim=edge_feat_dim,
    gnn_kwargs={"layers": chosen_layers},
)

def _latest_model_path(model_dir: str) -> tuple[str, int, int]:
    import re
    import glob
    pattern = os.path.join(model_dir, "model_episode*_ts*.zip")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(r"model_episode(\d+)_ts(\d+)\.zip$", os.path.basename(path))
        if m:
            ep = int(m.group(1))
            ts = int(m.group(2))
            candidates.append((ts, ep, path))
    if not candidates:
        raise FileNotFoundError(f"No saved models found in {model_dir}")
    candidates.sort(key=lambda x: x[0])
    ts, ep, path = candidates[-1]
    return path, ep, ts

if continue_training:
    latest_path, last_ep, last_ts = _latest_model_path(str(opt.logging.model_save_dir))
    print(f"[CONTINUE] Loading model: {latest_path}")
    model = PPO.load(latest_path, env=env)
    model.num_timesteps = int(last_ts)
    rp_logger.cfg.episode_index = int(last_ep) + 1
    controller._ep_idx = int(last_ep)
else:
    model = PPO(
        RTGNNPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=int(opt.ppo.n_steps),
        batch_size=int(opt.ppo.batch_size),
        learning_rate=float(opt.ppo.learning_rate),
        gamma=float(opt.ppo.gamma),
        clip_range=float(opt.ppo.clip_range),
        clip_range_vf=opt.ppo.clip_range_vf,
        vf_coef=float(opt.ppo.vf_coef),
        ent_coef=float(opt.ppo.ent_coef),
        gae_lambda=float(opt.ppo.gae_lambda),
        n_epochs=int(opt.ppo.n_epochs),
        verbose=1
    )

    print("Initial noop_logit:", model.policy.noop_logit.item())
    model.policy.noop_logit.data.fill_(float(opt.ppo.policy_kwargs.noop_init))
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
model_save_dir = str(opt.logging.model_save_dir)

callback = RPLoggerCallback(
    rp_logger,
    controller,
    metrics_log_path=metrics_log_path,
    logit_metrics_log_path=None,#logit_metrics_log_path,
    num_robots=R,
    reset_fn=reset_fn,  # Pass reset_fn to get current seed
    save_model_dir=model_save_dir,  # Enable model saving after each rollout
    continue_training=continue_training,
)

if continue_training:
    callback.ep_idx = rp_logger.cfg.episode_index

model.learn(total_timesteps=int(opt.ppo.total_timesteps), callback=callback, reset_num_timesteps=not continue_training)
model.save("ppo_rp_gnn.zip")
