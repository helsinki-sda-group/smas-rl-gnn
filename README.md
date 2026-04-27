# RT-GNN-RL for Robot-Task Allocation in Dynamic Environments

This repository implements a framework for robot-task (RT) allocation in dynamic environments using **graph-based reinforcement learning**.

The main application here is **ride-pooling / taxi task allocation** on SUMO, but the GNN+RL components are reusable for other robot–task scenarios.

**Development note.**  
The ride-pooling environment included here is an early-stage prototype.  
It reflects the structure of the RT-GNN-RL framework, but numerical behavior
(e.g., task deadlines, capacities, SUMO travel-time estimates, reward components)
has not been fully verified.  
The implementation is currently under refinement and should be considered
non-final.

## Repository structure

- `rt_gnn_rl/` – reusable GNN+RL library:
  - `graphs/builders.py` – build padded ego-graphs per robot from robots, tasks and candidate lists.
  - `policy/actor_critic.py` – GNN-based actor–critic operating on ego-graphs.
  - `policy/gnn_backbone.py` – GraphSAGE backbone and dummy MLP backbone.
  - `policy/sb3_gnn_policy.py` – Stable-Baselines3 policy wrapper (`RTGNNPolicy`).
  - `rollouts/rewards.py` – helper to combine reward components with weights.

- `sumo_rl_rs/` – SUMO-based ride-pooling environment:
  - `environment/rl_controller_adapter.py` – RL-friendly adapter around TraCI Taxi API:
    - tracks tasks (release time, waiting time, deadlines, flags),
    - selects candidate tasks per robot based on vicinity and capacity,
    - computes reward components per robot.
  - `environment/ridepool_rt_env.py` – `gymnasium.Env` that exposes the controller as
    an RL environment with batched ego-graph observations.
  - `logging/` – logging utilities and training callback.

- `utils/`:
  - `sumo_bootstrap.py` – start/reset SUMO (with or without GUI).
  - `feature_fns.py` – node feature construction for robots and tasks.

- `configs/` – example SUMO configuration and route files.

- `train.py` – example training script using PPO + `RTGNNPolicy`.

## Documentation (`readme/`)

- [readme/01_reward.md](readme/01_reward.md) - Reward system reference linked to implementation details and YAML knobs.
- [readme/02_architecture_variants.md](readme/02_architecture_variants.md) - Model and architecture variant notes used for experiments.
- [readme/03_robot_task_features.md](readme/03_robot_task_features.md) - Robot/task feature definitions and feature-engineering choices.
- [readme/04_training_metrics_log.md](readme/04_training_metrics_log.md) - Training metric logging format and metric interpretation.
- [readme/05_conflicts_log.md](readme/05_conflicts_log.md) - Conflict-resolution logging semantics and related diagnostics.
- [readme/06_comp_norms_log.md](readme/06_comp_norms_log.md) - Component norm logging for debugging/monitoring representation scale.
- [readme/07_eval_baselines.md](readme/07_eval_baselines.md) - Baseline evaluation workflow and policy behavior (`random`, `greedy`, `unique`).

