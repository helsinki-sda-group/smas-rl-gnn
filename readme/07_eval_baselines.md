# Baseline Evaluation (`eval_baselines.py`)

This document explains how baseline policies are implemented and evaluated in `eval_baselines.py`.

## Purpose

`eval_baselines.py` runs fixed, non-learning policies in the same SUMO-based environment used by RL training, then computes comparable episode metrics from logs.

At a high level:
1. Read config from `configs/rp_gnn.yaml`.
2. For each evaluation seed, start SUMO.
3. For each baseline policy, run one episode.
4. Compute metrics from per-episode CSV logs.
5. Aggregate mean/std across seeds by policy.

## Policies configured from YAML

From `configs/rp_gnn.yaml`:
- `baselines.policies`: list of policies to evaluate (default in this repo):
	- `random`
	- `unique`
	- `pickup_distance`
	- `pickup_deadline`
	- `pickup_deadline_distance`
	- `predicted_reward`
	- `predicted_reward_joint`
- `baselines.num_seeds`: number of seeds to use from `seeds.eval`
- `seeds.eval`: evaluation seed list

Current defaults evaluate all listed policies over the first 10 eval seeds.

## Action space convention used by baselines

For each robot, action indices are:
- `0..K_max-1`: pick candidate task at candidate slot `k`
- `K_max`: `NOOP` (do nothing)

`eval_baselines.py` sets:
- `NOOP = K_max`

At each decision step, policies use an action mask (`1` = valid, `0` = invalid) from environment info.

## Implementation notes

The script supports two policy families:
- `random` and `unique`: action-selection policies that compute per-robot actions from the current action mask.
- Slot-0 policies: `greedy`, `pickup_distance`, `pickup_deadline`, `pickup_deadline_distance`, `predicted_reward`, `predicted_reward_joint`.

Slot-0 policies choose candidate slot `0` when valid and rely on controller-side candidate ordering via `candidates_sorting`.

Current policy-to-sorting mapping used by `policy_candidates_sorting(...)`:
- `greedy` -> `pickup_distance`
- `pickup_distance` -> `pickup_distance`
- `pickup_deadline` -> `pickup_deadline`
- `pickup_deadline_distance` -> `pickup_deadline_distance`
- `predicted_reward` -> `predicted_reward`
- `predicted_reward_joint` -> `predicted_reward_joint`

If a selected sorting mode is not implemented in the controller, the script catches `NotImplementedError`, writes a `# SKIP ...` line in the metrics log, and continues with remaining policies.

## Baseline policy behavior

## 1) Slot-0 family (`greedy`, `pickup_distance`, `pickup_deadline`, `pickup_deadline_distance`, `predicted_reward`, `predicted_reward_joint`)

Implementation: `slot0_candidate_action(action_mask)`.

Behavior per robot:
- If candidate slot `0` is valid (`action_mask[r, 0] == 1`), choose action `0`.
- Otherwise choose `NOOP`.

Interpretation:
- These policies differ by candidate ordering mode, not by action-selection logic.
- Candidate ordering is controlled via `candidates_sorting` (per-policy mapping above).

## 2) `random`

Implementation: `random_valid_action(action_mask)`.

Behavior per robot:
- Collect all valid action indices (including `NOOP` when valid).
- Uniformly sample one valid index.
- If no valid index exists (rare fallback), pick `NOOP`.

Notes:
- RNG is seeded per episode seed (`np.random.default_rng(seed)`), so runs are reproducible per seed.
- Because sampling is independent per robot, multiple robots may choose actions that map to the same task before conflict resolution.

## 3) `unique`

Implementation: `greedy_unique_action(action_mask)`.

Behavior per decision step:
- Read candidate task IDs from `env.unwrapped._last_cand_task_ids`.
- Iterate robots in index order.
- For each robot, scan candidate slots `k = 0..K_max-1` and take the first valid candidate whose task ID has not already been chosen by another robot in this step.
- If none qualifies, choose `NOOP`.

Fallback:
- If candidate IDs are unavailable, falls back to `greedy` behavior.

Interpretation:
- This is a per-step greedy matching heuristic that avoids duplicate assignment attempts within the same decision step.
- Priority is robot-order dependent (lower robot index gets first claim).

## Evaluation loop details

For each `(seed, policy)`:
1. Build logger run directory: `runs/rp_eval_seed{seed}_{policy}`
2. Create `RLControllerAdapter` + `RidepoolRTEnv`
3. Reset env and loop until `done or trunc`
4. Compute one joint action vector (one action per robot) from policy
5. Step environment
6. After episode, flush logger files and compute metrics via `compute_episode_metrics_from_logs(...)`

## Logged outputs

## Per-run logs

Each `(seed, policy)` writes episode CSVs under its run directory, then `utils.metrics_calculator` computes:
- reward terms (`reward_sum`, `wait_sum`, `travel_sum`, `completion_sum`, etc.)
- service quality (`pickup_rate`, `completion_rate`, `obsolete_rate`, etc.)
- action/candidate diagnostics (`noop_fraction`, candidate stats, overload fraction)

## Aggregate text log

A summary log file is created in project root:
- `metrics_v{vicinity}_ms{max_steps}_mwd{max_wait}_mtd{max_travel}_cap{capacity}.log`

It contains:
- one line per `(seed, policy)` episode
- separator lines between seeds
- policy-wise mean+-std summary table over seeds
- metric legend mapping short names to full metrics

## Important implementation notes

- `--policies` can override YAML policy list and accepts all names in `SUPPORTED_BASELINE_POLICIES`.
- Candidate sorting default is resolved in this order: `--candidates-sorting`, then `--sorted` (deprecated alias for `pickup_distance`), then `env.candidates_sorting`, then legacy `env.sorted_candidates`, else `pickup_distance`.
- Slot-0 policy names override default sorting mode through `policy_candidates_sorting(...)`.
- `stable_baselines3.common.monitor.Monitor`, `pandas`, and `os` are imported but not used in this script.
- `G` is read from config but environment is created with `G=0` in this script.

These do not break baseline evaluation, but they are useful to know when interpreting behavior or refactoring.

## Quick run command

```bash
python eval_baselines.py --config configs/rp_gnn.yaml
```

Optional:
- `--sumoport <port>` to select SUMO remote port.
- `--policies <names...>` to override policy list for one run.
- `--candidates-sorting <mode>` to set default sorting mode for policies that do not override it.
