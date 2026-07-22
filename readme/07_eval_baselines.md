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

## Complete supported policy list

`eval_baselines.py` accepts the following `SUPPORTED_BASELINE_POLICIES`:
- `random`
- `unique`
- `greedy`
- `pickup_distance`
- `pickup_deadline`
- `pickup_deadline_distance`
- `predicted_reward`
- `predicted_reward_joint`

Notes:
- `greedy` is supported by CLI (`--policies greedy ...`) but is not part of the default YAML list in this repo.
- If `--policies` is provided, it overrides `baselines.policies` from YAML.

## Supported conflict resolution mechanisms
"capacity" | "closest" | "closest_then_capacity" | "logit_diff" | "random"

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

## Baseline policy behavior (all policies)

## 1) `random`

Implementation: `random_valid_action(action_mask)`.

Behavior per robot:
- Collect all valid action indices (including `NOOP` when valid).
- Uniformly sample one valid index.
- If no valid index exists (rare fallback), pick `NOOP`.

Notes:
- RNG is seeded per episode seed (`np.random.default_rng(seed)`), so runs are reproducible per seed.
- Because sampling is independent per robot, multiple robots may choose actions that map to the same task before conflict resolution.

## 2) `unique`

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

## 3) `greedy`

Implementation: `slot0_candidate_action(action_mask)` with `candidates_sorting=pickup_distance`.

Behavior per robot:
- If candidate slot `0` is valid (`action_mask[r, 0] == 1`), choose action `0`.
- Otherwise choose `NOOP`.

Interpretation:
- Equivalent action logic to other slot-0 policies.
- Uses distance-first candidate ordering.

## 4) `pickup_distance`

Implementation: `slot0_candidate_action(action_mask)` with `candidates_sorting=pickup_distance`.

Behavior per robot:
- If slot `0` is valid, take slot `0`; else `NOOP`.

Interpretation:
- Picks the top candidate under pickup-distance sorting.

## 5) `pickup_deadline`

Implementation: `slot0_candidate_action(action_mask)` with `candidates_sorting=pickup_deadline`.

Behavior per robot:
- If slot `0` is valid, take slot `0`; else `NOOP`.

Interpretation:
- Picks the top candidate under pickup-deadline sorting.

## 6) `pickup_deadline_distance`

Implementation: `slot0_candidate_action(action_mask)` with `candidates_sorting=pickup_deadline_distance`.

Behavior per robot:
- If slot `0` is valid, take slot `0`; else `NOOP`.

Interpretation:
- Picks the top candidate under combined deadline-and-distance sorting.

## 7) `predicted_reward`

Implementation: `slot0_candidate_action(action_mask)` with `candidates_sorting=predicted_reward`.

Behavior per robot:
- If slot `0` is valid, take slot `0`; else `NOOP`.

Interpretation:
- Picks the top candidate ranked by single-agent predicted reward.

How the score is computed:
- For each feasible candidate, the controller simulates a greedy pickup/dropoff sequence for the current taxi plan after inserting that candidate.
- It predicts candidate pickup and dropoff times from route travel-time estimates.
- It computes:

$$
s = w_{comp}\cdot \mathbf{1}_{\text{valid completion}} - w_{wait}\cdot \widehat{\text{wait}} - w_{travel}\cdot \widehat{\text{excess travel}}
$$

Where:
- $\widehat{\text{wait}} = \min(\text{predicted wait},\, \text{wait cap}) / \text{wait cap}$
- $\widehat{\text{excess travel}} = \min(\text{predicted ride time} - \text{direct est. travel},\, \text{travel cap}) / \text{travel cap}$
- `valid completion` is `1` only if both predicted pickup and dropoff satisfy their deadlines.

Candidate ordering (`predicted_reward`):
- Higher `score` first
- Earlier predicted pickup time
- Shorter pickup distance
- Reservation ID lexical order (final tie-break)

Meaning of this sorting:
- It is a local per-candidate objective: "which single candidate looks best for this taxi right now".
- It does not optimize by comparing total plan value before vs after insertion.

## 8) `predicted_reward_joint`

Implementation: `slot0_candidate_action(action_mask)` with `candidates_sorting=predicted_reward_joint`.

Behavior per robot:
- If slot `0` is valid, take slot `0`; else `NOOP`.

Interpretation:
- Picks the top candidate ranked by joint predicted reward.

How the score is computed:
- Let `before` be the taxi's current unfinished plan (shadow plan + onboard unfinished tasks).
- Let `after` be the same plan with the candidate inserted.
- For both plans, the controller predicts pickup/dropoff times for all tasks in a greedy pickup/dropoff sequence.
- It computes per-task predicted scores with the same components as above (completion, normalized wait penalty, normalized excess-travel penalty).
- It then sums task scores:

$$
R_{before} = \sum_{t \in \text{before}} \text{score}(t),\quad
R_{after} = \sum_{t \in \text{after}} \text{score}(t)
$$

and ranks by marginal improvement:

$$
\Delta R = R_{after} - R_{before}
$$

Candidate ordering (`predicted_reward_joint`):
- Higher marginal score $\Delta R$ first
- Higher absolute $R_{after}$
- Earlier candidate pickup time
- Shorter pickup distance
- Reservation ID lexical order (final tie-break)

Meaning of this sorting:
- It is a plan-aware objective: "which candidate improves the full current plan the most".
- It can prefer a candidate with lower standalone score if it interferes less with already onboard/assigned tasks.

Default reward parameters used by these predicted-reward sortings (unless overridden in config):
- $w_{comp}=1.0$
- $w_{wait}=1.5$
- $w_{travel}=2.0$
- `wait_cap=600`
- `travel_cap=90`

## Slot-0 family summary

Slot-0 policies are: `greedy`, `pickup_distance`, `pickup_deadline`, `pickup_deadline_distance`, `predicted_reward`, `predicted_reward_joint`.

All slot-0 policies share the same action-selection function (`slot0_candidate_action`). They differ only in how candidate slot `0` is produced by controller-side sorting (`candidates_sorting`).

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
