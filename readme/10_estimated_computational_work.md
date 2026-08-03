# Estimated Computational Work in metrics_wide.csv

This document explains the analytical work-proxy columns with prefix work_ that appear in [metrics_wide.csv](../metrics_wide.csv).

Primary implementation source:
- [estimate_work_metrics.py](../estimate_work_metrics.py)

Related RL evaluation estimator:
- [scripts/estimate_eval_work_reward.py](../scripts/estimate_eval_work_reward.py)

Scope note:
- These values are operation-count proxies for quality vs complexity comparisons.
- They are not wall-clock runtime benchmarks.
- For RL proposers such as 1hop, the current proxy is intentionally coarse: it models episode-level inference passes, not exact neural-network FLOPs.

## Why these columns exist

The project compares policy quality against expected computational effort. Raw runtime depends on hardware, SUMO load, logging overhead, and environment settings. The work_ fields normalize this by estimating how many algorithmic operations are likely done per episode.

The estimator writes these columns:
- work_model
- work_route_stops
- work_insertion_pairs
- work_candidate_scan
- work_active_proposals
- work_competition_factor
- work_proposer
- work_resolver
- work_total
- work_warning

## Inputs used by the estimator

For each row, the estimator uses:
- resolver and pol to select formula branches
- route_construction to decide whether insertion enumeration is active
- num_robots as robot count R (or inferred from instance if missing)
- max_robot_capacity to estimate route size unless overridden
- mcand as mean candidate count K
- msd as macro-step count M
- dstep as decision-step count D (fallback: D = M)
- noop as no-op fraction
- ovrlap and shared as fallback competition cues
- cemn as logged mean competitors per non-unique proposal

For RL evaluation work estimation, the related script also uses:
- proposer family, for example 1hop

## Core intermediate quantities
- R = max(num_robots, 0)
- K = max(mcand, 0)
- M = max(msd, 0)
- D = max(dstep, 0)
- n = clipped noop in [0, 1]
- K_actor = K + noop_candidates
- If not overridden, S = 2 * max_robot_capacity
- P = ((S + 1) * (S + 2)) / 2
- work_candidate_scan = M * R * K
- work_active_proposals = M * R * (1 - n)
- work_candidate_scan_actor = M * R * K_actor

In the current RL script, noop_candidates defaults to 1 to reflect one explicit NOOP action.

The default joint multiplier is 1.5.

| Column | Meaning |
|---|---|
| work_model | Name of the estimator model. Currently analytical_v1. |
| work_route_stops | Estimated route stop count S. |
| work_insertion_pairs | Number of insertion positions P per route. |
| work_candidate_scan | Cost proxy for candidate scanning across the episode. |
| work_active_proposals | Cost proxy for non-noop proposals across episode. |
| work_competition_factor | Extra pressure multiplier from proposal competition. |
| work_proposer | Policy-side proposal scoring cost proxy. |
| work_resolver | Resolver-side assignment/conflict-resolution cost proxy. |
| work_total | work_proposer + work_resolver. |
| work_warning | Optional flags about imputations or unknown names. |

For every row with recognized policy and resolver names:
- work_total = work_proposer + work_resolver


## Condensed formula table for `metrics_wide.csv`

This table is the shortest way to read the work columns in the two regimes now used by the plots:
- total-work regime: sum proposer and resolver costs
- parallel-time regime: assume decentralized proposer inference across robots, so proposer cost is divided by `R`
- `ra` means `reward_aligned`.
- The estimator branches on route construction for route-search-heavy modes.

Let:
- $R$ = number of robots
- $C = M \cdot R \cdot K$
- $A = M \cdot R \cdot (1-noop)$
- $P = \frac{(S+1)(S+2)}{2}$
- $I = 1$ for `nearest`, and $I = P$ for `reward_aligned`

| Component | `metrics_wide.csv` formula | Parallel-time proxy | Notes |
|---|---|---|---|
| `work_candidate_scan` | $C = M \cdot R \cdot K$ | same | Baseline candidate-scanning cost. |
| `work_active_proposals` | $A = M \cdot R \cdot (1-noop)$ | same | Number of non-NOOP proposals across the episode. |
| `work_proposer` (heuristics) | see policy table below | $work\_proposer / R$ | Heuristic proposer cost is treated as decentralized in the parallel regime. |
| `work_resolver` (predicted_reward_joint, `nearest`) | $A$ | same | Greedy route build, no insertion enumeration. |
| `work_resolver` (predicted_reward_joint, `reward_aligned`) | $A\cdot P\cdot joint\_multiplier$ | same | Default joint multiplier is 1.5. |

Important:

- The parallel-time proxy is a plotting/interpretation convenience; it is not stored in `metrics_wide.csv`.
- If `num_robots` is missing in the CSV, it can be inferred from `work_candidate_scan` and the episode statistics used to build the work columns.
- In the current implementation, the parallel proxy is only meant to rescale proposer cost, not resolver cost.

| Component family | `nr` / `nearest` case | `ra` / `reward_aligned` case | What changes in practice |
|---|---|---|---|
| Capacity / random resolver | $A$ | $A$ | No route-construction branch in code. |
| Closest-then-capacity resolver | $2A$ | $2A$ | Same algebra; different rows feed different `A`. |
| Predicted-reward resolver | $A$ | $A\cdot P$ | `nearest` skips insertion enumeration while `reward_aligned` enumerates route positions. |
| Predicted-reward-joint resolver | $A$ | $A\cdot P\cdot joint\_multiplier$ | Same route-construction split, with the joint multiplier only in `reward_aligned`. |
| Hungarian resolver | $D\cdot R\cdot T + D\cdot N^3$ | $D\cdot R\cdot T + D\cdot N^3$ | Same algebra; `nr` and `ra` rows can still differ through `M`, `K`, `D`, and `R`. |

Bottom line:
- `nr` and `ra` are plotted and compared as separate subsets.
- The route-search-heavy formulas differ: `nearest` uses greedy routing, while `reward_aligned` multiplies by `P`.
- The values also differ because the aggregated metrics in the row differ.

## Policy-specific work_proposer formulas

| Policy (pol) | work_proposer formula | Intuition |
|---|---|---|
| random | C | Constant-time style choice after candidate scan. |
| unique | C | Similar scan-level cost, little extra scoring. |
| pickup_distance | C | Single criterion per candidate. |
| pickup_deadline | C | Single criterion per candidate. |
| pickup_deadline_distance | 2 * C | Two criteria (deadline + distance) proxy as 2x scan effort. |
| predicted_reward | C * I | Route insertion evaluation over insertion positions only in `reward_aligned`. |
| predicted_reward_joint | C * I * joint_multiplier | Joint objective adds extra coordination/scoring overhead, again only in `reward_aligned`. |
| proposal_joint_competition | C * I * F | Joint scoring amplified by measured competition pressure. |

Notes:
- Alias closest is normalized to pickup_distance.
- joint_multiplier defaults to 1.5.
- F is data-driven from competition metrics:
	- F = 1 + max(max(cemn, 0), max(ovrlap, 0) * max(shared, 0))
- I = 1 for `nearest`, and I = P for `reward_aligned`.

## RL proposer formulas used for evaluation logs

The baseline table above applies to rows already stored in [metrics_wide.csv](../metrics_wide.csv). For RL evaluation logs, [scripts/estimate_eval_work_reward.py](../scripts/estimate_eval_work_reward.py) uses a separate proposer-side proxy.

Let:
- C_actor = work_candidate_scan_actor = M * R * (K + noop_candidates)

### 1hop proposer

Inference formula:
- work_actor = C_actor
- work_critic = 0
- work_proposer = work_actor

Training-style formula:
- work_actor = C_actor
- work_critic = C_actor
- work_proposer = 2 * C_actor

Interpretation:
- The 1hop proxy counts one actor inference pass per candidate slot, aggregated over robots and macro steps.
- By default it includes one extra NOOP action slot, so it is actually closer to M * R * (K + 1) than to M * R * K.
- In training mode the script assumes critic cost is roughly another forward pass of similar scale.

### 2hop proposer

Inference formula:
- work_actor = 2 * C_actor
- work_proposer = work_actor in inference mode

Interpretation:
- This is a simple depth multiplier proxy, not a measured network-cost model.


What is still not modeled explicitly:
- Graph message-passing cost
- Hidden dimension / layer width
- MLP head size
- Batching overhead
- Tensor-library / hardware effects

So the current proxy should be read as:
- pickup_deadline: one simple per-candidate scoring pass
- 1hop: one neural per-candidate inference pass of the same episode-scale order, but not the same true runtime

In other words, the present analytical model matches them in coarse scaling, not in exact constant factors.

If you want a stronger separation between heuristic and GNN proposer cost, the next refinement would be to multiply the 1hop actor term by an explicit architecture factor derived from hidden size, number of message-passing layers, or measured inference timing.

## Resolver-specific work_resolver formulas

Let A = work_active_proposals and P = work_insertion_pairs.

### 1) capacity and random resolvers

Formula:
- work_resolver = A

Interpretation:
- Resolver effort grows with how many active proposals exist.
- No extra route-insertion scoring multiplier is added.

Complexity trend:
- Roughly linear in M, R, and (1 - noop).

### 2) closest_then_capacity resolver

Formula:
- work_resolver = 2 * A

Interpretation:
- Two-stage selection logic is approximated as doubling base active-proposal effort.

Complexity trend:
- Same scaling as capacity, but constant factor 2x.

### 3) predicted_reward resolver

Formula:
- work_resolver = A * I

Interpretation:
- Resolver is modeled as evaluating insertion opportunities only when route_construction is `reward_aligned`.

Complexity trend:
- Linear in A and linear in I.
- Since P is quadratic in route_stops S, route size increases can strongly raise work.

### 4) predicted_reward_joint resolver

Formula:
- work_resolver = A * I * joint_multiplier
- default joint_multiplier = 1.5

Interpretation:
- Joint reward resolver assumes extra coordination/scoring overhead beyond predicted_reward.
- The extra insertion-enumeration factor is only active in `reward_aligned`.

Complexity trend:
- Same growth as predicted_reward, scaled by the joint multiplier.

### 5) hungarian resolver

Auxiliary terms:
- estimated_tasks_per_step T = max(R * K, 1)
- matrix_size N = max(R, T)

Formula:
- matrix_scoring_work = D * R * T
- assignment_work = D * N^3
- work_resolver = matrix_scoring_work + assignment_work

Interpretation:
- Includes matrix scoring plus cubic assignment solve proxy.
- This branch can become very large when K or R grows.

Complexity trend:
- Dominated by cubic term D * N^3 at moderate-to-large matrix sizes.

## End-to-end worked examples (policy + resolver)

These examples show how work_total is built from both components.

### Example A: policy = pickup_deadline, resolver = capacity

Given:
- C = 141.984
- A = 56.376

Then:
- work_proposer = C = 141.984
- work_resolver = A = 56.376
- work_total = 141.984 + 56.376 = 198.36

### Example B: policy = predicted_reward_joint, resolver = capacity

Given:
- C = 142.524
- P = 15
- joint_multiplier = 1.5
- A = 57.78

Then:
- work_proposer = C * P * 1.5 = 142.524 * 15 * 1.5 = 3206.79
- work_resolver = A = 57.78
- work_total = 3206.79 + 57.78 = 3264.57

Takeaway:
- Same resolver family can produce very different work_total if policy-side scoring becomes insertion-heavy.
- To isolate resolver impact, compare rows with same policy and similar C, P, A.

### Example C: proposer = 1hop, resolver = closest_then_capacity

Suppose evaluation aggregation gives:
- M = 49.4
- R = 6
- K = 0.657
- noop_candidates = 1
- noop = 0.6645

Then:
- C_actor = M * R * (K + 1) = 49.4 * 6 * 1.657 = 491.135
- A = M * R * (1 - noop) = 49.4 * 6 * (1 - 0.6645) = 99.4422
- work_actor = 491.135
- work_resolver = 2 * A = 198.884

Inference mode:
- work_critic = 0
- work_total = 491.135 + 198.884 = 690.019

Training-style mode:
- work_critic = 491.135
- work_total = 491.135 + 491.135 + 198.884 = 1181.154

Takeaway:
- 1hop is not estimated as identical to pickup_deadline here.
- It is estimated as a same-order scan-style proposer, but with an extra NOOP slot and optional critic duplication.

## How resolver choice changes work in practice

For similar scenario rows, resolver formulas create large separations:
- capacity-like branches keep resolver work near active proposal count.
- closest_then_capacity doubles that base.
- predicted_reward branches multiply by insertion_pairs.
- joint predicted variants add another 1.5x by default.
- hungarian can jump much higher due to cubic assignment proxy.

This is why two rows with similar reward metrics can show very different work_total values.

## Interpreting policy vs resolver effects

To separate effects cleanly:
- Compare work_proposer across policies while keeping resolver fixed.
- Compare work_resolver across resolvers while keeping policy fixed.
- Use work_total only for end-to-end computational burden.

Important:
- Changing only resolver may still change work_total less than expected if policy-side term dominates.
- In predicted_reward and joint variants, policy-side insertion scoring often dominates total cost.

## Interpreting real rows from metrics_wide.csv

Representative values visible in current data:
- A capacity row has work_active_proposals = 56.376 and work_resolver = 56.376.
- A closest_then_capacity row has work_active_proposals = 50.256 and work_resolver = 100.512 (exact 2x).
- A predicted_reward_joint row can reach work_resolver around 3000+, showing insertion-pair and joint amplification effects.
- A proposal_joint_competition setup with heavy competition can also produce large totals even when active proposals are moderate.

Use these as sanity checks when auditing resolver behavior.

## Warning flags and missing data behavior

work_warning may include:
- imputed_num_robots_from_instance
- imputed_num_robots_from_default
- missing_optional_metrics
- unknown_policy
- unknown_resolver

Behavior details:
- Unknown policy or resolver: all work_ numeric outputs are left blank for that row, and warning flags are set.
- Missing noop: defaults to 0.
- Missing dstep: falls back to msd.
- Missing ovrlap/shared/cemn: treated as 0 in competition factor fallback.

## Practical guidance for comparisons

When comparing resolvers, control for:
- Same scenario and instance.
- Same policy pol where possible.
- Similar R, K, M, D, and route capacity.

Then read:
- work_resolver for resolver-only complexity effect.
- work_total for end-to-end decision workload.

Recommended interpretation order:
1. Check work_warning first.
2. Compare work_resolver across resolvers.
3. Verify whether work_total differences are mostly resolver-side or proposer-side.
4. Relate complexity changes to reward/service metrics in the same row.

## Recomputing work columns

To recompute in place:
- python estimate_work_metrics.py path/to/metrics_wide.csv --in-place

Optional useful knobs:
- --joint-multiplier
- --route-stops
- --default-num-robots
- --unknown-mode warn|error

All formulas in this document correspond to analytical_v1 in [estimate_work_metrics.py](../estimate_work_metrics.py).