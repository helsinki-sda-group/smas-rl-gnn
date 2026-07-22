# Training Metrics Log: Full Column Reference

This document explains every column in `training_metrics_*.log` (for example: `training_metrics_v1250_ms2400_mwd240_mtd900_cap3.log`).

Primary implementation source:
- `utils/metrics_calculator.py`

## Where values come from

Each row is an `EpisodeMetrics` snapshot written by `metrics_to_string(...)`.

Data sources used in `compute_episode_metrics_from_logs(...)`:
- `rewards_macro.csv` (reward components and macro-step aggregates)
- `task_lifecycle.csv` (pickup/dropoff/obsolete lifecycle stats)
- `debug.csv` (NO-OP and overload/conflict stats)
- `candidates.csv` (candidate-list and overlap stats)
- `coordination.csv` (robot-level proposal and resolver-outcome counters)

If a denominator is zero (for example no picked tasks), the code writes `0.0` for that rate.

## Header blocks

The printed header is grouped as:

- `pol seed ts`
- reward sums: `rew cap step dln wait trav comp nsv`
- task lifecycle block: `pku pkr obs obsr pkv pkvr mwt cmp cmr anp anpr mtt pnc pncr`
- action/candidate block: `noop overld mcand cne_fr cne_mn dstep macmr msd ovrlap shared`
- coordination block: `ecr unop ncpr psur offpr`

## Column-by-column definition

### Identity columns

| Column | Field | Meaning |
|---|---|---|
| `pol` | `policy` | Policy label string (for example experiment index/name) set by the caller. |
| `seed` | `seed` | Random seed associated with that run/episode. |
| `ts` | `ts` | Model timestep identifier (commonly parsed from checkpoint name). |

### Reward aggregate columns (from `rewards_macro.csv`)

These are **sums over macro-steps** in the episode (except `macmr`, see later).

| Column | Field | Formula / meaning |
|---|---|---|
| `rew` | `reward_sum` | $\sum_t \text{reward}_t$ |
| `cap` | `capacity_sum` | $\sum_t \text{capacity\_avg}_t$ |
| `step` | `step_sum` | $\sum_t \text{step\_avg}_t$ |
| `dln` | `deadline_sum` | $\sum_t \text{deadline\_avg}_t$ (fallback: `missed_deadline_avg`) |
| `wait` | `wait_sum` | $\sum_t \text{wait\_avg}_t$ |
| `trav` | `travel_sum` | $\sum_t \text{travel\_avg}_t$ |
| `comp` | `completion_sum` | $\sum_t \text{completion\_avg}_t$ |
| `nsv` | `nonserved_sum` | $\sum_t \text{nonserved\_avg}_t$ |

### Task lifecycle columns (from `task_lifecycle.csv`)

Definitions:
- $N = \text{total\_tasks}$
- $N_{pku} = \text{picked\_up\_tasks}$
- $N_{cmp} = \text{completed\_tasks}$

| Column | Field | Formula / meaning |
|---|---|---|
| `pku` | `picked_up_tasks/total_tasks` | Displayed as `a/b` where $a=N_{pku}$ and $b=N$. |
| `pkr` | `pickup_rate` | $N_{pku} / N$ |
| `obs` | `obsolete_tasks` | Count of tasks with `was_obsolete == True`. |
| `obsr` | `obsolete_rate` | $\text{obsolete\_tasks} / N$ |
| `pkv` | `pickup_violated/picked_up_tasks` | Displayed as `a/b`, where violation means `actual_pickup_time > pickup_deadline` among picked tasks. |
| `pkvr` | `pickup_violated_rate` | $\text{pickup\_violated} / N_{pku}$ |
| `mwt` | `mean_wait_time` | Mean `actual_waiting_time` over picked-up tasks. |
| `cmp` | `completed_tasks/total_tasks` | Displayed as `a/b` where $a=N_{cmp}$ and $b=N$. |
| `cmr` | `completion_rate` | $N_{cmp} / N$ |
| `anp` | `assigned_never_picked/total_tasks` | Displayed as `a/b`; assigned taxi exists, but task was never picked up. |
| `anpr` | `assigned_never_picked_rate` | $\text{assigned\_never\_picked} / N$ |
| `mtt` | `mean_travel_time_completed` | Mean `actual_travel_time` over completed tasks. |
| `pnc` | `picked_not_completed/picked_up_tasks` | Displayed as `a/b`; picked up but never dropped off. |
| `pncr` | `picked_not_completed_rate` | $\text{picked\_not\_completed} / N_{pku}$ |

### Action/candidate columns (`debug.csv` + `candidates.csv` + `rewards_macro.csv`)

Definitions:
- $S = \text{total debug apply-input steps}$
- $R = \text{num robots}$

| Column | Field | Formula / meaning |
|---|---|---|
| `noop` | `noop_fraction` | Fraction of robot decisions that are NO-OP (assignment is `None`): $\text{noop\_count}/(S\cdot R)$. |
| `overld` | `overload_assignment_fraction` | $\text{overload\_count}/S$. `overload_count` increments on step pairs where both `winners` and `cand_counts` payloads are present in debug logs. |
| `mcand` | `mean_candidates_per_taxi` | Mean candidate count per row in `candidates.csv` (`cand_res_ids` split by `|`). |
| `cne_fr` | `cand_nonempty_frac` | Fraction of taxi-rows with at least 1 candidate. |
| `cne_mn` | `cand_mean_nonempty` | Mean candidate count conditioned on non-empty rows only. |
| `dstep` | `decision_steps` | Number of `time` groups where at least one taxi has non-empty candidates. |
| `macmr` | `macro_reward_mean` | Mean macro reward: $\frac{1}{T}\sum_t \text{reward}_t$ from `rewards_macro.csv`. |
| `msd` | `macro_steps_done` | Number of rows in `rewards_macro.csv` (macro steps logged). |
| `ovrlap` | `overlap_rate` | Fraction of candidate time-steps where at least one task appears in candidate lists of $\ge 2$ taxis. |
| `shared` | `mean_shared_tasks_per_step` | Mean count of tasks per time-step that are shared by $\ge 2$ taxis. |

### Coordination columns (`coordination.csv`)

`coordination.csv` contains one row per macro decision step. Episode-level rates are computed from **summed counts**, not from averaging per-step ratios.

Let:
- $N = \sum \text{robot\_decisions}$
- $E = \sum \text{empty\_candidate\_decisions}$
- $C = \sum \text{nonempty\_candidate\_decisions}$
- $U = \sum \text{unforced\_noops}$
- $P = \sum \text{real\_proposals}$
- $Q_1 = \sum \text{unique\_proposals}$
- $S = \sum \text{survived\_proposals}$
- $F = \sum \text{final\_assignments}$
- $O = \sum \text{off\_proposal\_assignments}$

Definitions used at decision-step level:
- `empty_candidate_decision`: robot had no valid candidate in final actor mask.
- `unforced_noop`: robot had candidates but selected explicit NOOP.
- `real_proposal`: robot had candidates, selected non-NOOP, and mapped to a valid task id.
- `unique_proposal`: real proposal in a bucket of size 1 by proposed task id.
- `conflicting_proposal`: real proposal in a bucket of size $\ge 2$ by proposed task id.
- `survived_proposal`: resolver output for that robot equals its original proposed task.

| Column | Field | Formula / meaning |
|---|---|---|
| `ecr` | `empty_candidate_rate` | $E/N$ if $N>0$, else $0.0$. |
| `unop` | `unforced_noop_rate` | $U/C$ if $C>0$, else $0.0$. |
| `ncpr` | `nonconflicting_proposal_rate` | $Q_1/P$ if $P>0$, else $0.0$. |
| `psur` | `proposal_survival_rate` | $S/P$ if $P>0$, else $0.0$. |
| `offpr` | `off_proposal_assignment_rate` | $O/F$ if $F>0$, else $0.0$. |

### Overlap vs conflict-ratio vs NCPR

- `ovrlap` and `shared` are candidate-set overlap diagnostics (who *could* compete).
- existing `conflict_ratio` in `conflicts.log` is a task-bucket conflict frequency (`conflicts_total / tasks_total`) and remains unchanged.
- `ncpr` is proposal/robot based (`unique_proposals / real_proposals`) and measures how many submitted proposals were nonconflicting.

### Additional dropoff metrics (summary only)
- `drp_ev`	dropoff_event_rate (all dropoffs / total tasks)
- `vcmr`	valid_completion_rate (pickup+dropoff before deadlines / total tasks)
- `invdr`	invalid_dropoff_rate (dropoffs failing validity / total tasks)
- `ddvr`	dropoff_deadline_violation_rate (dropped after dropoff_deadline / total tasks)

Invalid completion rate `1-vcmr` id calculated as rate of tasks which had dropoff but violated it `invdr` + rate of tasks which did not have dropoff:
$$1-vcmr = invdr + (1-drp\_ev)$$
Then invalide dropoff rate `invdr` is sum of rates of tasks violating only pickup $O_{pick}$, only dropoff $O_{drop}$, or both: $invdr=$O_{pick}+$O_{drop}+both$.
Given that $ddvr = $O_{drop}+both$, $O_pick = indvr - ddvr$.
$O_{pick}+both$ is pickup violation rate among all tasks, and it is $pvr_{tot} = pkr \cdot pkvr$.
Then $both = pvr_{tot}-indvr+ddvr$, $O_{drop}=ddvr-both$.

These quantities are related but not interchangeable.

## Practical interpretation

- High `noop` with low `mcand` often means robots frequently have nothing actionable.
- High `noop` with high `mcand` suggests policy hesitation or conservative NO-OP preference.
- `ovrlap` and `shared` describe competition pressure in candidate sets.
- `overld` indicates how often winner-selection/conflict handling path is active.
- `macmr` is the easiest single scalar for short-term trend, while `rew` is cumulative over the macro horizon.

## Code pointers

- Dataclass fields: `utils/metrics_calculator.py` → `EpisodeMetrics`
- Computation: `utils/metrics_calculator.py` → `compute_episode_metrics_from_logs(...)`
- Formatting/header: `metrics_to_string(...)`, `get_metrics_header(...)`
- Writer helpers: `ensure_metrics_log(...)`, `append_metrics_log(...)`, `append_metrics_summary(...)`
