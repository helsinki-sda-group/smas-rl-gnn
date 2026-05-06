# Episode-Level Quality Diagnostics

## Overview

The episode-level quality diagnostics system provides detailed per-episode metrics for ride-pooling RL experiments. It helps answer diagnostic questions such as:

- **Are wait penalties active?** → `rew_wait_event_pickup_mean`, `task_wait_time_mean`
- **Is the system actually pooling passengers?** → `pool_frac_multi_pax`, `pool_mean_onboard`
- **Is the policy choosing good tasks?** → `dec_noop_rate`, `dec_noop_with_candidates_count`
- **Is NOOP caused by no feasible tasks or conservative behavior?** → `dec_noop_no_candidates_count` vs `dec_noop_with_candidates_count`
- **How many tasks complete vs expire?** → `task_completed_rate`, `task_obsolete_rate`
- **Are raw dropoffs actually valid completions?** → `task_valid_completed_rate` vs `task_dropoff_event_rate`
- **Are deadlines being systematically violated?** → `task_pickup_deadline_violation_rate`, `task_dropoff_deadline_violation_rate`

This system **does not replace** any existing logs (`training_metrics_*.log`, `conflicts.log`, `rewards.csv`, etc.). It writes additional files to the run directory.

---

## Configuration

Add the following to your YAML config under the `logging:` key:

```yaml
logging:
  extended_quality_metrics: false        # Enable quality metrics computation
  extended_quality_plots: false          # (reserved for future use)
  extended_quality_include_task_level: false   # Write per-task rows to task_quality_events.csv
  extended_quality_include_decision_level: false  # Write per-decision rows to decision_quality_events.csv
```

When `extended_quality_metrics: true`, the following files are created in the run directory (NOT episode directories, so they survive pruning):

| File | Description |
|------|-------------|
| `quality_episode_metrics.csv` | One row per episode, flat metrics |
| `task_quality_events.csv` | Per-task rows (only if `extended_quality_include_task_level: true`) |
| `decision_quality_events.csv` | Per-decision rows (only if `extended_quality_include_decision_level: true`) |

Additionally, when enabled, two extra CSVs are written **per episode** (inside the episode directory, alongside existing files):

| File | Description |
|------|-------------|
| `robot_occupancy.csv` | Per-robot per-step occupancy state |
| `decisions.csv` | Per-robot per-decision-step action + candidate count |

---

## Metric Dictionary

### Identity Fields

| Column | Description |
|--------|-------------|
| `config_id` | Config prefix (from `cfg.prefix`) |
| `run_id` | Run name |
| `ts` | Training timestep at episode end |
| `episode` | Episode index |
| `reward_type` | `wait_travel` or `deadline` |
| `completion_mode` | Task completion mode: `pickup`, `dropoff`, or `valid_dropoff` |
| `num_robots` | Number of robots in this episode |
| `max_robot_capacity` | Robot capacity |

### Reward Subcomponents (Section: `rew_*`)

These come from internal accumulators in `RLControllerAdapter` (not stored in existing CSVs).

| Column | Description | Relevant reward_type |
|--------|-------------|---------------------|
| `rew_completion_sum` | Sum of completion bonuses (mode-dependent: counts pickups, all dropoffs, or valid dropoffs only) | both |
| `rew_completion_count` | Number of completions credited (depends on `completion_mode`) | both |
| `rew_completion_mean` | Mean completion bonus per credited event | both |
| `rew_dropoff_event_sum` | Reward assigned to all dropoff events (= 0 in `pickup` mode, = `rew_completion_sum` in `dropoff` mode, = `rew_valid_dropoff_sum` in `valid_dropoff` mode) | both |
| `rew_dropoff_event_count` | Count of all observed dropoff events regardless of mode | both |
| `rew_dropoff_event_mean` | Mean dropoff reward per event | both |
| `rew_valid_dropoff_sum` | Reward given for valid completions (pickup ≤ deadline AND dropoff ≤ deadline) | both |
| `rew_valid_dropoff_count` | Number of valid dropoff events | both |
| `rew_valid_dropoff_mean` | Mean reward per valid dropoff | both |
| `rew_invalid_dropoff_sum` | Reward given for invalid dropoffs (0.0 in `valid_dropoff` mode; > 0 in `dropoff` mode since all dropoffs are rewarded) | both |
| `rew_invalid_dropoff_count` | Number of invalid dropoff events (failed validity check) | both |
| `rew_invalid_dropoff_mean` | Mean reward per invalid dropoff | both |
| `rew_wait_event_pickup_sum` | Total wait penalty at pickups | `wait_travel` |
| `rew_wait_event_pickup_count` | Number of pickup wait events | `wait_travel` |
| `rew_wait_event_pickup_mean` | Mean wait penalty per pickup | `wait_travel` |
| `rew_wait_obsolete_pickup_sum` | Wait penalty for obsolete pickup events | `wait_travel` |
| `rew_wait_terminal_never_picked_sum` | Terminal wait penalty for never-picked tasks | `wait_travel` |
| `rew_wait_unattributed_sum` | Wait penalty not attributed to specific events | `wait_travel` |
| `rew_travel_event_dropoff_sum` | Total travel penalty at dropoffs | `wait_travel` |
| `rew_travel_event_dropoff_count` | Number of dropoff events | `wait_travel` |
| `rew_travel_event_dropoff_mean` | Mean travel penalty per dropoff | `wait_travel` |
| `rew_travel_terminal_picked_not_dropped_sum` | Terminal travel penalty for picked but not dropped tasks | `wait_travel` |
| `rew_deadline_pickup_lateness_sum` | Sum of pickup lateness penalties | `deadline` |
| `rew_deadline_pickup_lateness_count` | Number of pickup lateness events | `deadline` |
| `rew_deadline_pickup_lateness_mean` | Mean pickup lateness | `deadline` |
| `rew_deadline_dropoff_lateness_sum` | Sum of dropoff lateness penalties | `deadline` |
| `rew_deadline_dropoff_lateness_mean` | Mean dropoff lateness | `deadline` |
| `rew_obsolete_total_sum` | Total obsolete penalties | both |
| `rew_terminal_total_sum` | Total terminal penalties | both |

### Macro Reward (from `rewards_macro.csv`)

| Column | Description |
|--------|-------------|
| `macro_reward_mean` | Mean macro-step reward |
| `macro_reward_std` | Std of macro-step rewards |
| `macro_reward_sum` | Total reward for episode |
| `macro_steps` | Number of macro steps |

### Task Quality (Section: `task_*`)

Computed from `task_lifecycle.csv` in the episode directory.

| Column | Description |
|--------|-------------|
| `task_total_count` | Total tasks in episode |
| `task_completed_count` | Tasks dropped off while `was_obsolete=False` (backward-compat definition) |
| `task_completed_rate` | `completed / total` |
| `task_obsolete_count` | Tasks marked as obsolete (missed pickup window) |
| `task_obsolete_rate` | `obsolete / total` |
| `task_never_picked_count` | Tasks never picked up |
| `task_never_picked_rate` | `never_picked / total` |
| `task_picked_not_dropped_count` | Picked up but never dropped off |
| `task_picked_not_dropped_rate` | `picked_not_dropped / total` |
| `task_obs_dropoff_count` | Picked up but not dropped off before `dropoff_deadline` |
| `task_obs_dropoff_rate` | `obs_dropoff / total` |
| `task_dropoff_event_count` | All tasks with `actual_dropoff_time != None` (raw dropoff count) |
| `task_dropoff_event_rate` | `dropoff_event / total` |
| `task_valid_completed_count` | Tasks where `actual_pickup_time <= pickup_deadline AND actual_dropoff_time <= dropoff_deadline` |
| `task_valid_completed_rate` | `valid_completed / total` |
| `task_invalid_dropoff_count` | Dropped off tasks that failed the validity test (`dropoff_event - valid_completed`) |
| `task_invalid_dropoff_rate` | `invalid_dropoff / total` |
| `task_pickup_deadline_violation_count` | Tasks with `actual_pickup_time > pickup_deadline` |
| `task_pickup_deadline_violation_rate` | `pickup_violation / total` |
| `task_dropoff_deadline_violation_count` | Tasks with `actual_dropoff_time > dropoff_deadline` |
| `task_dropoff_deadline_violation_rate` | `dropoff_violation / total` |
| `task_obsolete_pickup_count` | Tasks where `was_obsolete=True` (missed pickup deadline) |
| `task_obsolete_pickup_rate` | `obsolete_pickup / total` |
| `task_obsolete_dropoff_count` | Dropped-off tasks that were already obsolete or missed dropoff deadline |
| `task_obsolete_dropoff_rate` | `obsolete_dropoff / total` |
| `task_wait_time_mean` | Mean actual wait time (pickup - reservation) |
| `task_wait_time_std` | Std of wait times |
| `task_wait_time_p50/p90/p95` | Percentile wait times |
| `task_wait_time_count` | Number of tasks with valid wait time |
| `task_travel_time_mean` | Mean actual travel time (dropoff - pickup) |
| `task_travel_time_p50/p90/p95` | Percentile travel times |
| `task_travel_ratio_mean` | Mean ratio of actual/estimated travel time |
| `task_pickup_lateness_mean` | Mean pickup lateness vs `pickup_deadline` (deadline mode) |

**`was_obsolete_dropoff` definition**: a task that was picked up but not dropped off before `dropoff_deadline`:
```python
actual_pickup_time is not None AND
dropoff_deadline is not None AND
(actual_dropoff_time is None OR actual_dropoff_time > dropoff_deadline)
```

### Pooling Metrics (Section: `pool_*`)

Computed from `robot_occupancy.csv` (written per-step when quality metrics enabled).

| Column | Description |
|--------|-------------|
| `pool_mean_onboard` | Mean passengers onboard across all robots/steps |
| `pool_max_onboard` | Max passengers onboard in any step |
| `pool_frac_multi_pax` | Fraction of robot-steps with ≥2 passengers (actual pooling) |
| `pool_frac_empty` | Fraction of robot-steps with 0 passengers |
| `pool_steps_total` | Total robot-step observations |
| `pool_robot_mean_onboard_mean` | Mean of per-robot mean occupancies |

### Decision / NOOP Metrics (Section: `dec_*`)

Computed from `decisions.csv` (written per-decision-step when quality metrics enabled).

| Column | Description |
|--------|-------------|
| `dec_total` | Total decisions made |
| `dec_noop_count` | Number of NOOP decisions |
| `dec_noop_rate` | `noop / total` |
| `dec_noop_no_candidates_count` | NOOPs with zero candidates (forced — no feasible tasks) |
| `dec_noop_with_candidates_count` | NOOPs with candidates available (conservative behavior) |
| `dec_mean_candidates` | Mean number of candidates per decision step |
| `dec_choice_eta_rank_mean` | **Unavailable** — ETAs per candidate not stored (always 0.0) |
| `dec_choice_pickup_margin_mean` | **Unavailable** — requires per-candidate ETAs (always 0.0) |

> **Note**: ETA-rank and feasibility margin metrics require storing per-candidate ETAs, which is not currently done. They are set to 0.0 and documented here for future implementation.

### Candidate Availability (Section: `cand_*`)

Computed from `candidates.csv`.

| Column | Description |
|--------|-------------|
| `cand_mean_slots` | Mean candidate slots available per dispatch step |
| `cand_zero_slots_rate` | Fraction of dispatch steps with zero candidates |

### Conflict Quality (Section: `conf_*`)

From in-memory `_conflict_stats` dict captured at episode end.

| Column | Description |
|--------|-------------|
| `conf_total` | Total conflicts in episode |
| `conf_tasks_total` | Total tasks processed in conflicts |
| `conf_winner_pickup` | Conflicts won by the robot with shortest pickup ETA |
| `conf_winner_pickup_rate` | `winner_pickup / total` |
| `conf_resolver_override` | Number of conflict **events** where the margin-based winner differed from the pickup-distance-based winner. Increments by at most 1 per conflict event regardless of how many robots competed. |
| `conf_resolver_override_rate` | `resolver_override / total` — how often the margin ranking disagreed with the distance ranking per conflict event |
| `conf_policy_matches_resolver_rate` | Rate at which the robot with the highest raw policy logit for the contested task also won the conflict (i.e., policy and resolver agreed on the winner). **Note:** these two rates are NOT complementary — `override_rate + policy_matches_rate` does not equal 1. `override_rate` measures distance-vs-margin disagreement; `policy_matches_rate` measures policy-vs-resolver disagreement. |
| `conf_margin_win_count` | Conflicts where winner had highest margin logit |
| `conf_margin_lose_count` | Conflicts where winner did NOT have highest margin logit |

---

## Completion Modes

The `completion_mode` config parameter controls when the `+w_comp` bonus is awarded:

| Mode | When `+w_comp` is given | Typical use |
|------|------------------------|-------------|
| `pickup` | At pickup event | Reward pickup speed |
| `dropoff` | At every dropoff event (regardless of deadlines) | Default; matches `rew_completion_count > task_valid_completed_count` |
| `valid_dropoff` | Only when `actual_pickup_time <= pickup_deadline AND actual_dropoff_time <= dropoff_deadline` | Aligns reward with service-quality objective |

### Valid completion definition

```text
valid_completion(task) =
    actual_pickup_time is not None
    AND actual_dropoff_time is not None
    AND actual_pickup_time <= pickup_deadline
    AND actual_dropoff_time <= dropoff_deadline
```

### Backward compatibility

- `completion_mode=dropoff` preserves exact pre-existing behavior.
- Old CSV columns (`rew_completion_*`, `task_completed_*`) are still present for all modes.
- New columns (`rew_valid_dropoff_*`, `task_valid_completed_*`, etc.) are appended; parsers expecting old columns are unaffected.
- Old logs that lack the new columns can still be read by plotting scripts — missing columns are handled gracefully.

### Why `rew_completion_count > task_valid_completed_count` in `dropoff` mode

Under `completion_mode=dropoff`, all dropoffs receive completion reward — including tasks that had already been marked obsolete or that missed their pickup deadline. The `task_valid_completed_count` metric only counts tasks satisfying both deadline constraints, so the two numbers legitimately diverge when late-pickup tasks are still served.

---

## Reward Type Mapping

| `reward_type` | Active reward subcomponents |
|---------------|----------------------------|
| `wait_travel` | `rew_wait_*`, `rew_travel_*`, `rew_completion_*`, `rew_obsolete_*`, `rew_terminal_*` |
| `deadline` | `rew_deadline_*`, `rew_completion_*`, `rew_obsolete_*`, `rew_terminal_*` |

Columns from the other mode will be zero-valued but present in the CSV for schema consistency.

---

## Plotting

```bash
# Plot a single run
python plot_quality_episode_metrics.py \
    --metrics runs/my_run/quality_episode_metrics.csv \
    --out plots/quality

# Overlay multiple runs
python plot_quality_episode_metrics.py \
    --metrics runs/run1/quality_episode_metrics.csv runs/run2/quality_episode_metrics.csv \
    --label-from run_id \
    --out plots/quality_comparison

# Plot specific metric groups only
python plot_quality_episode_metrics.py \
    --metrics runs/my_run/quality_episode_metrics.csv \
    --groups task_rates pooling decisions \
    --out plots/quality
```

Available metric groups: `reward_components`, `task_rates`, `task_counts`, `task_wait_time`, `task_travel_time`, `pooling`, `decisions`, `conflicts`.

X-axis is `ts` (training timestep) when available, otherwise `episode` index.

---

## Interpretation Guide

### Is the system pooling?

Check `pool_frac_multi_pax`. If near 0, passengers are rarely sharing rides — either demand is low or robot capacity is not being used. Compare with `pool_frac_empty` to understand idle time.

### Why is NOOP rate high?

- If `dec_noop_no_candidates_count` is high → forced NOOP (no feasible tasks); may indicate supply-demand mismatch or very tight feasibility constraints.
- If `dec_noop_with_candidates_count` is high → conservative policy behavior; policy is choosing NOOP over available tasks.

### Is the policy learning to complete tasks?

Watch `task_completed_rate` increasing over training (x-axis = `ts`). A stagnant or decreasing rate indicates reward signal problems.

### Are wait penalties being triggered?

`rew_wait_event_pickup_mean` should be negative if wait penalties are active. If zero, check `reward_type == "wait_travel"` and `w_wait` weight in context.

### Are deadlines being violated?

`task_pickup_lateness_mean` > 0 indicates average deadline violations. In `deadline` mode, `rew_deadline_pickup_lateness_mean` should track this. High `task_obs_dropoff_rate` indicates dropoff deadline violations.

### Is the policy actually solving the real objective?

Compare `task_valid_completed_rate` vs `task_dropoff_event_rate`. If the gap is large, vehicles are reaching destinations but failing deadline constraints — the reward signal (`completion_mode=dropoff`) may be rewarding low-quality service. Switch to `completion_mode=valid_dropoff` to gate the reward on genuine deadline compliance.

### Are dropoff deadline violations driving the gap?

Check `task_dropoff_deadline_violation_rate` and `task_pickup_deadline_violation_rate` separately. Pickup violations (`actual_pickup_time > pickup_deadline`) indicate slow dispatch. Dropoff violations indicate the task was picked up but travel took too long.

### Conflict resolution quality

**`conf_resolver_override_rate`** measures how often the margin-based ranking disagrees with the pickup-distance ranking — i.e., the robot selected by margin was NOT the closest robot. A value of 0.64–0.78 is normal and means margin and distance frequently pick different winners. It counts per conflict *event* (not per competing robot), so a 3-robot conflict adds at most 1.

**`conf_policy_matches_resolver_rate`** measures how often the robot with the highest raw policy logit for the contested task actually won the resolver decision. A value of 0.0 was previously caused by a bug where raw logits were never forwarded to the conflict resolver. This should now report a non-zero value reflecting genuine policy-resolver alignment.
