# Reward System (Code-Linked)

This document describes how rewards are computed in the current codebase and how they are controlled from `configs/rp_gnn.yaml`.

## Source-of-truth in code

- Core reward config parsing:
  - `sumo_rl_rs/environment/rl_controller_adapter.py` → `RLControllerAdapter.__init__`
- Non-terminal reward computation:
  - `sumo_rl_rs/environment/rl_controller_adapter.py` → `_compute_rewards_per_robot_from_events(...)`
- Terminal reward computation:
  - `sumo_rl_rs/environment/rl_controller_adapter.py` → `_compute_terminal_penalty_terms(...)`
- Merge terminal into per-step terms:
  - `sumo_rl_rs/environment/rl_controller_adapter.py` → `apply_and_step(...)`
- Macro accumulation and logging:
  - `sumo_rl_rs/environment/ridepool_rt_env.py` → `RidepoolRTEnv.step(...)`
  - `sumo_rl_rs/logging/ridepool_logger.py` → `log_rewards(...)`, `log_macro_step(...)`

## YAML flags (`configs/rp_gnn.yaml`)

Under `env.reward_params`:

- `reward_type`: `deadline` or `wait_travel`
- `w_comp`, `w_wait`, `w_travel`, `w_deadline`
- `wait_cap`, `travel_cap`, `deadline_cap`
- `terminal_wait_share_unowned`

Also relevant:

- `env.max_wait_delay_s`: pickup deadline horizon for task creation
- `env.max_travel_delay_s`: dropoff deadline horizon for task creation
- `env.decision_dt`: macro action spacing (rewards are still computed at each internal sim-step)

---

## 1) Non-terminal reward (every internal sim step)

Computed in `_compute_rewards_per_robot_from_events(...)` from new pickup/dropoff events.

For robot $i$ at step $t$:

$$
r_i(t) = r^{comp}_i + r^{wait}_i + r^{deadline}_i + r^{travel}_i
$$

(with inactive terms equal to 0 depending on `reward_type`).

### 1.1 Completion term

If `completion_mode == "pickup"`:

$$
r^{comp}_i = w_{comp} \cdot |\text{new\_pickups}_i|
$$

Else (`dropoff`, current default):

$$
r^{comp}_i = w_{comp} \cdot |\text{new\_dropoffs}_i|
$$

### 1.2 Wait term (event at pickup)

For each newly picked task $j$ by robot $i$:

$$
\Delta r^{wait}_{i,j} = -\frac{\min(\text{wait}_j,\, \text{wait\_cap})}{\text{wait\_cap}}
$$

where $\text{wait}_j = now - reservationTime_j$.

Then:

$$
r^{wait}_i = w_{wait} \cdot \sum_j \Delta r^{wait}_{i,j}
$$

### 1.3 Deadline mode (`reward_type == deadline`)

Pickup lateness:

$$
late^{pu}_j = \max(0,\, now - pickupDeadline_j)
$$

Dropoff lateness:

$$
late^{do}_j = \max(0,\, now - dropoffDeadline_j)
$$

Per-event normalized penalty:

$$
\Delta d = -\frac{\min(late,\, deadline\_cap)}{deadline\_cap}
$$

Then:

$$
r^{deadline}_i = w_{deadline} \cdot \sum \Delta d
$$

### 1.4 Wait-travel mode (`reward_type == wait_travel`)

Travel penalty at dropoff only:

$$
over_j = \max(0,\, actualTravel_j - estTravel_j)
$$

$$
\Delta r^{travel}_{i,j} = -\frac{\min(over_j,\, travel\_cap)}{travel\_cap}
$$

$$
r^{travel}_i = w_{travel} \cdot \sum_j \Delta r^{travel}_{i,j}
$$

---

## 2) Newly-obsolete penalty (one-time when task becomes obsolete)

A task is considered in the newly-obsolete set once:

$$
newly\_obsolete = obsolete\_now \setminus obsolete\_{prev}
$$

So this branch is one-shot per task.

### 2.1 In `deadline` mode

For each newly obsolete task $j$:

$$
late_j = \max(1.0,\, now - pickupDeadline_j)
$$

$$
p_j = -\max\left(0.05,\, \frac{\min(late_j,\, deadline\_cap)}{deadline\_cap}\right)
$$

Added to `deadline` component (`deadline_penalty_by_robot`).

### 2.2 In `wait_travel` mode

For each newly obsolete task $j$:

$$
late_j = \max(1.0,\, now - pickupDeadline_j)
$$

$$
p_j = -\max\left(0.05,\, \frac{\min(late_j,\, wait\_cap)}{wait\_cap}\right)
$$

Added to `wait` component (`wait_penalty_by_robot`).

### 2.3 Owner attribution

Order:

1. `self._res_owner_by_res[res_id]`
2. owner from shadow plan (`owner_from_shadow`)
3. fallback equal split across robots

So yes, unknown ownership can produce equal split penalties.

---

## 3) Terminal reward (applied once at episode end)

In `apply_and_step(...)`:

- if `done` and not `_terminal_penalties_applied`:
  - compute `_compute_terminal_penalty_terms(...)`
  - add into current step `terms` and `per_robot`
  - set `_terminal_penalties_applied = True`

Only active for `reward_type == wait_travel`.

### 3.1 Task filtering

Terminal loop includes tasks with:

- `actual_dropoff_time is None`

So unfinished tasks are penalized at end.

### 3.2 Never-picked unfinished task

Current behavior: **terminal wait only** (terminal travel removed for this case).

$$
\Delta r^{wait}_{term,j} = -\frac{\min(now - reservationTime_j,\, wait\_cap)}{wait\_cap} \cdot w_{wait}
$$

Added to `terms[rid]["wait"]`.

### 3.3 Picked-but-not-dropped task

Terminal travel overtime:

$$
elapsed_j = now - pickupTime_j
$$

$$
over_j = \max(0,\, elapsed_j - estTravel_j)
$$

$$
\Delta r^{travel}_{term,j} = -\frac{\min(over_j,\, travel\_cap)}{travel\_cap} \cdot w_{travel}
$$

Added to `terms[rid]["travel"]`.

### 3.4 Unowned terminal wait sharing

If owner cannot be resolved and `terminal_wait_share_unowned == true`:

- split terminal wait penalty evenly across all robots.

If `false`:

- that unowned terminal wait penalty is discarded.

---

## 4) Which CSV columns receive which components

- Per robot per internal step:
  - `rewards.csv` columns: `reward`, `deadline`, `wait`, `travel`, `completion`, ...
- Macro step sums (in env step):
  - `rewards_macro.csv` columns: `reward`, `deadline_avg`, `wait_avg`, `travel_avg`, `completion_avg`, ...

(Names contain `_avg` historically but values are cumulative sums over internal steps of one macro action.)

---

## 5) Practical interpretation with your current YAML values

From `configs/rp_gnn.yaml`:

- `w_comp = 2.0`
- `w_wait = 1.5`
- `w_travel = 1.5`
- `wait_cap = 600`
- `travel_cap = 90`

Per single normalized event contribution:

- wait term in $[-1, 0]$, weighted to $[-1.5, 0]$
- travel term in $[-1, 0]$, weighted to $[-1.5, 0]$
- completion contributes `+2.0` per counted completion

Totals are sums over robots, tasks, and internal steps.
