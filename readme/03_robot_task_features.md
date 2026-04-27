# Robot & Task Features (Code + YAML Mapping)

This document describes feature definitions, normalization, edge features, and route-slot extensions.

## Source-of-truth in code

- Feature construction:
  - `utils/feature_fns.py`
    - `compute_feature_dim(...)`
    - `get_feature_names(...)`
    - `expand_edge_features(...)`
    - `make_feature_fn(...)`
- Graph packing:
  - `rt_gnn_rl/graphs/builders.py` → `build_padded_ego_batch(...)`
- Environment observation spaces:
  - `sumo_rl_rs/environment/ridepool_rt_env.py`

---

## 1) YAML flags controlling features

From `configs/rp_gnn.yaml`:

- `features.use_xy_pickup`
- `features.normalize_features`
- `features.use_node_type`
- `features.use_ego_robot`
- `features.use_edge_rt`
- `features.edge_features`
- `features.robot_commitment`
- `features.route_slots_k`

Environment flags that affect scaling/context:

- `env.vicinity_m`
- `env.max_robot_capacity`
- `env.max_wait_delay_s`
- `env.max_travel_delay_s`
- `env.max_steps`

---

## 2) Node feature dimensionality

Base node dimension is 9 (`compute_feature_dim`).

Let:

- $d_{base}=9$
- add +2 if `use_xy_pickup=true` and `use_edge_rt=false`
- add +2 if `use_node_type=true`
- add +1 if `use_ego_robot=true`

So:

$$
F = 9 + 2\,\mathbb{1}[use\_xy\_pickup \land \neg use\_edge\_rt] + 2\,\mathbb{1}[use\_node\_type] + \mathbb{1}[use\_ego\_robot]
$$

---

## 3) Robot node features

For node types `robot`, `robot_ego`, `robot_other` in `feature_fn`:

### Core slots

- `out[0], out[1]`: robot position $(x,y)$
- `out[2]`: free capacity = `(vehicle_capacity - onboard_count)`
- remaining base slots are pads unless optional flags fill tail slots

### Optional one-hot tails

If `use_node_type=true`:

- append `is_robot`, `is_task`

If `use_ego_robot=true`:

- append `is_ego_robot` (1 for `robot_ego`, 0 for others)

### Normalization

When `normalize_features=true`:

- positions divided by `pos_scale = max(1, vicinity_m)`
- free capacity divided by `cap_scale = max(1, max_robot_capacity)`

---

## 4) Task node features

For node type `task` in `feature_fn`:

### Time features

- reservation time
- waiting time: $\max(0, now - reservationTime)$
- estimated travel time (`estTravelTime`)

Normalized mode:

- reservation time / `time_scale`
- waiting / `wait_scale=max(1,max_wait_delay_s)`
- est travel / `travel_scale=max(1,max_travel_delay_s)`

where

- `time_scale = max_steps` if present, else `wait_scale`

### Spatial features

- pickup $(x_p, y_p)$ from `fromEdge`
- dropoff $(x_d, y_d)$ from `toEdge`

Branch A (`use_xy_pickup=true` and `use_edge_rt=false`):

- includes pickup deltas relative to robot:
  - $\Delta x_p = x_p - x_r$
  - $\Delta y_p = y_p - y_r$
- includes dropoff absolute coords
- `is_obsolete`, `is_assigned`

Branch B (otherwise):

- includes dropoff coords directly
- `is_obsolete`, `is_assigned`

All spatial terms are scaled by `pos_scale` if normalization is enabled.

---

## 5) Edge features (`edge_rt`) and route-slot augmentation

When `features.use_edge_rt=true`:

- edge feature dimension is `len(edge_features)`
- edge features are produced by `feature_fn(..., node_type="edge_rt")`

Supported base edge names in code:

- `dx`: pickup delta x (task pickup minus robot position)
- `dy`: pickup delta y
- `eta`: road-distance-based ETA proxy, normalized by `travel_scale` if enabled
- `is_ego_edge`: set to 1 for ego star edges, 0 for competitor edges

### Route-slot commitment features

If `features.robot_commitment == "route_slots"`:

`expand_edge_features(...)` appends for each slot index $s \in [0, route\_slots\_k-1]$:

- `slot{s}_pu_dx`, `slot{s}_pu_dy`
- `slot{s}_do_dx`, `slot{s}_do_dy`
- `slot{s}_valid`

These are computed from robot shadow route reservations:

- pickup slot deltas relative to candidate pickup
- drop slot deltas relative to candidate dropoff

Missing slots are zero-filled with `slot{s}_valid=0`.

---

## 6) How features are assembled into observation tensors

`build_padded_ego_batch(...)` outputs:

- `x`: `[R, N_max, F]`
- `node_mask`: `[R, N_max]`
- `edge_index`: `[R, 2, E_max]`
- `edge_mask`: `[R, E_max]`
- optional `edge_attr`: `[R, E_max, edge_feat_dim]`
- `cand_idx`: `[R, K_max]`
- `cand_mask`: `[R, K_max]`

Node/edge construction per robot:

1. node 0 = robot
2. candidate tasks = nodes 1..k
3. star edges robot$\leftrightarrow$task
4. optional competitor robot nodes/edges in 2-hop mode

Padding uses zeros + masks.

---

## 7) 1-hop vs 2-hop impact on feature usage

- Node features are always generated the same way by `feature_fn`.
- 2-hop changes graph topology (adds competitor nodes/edges), not node feature schema itself.
- Edge features (`edge_rt`) are available to both 1-hop and 2-hop edges when enabled.

---

## 8) Quick mapping from current `rp_gnn.yaml`

Current key settings imply:

- `use_edge_rt: true` → model uses edge attributes (`edge_attr`)
- `edge_features: [dx, dy, eta, is_ego_edge]`
- `robot_commitment: route_slots`, `route_slots_k: 2`
  - effective edge feature list becomes base + 10 slot fields
- `use_node_type: true`, `use_ego_robot: true`
  - node tail includes robot/task type indicators and ego flag
- `normalize_features: true`
  - position/time/capacity-related feature scaling is active
