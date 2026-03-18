# 1-Hop / 2-Hop Architecture Variants (Code + Config)

This document explains architecture variants and how `configs/rp_gnn.yaml` flags map into the actual model behavior.

## Source-of-truth in code

- Graph construction (1-hop + 2-hop neighbors):
  - `rt_gnn_rl/graphs/builders.py` → `build_padded_ego_batch(...)`
- Policy wrapper + flag wiring:
  - `rt_gnn_rl/policy/sb3_gnn_policy.py` → `RTGNNPolicy`
- Actor/critic internals:
  - `rt_gnn_rl/policy/actor_critic.py` → `EgoActorCritic`
- GNN encoder:
  - `rt_gnn_rl/policy/gnn_backbone.py` → `EgoGraphEncoder`, `EdgeSAGEConv`
- Config-to-policy wiring:
  - `train.py` (same logic mirrored in eval scripts)

---

## 1) Relevant YAML flags

In `configs/rp_gnn.yaml`:

### Environment graph flags

- `env.two_hop` (bool)
- `env.two_hop_directed` (bool)
- `env.two_hop_critic` (bool)
- `env.two_hop_arch` in `{plain, comp_corr}` (also accepts alias `arch-3.2` → `comp_corr`)

### Feature/edge flags

- `features.use_edge_rt`
- `features.edge_features`

### Policy/GNN flags

- `ppo.policy_kwargs.gnn_layers`
- optional `ppo.policy_kwargs.gnn_layers_two_hop`
- `ppo.policy_kwargs.lambda_init`
- `ppo.policy_kwargs.k_max`

---

## 2) Graph semantics

For each ego-graph (per robot):

- node 0: ego robot
- nodes 1..k: candidate tasks
- optional extra nodes: competitor robots (in 2-hop mode)

Base star edges always present (if candidates exist):

- robot $\leftrightarrow$ task (two directed edges)

When `two_hop=true`, for each candidate task $t$:

- add competitor robot nodes whose distance to task pickup is within `vicinity_m`
- add edges task $\to$ competitor, and optionally competitor $\to$ task if `two_hop_directed=false`

So 2-hop introduces robot-competition context into each ego graph.

---

## 3) Variant mapping from YAML to model

In `train.py`:

- `use_competitor_fusion = (two_hop && two_hop_arch == "comp_corr")`
- `use_two_hop_actor = (two_hop && two_hop_arch == "plain")`
- `use_two_hop_critic = (two_hop && two_hop_critic)`

These flags are passed into `RTGNNPolicy` / `EgoActorCritic`.

### 3.1 Variant A: 1-hop baseline

Condition:

- `two_hop=false`

Behavior:

- actor uses 1-hop pruned graph (ego + candidate tasks)
- critic also 1-hop unless separately configured otherwise (but with `two_hop=false`, no competitor nodes exist)

### 3.2 Variant B: 2-hop plain

Condition:

- `two_hop=true`, `two_hop_arch=plain`

Behavior:

- actor encoder runs on full 2-hop graph (competitor nodes included)
- no explicit competitor-correction head
- logits come from task embedding head directly:

$$
\ell_t = W_{act} h_t + b_{act}
$$

### 3.3 Variant C: 2-hop competitor-corrected (`comp_corr` / `arch-3.2`)

Condition:

- `two_hop=true`, `two_hop_arch=comp_corr`

Behavior:

- actor task embeddings are from 1-hop path
- competitor context is computed from full 2-hop graph through attention over competitors
- final logit per candidate task:

$$
\ell_t = \ell_t^{base} + \ell_t^{comp} + \ell_t^{ind}
$$

with

$$
\ell_t^{base} = W_{act} h_t + b_{act}
$$

Competitor context pipeline:

1. collect competitor nodes $\mathcal{C}(t)$ for task $t$
2. build pair features via

$$
u_{tc} = \phi_{comp}([h_t, x_c, a_{tc}])
$$

3. attention scores

$$
s_{tc} = w_s^\top \nu_{tc},\quad
\alpha_{tc} = \text{softmax}_{c\in\mathcal{C}(t)}(s_{tc})
$$

4. aggregate context

$$
z_t = \sum_{c\in\mathcal{C}(t)} \alpha_{tc} \nu_{tc}
$$

5. correction term

$$
\ell_t^{comp} = w_c^\top z_t
$$

6. competitor-exists indicator term

$$
\ell_t^{ind} = b_{comp}\cdot \mathbb{1}[|\mathcal{C}(t)|>0]
$$

---

## 4) Critic variants

Configured by `critic_aggregation` inside `EgoActorCritic` (currently set by policy defaults).

Given robot embeddings $E_i$:

- `per_robot`: $V_i = f(E_i)$
- `joint_mean`: $V = f\left(\frac{1}{R}\sum_i E_i\right)$
- `joint_attn`: 
  $$
a_i = w_a^\top E_i,\; \beta_i = \text{softmax}(a_i),\; V = f\left(\sum_i \beta_i E_i\right)
  $$

And graph scope for critic is controlled by:

- `use_two_hop_critic` (wired from `env.two_hop_critic`)

---

## 5) Encoder and edge attributes

When `edge_dim>0` (`features.use_edge_rt=true`):

- `EgoGraphEncoder` uses `EdgeSAGEConv`
- message function:

$$
m_{i\leftarrow j} = \text{MLP}([x_i, x_j, e_{ij}])
$$

- aggregated with mean (`MessagePassing(aggr="mean")`)

Otherwise plain `SAGEConv` is used.

---

## 6) NO-OP and masked action head (SB3 wrapper)

`RTGNNPolicy` appends one learnable NO-OP logit to each robot candidate logit vector:

- base logits from actor: shape `[R, K_max]`
- append scalar `noop_logit` → `[R, K_max+1]`
- apply action mask (`cand_mask` + NO-OP always valid)

This is independent of 1-hop vs 2-hop selection.

---

## 7) Practical recipe from YAML

- **Pure 1-hop**:
  - `two_hop: false`
- **2-hop plain**:
  - `two_hop: true`
  - `two_hop_arch: plain`
- **2-hop competitor-corrected**:
  - `two_hop: true`
  - `two_hop_arch: comp_corr`
- **Critic on 2-hop graph**:
  - `two_hop_critic: true`
