# Evaluation Timing Protocol

This document defines the shared timing protocol used by baseline proposer-resolver methods and the current GNN actor-plus-critic evaluation path.

## Why this protocol exists

Fair cross-method timing compares the same outer decision boundaries:

1. Proposal generation
2. Conflict resolution / assignment
3. Simulation / state transition

Internal GNN forward timings are diagnostic and must not be directly compared with full baseline proposal time. The primary paper comparison uses shared outer proposal, resolution, simulation, and decision_total timings.

## Timer and units

- Monotonic timer: time.perf_counter_ns()
- Internal storage: integer nanoseconds
- Summary output: milliseconds (_ms columns)

## Outer phase boundaries

### Proposal generation

Starts when decision-step proposal work starts and ends when full proposals are ready for resolver input.

Includes, depending on method:

- candidate list preparation
- proposer scoring
- policy inference (GNN actor-plus-critic for current benchmark)
- action selection
- action-to-task mapping
- proposal data creation

### Conflict resolution

Starts after proposal set is ready.
Ends when conflict-free winners/final assignments are computed.

Includes:

- grouping proposals by task
- conflict detection
- resolver scoring
- winner selection
- assignment-map construction

### Simulation / state transition

Starts immediately before assignments are applied to controller/environment simulation work.
Ends after corresponding simulation/state transition work for the decision step is complete.

### Total decision

Decision total is measured independently and logged as decision_total_ns.

- proposal_ns
- resolution_ns
- simulation_ns
- decision_total_ns
- other_ns = max(0, decision_total_ns - proposal_ns - resolution_ns - simulation_ns)

## GNN internal timing fields (diagnostic)

Current benchmark path is actor-plus-critic inference.

- gnn_obs_build_ns
- gnn_tensor_prepare_ns
- gnn_graph_prepare_ns
- gnn_policy_total_ns
- gnn_actor_ns
- gnn_critic_ns
- gnn_action_ns
- gnn_action_mapping_ns

No additional model forward passes are executed only for timing.

## Actor workload observables

Per decision, for the actual actor graph batch:

- n_actor_graphs
- n_actor_nonempty_graphs
- n_actor_nodes
- n_actor_edges
- n_actor_candidates

Conventions:

- n_actor_graphs includes empty-candidate robots if actor still receives their graph (current behavior).
- n_actor_edges counts edges before any implicit self-loop insertion by the encoder.

## Per-robot amortized GNN timing fields

These are amortized costs under batched fleet execution, not standalone one-robot latency.

### Actor-network cost

gnn_actor_amortized_robot_ns = gnn_actor_ns / n_actor_graphs

Meaning:
Amortized actor-network computation per robot under batched execution.

### Accounted actor proposal path

gnn_actor_path_accounted_amortized_robot_ns =
(gnn_obs_build_ns + gnn_tensor_prepare_ns + gnn_graph_prepare_ns + gnn_actor_ns + gnn_action_ns + gnn_action_mapping_ns) / n_actor_graphs

Meaning:
Amortized time per robot for explicitly instrumented actor-side proposal path.

### Broad actor-only proposal estimate

gnn_actor_only_proposal_est_amortized_robot_ns = max(0, proposal_ns - gnn_critic_ns) / n_actor_graphs

Meaning:
Estimated complete amortized actor-only proposal time per robot under batched fleet execution.

This is not measured standalone single-robot latency.

## Warm-up handling

Config:

- timing.enabled (default false)
- timing.internal_gnn (default true)
- timing.warmup_episodes (default 0)

Warm-up rows are logged with warmup=1 in timing_steps.csv.
Primary summary aggregates exclude warm-up rows and report warmup_decisions separately.

## Output files

When timing is enabled:

- timing_steps.csv: per-decision rows
- timing_summary.csv: grouped aggregates + percentile stats

Outputs are written to each evaluation run directory.

## Enable timing

### Baseline evaluation

Use existing baseline command/config and set:

- timing.enabled: true

Example:

python eval_baselines.py --config configs/rp_gnn.yaml

### GNN checkpoint evaluation (actor-plus-critic)

Use existing command and set:

- timing.enabled: true

Example:

python eval_saved_models.py --config configs/rp_gnn.yaml --model-dir runs/rp_gnn_debug/saved_models

inference_mode is logged as actor_critic for current GNN timing benchmark.

Future actor-only benchmarking should reuse the same schema/protocol and compare against gnn_actor_only_proposal_est_amortized_robot_ms.
