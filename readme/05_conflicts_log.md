# `conflicts.log` Metrics Reference

This document describes every column in root-level `conflicts.log`.

Primary sources:
- `sumo_rl_rs/logging/ridepool_logger.py`
- `sumo_rl_rs/environment/rl_controller_adapter.py`
- `plot_comp_norms.py` (consumer/plotter)

## What this log is

`conflicts.log` is an **episode-level summary** of assignment conflicts (one row per episode).

A row is written at episode end by `RidepoolLogger._append_conflicts_summary(...)`.

## Header

```text
episode,conflicts_total,tasks_total,conflict_ratio,winner_pickup,winner_margin,resolver_override,resolver_override_rate,avg_margin_win,avg_margin_lose,avg_margin_gap
```

## How counters are accumulated

During assignment conflict resolution (`RLControllerAdapter._resolve_assignment_conflicts(...)`):

- `log_conflict_task_count(len(buckets))` is called once per resolution step.
  - `buckets` = reservation IDs that were chosen by at least one robot in that step.
- For each reservation with multiple candidate robots (`len(rids) > 1`), exactly one conflict event is recorded via:
  - `log_conflict_metrics_event(...)`

So `tasks_total` and `conflicts_total` are episode accumulators over all decision steps.

## Column definitions

Let:
- $T = \text{tasks\_total}$
- $C = \text{conflicts\_total}$
- $C_{wp} = \text{winner\_pickup}$
- $C_{wm} = \text{winner\_margin}$
- $C_{ovr} = \text{resolver\_override}$

| Column | Meaning | Formula / rule |
|---|---|---|
| `episode` | Episode index. | Integer episode ID at write time. |
| `tasks_total` | Number of conflict-buckets seen across the episode. | Sum of `len(buckets)` each time `_resolve_assignment_conflicts(...)` runs. |
| `conflicts_total` | Number of **actual multi-candidate conflicts**. | Incremented by 1 for each reservation where at least 2 robots competed. |
| `conflict_ratio` | Share of tasks that were conflicting. | $C/T$ if $T>0$, else $0$. |
| `winner_pickup` | Count of conflicts where final winner is in pickup-distance winner set. | Increment if selected winner belongs to `pickup_winners`. |
| `winner_margin` | Count of conflicts where final winner is in margin winner set. | Increment if selected winner belongs to `margin_winners`. |
| `resolver_override` | Count of conflicts where pickup and margin recommendations disagree. | If both winner sets are non-empty and first sorted element differs, increment by 1. |
| `resolver_override_rate` | Override rate among conflicts. | $C_{ovr}/C$ if $C>0$, else $0$. |
| `avg_margin_win` | Mean winner margin over events with valid margin. | $\sum m_{win}/N_{win}$. |
| `avg_margin_lose` | Mean loser margin over valid loser margins. | $\sum m_{lose}/N_{lose}$. |
| `avg_margin_gap` | Mean winner-loser margin difference. | $\sum (m_{win}-m_{lose})/N_{gap}$. |

## Important nuances

- `winner_pickup` and `winner_margin` are **counts**, not rates.
- `winner_margin` can equal `conflicts_total` when margin-based winner set always includes final winner.
- `resolver_override` is computed from representative set members, not from all tied winners.
- CSV values are written with two decimals (`:.2f`) in the logger.

## Relationship to per-event conflict details

This file is a summary only. Per-conflict details are in episode CSV:
- `runs/<run>/episode_xxxx/conflicts.csv`

That per-episode file has row-level fields:
- `time,res_id,taxi_candidates,remaining_caps,distances,winner`

## Plot usage

`plot_comp_norms.py --conflicts-log conflicts.log` expects exactly:

- `episode`
- `conflicts_total`
- `tasks_total`
- `conflict_ratio`
- `winner_pickup`
- `winner_margin`
- `resolver_override`
- `resolver_override_rate`
- `avg_margin_win`
- `avg_margin_lose`
- `avg_margin_gap`

and generates:
- `conflicts_total_vs_tasks_total.png`
- `conflict_ratio.png`
- `winner_pickup_margin_override.png`
- `resolver_override_rate.png`
- `avg_margins.png`
