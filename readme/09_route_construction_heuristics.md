# Route Construction Heuristics

This document describes how robot pickup/dropoff route order is reconstructed in the controller.

## Where It Lives

Implementation: [sumo_rl_rs/environment/rl_controller_adapter.py](../sumo_rl_rs/environment/rl_controller_adapter.py)

Main route-rebuild path: `apply_and_step(...)`.

## Route Sequence Representation

The dispatch sequence is a list of reservation IDs, for example:

`[r1, r2, r1, r2]`

Interpretation is by occurrence count per reservation:

- first occurrence: pickup
- second occurrence: dropoff

For passengers already picked up, SUMO still expects the reservation to appear in the dispatch list. The first occurrence is a synthetic pickup placeholder and is ignored by dispatcher semantics; the second occurrence is the real dropoff.

## Existing Heuristic: nearest

`route_construction: nearest`

This is the legacy behavior.

- Build pickup/dropoff stop candidates with pickup-before-dropoff constraints.
- Repeatedly choose next stop with minimum road distance from current edge.
- Start from taxi current edge, then from each selected stop edge.

This heuristic optimizes local proximity and is fast, but can keep an onboard passenger in-vehicle longer than desired if near-term pickups keep getting preferred.

## New Heuristic: deadline_travel (Reward-Aligned)

`route_construction: deadline_travel`

Alias: `route_construction: reward_aligned`

This mode performs exhaustive search over valid remaining stop orders (small problem sizes only), predicts route outcomes, then selects using a reward-aligned objective:

1. maximize predicted route reward from configured environment reward terms;
2. among equal rewards, minimize maximum travel-time excess for already onboard passengers;
3. then minimize maximum travel-time excess overall;
4. then minimize total route duration;
5. finally apply deterministic sequence ordering.

Continuous values are rounded before tuple comparison to reduce instability from tiny floating-point differences.

### Feasibility Rules

A sequence is valid only if:

- pickup precedes dropoff for each unpicked reservation,
- capacity is never exceeded,
- initial occupancy includes already picked-up passengers,
- no stop appears more than once.

### Predicted Reward Terms

For `reward_type: wait_travel`, each feasible sequence computes:

- predicted completion reward from configured completion mode;
- predicted waiting penalty for pending pickups only;
- predicted travel-excess penalty for all unfinished reservations with predicted dropoff.

Waiting and travel penalties use the same caps and weights as runtime reward:

- wait cap: `wait_cap`, weight: `w_wait`
- travel cap: `travel_cap`, weight: `w_travel`
- completion weight: `w_comp`

For onboard passengers, travel duration is measured from `actual_pickup_time`, and no additional pickup waiting penalty is added.

For `reward_type: deadline`, exhaustive scoring reuses the existing weighted pickup/dropoff lateness normalization with `deadline_cap` and `w_deadline`.

Unsupported reward modes raise a clear configuration error in exhaustive route-construction mode.

### Travel-Time Excess

For each unfinished reservation with predicted dropoff:

- unpicked: `predicted_dropoff - predicted_pickup - estTravelTime`
- already picked: `predicted_dropoff - actual_pickup_time - estTravelTime`

All excess values are clamped at minimum zero and capped in the reward penalty term.

## Complexity Safeguard

`route_exhaustive_max_stops` limits exhaustive search.

- If the number of remaining stops exceeds this threshold, the controller deterministically falls back to `nearest`.
- Episode continues normally; no hard failure.

## Configuration

Recommended config fields under `env`:

```yaml
env:
  route_construction: nearest               # nearest | deadline_travel | reward_aligned
  route_exhaustive_max_stops: 8             # exhaustive safeguard
  route_construction_debug: false            # optional diagnostic logging
```

Defaults preserve backward compatibility:

- `route_construction: nearest`
- `route_exhaustive_max_stops: 8`
- `route_construction_debug: false`

## Diagnostics

Per robot route rebuild diagnostics track:

- whether exhaustive search was used,
- fallback usage and reason,
- number of valid sequences evaluated,
- selected score tuple when exhaustive mode succeeds.

With `route_construction_debug: true`, diagnostics are emitted via logger debug channel.
