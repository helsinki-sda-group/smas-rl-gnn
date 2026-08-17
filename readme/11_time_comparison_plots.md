# Time-comparison plots: baselines vs. RL saved-model evaluation

This document explains how to produce the `--time` plots from `plot_metrics_wide.py`
(measured proposer/resolver latency vs. reward, decision-phase breakdowns, and
proposer/resolver latency diagnostics), how the RL saved-model evaluation overlay
is aggregated, and how to run everything on Mahti with `scripts/plot_time_cmp_mahti.sh`.

It supplements [09_route_construction_heuristics.md](09_route_construction_heuristics.md),
[10_estimated_computational_work.md](10_estimated_computational_work.md), and
[timing_evaluation.md](timing_evaluation.md); see those for the phase-boundary
definitions and the analytical work-estimate model (`--work-cmp`), which is a
different, non-measured comparison.

## Why this exists

`--time` compares **measured** wall-clock latency (from `timing_summary*.csv`,
produced by the shared timing protocol in [timing_evaluation.md](timing_evaluation.md))
against quality metrics (from `metrics_wide.csv`) for every proposer/resolver
combination found in the data. The RL policy is a single trained (proposer, resolver)
combination -- the resolver is fixed at training time and the "proposer" is effectively
the GNN policy itself -- so it can be overlaid onto the same plots as just another
proposer under its trained resolver, using its own **real measured** timing data
(no analytical estimate is needed, unlike `--work-cmp-with-rl`).

## Inputs

- `metrics_wide.csv`: produced by `aggregate_metrics_logs.py` from baseline
  `metrics*.log` files (one row per proposer/resolver/scenario/protocol/route
  combination, with `_std` companion columns).
- `timing_summary*.csv`: produced by `eval_baselines.py` (baselines) and by
  `eval_saved_models.py` (RL), one row per `(seed, episode)` using the shared
  `TimingRunCollector` (`utils/eval_timing.py`), so both sources share the exact
  same column schema.
- RL saved-model evaluation run directory (`evaluation_<date>_<time>/`, from
  `eval_saved_models.py`): contains `evaluation.log`, `evaluation_metrics.log`,
  `logit_metrics.log`, `evaluation_results.csv`, `timing_steps.csv`,
  `timing_summary.csv`.

## Basic usage: `plot_metrics_wide.py --time` (baselines only)

```bash
python plot_metrics_wide.py metrics_wide.csv \
  --output-dir metrics_wide_plots \
  --time --pareto --time-diagnose-proposer-resolver \
  --timing-dir /path/to/baseline_job_dir \
  --scenario randdest
```

This produces, per `<scenario>/time_cmp/<route>_<protocol>/`:

- `rew_time_cmp.png` -- reward vs. measured proposal+resolution latency (the
  Pareto plot; central comparison plot).
- `proposal_resolution_time_cmp.png`, `proposal_time_by_resolver.png`,
  `resolution_time_by_proposer.png`
- `decision_phase_breakdown_by_combination.png`,
  `decision_phase_breakdown_overall.png`
- `decision_phase_totals_overall.png`
- diagnostics (with `--time-diagnose-proposer-resolver`):
  `proposal_latency_vs_candidate_workload.png`,
  `proposal_latency_by_proposer_and_resolver.png`, matched-replicate/spread plots.
- `time_cmp_data.csv`, `time_phase_data.csv` (audit CSVs)

## Adding the RL overlay: `--rl-eval-dirs`

```bash
python plot_metrics_wide.py metrics_wide.csv \
  --output-dir metrics_wide_plots \
  --time --pareto --time-diagnose-proposer-resolver \
  --timing-dir /path/to/baseline_job_dir \
  --scenario randdest \
  --rl-eval-dirs /path/to/eval_saved_models_jobs/1hop_noop-w3-1_ctc_cap2/evaluation_20260806_133415 \
                 /path/to/eval_saved_models_jobs/1hop_noop-w3-1_ctc_cap2/evaluation_20260807_091200 \
  --rl-mode best --rl-quality-window 5 --rl-time-scope all \
  --eval-proposer 1hop --eval-resolver closest_then_capacity \
  --eval-scenario randdest --eval-route-construction nearest --eval-protocol aa
```

`--rl-eval-dirs` takes one or more `evaluation_<date>_<time>` directories for the
**same trained checkpoint** (repeated evaluation invocations). Internally,
`plot_metrics_wide.py`:

1. Parses each run's `evaluation_metrics.log` (per-episode `rew`, `mcand`, `msd`,
   `dstep`, `noop`, `ovrlap`, `shared`, ..., and the training-checkpoint `ts` each
   episode was evaluated at), dropping `MEAN`/`STD` summary rows.
2. Selects a **quality window** per run over the **training-checkpoint (`ts`) axis**:
   groups episodes by `ts`, then slides a window of `--rl-quality-window` (default 5)
   consecutive checkpoints; the window whose pooled episodes have the highest mean
   reward is kept, and those episodes are averaged (mean/std) to produce that run's
   quality point. If a run only ever evaluates a single fixed checkpoint (all rows
   share one `ts`), the window trivially covers that checkpoint's episodes.
3. Combines runs per `--rl-mode`:
   - `best` (default): use the run whose quality-window mean reward is highest;
     its timing data alone is used.
   - `mean`: average the per-run quality-window points across all runs; pool
     timing rows from *all* runs.
   - `distinct`: keep every run as its own labeled series (`rl_<proposer>_r1`,
     `rl_<proposer>_r2`, ...), each with its own quality point and timing rows.
4. Builds a synthetic "metrics_wide row" (same columns as `metrics_wide.csv`) for
   the resulting quality point(s), labeled `pol=rl_<--eval-proposer>[_r<N>]`, with
   `resolver` taken from the RL run's own `timing_summary.csv` (falls back to
   `--eval-resolver`), and `scenario`/`route_construction`/`protocol` taken from
   `--eval-scenario`/`--eval-route-construction`/`--eval-protocol` (the RL run's
   raw timing CSV does not carry scenario/route metadata, so these must be told
   to the script explicitly -- they should match the scenario the model was
   evaluated on).
5. Rewrites the matching RL `timing_summary.csv` rows' `proposer` field to the
   same label and concatenates them onto the baseline timing rows.

Because these synthetic rows use exactly the same columns/keys as ordinary
baseline rows, **every per-combination plot automatically includes the RL point**
without special-casing: `rew_time_cmp.png`, `decision_phase_breakdown_by_combination.png`,
`proposal_latency_vs_candidate_workload.png`, resolver/policy comparison plots, etc.
`closest` and `closest_than_capacity` (and `ctc`/`closest_then_capacity`) are
already treated as aliases of one canonical resolver name throughout the script
(`_canonical_resolver_name`), so RL and baseline points for that resolver are
grouped and colored together.

### Visual distinction

- On `rew_time_cmp.png` and other per-point scatter plots, RL points always use a
  black-edged star marker (larger than baseline markers) and are annotated with
  their label, regardless of `--annotate-time`.
- On the two **aggregated** plots that average or sum across all proposer/resolver
  combinations -- `decision_phase_breakdown_overall.png` ("average decision-phase
  latency across combinations") and `decision_phase_totals_overall.png` ("total
  measured time by decision phase") -- RL is **never averaged/summed together
  with baselines**. Instead it is computed separately and drawn as a second,
  hatched bar group next to the baseline bars.

### Timing scope: `--rl-time-scope`

- `all` (default, recommended): use every non-warmup measured episode across every
  evaluated checkpoint for timing. Latency is mostly a property of the (fixed-size)
  model architecture and hardware rather than of which checkpoint/episodes happened
  to score well, so pooling more episodes reduces noise without biasing the estimate.
- `window`: restrict RL timing rows to the same training checkpoints (`ts`) selected
  for the quality window. `timing_summary.csv` rows don't carry a `ts` column directly
  -- it's recovered from the `policy` column (the checkpoint filename, e.g.
  `model_episode12_ts5000000.zip`); rows whose filename doesn't match that pattern are
  excluded from `window` scope with a warning.

Note: because multiple checkpoints reuse the same `(seed, episode)` attempt numbers,
RL timing rows are internally offset per checkpoint before being merged with baseline
timing rows, so pooling a checkpoint sweep never trips the duplicate-timing-row check.

## Mahti wrapper: `scripts/plot_time_cmp_mahti.sh`

```bash
bash scripts/plot_time_cmp_mahti.sh 1hop_noop-w3-1_ctc_cap2 job_eval_baseline_matrix_7162319 \
  --rl-mode best --eval-resolver closest_then_capacity --pareto --time-diagnose-proposer-resolver
```

- `1hop_noop-w3-1_ctc_cap2` is a run folder under
  `/scratch/project_2012159/kbocheni/smas-rl-gnn/eval_saved_models_jobs/`; the
  script auto-discovers every `evaluation_<date>_<time>` subdirectory inside it
  and passes all of them as `--rl-eval-dirs`.
- `job_eval_baseline_matrix_7162319` is a job folder under
  `/scratch/project_2012159/kbocheni/smas-rl-gnn/eval_baseline_matrix/`; the
  script runs `aggregate_metrics_logs.py` on it to build `metrics_wide.csv`
  (skippable with `--skip-aggregate` if already built), then passes only the
  **top-level** `timing_summary_*.csv` files via `--timing-files` (not a recursive
  `--timing-dir` search) -- baseline job dirs can also contain a `runs/rp_eval_seed*/`
  subfolder with raw per-seed copies of the same timing data, and including both
  copies would trip the duplicate-timing-row check.
- Output goes to
  `/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plot_time_comparison/<RL_RUN_NAME>__vs__<BASELINE_JOB_NAME>/`.
- Run `bash scripts/plot_time_cmp_mahti.sh --help` (or with no arguments) for the
  full option list; `--dry-run` prints the underlying `aggregate_metrics_logs.py`
  and `plot_metrics_wide.py` commands without executing them.

To compare a *different* trained checkpoint (e.g. `1hop_noop-w3-2_ctc_cap2`)
against the same baseline job, re-run the wrapper with that run name; each
trained model is its own comparison and is not mixed with other trained models.

## Column/argument reference

| Argument | Meaning |
| --- | --- |
| `--rl-eval-dirs` | One or more `evaluation_<date>_<time>` directories for one trained checkpoint |
| `--rl-mode` | `best` (default) / `mean` / `distinct` -- see above |
| `--rl-quality-window` | Sliding window size (episodes) for the best-reward quality window (default 5) |
| `--rl-time-scope` | `all` (default) or `window` -- which episodes' timing rows are used |
| `--rl-quality-file` | Filename of the per-episode quality log inside each RL dir (default `evaluation_metrics.log`) |
| `--rl-timing-file` | Filename of the per-episode timing summary inside each RL dir (default `timing_summary.csv`) |
| `--eval-proposer` | Label used for the RL series, e.g. `1hop` -> `rl_1hop` |
| `--eval-resolver` | Fallback trained resolver name if the RL timing CSV lacks a `resolver` column |
| `--eval-scenario`, `--eval-route-construction`, `--eval-protocol` | Metadata to attach to the synthetic RL row/timing rows (must match the scenario/route/protocol the model was evaluated under) |

See also [plot_metrics_wide.py](../plot_metrics_wide.py) (`load_rl_time_overlay`,
`_load_rl_run`, `_rolling_best_window_rows`, `plot_time_cmp`,
`plot_time_phase_breakdown_mean`, `plot_time_phase_totals`) and
[scripts/plot_time_cmp_mahti.sh](../scripts/plot_time_cmp_mahti.sh).
