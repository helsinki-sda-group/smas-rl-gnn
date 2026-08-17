#!/bin/bash
# scripts/plot_time_cmp_mahti.sh
#
# Wraps plot_metrics_wide.py --time to compare measured proposer/resolver latency
# (and reward) between:
#   - a baseline matrix job (bash scripts/run_eval_baselines_matrix.sh output), and
#   - one RL saved-model evaluation run (eval_saved_models.py output; a single
#     trained checkpoint, possibly evaluated multiple times -> K evaluation_<ts> dirs)
#
# RL evaluation runs: /scratch/project_2012159/kbocheni/smas-rl-gnn/eval_saved_models_jobs/<RL_RUN_NAME>/evaluation_<ts>/
# Baseline matrix job: /scratch/project_2012159/kbocheni/smas-rl-gnn/eval_baseline_matrix/<BASELINE_JOB_NAME>/
# Output:              /projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plot_time_comparison/<RL_RUN_NAME>__vs__<BASELINE_JOB_NAME>/
#
# Usage:
#   bash scripts/plot_time_cmp_mahti.sh <RL_RUN_NAME> <BASELINE_JOB_NAME> [OPTIONS]
#
# Example:
#   bash scripts/plot_time_cmp_mahti.sh 1hop_noop-w3-1_ctc_cap2 job_eval_baseline_matrix_7162319 \
#     --rl-mode best --eval-resolver closest_then_capacity --pareto --time-diagnose-proposer-resolver
#
# Options (all optional, passed through to plot_metrics_wide.py unless noted):
#   --rl-mode {mean,best,distinct}      How to combine multiple RL evaluation_<ts> dirs (default: best)
#   --rl-quality-window N               Sliding reward-window size in episodes (default: 5)
#   --rl-time-scope {all,window}        Episodes contributing to RL timing (default: all)
#   --eval-proposer NAME                RL proposer label (default: 1hop)
#   --eval-resolver NAME                RL trained resolver (default: closest_then_capacity)
#   --eval-scenario NAME                Scenario RL was evaluated on (default: randdest)
#   --eval-route-construction NAME      Route construction used during RL training/eval (default: nearest)
#   --eval-protocol {aa,admission,forced}  Protocol RL was trained/evaluated under (default: aa)
#   --metrics M [M ...]                 Quality metrics to plot besides reward (default: rew)
#   --scenario NAME                     Scenario subset to plot (default: all)
#   --pareto / --no-pareto              Draw Pareto frontier on rew_time_cmp (default: on)
#   --time-diagnose-proposer-resolver   Add proposer/resolver latency diagnostics
#   --skip-aggregate                    Reuse an existing metrics_wide.csv under the baseline job dir
#   --output-dir DIR                    Override output directory
#   --dry-run                           Print the commands without executing them
#   -h, --help                          Show this help
#
# Requires plot_metrics_wide.py's --rl-eval-dirs/--rl-mode/--rl-quality-window/--rl-time-scope support.
# See readme/11_time_comparison_plots.md for how RL and baseline data are combined.

set -euo pipefail

REPO=/projappl/project_2012159/kbocheni_temp/smas-rl-gnn
RL_EVAL_ROOT=/scratch/project_2012159/kbocheni/smas-rl-gnn/eval_saved_models_jobs
BASELINE_ROOT=/scratch/project_2012159/kbocheni/smas-rl-gnn/eval_baseline_matrix
PLOTS_ROOT=/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plot_time_comparison

usage() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

RL_RUN_NAME="$1"; shift
BASELINE_JOB_NAME="$1"; shift

RL_MODE=best
RL_QUALITY_WINDOW=5
RL_TIME_SCOPE=all
EVAL_PROPOSER=1hop
EVAL_RESOLVER=closest_then_capacity
EVAL_SCENARIO=randdest
EVAL_ROUTE_CONSTRUCTION=nearest
EVAL_PROTOCOL=aa
METRICS=(rew)
SCENARIO=all
PARETO=1
TIME_DIAG=0
SKIP_AGGREGATE=0
OUTPUT_DIR_OVERRIDE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rl-mode) RL_MODE="$2"; shift 2 ;;
    --rl-quality-window) RL_QUALITY_WINDOW="$2"; shift 2 ;;
    --rl-time-scope) RL_TIME_SCOPE="$2"; shift 2 ;;
    --eval-proposer) EVAL_PROPOSER="$2"; shift 2 ;;
    --eval-resolver) EVAL_RESOLVER="$2"; shift 2 ;;
    --eval-scenario) EVAL_SCENARIO="$2"; shift 2 ;;
    --eval-route-construction) EVAL_ROUTE_CONSTRUCTION="$2"; shift 2 ;;
    --eval-protocol) EVAL_PROTOCOL="$2"; shift 2 ;;
    --metrics)
      shift
      METRICS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do METRICS+=("$1"); shift; done
      ;;
    --scenario) SCENARIO="$2"; shift 2 ;;
    --pareto) PARETO=1; shift ;;
    --no-pareto) PARETO=0; shift ;;
    --time-diagnose-proposer-resolver) TIME_DIAG=1; shift ;;
    --skip-aggregate) SKIP_AGGREGATE=1; shift ;;
    --output-dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
  esac
done

RL_RUN_DIR="${RL_EVAL_ROOT}/${RL_RUN_NAME}"
BASELINE_JOB_DIR="${BASELINE_ROOT}/${BASELINE_JOB_NAME}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${PLOTS_ROOT}/${RL_RUN_NAME}__vs__${BASELINE_JOB_NAME}}"

if [[ ! -d "$RL_RUN_DIR" ]]; then
  echo "[ERROR] RL run directory not found: $RL_RUN_DIR"
  exit 1
fi
if [[ ! -d "$BASELINE_JOB_DIR" ]]; then
  echo "[ERROR] Baseline job directory not found: $BASELINE_JOB_DIR"
  exit 1
fi

# Match evaluation_<date>_<time> output dirs only (e.g. evaluation_20260806_133415);
# excludes sibling dirs like evaluation_runs/ (raw per-episode scratch space, not a run output).
mapfile -t RL_EVAL_DIRS < <(find "$RL_RUN_DIR" -maxdepth 1 -type d -regextype posix-extended -regex '.*/evaluation_[0-9]{8}_[0-9]{6}' | sort)
if [[ ${#RL_EVAL_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No evaluation_<date>_<time> directories found under: $RL_RUN_DIR"
  exit 1
fi
echo "[INFO] Found ${#RL_EVAL_DIRS[@]} RL evaluation run(s) for ${RL_RUN_NAME}:"
printf '  - %s\n' "${RL_EVAL_DIRS[@]}"

METRICS_WIDE_CSV="${BASELINE_JOB_DIR}/metrics_wide.csv"
AGGREGATE_CMD=(python "${REPO}/aggregate_metrics_logs.py" "$BASELINE_JOB_DIR" -o "$METRICS_WIDE_CSV")
if [[ "$SKIP_AGGREGATE" -eq 1 && -f "$METRICS_WIDE_CSV" ]]; then
  echo "[INFO] --skip-aggregate: reusing existing $METRICS_WIDE_CSV"
else
  echo "[INFO] Aggregating baseline metrics logs -> $METRICS_WIDE_CSV"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  %q' "${AGGREGATE_CMD[@]}"; echo
  else
    "${AGGREGATE_CMD[@]}"
  fi
fi

mapfile -t BASELINE_TIMING_FILES < <(find "$BASELINE_JOB_DIR" -maxdepth 1 -type f -name 'timing_summary*.csv' | sort)
if [[ ${#BASELINE_TIMING_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No top-level timing_summary*.csv files found under: $BASELINE_JOB_DIR"
  exit 1
fi

PLOT_CMD=(
  python "${REPO}/plot_metrics_wide.py" "$METRICS_WIDE_CSV"
  --output-dir "$OUTPUT_DIR"
  --time
  --timing-files "${BASELINE_TIMING_FILES[@]}"
  --scenario "$SCENARIO"
  --metrics "${METRICS[@]}"
  --rl-eval-dirs "${RL_EVAL_DIRS[@]}"
  --rl-mode "$RL_MODE"
  --rl-quality-window "$RL_QUALITY_WINDOW"
  --rl-time-scope "$RL_TIME_SCOPE"
  --eval-proposer "$EVAL_PROPOSER"
  --eval-resolver "$EVAL_RESOLVER"
  --eval-scenario "$EVAL_SCENARIO"
  --eval-route-construction "$EVAL_ROUTE_CONSTRUCTION"
  --eval-protocol "$EVAL_PROTOCOL"
)
if [[ "$PARETO" -eq 1 ]]; then
  PLOT_CMD+=(--pareto)
fi
if [[ "$TIME_DIAG" -eq 1 ]]; then
  PLOT_CMD+=(--time-diagnose-proposer-resolver)
fi

echo "[INFO] Plotting time comparison -> $OUTPUT_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '  %q' "${PLOT_CMD[@]}"; echo
else
  "${PLOT_CMD[@]}"
fi
