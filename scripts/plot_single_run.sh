#!/bin/bash
set -euo pipefail

# Plot helper for one training run identified by run_name.
# Finds the latest matching job directory under scratch, then runs:
# - plot_training_results.py
# - plot_comp_norms.py
# - plot_action_candidates.py

usage() {
  cat <<'EOF'
Usage:
  scripts/plot_single_run.sh <run_name> [options]

Options:
  --job-id <id>           Use a specific job id instead of latest match.
  --repo <path>           Repo root. Default: /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
  --scratch-root <path>   Scratch run root. Default: /scratch/project_2012159/kbocheni/smas-rl-gnn
  --out-root <path>       Where to save plots. Default: /projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_single_run
  --config <path>         Config path passed to plot_training_results.py.
                          Default: <job_dir>/rp_gnn.yaml if present, otherwise <repo>/configs/rp_gnn.yaml
  --ma-window <n>         Moving-average window for per-seed reward plots (default: 20).
  --action-window <n>     Smoothing window for action candidate plots (default: 10).
  -h, --help              Show this help.

Examples:
  scripts/plot_single_run.sh rp_gnn_debug
  scripts/plot_single_run.sh rp_gnn_debug --job-id 6574582
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RUN_NAME="$1"
shift

REPO="/projappl/project_2012159/kbocheni_temp/smas-rl-gnn"
SCRATCH_ROOT="/scratch/project_2012159/kbocheni/smas-rl-gnn"
OUT_ROOT="/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_single_run"
JOB_ID=""
CFG=""
MA_WINDOW=20
ACTION_WINDOW=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      JOB_ID="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --scratch-root)
      SCRATCH_ROOT="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --config)
      CFG="$2"
      shift 2
      ;;
    --ma-window)
      MA_WINDOW="$2"
      shift 2
      ;;
    --action-window)
      ACTION_WINDOW="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

JOBS_ROOT="$SCRATCH_ROOT/jobs"

if [[ -n "$JOB_ID" ]]; then
  JOBDIR="$JOBS_ROOT/job_${RUN_NAME}_${JOB_ID}"
  if [[ ! -d "$JOBDIR" ]]; then
    echo "[ERROR] Job directory not found: $JOBDIR"
    exit 1
  fi
else
  latest_id=""
  latest_dir=""
  shopt -s nullglob
  for d in "$JOBS_ROOT"/job_"$RUN_NAME"_*; do
    [[ -d "$d" ]] || continue
    id="${d##*_}"
    [[ "$id" =~ ^[0-9]+$ ]] || continue
    if [[ -z "$latest_id" || "$id" -gt "$latest_id" ]]; then
      latest_id="$id"
      latest_dir="$d"
    fi
  done
  shopt -u nullglob

  if [[ -z "$latest_dir" ]]; then
    echo "[ERROR] No job directory found for run_name='$RUN_NAME' under $JOBS_ROOT"
    exit 1
  fi

  JOB_ID="$latest_id"
  JOBDIR="$latest_dir"
fi

if [[ -z "$CFG" ]]; then
  if [[ -f "$JOBDIR/rp_gnn.yaml" ]]; then
    CFG="$JOBDIR/rp_gnn.yaml"
  else
    CFG="$REPO/configs/rp_gnn.yaml"
  fi
fi

METRICS_LOG="$(ls -1t "$JOBDIR"/training_metrics_*.log 2>/dev/null | head -n 1 || true)"
TRAIN_OUTPUT="$JOBDIR/train_output.txt"
COMP_LOG="$JOBDIR/comp_norms.log"
CONFLICTS_LOG="$JOBDIR/conflicts.log"

if [[ -z "$METRICS_LOG" ]]; then
  echo "[ERROR] No training_metrics_*.log found in $JOBDIR"
  exit 1
fi
if [[ ! -f "$TRAIN_OUTPUT" ]]; then
  echo "[ERROR] Missing train_output.txt in $JOBDIR"
  exit 1
fi

OUTDIR="$OUT_ROOT/${RUN_NAME}_job_${JOB_ID}"
mkdir -p "$OUTDIR"

if [[ -x "$REPO/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

echo "[INFO] run_name: $RUN_NAME"
echo "[INFO] job_id:   $JOB_ID"
echo "[INFO] job_dir:  $JOBDIR"
echo "[INFO] config:   $CFG"
echo "[INFO] out_dir:  $OUTDIR"

pushd "$OUTDIR" > /dev/null

"$PYTHON_BIN" "$REPO/plot_training_results.py" \
  "$METRICS_LOG" \
  "$TRAIN_OUTPUT" \
  --config "$CFG" \
  --ma-window "$MA_WINDOW"

"$PYTHON_BIN" "$REPO/plot_comp_norms.py" \
  --log "$COMP_LOG" \
  --conflicts-log "$CONFLICTS_LOG" \
  --out "$OUTDIR/comp_norms_plots"

"$PYTHON_BIN" "$REPO/plot_action_candidates.py" \
  "$METRICS_LOG" \
  --window "$ACTION_WINDOW" \
  --out "$OUTDIR/action_candidate_plots"

popd > /dev/null

echo "[OK] Plots written to: $OUTDIR"
