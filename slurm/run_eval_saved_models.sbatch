#!/bin/bash
#SBATCH --job-name=rp-eval-saved
#SBATCH --account=project_2012159
#SBATCH --partition=small
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/%x-%j.out
#SBATCH --error=/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/%x-%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RUN_NAME CONFIG_PATH [EVAL_ARGS...]"
  echo "Example: $0 2hop-maxpool-3_ctc_cap2 configs/rp_gnn_2hop-maxpool-3_ctc_cap2.yaml --seeds eval --eval-runs 1 --model-sample 0.5 --deterministic"
  exit 1
fi

REPO=/projappl/project_2012159/kbocheni_temp/smas-rl-gnn
RUNROOT=/scratch/project_2012159/kbocheni/smas-rl-gnn
EVAL_ROOT=${RUNROOT}/eval_saved_models_jobs

RUN_NAME="$1"
CFG="$2"
shift 2

MODEL_DIR=${RUNROOT}/runs/${RUN_NAME}/!saved_models
JOBDIR=${EVAL_ROOT}/${RUN_NAME}

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[ERROR] Model directory not found: $MODEL_DIR"
  exit 1
fi

if [[ ! -f "$REPO/$CFG" ]]; then
  echo "[ERROR] Config file not found: $REPO/$CFG"
  exit 1
fi

mkdir -p "$JOBDIR"

SNAP_DIR="$JOBDIR/config_snapshots"
mkdir -p "$SNAP_DIR"
CFG_BASENAME="$(basename "$CFG")"
TS="$(date +%Y%m%d_%H%M%S)"
SNAP_CFG="$SNAP_DIR/${TS}_${CFG_BASENAME}"
cp "$REPO/$CFG" "$SNAP_CFG"

echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Config: $REPO/$CFG"
echo "[INFO] Model dir: $MODEL_DIR"
echo "[INFO] Eval output root: $JOBDIR"
echo "[INFO] Config snapshot: $SNAP_CFG"

module purge
module load python-data
source "$REPO/.venv/bin/activate"

export SUMO_HOME=/projappl/project_2012159/kbocheni_temp/sumo-local
export XERCES_HOME=/projappl/project_2012159/kbocheni_temp/xerces-local
export SUMO_TOOLS=/projappl/project_2012159/kbocheni_temp/sumo-local/share/sumo/tools
export PYTHONPATH="$REPO:$SUMO_TOOLS:${PYTHONPATH:-}"
export PATH="$SUMO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$XERCES_HOME/lib64:$SUMO_HOME/lib64:$SUMO_HOME/lib:${LD_LIBRARY_PATH:-}"

PORT=$((8900 + SLURM_JOB_ID % 1000))

cmd=(
  python "$REPO/eval_saved_models.py"
  --config "$REPO/$CFG"
  --model-dir "$MODEL_DIR"
  --output-dir "$JOBDIR"
  --sumoport "$PORT"
)

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

"${cmd[@]}"
