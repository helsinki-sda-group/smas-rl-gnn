#!/bin/bash
#SBATCH --job-name=rp-eval-baselines
#SBATCH --account=project_2012159
#SBATCH --partition=small
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/%x-%j.out
#SBATCH --error=/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/%x-%j.err

set -euo pipefail

REPO=/projappl/project_2012159/kbocheni_temp/smas-rl-gnn
RUNROOT=/scratch/project_2012159/kbocheni/smas-rl-gnn/eval_jobs
CFG=${1:-configs/rp_gnn.yaml}

RUN_NAME=$(grep 'run_name:' "$REPO/$CFG" | awk '{print $2}' || echo "unknown")
JOBDIR=${RUNROOT}/job_eval_${RUN_NAME}_${SLURM_JOB_ID}

mkdir -p "$JOBDIR"

SNAP_DIR="$JOBDIR/config_snapshots"
mkdir -p "$SNAP_DIR"
CFG_BASENAME="$(basename "$CFG")"
TS="$(date +%Y%m%d_%H%M%S)"
SNAP_CFG="$SNAP_DIR/${TS}_${CFG_BASENAME}"
cp "$REPO/$CFG" "$SNAP_CFG"

cp "$REPO/$CFG" "$JOBDIR/rp_gnn_eval.yaml"

echo "[INFO] Config snapshot saved: $SNAP_CFG"
echo "[INFO] Job directory: $JOBDIR"

cd "$JOBDIR"

module purge
module load python-data
source "$REPO/.venv/bin/activate"

export SUMO_HOME=/projappl/project_2012159/kbocheni_temp/sumo-local
export XERCES_HOME=/projappl/project_2012159/kbocheni_temp/xerces-local
export SUMO_TOOLS=/projappl/project_2012159/kbocheni_temp/sumo-local/share/sumo/tools
export PYTHONPATH="$REPO:$SUMO_TOOLS:${PYTHONPATH:-}"
export PATH="$SUMO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$XERCES_HOME/lib64:$SUMO_HOME/lib64:$SUMO_HOME/lib:${LD_LIBRARY_PATH:-}"

PORT=$((8500 + SLURM_JOB_ID % 1000))
python "$REPO/eval_baselines.py" --config "$REPO/$CFG" --sumoport "$PORT"
