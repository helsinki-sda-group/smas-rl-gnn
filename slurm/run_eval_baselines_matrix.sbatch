#!/bin/bash
#SBATCH --job-name=rp-eval-baseline-matrix
#SBATCH --account=project_2012159
#SBATCH --partition=small
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/%x-%j.out
#SBATCH --error=/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/%x-%j.err

set -euo pipefail

REPO=/projappl/project_2012159/kbocheni_temp/smas-rl-gnn
RUNROOT=/scratch/project_2012159/kbocheni/smas-rl-gnn/eval_baseline_matrix
JOBDIR=${RUNROOT}/job_eval_baseline_matrix_${SLURM_JOB_ID}

mkdir -p "$JOBDIR"
mkdir -p "$JOBDIR/configs"

SNAP_DIR="$JOBDIR/config_snapshots"
mkdir -p "$SNAP_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
cp "$REPO/configs/rp_gnn.yaml" "$SNAP_DIR/${TS}_rp_gnn.yaml"

echo "[INFO] Job directory: $JOBDIR"
echo "[INFO] Matrix root: $RUNROOT"

action_usage() {
  cat <<'EOF'
Optional args are forwarded to scripts/run_eval_baselines_matrix.sh, for example:
  sbatch slurm/run_eval_baselines_matrix.sbatch --scenarios "randdest,wave" --resolvers "capacity,closest"
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  action_usage
  exit 0
fi

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

bash "$REPO/scripts/run_eval_baselines_matrix.sh" \
  --repo-root "$REPO" \
  --base-yaml "$REPO/configs/rp_gnn.yaml" \
  --base-sumocfg "$REPO/configs/small_net.sumocfg" \
  --output-dir "configs" \
  --sumoport "$PORT" \
  --python python \
  "$@"
