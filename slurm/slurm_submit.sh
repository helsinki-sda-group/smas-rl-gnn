#!/bin/bash
# slurm_submit.sh - Submit training jobs to Slurm for specified methods
# Usage: ./slurm_submit.sh METHOD1 [METHOD2 ...] [--mode continue] [--dry-run]
#
# Examples:
#   ./slurm_submit.sh 1hop_rnd 2hop 1hop_ctc
#   ./slurm_submit.sh 1hop --mode continue
#   ./slurm_submit.sh 2hop --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLURM_DIR="$REPO_ROOT/slurm"
CONFIG_DIR="$REPO_ROOT/configs"
SBATCH_TEMPLATE="$SLURM_DIR/run_train.sbatch"

# Parse arguments
METHODS=()
MODE="new"
DRY_RUN=0
i=1

while [[ $i -le $# ]]; do
  arg="${!i}"
  case "$arg" in
    --mode)
      i=$((i + 1))
      MODE="${!i}"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      METHODS+=("$arg")
      ;;
  esac
  i=$((i + 1))
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
  echo "[ERROR] No methods specified. Usage: $0 METHOD1 [METHOD2 ...] [--mode new|continue] [--dry-run]"
  exit 1
fi

if [[ ! -f "$SBATCH_TEMPLATE" ]]; then
  echo "[ERROR] Slurm template not found: $SBATCH_TEMPLATE"
  exit 1
fi

# Function to match method pattern to config files
# Input: method pattern (e.g., "1hop_rnd", "2hop", "1hop_ctc")
# Output: list of matching config file paths
find_configs_for_method() {
  local method="$1"
  local pattern=""
  local normalize_base

  normalize_base() {
    local base_in="$1"
    case "$base_in" in
      1hop_1hop_critic) echo "1hop-critic" ;;
      *) echo "$base_in" ;;
    esac
  }

  # Parse method pattern to extract base and variant
  # Patterns:
  #   "1hop_rnd" -> base="1hop", variant="rnd" -> match "rp_gnn_1hop-*_rnd.yaml"
  #   "2hop" -> base="2hop", no variant -> match "rp_gnn_2hop-[0-9]*.yaml" (base only, no variants)
  #   "1hop_ctc" -> base="1hop", variant="ctc" -> match "rp_gnn_1hop-*_ctc.yaml"
  #   "1hop_ctc_cap2" -> base="1hop", variant="ctc", postfix="cap2" -> match "rp_gnn_1hop-*_ctc_cap2.yaml"
  #   "2hop_cap2" -> base="2hop", postfix="cap2" -> match "rp_gnn_2hop-[0-9]*_cap2.yaml" (base-only cap2)
  #   "1hop_1hop_critic" -> normalized base="1hop-critic" -> match "rp_gnn_1hop-critic-[0-9]*.yaml" (base only)
  #   "1hop_1hop_critic_rnd" -> normalized base="1hop-critic", variant="rnd" -> match "rp_gnn_1hop-critic-*_rnd.yaml"
  #   "1hop-critic_ctc" -> base="1hop-critic", variant="ctc" -> match "rp_gnn_1hop-critic-*_ctc.yaml"

  if [[ "$method" =~ ^(.+)_(rnd|ctc)_cap2$ ]]; then
    # Method has variant and cap2 suffix (e.g., "1hop_ctc_cap2")
    local base="$(normalize_base "${BASH_REMATCH[1]}")"
    local variant="${BASH_REMATCH[2]}"
    pattern="rp_gnn_${base}-[0-9]*_${variant}_cap2.yaml"
    find "$CONFIG_DIR" -maxdepth 1 -name "$pattern" -type f | sort
  elif [[ "$method" =~ ^(.+)_(rnd|ctc)$ ]]; then
    # Method has variant suffix (e.g., "1hop_rnd" -> base="1hop", variant="rnd")
    local base="$(normalize_base "${BASH_REMATCH[1]}")"
    local variant="${BASH_REMATCH[2]}"
    pattern="rp_gnn_${base}-[0-9]*_${variant}.yaml"
    find "$CONFIG_DIR" -maxdepth 1 -name "$pattern" -type f | sort
  elif [[ "$method" =~ ^(.+)_cap2$ ]]; then
    # No variant, cap2 postfix (e.g., "2hop_cap2")
    local base="$(normalize_base "${BASH_REMATCH[1]}")"
    pattern="rp_gnn_${base}-[0-9]*_cap2.yaml"
    find "$CONFIG_DIR" -maxdepth 1 -name "$pattern" -type f | sort
  else
    # No variant (e.g., "2hop" or "1hop_1hop_critic"/"1hop-critic" -> match only base configs)
    # Pattern matches -1.yaml, -2.yaml, etc. but NOT -1_rnd.yaml or -1_ctc.yaml
    local base="$(normalize_base "$method")"
    pattern="rp_gnn_${base}-[0-9].yaml"
    find "$CONFIG_DIR" -maxdepth 1 -name "$pattern" -type f | sort
  fi
}

# Track submission summary
TOTAL_JOBS=0
SUBMITTED_JOBS=0
FAILED_JOBS=0

echo "[INFO] Starting Slurm job submission for methods: ${METHODS[@]}"
echo "[INFO] Mode: $MODE"
echo "[INFO] Dry-run: $([ $DRY_RUN -eq 1 ] && echo "enabled" || echo "disabled")"
echo ""

# For each method, find matching configs and submit jobs
for method in "${METHODS[@]}"; do
  mapfile -t configs < <(find_configs_for_method "$method")

  if [[ ${#configs[@]} -eq 0 ]]; then
    echo "[WARN] No configs found for method: $method"
    continue
  fi

  echo "[INFO] Found ${#configs[@]} config(s) for method: $method"

  for config in "${configs[@]}"; do
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    config_rel="${config#$REPO_ROOT/}"
    run_name=$(grep 'run_name:' "$config" | awk '{print $2}' || echo "unknown")

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "  [DRY-RUN] sbatch --job-name='rp-train-$run_name' $SBATCH_TEMPLATE '$config_rel' '$MODE'"
      SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
    else
      echo "  [SUBMIT] $config_rel (run_name=$run_name)"
      if sbatch --job-name="rp-train-$run_name" "$SBATCH_TEMPLATE" "$config_rel" "$MODE"; then
        SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
      else
        FAILED_JOBS=$((FAILED_JOBS + 1))
        echo "    [ERROR] Failed to submit job"
      fi
    fi
  done
done

echo ""
echo "[INFO] Submission Summary"
echo "  Total configs: $TOTAL_JOBS"
echo "  Submitted: $SUBMITTED_JOBS"
echo "  Failed: $FAILED_JOBS"

if [[ $FAILED_JOBS -gt 0 ]]; then
  exit 1
fi

exit 0
