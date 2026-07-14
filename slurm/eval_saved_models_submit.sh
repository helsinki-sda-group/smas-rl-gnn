#!/bin/bash
# eval_saved_models_submit.sh - Submit eval_saved_models.py jobs on Mahti.
#
# Supports both exact run names and model family names that expand to multiple runs.
#
# Family examples:
#   2hop-maxpool_ctc_cap2 -> 2hop-maxpool-1_ctc_cap2, 2hop-maxpool-3_ctc_cap2, ...
#   1hop_critic_ctc       -> 1hop_critic-1_ctc, 1hop_critic-2_ctc, ...
#
# Usage examples:
#   bash slurm/eval_saved_models_submit.sh 2hop-maxpool_ctc_cap2 1hop_critic_ctc --eval-runs 1 --seeds eval --model-sample 0.5 --deterministic
#   bash slurm/eval_saved_models_submit.sh 2hop-maxpool-3_ctc_cap2 --dry-run --eval-runs 1 --seeds eval

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLURM_DIR="$REPO_ROOT/slurm"
CONFIG_DIR="$REPO_ROOT/configs"
SBATCH_TEMPLATE="$SLURM_DIR/run_eval_saved_models.sbatch"
RUNS_ROOT="/scratch/project_2012159/kbocheni/smas-rl-gnn/runs"

if [[ ! -f "$SBATCH_TEMPLATE" ]]; then
  echo "[ERROR] Slurm template not found: $SBATCH_TEMPLATE"
  exit 1
fi

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "[ERROR] Config directory not found: $CONFIG_DIR"
  exit 1
fi

# Parse arguments.
TARGETS=()
DRY_RUN=0
EXTRA_ARGS=()
SUMOPORT_BASE=8813

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --eval-runs|--seeds|--model-sample|--ma-window)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] Missing value for $1"
        exit 1
      fi
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --sumoport-base)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] Missing value for $1"
        exit 1
      fi
      SUMOPORT_BASE="$2"
      shift 2
      ;;
    --sumoport)
      echo "[ERROR] --sumoport is not supported in batch mode because each run needs a unique port. Use --sumoport-base instead."
      exit 1
      ;;
    --deterministic|--sorted|--gui|--print-steps)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --*)
      echo "[ERROR] Unsupported option: $1"
      exit 1
      ;;
    *)
      TARGETS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[ERROR] No run names or model families provided."
  echo "Usage: $0 RUN_OR_FAMILY [RUN_OR_FAMILY ...] [--eval-runs N] [--seeds train|eval|both] [--model-sample X] [--sumoport-base P] [--deterministic] [--dry-run]"
  exit 1
fi

if [[ ! -d "$RUNS_ROOT" ]]; then
  echo "[ERROR] Runs root not found: $RUNS_ROOT"
  exit 1
fi

escape_ere() {
  sed -e 's/[.[\*^$()+?{}|]/\\&/g' <<<"$1"
}

family_to_regex() {
  local family="$1"
  local core
  local suffix

  if [[ "$family" =~ ^(.+)_(ctc|rnd)_cap2$ ]]; then
    core="${BASH_REMATCH[1]}"
    suffix="_${BASH_REMATCH[2]}_cap2"
  elif [[ "$family" =~ ^(.+)_(ctc|rnd)$ ]]; then
    core="${BASH_REMATCH[1]}"
    suffix="_${BASH_REMATCH[2]}"
  else
    core="$family"
    suffix=""
  fi

  core="$(escape_ere "$core")"
  suffix="$(escape_ere "$suffix")"

  # Strict: require exact family text and a hyphen before numeric instance index.
  echo "^${core}-[0-9]+${suffix}$"
}

find_matching_runs_for_target() {
  local target="$1"
  local run_dir="$RUNS_ROOT/$target"

  # Exact run name wins.
  if [[ -d "$run_dir" && -d "$run_dir/!saved_models" ]]; then
    printf '%s\n' "$target"
    return
  fi

  local regex
  regex="$(family_to_regex "$target")"

  find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
    grep -E "$regex" |
    while IFS= read -r rn; do
      if [[ -d "$RUNS_ROOT/$rn/!saved_models" ]]; then
        printf '%s\n' "$rn"
      fi
    done |
    sort -u
}

find_config_for_run_name() {
  local run_name="$1"
  local cfg
  while IFS= read -r cfg; do
    if grep -q -E "^[[:space:]]*run_name:[[:space:]]*${run_name}[[:space:]]*$" "$cfg"; then
      printf '%s\n' "$cfg"
      return
    fi
  done < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name 'rp_gnn*.yaml' | sort)
}

ALL_RUNS=()
for target in "${TARGETS[@]}"; do
  mapfile -t matches < <(find_matching_runs_for_target "$target")
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "[WARN] No matching runs for target: $target"
    continue
  fi
  echo "[INFO] Target '$target' matched ${#matches[@]} run(s)."
  for rn in "${matches[@]}"; do
    ALL_RUNS+=("$rn")
  done
done

if [[ ${#ALL_RUNS[@]} -eq 0 ]]; then
  echo "[ERROR] No runs matched."
  exit 1
fi

mapfile -t UNIQUE_RUNS < <(printf '%s\n' "${ALL_RUNS[@]}" | sort -u)

echo "[INFO] Submitting ${#UNIQUE_RUNS[@]} evaluation job(s)."
echo "[INFO] Sumo port base: $SUMOPORT_BASE"
echo "[INFO] Extra eval args: ${EXTRA_ARGS[*]:-(none)}"

TOTAL=0
SUBMITTED=0
FAILED=0

for run_name in "${UNIQUE_RUNS[@]}"; do
  TOTAL=$((TOTAL + 1))
  assigned_port=$((SUMOPORT_BASE + TOTAL - 1))
  cfg_path="$(find_config_for_run_name "$run_name")"

  if [[ -z "$cfg_path" ]]; then
    FAILED=$((FAILED + 1))
    echo "  [ERROR] No config found with run_name=$run_name"
    continue
  fi

  cfg_rel="${cfg_path#$REPO_ROOT/}"
  cmd=(
    sbatch
    --job-name="rp-eval-${run_name}"
    "$SBATCH_TEMPLATE"
    "$run_name"
    "$cfg_rel"
    --sumoport
    "$assigned_port"
    "${EXTRA_ARGS[@]}"
  )

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  [DRY-RUN] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    SUBMITTED=$((SUBMITTED + 1))
  else
    echo "  [SUBMIT] run=$run_name config=$cfg_rel sumoport=$assigned_port"
    if "${cmd[@]}"; then
      SUBMITTED=$((SUBMITTED + 1))
    else
      FAILED=$((FAILED + 1))
    fi
  fi
done

echo ""
echo "[INFO] Summary"
echo "  Total matched runs: $TOTAL"
echo "  Submitted: $SUBMITTED"
echo "  Failed: $FAILED"

if [[ $FAILED -gt 0 ]]; then
  exit 1
fi

exit 0
