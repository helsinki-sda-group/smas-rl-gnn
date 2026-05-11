#!/bin/bash
# scripts/plot_eval_results.sh
#
# Plot evaluation results from eval_saved_models.py runs on Mahti.
#
# Supports three usage modes:
#   1. Model family name (expands to all matching run instances):
#        bash scripts/plot_eval_results.sh 1hop_critic_ctc_cap2 [OPTIONS]
#   2. Single run name (exact run in eval_saved_models_jobs):
#        bash scripts/plot_eval_results.sh 1hop_critic-1_ctc_cap2 [OPTIONS]
#   3. Backward-compatible direct file path:
#        bash scripts/plot_eval_results.sh path/to/evaluation_metrics.log [OPTIONS]
#
# Eval runs:   /scratch/project_2012159/kbocheni/smas-rl-gnn/eval_saved_models_jobs/<run>/evaluation_<ts>/evaluation_metrics.log
# Baseline:    /scratch/project_2012159/kbocheni/smas-rl-gnn/eval_jobs/job_eval_<run>_<jobid>/metrics_*.log
# Output:      /projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_evaluation/<subdir>/
#
# Family example: 1hop_critic_ctc_cap2 -> runs 1hop_critic-1_ctc_cap2, 1hop_critic-2_ctc_cap2, ...
#   output -> plots_evaluation/1hop_critic_ctc_cap2/1hop_critic-1_ctc_cap2/, ...
# Single run:     1hop_critic-1_ctc_cap2
#   output -> plots_evaluation/1hop_critic-1_ctc_cap2/
#
# Options:
#   --ma N             Moving average window (default: 10)
#   --baseline-std     Show baseline +/-1 std bands
#   --baseline-log F   Override baseline log path (applies to all runs)
#   --output-dir D     Override plots output root
#   --dry-run          Print commands without executing
#   -h, --help         Show this help

set -euo pipefail

REPO=/projappl/project_2012159/kbocheni_temp/smas-rl-gnn
EVAL_ROOT=/scratch/project_2012159/kbocheni/smas-rl-gnn/eval_saved_models_jobs
BASELINE_ROOT=/scratch/project_2012159/kbocheni/smas-rl-gnn/eval_jobs
PLOTS_ROOT=/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_evaluation

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

usage() {
  cat <<'EOF'
Usage:
  scripts/plot_eval_results.sh <RUN_OR_MODEL> [<RUN_OR_MODEL> ...] [OPTIONS]
  scripts/plot_eval_results.sh path/to/evaluation_metrics.log [OPTIONS]  # backward compat

Options:
  --ma N             Moving average window (default: 10)
  --baseline-std     Show baseline +/-1 std bands
  --baseline-log F   Override baseline log path for all runs
  --output-dir D     Override plots output root (default: plots_evaluation/)
  --dry-run          Print commands without executing
  -h, --help         Show this help

Family examples:
  1hop_critic_ctc_cap2   ->  1hop_critic-1_ctc_cap2, 1hop_critic-2_ctc_cap2, ...
  2hop-maxpool_ctc_cap2  ->  2hop-maxpool-1_ctc_cap2, 2hop-maxpool-3_ctc_cap2, ...
EOF
}

TARGETS=()
MA_WINDOW=10
BASELINE_STD=0
BASELINE_LOG_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
DRY_RUN=0

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ma)
      MA_WINDOW="$2"; shift 2 ;;
    --baseline-std)
      BASELINE_STD=1; shift ;;
    --baseline-log)
      BASELINE_LOG_OVERRIDE="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --*)
      echo "[ERROR] Unknown option: $1"; exit 1 ;;
    *)
      TARGETS+=("$1"); shift ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[ERROR] No run name, model family, or file path specified."
  usage
  exit 1
fi

PLOTS_ROOT="${OUTPUT_DIR_OVERRIDE:-$PLOTS_ROOT}"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Escape special ERE characters in a literal string
escape_ere() {
  sed -e 's/[.[\*^$()+?{}|]/\\&/g' <<<"$1"
}

# Build a regex that matches all run instances of a model family.
# Same logic as eval_saved_models_submit.sh:family_to_regex().
family_to_regex() {
  local family="$1"
  local core suffix
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
  echo "^${core}-[0-9]+${suffix}$"
}

# Derive model family name from a single run name.
# e.g. 1hop_critic-1_ctc_cap2 -> 1hop_critic_ctc_cap2
run_to_family() {
  local run="$1"
  if [[ "$run" =~ ^(.+)-([0-9]+)(_.*)?$ ]]; then
    local core="${BASH_REMATCH[1]}"
    local suffix="${BASH_REMATCH[3]:-}"
    printf '%s%s\n' "$core" "$suffix"
  else
    printf '%s\n' "$run"
  fi
}

# Resolve a target name to a list of run names present in EVAL_ROOT.
# If the target is already an exact directory, returns it as-is.
# Otherwise treats it as a family name and expands to all matching instances.
resolve_target() {
  local target="$1"
  if [[ -d "$EVAL_ROOT/$target" ]]; then
    printf '%s\n' "$target"
    return
  fi
  local regex
  regex="$(family_to_regex "$target")"
  find "$EVAL_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
    | grep -E "$regex" \
    | sort -u
}

# Find the most recent evaluation_metrics.log under a run directory.
find_metrics_log() {
  local run_dir="$1"
  find "$run_dir" -mindepth 2 -maxdepth 2 -name 'evaluation_metrics.log' 2>/dev/null \
    | sort -u | tail -n1
}

# Find a baseline metrics_*.log for a given run name, trying exact match then
# model-family match in BASELINE_ROOT (folders named job_eval_<run>_<jobid>).
# Prints the path on success; returns 1 on failure.
find_baseline_log() {
  local run_name="$1"
  local family
  family="$(run_to_family "$run_name")"

  local match_dir
  match_dir="$(
    find "$BASELINE_ROOT" -mindepth 1 -maxdepth 1 -type d \
         -name "job_eval_${run_name}_*" 2>/dev/null \
      | sort | tail -n1
  )"

  if [[ -z "$match_dir" ]]; then
    local family_regex
    family_regex="$(family_to_regex "$family")"
    match_dir="$(
      find "$BASELINE_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
        | sed -E 's/^job_eval_(.+)_[0-9]+$/\1/' \
        | grep -E "$family_regex" \
        | sort -u \
        | while IFS= read -r matched_run; do
            find "$BASELINE_ROOT" -mindepth 1 -maxdepth 1 -type d \
                 -name "job_eval_${matched_run}_*" 2>/dev/null
          done \
        | sort | tail -n1
    )"
  fi

  if [[ -n "$match_dir" ]]; then
    local mf
    mf="$(find "$match_dir" -maxdepth 1 -name 'metrics_*.log' 2>/dev/null \
            | sort | tail -n1)"
    if [[ -n "$mf" ]]; then
      printf '%s\n' "$mf"
      return 0
    else
      echo "  [WARN] Baseline dir found but no metrics_*.log inside: $match_dir" >&2
    fi
  fi

  return 1
}

# Read a scalar value for a key from a YAML file (simple grep, works for flat
# and one-level-indented keys).
read_yaml_scalar() {
  local yaml_file="$1"
  local key="$2"
  grep -E "^\s*${key}:" "$yaml_file" 2>/dev/null \
    | head -n1 \
    | sed -E "s/.*${key}:[[:space:]]*//"
}

# Inspect the most recent config snapshot for a run and print key settings.
check_config_snapshot() {
  local run_name="$1"
  local snap_dir="$EVAL_ROOT/$run_name/config_snapshots"
  if [[ ! -d "$snap_dir" ]]; then
    echo "  [CONFIG] No config_snapshots dir found for $run_name — skipping config check."
    return
  fi
  local snap_file
  snap_file="$(find "$snap_dir" -maxdepth 1 -name '*.yaml' 2>/dev/null | sort | tail -n1)"
  if [[ -z "$snap_file" ]]; then
    echo "  [CONFIG] No YAML snapshot found in $snap_dir — skipping config check."
    return
  fi
  echo "  [CONFIG] Snapshot: $(basename "$snap_file")"

  local completion_mode w_comp w_wait w_travel
  completion_mode="$(read_yaml_scalar "$snap_file" "completion_mode")"
  w_comp="$(read_yaml_scalar "$snap_file" "w_comp")"
  w_wait="$(read_yaml_scalar "$snap_file" "w_wait")"
  w_travel="$(read_yaml_scalar "$snap_file" "w_travel")"

  if [[ -n "$completion_mode" ]]; then
    echo "  [CONFIG] completion_mode = $completion_mode"
    if [[ "$completion_mode" == "valid_dropoff" ]]; then
      echo "  [CONFIG]   NOTE: valid_dropoff — only valid completions (within time constraints) earn reward."
    fi
  else
    echo "  [CONFIG] completion_mode not found in snapshot."
  fi
  [[ -n "$w_comp"   ]] && echo "  [CONFIG] w_comp   = $w_comp"
  [[ -n "$w_wait"   ]] && echo "  [CONFIG] w_wait   = $w_wait"
  [[ -n "$w_travel" ]] && echo "  [CONFIG] w_travel = $w_travel"
}

# --------------------------------------------------------------------------- #
# Activate Python environment
# --------------------------------------------------------------------------- #

module purge 2>/dev/null || true
module load python-data 2>/dev/null || true
if [[ -f "$REPO/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
fi

# --------------------------------------------------------------------------- #
# Resolve all targets -> (run_name, plot_subdir) pairs
# --------------------------------------------------------------------------- #

declare -a RUN_LIST       # run names
declare -a SUBDIR_LIST    # output subfolder per run

for target in "${TARGETS[@]}"; do

  # ---- Backward compat: direct file path ---------------------------------- #
  if [[ -f "$target" ]]; then
    echo "[COMPAT] Detected file path: $target — passing directly to plot_eval_results.py"
    py_cmd=(
      python "$REPO/plot_eval_results.py" "$target"
      --ma "$MA_WINDOW"
    )
    [[ $BASELINE_STD -eq 1 ]] && py_cmd+=("--baseline-std")
    [[ -n "$BASELINE_LOG_OVERRIDE" ]] && py_cmd+=("--baseline-log" "$BASELINE_LOG_OVERRIDE")

    if [[ $DRY_RUN -eq 1 ]]; then
      printf '[DRY-RUN] '
      printf '%q ' "${py_cmd[@]}"
      printf '\n'
    else
      "${py_cmd[@]}"
    fi
    continue
  fi

  # ---- Resolve to run names ----------------------------------------------- #
  mapfile -t resolved < <(resolve_target "$target")

  if [[ ${#resolved[@]} -eq 0 ]]; then
    echo "[WARN] No eval run directories found for target '$target' in $EVAL_ROOT"
    continue
  fi

  echo "[INFO] Target '$target' -> ${#resolved[@]} run(s): ${resolved[*]}"

  # Determine output subfolder:
  #   - exact single run whose name == target -> plots_evaluation/<run>/
  #   - family expansion (or multiple) -> plots_evaluation/<family>/<run>/
  local_subdir="$target"

  for rn in "${resolved[@]}"; do
    RUN_LIST+=("$rn")
    SUBDIR_LIST+=("$local_subdir")
  done
done

# --------------------------------------------------------------------------- #
# Plot each run
# --------------------------------------------------------------------------- #

SUCCESS=0
FAILED=0

for idx in "${!RUN_LIST[@]}"; do
  run_name="${RUN_LIST[$idx]}"
  plot_subdir="${SUBDIR_LIST[$idx]}"

  echo ""
  echo "=== Run: $run_name ==="

  # Determine output directory
  if [[ "$run_name" == "$plot_subdir" ]]; then
    # Exact single run
    out_dir="$PLOTS_ROOT/$run_name"
  else
    # Family expansion
    out_dir="$PLOTS_ROOT/$plot_subdir/$run_name"
  fi

  # Find metrics log
  metrics_log="$(find_metrics_log "$EVAL_ROOT/$run_name")"
  if [[ -z "$metrics_log" ]]; then
    echo "  [ERROR] No evaluation_metrics.log found under $EVAL_ROOT/$run_name"
    FAILED=$((FAILED + 1))
    continue
  fi
  echo "  Metrics log : $metrics_log"

  # Find baseline log
  baseline_arg=""
  if [[ -n "$BASELINE_LOG_OVERRIDE" ]]; then
    baseline_arg="$BASELINE_LOG_OVERRIDE"
    echo "  Baseline log: $baseline_arg  (override)"
  else
    if baseline_log="$(find_baseline_log "$run_name")"; then
      baseline_arg="$baseline_log"
      echo "  Baseline log: $baseline_arg"
    else
      family_hint="$(run_to_family "$run_name")"
      echo "  [WARN] No baseline log found for run '$run_name' or family '$family_hint' in $BASELINE_ROOT"
    fi
  fi

  # Config check
  check_config_snapshot "$run_name"

  # Build Python command
  mkdir -p "$out_dir"
  py_cmd=(
    python "$REPO/plot_eval_results.py" "$metrics_log"
    --ma "$MA_WINDOW"
    --output-dir "$out_dir"
  )
  [[ $BASELINE_STD -eq 1 ]] && py_cmd+=("--baseline-std")
  [[ -n "$baseline_arg" ]] && py_cmd+=("--baseline-log" "$baseline_arg")

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  [DRY-RUN] '
    printf '%q ' "${py_cmd[@]}"
    printf '\n'
    SUCCESS=$((SUCCESS + 1))
  else
    if "${py_cmd[@]}"; then
      echo "  [OK] Plots saved to: $out_dir"
      SUCCESS=$((SUCCESS + 1))
    else
      echo "  [ERROR] Plotting failed for $run_name"
      FAILED=$((FAILED + 1))
    fi
  fi
done

echo ""
echo "=== Done: $SUCCESS succeeded, $FAILED failed ==="
