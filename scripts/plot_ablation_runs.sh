#!/bin/bash
set -euo pipefail

# Compare several Mahti training runs identified by run_name.
# For each run_name, the latest matching job directory is selected unless job ids are given.
# The script then:
# 1. writes a generated ablation config
# 2. runs aggregate_ablation_results.py with that config
# 3. runs plot_action_candidates.py across the selected runs

usage() {
  cat <<'EOF'
Usage:
  scripts/plot_ablation_runs.sh <run_name1> <run_name2> [run_name3 ...] [options]

Options:
  --job-ids <id1,id2,...>     Optional explicit job ids, same order as run names.
  --labels <l1,l2,...>        Optional display labels, same order as run names.
  --repo <path>               Repo root.
                              Default: /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
  --template-config <path>    YAML template for ablation parameters.
                              Default: <repo>/ablation_conf_mahti.yaml
  --scratch-root <path>       Scratch root.
                              Default: /scratch/project_2012159/kbocheni/smas-rl-gnn
  --out-root <path>           Output root for comparison plots and generated config.
                              Default: /projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_ablation
  --action-window <n>         Override action smoothing window from YAML template.
  -h, --help                  Show this help.

Examples:
  scripts/plot_ablation_runs.sh rp_gnn_debug_1hop rp_gnn_debug_2hop
  scripts/plot_ablation_runs.sh rp_gnn_debug_1hop rp_gnn_debug_2hop --labels "1 hop,2 hop"
  scripts/plot_ablation_runs.sh rp_gnn_debug_1hop rp_gnn_debug_2hop --job-ids 6574001,6574582
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

REPO="/projappl/project_2012159/kbocheni_temp/smas-rl-gnn"
TEMPLATE_CONFIG=""
SCRATCH_ROOT="/scratch/project_2012159/kbocheni/smas-rl-gnn"
OUT_ROOT="/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_ablation"
ACTION_WINDOW_OVERRIDE=""
JOB_IDS_CSV=""
LABELS_CSV=""

RUN_NAMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-ids)
      JOB_IDS_CSV="$2"
      shift 2
      ;;
    --labels)
      LABELS_CSV="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --template-config)
      TEMPLATE_CONFIG="$2"
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
    --action-window)
      ACTION_WINDOW_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      RUN_NAMES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#RUN_NAMES[@]} -lt 2 ]]; then
  echo "[ERROR] Provide at least two run names to compare."
  exit 1
fi

if [[ -z "$TEMPLATE_CONFIG" ]]; then
  TEMPLATE_CONFIG="$REPO/ablation_conf_mahti.yaml"
fi
if [[ ! -f "$TEMPLATE_CONFIG" ]]; then
  echo "[ERROR] Template config not found: $TEMPLATE_CONFIG"
  exit 1
fi

JOBS_ROOT="$SCRATCH_ROOT/jobs"

split_csv_to_array() {
  local csv="$1"
  local -n out_arr=$2
  out_arr=()
  if [[ -z "$csv" ]]; then
    return 0
  fi
  IFS=',' read -r -a out_arr <<< "$csv"
  for i in "${!out_arr[@]}"; do
    out_arr[$i]="$(echo "${out_arr[$i]}" | sed 's/^ *//; s/ *$//')"
  done
}

JOB_IDS=()
LABELS=()
split_csv_to_array "$JOB_IDS_CSV" JOB_IDS
split_csv_to_array "$LABELS_CSV" LABELS

if [[ ${#JOB_IDS[@]} -gt 0 && ${#JOB_IDS[@]} -ne ${#RUN_NAMES[@]} ]]; then
  echo "[ERROR] --job-ids must have the same number of entries as run names."
  exit 1
fi
if [[ ${#LABELS[@]} -gt 0 && ${#LABELS[@]} -ne ${#RUN_NAMES[@]} ]]; then
  echo "[ERROR] --labels must have the same number of entries as run names."
  exit 1
fi

resolve_jobdir() {
  local run_name="$1"
  local explicit_job_id="${2:-}"

  if [[ -n "$explicit_job_id" ]]; then
    local explicit_dir="$JOBS_ROOT/job_${run_name}_${explicit_job_id}"
    [[ -d "$explicit_dir" ]] || return 1
    printf '%s\n' "$explicit_dir"
    return 0
  fi

  local latest_id=""
  local latest_dir=""
  shopt -s nullglob
  for d in "$JOBS_ROOT"/job_"$run_name"_*; do
    [[ -d "$d" ]] || continue
    local id="${d##*_}"
    [[ "$id" =~ ^[0-9]+$ ]] || continue
    if [[ -z "$latest_id" || "$id" -gt "$latest_id" ]]; then
      latest_id="$id"
      latest_dir="$d"
    fi
  done
  shopt -u nullglob

  [[ -n "$latest_dir" ]] || return 1
  printf '%s\n' "$latest_dir"
}

SELECTED_JOBDIRS=()
SELECTED_METRICS=()
SELECTED_LABELS=()
SELECTED_IDS=()

for i in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$i]}"
  explicit_id=""
  if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    explicit_id="${JOB_IDS[$i]}"
  fi
  jobdir="$(resolve_jobdir "$run_name" "$explicit_id")" || {
    echo "[ERROR] Could not resolve job directory for run_name='$run_name'"
    exit 1
  }
  metrics_log="$(ls -1t "$jobdir"/training_metrics_*.log 2>/dev/null | head -n 1 || true)"
  if [[ -z "$metrics_log" ]]; then
    echo "[ERROR] No training_metrics_*.log found in $jobdir"
    exit 1
  fi

  job_id="${jobdir##*_}"
  SELECTED_JOBDIRS+=("$jobdir")
  SELECTED_METRICS+=("$metrics_log")
  SELECTED_IDS+=("$job_id")
  if [[ ${#LABELS[@]} -gt 0 ]]; then
    SELECTED_LABELS+=("${LABELS[$i]}")
  else
    SELECTED_LABELS+=("$run_name")
  fi
done

comparison_slug="${RUN_NAMES[0]}"
for ((i=1; i<${#RUN_NAMES[@]}; i++)); do
  comparison_slug+="__${RUN_NAMES[$i]}"
done

OUTDIR="$OUT_ROOT/$comparison_slug"
mkdir -p "$OUTDIR"

GENERATED_CONF="$OUTDIR/ablation_conf_generated.yaml"
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

MAPPING_FILE="$OUTDIR/run_mapping.tsv"
: > "$MAPPING_FILE"
for i in "${!SELECTED_JOBDIRS[@]}"; do
  base_name="$(basename "${SELECTED_JOBDIRS[$i]}")"
  printf '%s\t%s\t%s\n' "$base_name" "${SELECTED_LABELS[$i]}" "${SELECTED_JOBDIRS[$i]}" >> "$MAPPING_FILE"
done

ACTION_PARAMS="$($PYTHON_BIN - "$TEMPLATE_CONFIG" "$GENERATED_CONF" "$MAPPING_FILE" "$OUTDIR" "${ACTION_WINDOW_OVERRIDE:-}" <<'PY'
import sys
from omegaconf import OmegaConf

template_path, generated_path, mapping_path, outdir, action_override = sys.argv[1:6]
cfg = OmegaConf.load(template_path)

model_dirs = []
experiment_names = {}
with open(mapping_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        base_name, label, model_dir = line.split("\t", 2)
        model_dirs.append(model_dir)
        experiment_names[base_name] = label

cfg.model_dirs = model_dirs
cfg.experiment_names = experiment_names
cfg.output_dir = f"{outdir}/ablation_results"

OmegaConf.save(cfg, generated_path)

script_cfg = cfg.get("script") or {}
action_window = int(action_override) if action_override else int(script_cfg.get("action_window", 10))
action_out_dirname = str(script_cfg.get("action_out_dirname", "action_comparison"))
print(f"{action_window}\t{action_out_dirname}")
PY
)"

ACTION_WINDOW="${ACTION_PARAMS%%$'\t'*}"
ACTION_OUT_DIRNAME="${ACTION_PARAMS#*$'\t'}"

echo "[INFO] Generated config: $GENERATED_CONF"
echo "[INFO] Comparison output dir: $OUTDIR"
echo "[INFO] Action window: $ACTION_WINDOW"

"$PYTHON_BIN" "$REPO/aggregate_ablation_results.py" --config "$GENERATED_CONF"

"$PYTHON_BIN" "$REPO/plot_action_candidates.py" \
  "${SELECTED_METRICS[@]}" \
  --labels "$(IFS=,; echo "${SELECTED_LABELS[*]}")" \
  --window "$ACTION_WINDOW" \
  --out "$OUTDIR/$ACTION_OUT_DIRNAME"

echo "[OK] Ablation comparison written to: $OUTDIR"
