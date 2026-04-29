#!/bin/bash
set -euo pipefail

# Compare several Mahti training methods and auto-discover matching runs.
# For each discovered run_name (e.g. 1hop-1, 1hop-2), the latest matching job
# directory is selected, optionally filtered by --job-ids.
# The script then:
# 1. writes a generated ablation config
# 2. runs aggregate_ablation_results.py with that config
# 3. runs plot_action_candidates.py across the selected runs
# 4. runs plot_conflicts_comparison.py across the selected runs

usage() {
  cat <<'EOF'
Usage:
  scripts/plot_ablation_runs.sh <method1> [method2 ...] [options]

Options:
  --job-ids <id1,id2,...>     Optional job-id filter. Keeps only discovered runs whose
                              selected job id is in this list.
  --labels <l1,l2,...>        Optional display labels, same order as methods.
  --repo <path>               Repo root.
                              Default: /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
  --template-config <path>    YAML template for ablation parameters.
                              Default: <repo>/ablation_conf_mahti.yaml
  --scratch-root <path>       Scratch root.
                              Default: /scratch/project_2012159/kbocheni/smas-rl-gnn
  --out-root <path>           Output root for comparison plots and generated config.
                              Default: /projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_ablation
  --action-window <n>         Override action smoothing window from YAML template.
  --conflicts-window <n>      Override conflict smoothing window from YAML template.
  -h, --help                  Show this help.

Examples:
  scripts/plot_ablation_runs.sh 1hop 1hop_critic 2hop
  scripts/plot_ablation_runs.sh 1hop 1hop_critic 2hop --labels "1 hop,1 hop critic,2 hop"
  scripts/plot_ablation_runs.sh 1hop 1hop_critic 2hop --job-ids 6574001,6574582
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

REPO="/projappl/project_2012159/kbocheni_temp/smas-rl-gnn"
TEMPLATE_CONFIG=""
SCRATCH_ROOT="/scratch/project_2012159/kbocheni/smas-rl-gnn"
OUT_ROOT="/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_ablation"
ACTION_WINDOW_OVERRIDE=""
CONFLICTS_WINDOW_OVERRIDE=""
JOB_IDS_CSV=""
LABELS_CSV=""

METHODS=()
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
    --conflicts-window)
      CONFLICTS_WINDOW_OVERRIDE="$2"
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
      METHODS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#METHODS[@]} -lt 1 ]]; then
  echo "[ERROR] Provide at least one method to compare."
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

if [[ ${#LABELS[@]} -gt 0 && ${#LABELS[@]} -ne ${#METHODS[@]} ]]; then
  echo "[ERROR] --labels must have the same number of entries as methods."
  exit 1
fi

declare -A METHOD_LABEL_MAP=()
for i in "${!METHODS[@]}"; do
  method="${METHODS[$i]}"
  if [[ ${#LABELS[@]} -gt 0 ]]; then
    METHOD_LABEL_MAP["$method"]="${LABELS[$i]}"
  else
    METHOD_LABEL_MAP["$method"]="$method"
  fi
done

declare -A JOB_FILTER_SET=()
if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
  for id in "${JOB_IDS[@]}"; do
    if [[ "$id" =~ ^[0-9]+$ ]]; then
      JOB_FILTER_SET["$id"]=1
    else
      echo "[WARN] Ignoring non-numeric job id filter: $id"
    fi
  done
fi

declare -A MATCHED_FILTER_ID=()
declare -A METHOD_FOR_RUN=()
declare -A BEST_ID_FOR_RUN=()

shopt -s nullglob
for d in "$JOBS_ROOT"/job_*_*; do
  [[ -d "$d" ]] || continue
  b="$(basename "$d")"
  id="${b##*_}"
  [[ "$id" =~ ^[0-9]+$ ]] || continue
  run_name="${b#job_}"
  run_name="${run_name%_"$id"}"

  for method in "${METHODS[@]}"; do
    if [[ "$run_name" != *"${method}-"* ]]; then
      continue
    fi

    if [[ ${#JOB_FILTER_SET[@]} -gt 0 ]]; then
      if [[ -z "${JOB_FILTER_SET[$id]+x}" ]]; then
        continue
      fi
      MATCHED_FILTER_ID["$id"]=1
    fi

    prev="${BEST_ID_FOR_RUN[$run_name]:-}"
    if [[ -z "$prev" || "$id" -gt "$prev" ]]; then
      BEST_ID_FOR_RUN["$run_name"]="$id"
      METHOD_FOR_RUN["$run_name"]="$method"
    fi
  done
done
shopt -u nullglob

if [[ ${#JOB_FILTER_SET[@]} -gt 0 ]]; then
  for id in "${JOB_IDS[@]}"; do
    [[ "$id" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${MATCHED_FILTER_ID[$id]+x}" ]]; then
      echo "[WARN] --job-ids entry '$id' did not match any discovered run for requested methods; skipped."
    fi
  done
fi

SELECTED_JOBDIRS=()
SELECTED_METRICS=()
SELECTED_CONFLICTS=()
SELECTED_LABELS=()
SELECTED_IDS=()

for method in "${METHODS[@]}"; do
  method_runs=()
  for run_name in "${!BEST_ID_FOR_RUN[@]}"; do
    if [[ "${METHOD_FOR_RUN[$run_name]}" == "$method" ]]; then
      method_runs+=("$run_name")
    fi
  done

  if [[ ${#method_runs[@]} -eq 0 ]]; then
    echo "[WARN] No discovered runs for method '$method' after filtering; skipping method."
    continue
  fi

  IFS=$'\n' method_runs_sorted=($(printf '%s\n' "${method_runs[@]}" | sort))
  unset IFS

  method_label_base="${METHOD_LABEL_MAP[$method]}"
  for run_name in "${method_runs_sorted[@]}"; do
    job_id="${BEST_ID_FOR_RUN[$run_name]}"
    jobdir="$JOBS_ROOT/job_${run_name}_${job_id}"

    metrics_log="$(ls -1t "$jobdir"/training_metrics_*.log 2>/dev/null | head -n 1 || true)"
    if [[ -z "$metrics_log" ]]; then
      echo "[WARN] No training_metrics_*.log found in $jobdir; skipping run."
      continue
    fi
    conflicts_log="$jobdir/conflicts.log"

    SELECTED_JOBDIRS+=("$jobdir")
    SELECTED_METRICS+=("$metrics_log")
    SELECTED_CONFLICTS+=("$conflicts_log")
    SELECTED_IDS+=("$job_id")

    if [[ ${#LABELS[@]} -gt 0 ]]; then
      run_suffix="${run_name#*${method}}"
      SELECTED_LABELS+=("${method_label_base}${run_suffix}")
    else
      SELECTED_LABELS+=("$run_name")
    fi
  done
done

if [[ ${#SELECTED_JOBDIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No valid runs found for requested methods."
  exit 1
fi

comparison_slug="${METHODS[0]}"
for ((i=1; i<${#METHODS[@]}; i++)); do
  comparison_slug+="__${METHODS[$i]}"
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

ACTION_PARAMS="$($PYTHON_BIN - "$TEMPLATE_CONFIG" "$GENERATED_CONF" "$MAPPING_FILE" "$OUTDIR" "${ACTION_WINDOW_OVERRIDE:-}" "${CONFLICTS_WINDOW_OVERRIDE:-}" <<'PY'
import sys
from omegaconf import OmegaConf

template_path, generated_path, mapping_path, outdir, action_override, conflicts_override = sys.argv[1:7]
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
default_window = int(cfg.get("k_eval", 10))
action_window = int(action_override) if action_override else int(script_cfg.get("action_window", default_window))
action_out_dirname = str(script_cfg.get("action_out_dirname", "action_comparison"))
conflicts_window = int(conflicts_override) if conflicts_override else int(script_cfg.get("conflicts_window", default_window))
conflicts_out_dirname = str(script_cfg.get("conflicts_out_dirname", "conflicts_comparison"))
action_grouped_only = int(bool(cfg.get("action_grouped_only", True)))
action_plot_std = int(bool(cfg.get("action_plot_std", True)))
conflicts_plot_std = int(bool(cfg.get("conflicts_plot_std", True)))
mean_runs = int(bool(cfg.get("mean_runs", True)))
print(f"{action_window}\t{action_out_dirname}\t{conflicts_window}\t{conflicts_out_dirname}\t{action_grouped_only}\t{action_plot_std}\t{conflicts_plot_std}\t{mean_runs}")
PY
)"

IFS=$'\t' read -r ACTION_WINDOW ACTION_OUT_DIRNAME CONFLICTS_WINDOW CONFLICTS_OUT_DIRNAME ACTION_GROUPED_ONLY ACTION_PLOT_STD CONFLICTS_PLOT_STD MEAN_RUNS <<< "$ACTION_PARAMS"

echo "[INFO] Generated config: $GENERATED_CONF"
echo "[INFO] Comparison output dir: $OUTDIR"
echo "[INFO] Action window: $ACTION_WINDOW"
echo "[INFO] Conflicts window: $CONFLICTS_WINDOW"

"$PYTHON_BIN" "$REPO/aggregate_ablation_results.py" --config "$GENERATED_CONF"

ACTION_FLAGS=""
if [[ "$ACTION_GROUPED_ONLY" == "1" ]]; then
  ACTION_FLAGS="--grouped-only"
fi
if [[ "$ACTION_PLOT_STD" != "1" ]]; then
  ACTION_FLAGS="$ACTION_FLAGS --no-plot-std"
fi
if [[ "$MEAN_RUNS" != "1" ]]; then
  ACTION_FLAGS="$ACTION_FLAGS --no-mean-runs"
fi

CONFLICT_FLAGS=""
if [[ "$CONFLICTS_PLOT_STD" != "1" ]]; then
  CONFLICT_FLAGS="--no-plot-std"
fi
if [[ "$MEAN_RUNS" != "1" ]]; then
  CONFLICT_FLAGS="$CONFLICT_FLAGS --no-mean-runs"
fi

"$PYTHON_BIN" "$REPO/plot_action_candidates.py" \
  "${SELECTED_METRICS[@]}" \
  --labels "$(IFS=,; echo "${SELECTED_LABELS[*]}")" \
  --window "$ACTION_WINDOW" \
  $ACTION_FLAGS \
  --out "$OUTDIR/$ACTION_OUT_DIRNAME"

"$PYTHON_BIN" "$REPO/plot_conflicts_comparison.py" \
  "${SELECTED_CONFLICTS[@]}" \
  --labels "$(IFS=,; echo "${SELECTED_LABELS[*]}")" \
  --window "$CONFLICTS_WINDOW" \
  $CONFLICT_FLAGS \
  --out "$OUTDIR/$CONFLICTS_OUT_DIRNAME"

echo "[OK] Ablation comparison written to: $OUTDIR"
