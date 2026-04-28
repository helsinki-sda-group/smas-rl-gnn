#!/bin/bash
set -euo pipefail

# Generate conflict-resolution variants for rp_gnn experiment configs.
#
# This script creates new YAML files in configs/ with suffix _ctc.yaml,
# updates:
#   env.conflict_resolution -> closest_then_capacity
#   logging.run_name        -> <old_run_name>_ctc
#   logging.model_save_dir  -> <logging.out_dir>/<new_run_name>/!saved_models
#
# By default it only generates files. Use --submit to also submit all generated
# configs via slurm/run_train.sbatch.

usage() {
  cat <<'EOF'
Usage:
  scripts/generate_ctc_configs.sh [options]

Options:
  --submit                 Submit generated configs to Slurm.
  --config-dir <path>      Config directory (default: configs)
  --sbatch-script <path>   Sbatch script path (default: slurm/run_train.sbatch)
  --suffix <text>          Suffix for generated configs/run_name (default: _ctc)
  --dry-run                Show actions without writing files or submitting jobs.
  -h, --help               Show this help.

Notes:
  - Resolver value is closest_then_capacity (supported by current code).
  - Source files selected: configs/rp_gnn_*.yaml excluding rp_gnn.yaml and rp_toy.yaml.
EOF
}

SUBMIT=0
DRY_RUN=0
CONFIG_DIR="configs"
SBATCH_SCRIPT="slurm/run_train.sbatch"
SUFFIX="_ctc"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      SUBMIT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    --sbatch-script)
      SBATCH_SCRIPT="$2"
      shift 2
      ;;
    --suffix)
      SUFFIX="$2"
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

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "[ERROR] Config directory not found: $CONFIG_DIR"
  exit 1
fi

if [[ "$SUBMIT" -eq 1 && ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[ERROR] Sbatch script not found: $SBATCH_SCRIPT"
  exit 1
fi

PYTHON_BIN="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "[INFO] Config dir: $CONFIG_DIR"
echo "[INFO] Suffix: $SUFFIX"
if [[ "$SUBMIT" -eq 1 ]]; then
  echo "[INFO] Submit: yes"
else
  echo "[INFO] Submit: no"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[INFO] Dry run: yes"
fi

TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

"$PYTHON_BIN" - "$CONFIG_DIR" "$SUFFIX" "$DRY_RUN" "$TMP_LIST" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

config_dir = Path(sys.argv[1])
suffix = sys.argv[2]
dry_run = sys.argv[3] == "1"
out_list = Path(sys.argv[4])

files = sorted(config_dir.glob("rp_gnn_*.yaml"))
files = [p for p in files if p.name not in {"rp_gnn.yaml", "rp_toy.yaml"}]

if not files:
    print("[WARN] No source files found matching rp_gnn_*.yaml")
    out_list.write_text("", encoding="utf-8")
    raise SystemExit(0)

generated = []
for src in files:
    cfg = OmegaConf.load(src)
    old_run = str(getattr(cfg.logging, "run_name", src.stem)).strip()
    out_dir = str(getattr(cfg.logging, "out_dir", "runs")).rstrip("/")

    new_run = f"{old_run}{suffix}"
    cfg.env.conflict_resolution = "closest_then_capacity"
    cfg.logging.run_name = new_run
    cfg.logging.model_save_dir = f"{out_dir}/{new_run}/!saved_models"

    dst = src.with_name(src.stem + suffix + ".yaml")
    generated.append(dst)

    if dry_run:
        print(f"[DRY] {src} -> {dst}  run_name={new_run}")
    else:
        OmegaConf.save(cfg, dst)
        print(f"[OK]  {src} -> {dst}  run_name={new_run}")

out_list.write_text("\n".join(str(p) for p in generated), encoding="utf-8")
print(f"[INFO] Total generated: {len(generated)}")
PY

if [[ "$SUBMIT" -eq 1 ]]; then
  while IFS= read -r cfg; do
    [[ -n "$cfg" ]] || continue
    job_name="$(basename "$cfg" .yaml)"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] sbatch --job-name=$job_name $SBATCH_SCRIPT $cfg"
    else
      echo "[RUN] sbatch --job-name=$job_name $SBATCH_SCRIPT $cfg"
      sbatch --job-name="$job_name" "$SBATCH_SCRIPT" "$cfg"
    fi
  done < "$TMP_LIST"
fi

echo "[DONE]"
