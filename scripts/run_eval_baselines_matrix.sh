#!/usr/bin/env bash
set -euo pipefail

# Run eval_baselines.py across scenario/resolver combinations.
# For each pair, regenerate config files first (overwrite), then execute eval.

usage() {
  cat <<'EOF'
Usage:
  scripts/run_eval_baselines_matrix.sh [options]

Options:
  --scenarios "s1,s2,..."     Scenario aliases (default: randdest,corridor_asymmetric,wave,corridor_wave,corridor_mixed,corridor_noisy,corridor_hard)
  --resolvers "r1,r2,..."     Resolver names/aliases (default: capacity,closest,ctc)
  --policies "p1,p2,..."      Optional proposer policy list for generated YAML.
                               If omitted, generator uses all supported policies.
  --base-yaml <path>           Base YAML template (default: configs/rp_gnn.yaml)
  --base-sumocfg <path>        Base SUMO cfg template (default: configs/small_net.sumocfg)
  --output-dir <path>          Generated config output dir (default: configs)
  --python <bin>               Python executable (default: python)
  --dry-run                    Print commands without running eval.
  -h, --help                   Show this help.

Notes:
  - Generated files are overwritten each time.
  - eval_baselines.py reads proposer policies from generated YAML baselines.policies.
EOF
}

SCENARIOS="randdest,corridor_asymmetric,wave,corridor_wave,corridor_mixed,corridor_noisy,corridor_hard"
RESOLVERS="capacity,closest,ctc"
POLICIES=""
BASE_YAML="configs/rp_gnn.yaml"
BASE_SUMOCFG="configs/small_net.sumocfg"
OUTPUT_DIR="configs"
PYTHON_BIN="python"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenarios)
      SCENARIOS="$2"
      shift 2
      ;;
    --resolvers)
      RESOLVERS="$2"
      shift 2
      ;;
    --policies)
      POLICIES="$2"
      shift 2
      ;;
    --base-yaml)
      BASE_YAML="$2"
      shift 2
      ;;
    --base-sumocfg)
      BASE_SUMOCFG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

IFS=',' read -r -a SCENARIO_LIST <<< "$SCENARIOS"
IFS=',' read -r -a RESOLVER_LIST <<< "$RESOLVERS"

if [[ ${#SCENARIO_LIST[@]} -eq 0 || ${#RESOLVER_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] scenarios and resolvers must be non-empty"
  exit 1
fi

echo "[INFO] scenarios: ${SCENARIOS}"
echo "[INFO] resolvers: ${RESOLVERS}"
echo "[INFO] base-yaml: ${BASE_YAML}"
echo "[INFO] base-sumocfg: ${BASE_SUMOCFG}"
echo "[INFO] output-dir: ${OUTPUT_DIR}"

for scenario in "${SCENARIO_LIST[@]}"; do
  scenario="${scenario//[[:space:]]/}"
  [[ -n "$scenario" ]] || continue

  for resolver in "${RESOLVER_LIST[@]}"; do
    resolver="${resolver//[[:space:]]/}"
    [[ -n "$resolver" ]] || continue

    echo "[RUN] scenario=${scenario} resolver=${resolver}"

    GEN_ARGS=(
      "scripts/generate_baseline_eval_configs.py"
      "--scenario" "$scenario"
      "--resolver" "$resolver"
      "--base-yaml" "$BASE_YAML"
      "--base-sumocfg" "$BASE_SUMOCFG"
      "--output-dir" "$OUTPUT_DIR"
    )

    if [[ -n "$POLICIES" ]]; then
      IFS=',' read -r -a POLICY_LIST <<< "$POLICIES"
      GEN_ARGS+=("--policies")
      for p in "${POLICY_LIST[@]}"; do
        p="${p//[[:space:]]/}"
        [[ -n "$p" ]] && GEN_ARGS+=("$p")
      done
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] $PYTHON_BIN ${GEN_ARGS[*]}"
      continue
    fi

    "$PYTHON_BIN" "${GEN_ARGS[@]}"

    # Resolver alias for generated file naming.
    resolver_alias="$resolver"
    if [[ "$resolver" == "ctc" || "$resolver" == "closest_then_capacity" ]]; then
      resolver_alias="closest"
    fi

    generated_yaml="${OUTPUT_DIR}/rp_baseline_${scenario}_${resolver_alias}.yaml"
    if [[ ! -f "$generated_yaml" ]]; then
      echo "[ERROR] Generated YAML not found: $generated_yaml"
      exit 1
    fi

    echo "[RUN] $PYTHON_BIN eval_baselines.py --config $generated_yaml"
    "$PYTHON_BIN" eval_baselines.py --config "$generated_yaml"
  done
done

echo "[DONE] baseline evaluation matrix completed"
