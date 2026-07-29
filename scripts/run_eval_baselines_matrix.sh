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
  --resolvers "r1,r2,..."     Resolver names/aliases (default: capacity,closest,ctc,predicted_reward,predicted_reward_joint,hungarian)
  --admission-aware "v1,v2"   Admission-aware values (default: false). Allowed: false,true
  --route-construction "m1,m2" Route-construction values. Allowed: nearest,reward_aligned,all (default: all)
  --policies "p1,p2,..."      Optional proposer policy list for generated YAML.
                               If omitted, generator uses canonical default policies
                               (alias duplicates greedy and predicted_reward_joint_competition are excluded).
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
RESOLVERS="capacity,closest,ctc,predicted_reward,predicted_reward_joint,hungarian"
ADMISSION_AWARE="false"
ROUTE_CONSTRUCTION="all"
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
    --admission-aware)
      ADMISSION_AWARE="$2"
      shift 2
      ;;
    --route-construction)
      ROUTE_CONSTRUCTION="$2"
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
IFS=',' read -r -a ADMISSION_LIST <<< "$ADMISSION_AWARE"
IFS=',' read -r -a ROUTE_LIST_RAW <<< "$ROUTE_CONSTRUCTION"

ROUTE_LIST=()
for route_mode in "${ROUTE_LIST_RAW[@]}"; do
  route_mode="${route_mode//[[:space:]]/}"
  [[ -n "$route_mode" ]] || continue
  if [[ "$route_mode" == "all" ]]; then
    ROUTE_LIST+=("nearest" "reward_aligned")
    continue
  fi
  if [[ "$route_mode" != "nearest" && "$route_mode" != "reward_aligned" ]]; then
    echo "[ERROR] route-construction value must be nearest, reward_aligned, or all (got: $route_mode)"
    exit 1
  fi
  ROUTE_LIST+=("$route_mode")
done

if [[ ${#ROUTE_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] route-construction values must be non-empty"
  exit 1
fi

if [[ ${#SCENARIO_LIST[@]} -eq 0 || ${#RESOLVER_LIST[@]} -eq 0 || ${#ADMISSION_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] scenarios, resolvers, and admission-aware values must be non-empty"
  exit 1
fi

echo "[INFO] scenarios: ${SCENARIOS}"
echo "[INFO] resolvers: ${RESOLVERS}"
echo "[INFO] admission-aware: ${ADMISSION_AWARE}"
echo "[INFO] route-construction: ${ROUTE_CONSTRUCTION}"
echo "[INFO] base-yaml: ${BASE_YAML}"
echo "[INFO] base-sumocfg: ${BASE_SUMOCFG}"
echo "[INFO] output-dir: ${OUTPUT_DIR}"

for scenario in "${SCENARIO_LIST[@]}"; do
  scenario="${scenario//[[:space:]]/}"
  [[ -n "$scenario" ]] || continue

  for resolver in "${RESOLVER_LIST[@]}"; do
    resolver="${resolver//[[:space:]]/}"
    [[ -n "$resolver" ]] || continue

    for admission in "${ADMISSION_LIST[@]}"; do
      admission="${admission//[[:space:]]/}"
      [[ -n "$admission" ]] || continue
      if [[ "$admission" != "false" && "$admission" != "true" ]]; then
        echo "[ERROR] admission-aware value must be false or true (got: $admission)"
        exit 1
      fi

      for route_mode in "${ROUTE_LIST[@]}"; do
        echo "[RUN] scenario=${scenario} resolver=${resolver} admission_aware=${admission} route_construction=${route_mode}"

        GEN_ARGS=(
          "scripts/generate_baseline_eval_configs.py"
          "--scenario" "$scenario"
          "--resolver" "$resolver"
          "--admission-aware" "$admission"
          "--route-construction" "$route_mode"
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

        echo "[RUN] $PYTHON_BIN eval_baselines.py --config $generated_yaml (admission_aware=$admission route_construction=$route_mode)"
        "$PYTHON_BIN" eval_baselines.py --config "$generated_yaml"
      done
    done
  done
done

echo "[DONE] baseline evaluation matrix completed"
