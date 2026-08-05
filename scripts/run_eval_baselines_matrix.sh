#!/usr/bin/env bash
set -euo pipefail

# Run eval_baselines.py across scenario/resolver combinations.
# For each pair, regenerate config files first (overwrite), then execute eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
  --repo-root <path>           Repository root (default: auto-detected from script location)
  --base-yaml <path>           Base YAML template (default: configs/rp_gnn.yaml)
  --base-sumocfg <path>        Base SUMO cfg template (default: configs/small_net.sumocfg)
  --output-dir <path>          Generated config output dir (default: configs)
  --sumoport <int>             Optional SUMO remote port forwarded to eval_baselines.py
  --python <bin>               Python executable (default: python)
  --dry-run                    Print commands without running eval.
  -h, --help                   Show this help.

Notes:
  - Generated files are overwritten each time.
  - eval_baselines.py reads proposer policies from generated YAML baselines.policies.
  - eval outputs are written under the current working directory.
EOF
}

SCENARIOS="randdest,corridor_asymmetric,wave,corridor_wave,corridor_mixed,corridor_noisy,corridor_hard"
RESOLVERS="capacity,closest,ctc,predicted_reward,predicted_reward_joint,hungarian"
ADMISSION_AWARE="false"
ROUTE_CONSTRUCTION="all"
POLICIES=""
REPO_ROOT="$REPO_ROOT_DEFAULT"
BASE_YAML="${REPO_ROOT_DEFAULT}/configs/rp_gnn.yaml"
BASE_SUMOCFG="${REPO_ROOT_DEFAULT}/configs/small_net.sumocfg"
OUTPUT_DIR="configs"
SUMO_PORT=""
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
    --repo-root)
      REPO_ROOT="$2"
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
    --sumoport)
      SUMO_PORT="$2"
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

GENERATOR_SCRIPT="${REPO_ROOT}/scripts/generate_baseline_eval_configs.py"
EVAL_SCRIPT="${REPO_ROOT}/eval_baselines.py"

if [[ ! -f "$GENERATOR_SCRIPT" ]]; then
  echo "[ERROR] Generator script not found: $GENERATOR_SCRIPT"
  exit 1
fi
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] Eval script not found: $EVAL_SCRIPT"
  exit 1
fi
if [[ ! -f "$BASE_YAML" ]]; then
  echo "[ERROR] Base YAML not found: $BASE_YAML"
  exit 1
fi
if [[ ! -f "$BASE_SUMOCFG" ]]; then
  echo "[ERROR] Base SUMO cfg not found: $BASE_SUMOCFG"
  exit 1
fi

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
echo "[INFO] repo-root: ${REPO_ROOT}"
echo "[INFO] base-yaml: ${BASE_YAML}"
echo "[INFO] base-sumocfg: ${BASE_SUMOCFG}"
echo "[INFO] output-dir: ${OUTPUT_DIR}"
if [[ -n "$SUMO_PORT" ]]; then
  echo "[INFO] sumoport: ${SUMO_PORT}"
fi

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
          "$GENERATOR_SCRIPT"
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

        EVAL_ARGS=("$EVAL_SCRIPT" "--config" "$generated_yaml")
        if [[ -n "$SUMO_PORT" ]]; then
          EVAL_ARGS+=("--sumoport" "$SUMO_PORT")
        fi

        echo "[RUN] $PYTHON_BIN ${EVAL_ARGS[*]} (admission_aware=$admission route_construction=$route_mode)"
        "$PYTHON_BIN" "${EVAL_ARGS[@]}"
      done
    done
  done
done

echo "[DONE] baseline evaluation matrix completed"
