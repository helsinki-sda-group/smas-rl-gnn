#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

SUPPORTED_BASELINE_POLICIES: List[str] = [
    "random",
    "unique",
    "greedy",
    "pickup_distance",
    "pickup_deadline",
    "pickup_deadline_distance",
    "predicted_reward",
    "predicted_reward_joint",
    "predicted_reward_joint_competition",
    "proposal_joint_competition",
]

SCENARIO_ROUTE_MAP: Dict[str, str] = {
    "randdest": "coordination_medium_rand_dest_cap2.xml",
    "corridor_asymmetric": "corridor_asymmetric_cap2_taxis6.rou.xml",
    "wave": "wave_demand_cap2_taxis6.rou.xml",
    "corridor_wave": "corridor_wave_cap2_taxis6.rou.xml",
    "corridor_mixed": "corridor_mixed_cap2_taxis6.rou.xml",
    "corridor_noisy": "corridor_noisy_cap2_taxis6.rou.xml",
    "corridor_hard": "corridor_hard_cap2_taxis6.rou.xml",
}

SCENARIO_ALIASES: Dict[str, str] = {
    "rand_dest": "randdest",
    "randdest": "randdest",
    "asymmetric": "corridor_asymmetric",
    "corridor_asymmetric": "corridor_asymmetric",
    "wave": "wave",
    "corridor_wave": "corridor_wave",
    "mixed": "corridor_mixed",
    "corridor_mixed": "corridor_mixed",
    "noisy": "corridor_noisy",
    "corridor_noisy": "corridor_noisy",
    "hard": "corridor_hard",
    "corridor_hard": "corridor_hard",
}

RESOLVER_ALIASES: Dict[str, str] = {
    "capacity": "capacity",
    "closest": "closest",
    "ctc": "closest_then_capacity",
    "closest_then_capacity": "closest_then_capacity",
    "logit_diff": "logit_diff",
    "logitdiff": "logit_diff",
    "random": "random",
}


def _normalize_scenario(value: str) -> str:
    key = str(value).strip().lower()
    if key in SCENARIO_ALIASES:
        return SCENARIO_ALIASES[key]

    for token, alias in [
        ("rand_dest", "randdest"),
        ("corridor_wave", "corridor_wave"),
        ("asymmetric", "corridor_asymmetric"),
        ("wave", "wave"),
        ("mixed", "corridor_mixed"),
        ("noisy", "corridor_noisy"),
        ("hard", "corridor_hard"),
    ]:
        if token in key:
            return alias

    raise ValueError(
        f"Unsupported scenario '{value}'. Supported aliases: {sorted(SCENARIO_ROUTE_MAP.keys())}."
    )


def _normalize_resolver(value: str) -> str:
    key = str(value).strip().lower()
    if key not in RESOLVER_ALIASES:
        raise ValueError(
            "Unsupported resolver "
            f"'{value}'. Supported: {sorted(set(RESOLVER_ALIASES.keys()))}."
        )
    return RESOLVER_ALIASES[key]


def _resolver_filename_alias(resolver: str) -> str:
    return "closest" if resolver == "closest_then_capacity" else resolver


def _resolve_policies(raw_policies: List[str] | None) -> List[str]:
    if not raw_policies:
        return list(SUPPORTED_BASELINE_POLICIES)

    policies = [p.strip() for p in raw_policies if p.strip()]
    invalid = sorted(set(p for p in policies if p not in SUPPORTED_BASELINE_POLICIES))
    if invalid:
        raise ValueError(
            f"Unsupported policy names: {invalid}. Supported: {SUPPORTED_BASELINE_POLICIES}."
        )
    return policies


def _load_and_patch_sumocfg(base_sumocfg: Path, route_file_name: str) -> ET.ElementTree:
    tree = ET.parse(base_sumocfg)
    root = tree.getroot()
    route_node = root.find("./input/route-files")
    if route_node is None:
        raise ValueError(f"No <route-files> element found in {base_sumocfg}")
    route_node.set("value", route_file_name)
    return tree


def generate_configs(
    *,
    scenario: str,
    resolver_input: str,
    base_yaml: Path,
    base_sumocfg: Path,
    output_dir: Path,
    policies: List[str] | None,
) -> tuple[Path, Path, str, str]:
    scenario_alias = _normalize_scenario(scenario)
    resolver = _normalize_resolver(resolver_input)
    resolver_alias = _resolver_filename_alias(resolver)

    if scenario_alias not in SCENARIO_ROUTE_MAP:
        raise ValueError(
            f"Scenario alias '{scenario_alias}' has no route mapping. "
            f"Known: {sorted(SCENARIO_ROUTE_MAP.keys())}."
        )

    selected_policies = _resolve_policies(policies)
    route_file_name = SCENARIO_ROUTE_MAP[scenario_alias]

    configs_dir = output_dir
    configs_dir.mkdir(parents=True, exist_ok=True)

    generated_sumocfg_name = f"small_net_cap2_{scenario_alias}_{resolver_alias}.sumocfg"
    generated_sumocfg_path = configs_dir / generated_sumocfg_name

    generated_yaml_name = f"rp_baseline_{scenario_alias}_{resolver_alias}.yaml"
    generated_yaml_path = configs_dir / generated_yaml_name

    route_path = base_sumocfg.parent / route_file_name
    if not route_path.exists():
        raise FileNotFoundError(f"Mapped route file not found: {route_path}")

    sumo_tree = _load_and_patch_sumocfg(base_sumocfg, route_file_name)
    sumo_tree.write(generated_sumocfg_path, encoding="UTF-8", xml_declaration=True)

    cfg = OmegaConf.load(base_yaml)
    cfg.env.sumo_cfg = str(Path(configs_dir.name) / generated_sumocfg_name).replace("\\", "/")
    cfg.env.conflict_resolution = resolver
    cfg.baselines.policies = selected_policies

    OmegaConf.save(cfg, generated_yaml_path)

    return generated_yaml_path, generated_sumocfg_path, scenario_alias, resolver_alias


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate baseline evaluation YAML + SUMO config for one scenario/resolver pair. "
            "Outputs: rp_baseline_[scenario]_[resolver].yaml and "
            "small_net_cap2_[scenario]_[resolver].sumocfg"
        )
    )
    parser.add_argument("--scenario", required=True, help="Scenario alias (e.g. randdest, corridor_asymmetric)")
    parser.add_argument("--resolver", required=True, help="Resolver name or alias (e.g. capacity, closest, ctc)")
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help="Baseline proposer policy names. Default: all supported policies.",
    )
    parser.add_argument(
        "--base-yaml",
        default="configs/rp_gnn.yaml",
        help="Template YAML config. Default: configs/rp_gnn.yaml",
    )
    parser.add_argument(
        "--base-sumocfg",
        default="configs/small_net.sumocfg",
        help="Template SUMO cfg. Default: configs/small_net.sumocfg",
    )
    parser.add_argument(
        "--output-dir",
        default="configs",
        help="Directory for generated files. Default: configs",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        yaml_path, sumocfg_path, scenario_alias, resolver_alias = generate_configs(
            scenario=args.scenario,
            resolver_input=args.resolver,
            base_yaml=Path(args.base_yaml),
            base_sumocfg=Path(args.base_sumocfg),
            output_dir=Path(args.output_dir),
            policies=args.policies,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Generated YAML: {yaml_path}")
    print(f"[OK] Generated SUMO cfg: {sumocfg_path}")
    print(f"[INFO] scenario={scenario_alias} resolver={resolver_alias}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
