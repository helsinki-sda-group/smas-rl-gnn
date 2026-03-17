from __future__ import annotations

import argparse
import random
from pathlib import Path
import xml.etree.ElementTree as ET


def load_valid_edges(net_path: Path) -> list[str]:
    tree = ET.parse(net_path)
    root = tree.getroot()
    valid_edges: list[str] = []
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id:
            continue
        if edge_id.startswith(":"):
            continue
        edge_function = edge.get("function", "")
        if edge_function == "internal":
            continue
        valid_edges.append(edge_id)
    if not valid_edges:
        raise ValueError(f"No valid non-internal edges found in net file: {net_path}")
    return valid_edges


def randomize_destinations(
    baseline_rou: Path,
    net_path: Path,
    output_path: Path,
    seed: int | None = None,
) -> None:
    rng = random.Random(seed)
    valid_edges = load_valid_edges(net_path)

    tree = ET.parse(baseline_rou)
    root = tree.getroot()

    num_input_rides = 0
    num_updated = 0
    for person in root.findall("person"):
        for ride in person.findall("ride"):
            num_input_rides += 1
            to_edge = rng.choice(valid_edges)
            ride.set("to", to_edge)
            ride.set("parkingArea", f"pa{to_edge}")
            num_updated += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    if num_updated != num_input_rides:
        raise RuntimeError(
            f"Ride count changed unexpectedly: input={num_input_rides}, updated={num_updated}"
        )
    print(f"[OK] Updated {num_updated} ride destinations")
    print(f"[OK] Wrote: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomize passenger ride destinations in a SUMO routes file"
    )
    parser.add_argument(
        "baseline_rou",
        type=str,
        help="Path to baseline .rou.xml file",
    )
    parser.add_argument(
        "net_xml",
        type=str,
        help="Path to SUMO .net.xml file used for valid edge ids",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: <baseline_stem>_rand_dest.xml in same folder)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility",
    )
    args = parser.parse_args()

    baseline_rou = Path(args.baseline_rou)
    net_xml = Path(args.net_xml)
    if not baseline_rou.exists():
        raise FileNotFoundError(f"Baseline routes file not found: {baseline_rou}")
    if not net_xml.exists():
        raise FileNotFoundError(f"Net file not found: {net_xml}")

    if args.output is None:
        output_path = baseline_rou.with_name(f"{baseline_rou.stem}_rand_dest.xml")
    else:
        output_path = Path(args.output)

    randomize_destinations(baseline_rou, net_xml, output_path, seed=args.seed)


if __name__ == "__main__":
    main()
