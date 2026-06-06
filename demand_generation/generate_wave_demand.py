from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Trip:
    person_id: str
    wave_idx: int
    depart: float
    trip_type: str
    template: str
    origin_cluster: str
    destination_cluster: str
    from_edge: str
    to_edge: str
    parking_area: str | None


def load_clusters(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Clusters JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("clusters-json must be a mapping {cluster_name: [edge_ids]}")

    clusters: dict[str, list[str]] = {}
    for name, edges in data.items():
        if not isinstance(name, str):
            raise ValueError("Cluster names must be strings")
        if not isinstance(edges, list):
            raise ValueError(f"Cluster '{name}' must map to a list of edge IDs")
        clusters[name] = [str(e) for e in edges if str(e).strip()]

    if not clusters:
        raise ValueError("No clusters loaded from clusters-json")
    return clusters


def get_builtin_templates() -> dict[str, list[tuple[str, str, float]]]:
    return {
        "AB": [("A", "B", 1.0)],
        "BA": [("B", "A", 1.0)],
        "AC": [("A", "C", 1.0)],
        "CA": [("C", "A", 1.0)],
        "BC": [("B", "C", 1.0)],
        "CB": [("C", "B", 1.0)],
        "AB_AC": [("A", "B", 0.5), ("A", "C", 0.5)],
        "BA_CA": [("B", "A", 0.5), ("C", "A", 0.5)],
        "AB_BC": [("A", "B", 0.5), ("B", "C", 0.5)],
        "CB_BA": [("C", "B", 0.5), ("B", "A", 0.5)],
        "AC_BC": [("A", "C", 0.5), ("B", "C", 0.5)],
        "CA_CB": [("C", "A", 0.5), ("C", "B", 0.5)],
    }


def allocate_counts(total: int, weighted_relations: list[tuple[str, str, float]]) -> list[int]:
    if total <= 0:
        return [0 for _ in weighted_relations]

    raw = [total * w for (_o, _d, w) in weighted_relations]
    base = [int(x) for x in raw]
    remainder = total - sum(base)

    frac_idx = sorted(
        range(len(raw)),
        key=lambda i: (raw[i] - base[i]),
        reverse=True,
    )
    for i in range(remainder):
        base[frac_idx[i % len(base)]] += 1
    return base


def sample_trip(
    clusters: dict[str, list[str]],
    origin_cluster: str,
    destination_cluster: str,
    rng: random.Random,
    max_tries: int = 30,
) -> tuple[str, str]:
    o_edges = clusters[origin_cluster]
    d_edges = clusters[destination_cluster]

    if not o_edges:
        raise ValueError(f"Origin cluster '{origin_cluster}' has no edges")
    if not d_edges:
        raise ValueError(f"Destination cluster '{destination_cluster}' has no edges")

    for _ in range(max_tries):
        from_edge = rng.choice(o_edges)
        to_edge = rng.choice(d_edges)
        if from_edge != to_edge:
            return from_edge, to_edge

    from_edge = rng.choice(o_edges)
    alt_to = next((e for e in d_edges if e != from_edge), None)
    if alt_to is None:
        raise RuntimeError(
            f"Could not sample distinct from/to for {origin_cluster}->{destination_cluster}"
        )
    return from_edge, alt_to


def random_trip_relation(
    clusters: dict[str, list[str]],
    mode: str,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    names = sorted(clusters.keys())

    if mode == "cluster-pairs":
        for _ in range(40):
            o = rng.choice(names)
            d = rng.choice(names)
            if o == d and len(clusters[o]) < 2:
                continue
            from_edge, to_edge = sample_trip(clusters, o, d, rng)
            return o, d, from_edge, to_edge
        raise RuntimeError("Could not sample random trip in cluster-pairs mode")

    # cluster-edges mode
    all_edges: list[tuple[str, str]] = []
    for cname, edges in clusters.items():
        all_edges.extend((cname, e) for e in edges)

    if len(all_edges) < 2:
        raise ValueError("Need at least 2 total edges for random-trip-mode=cluster-edges")

    for _ in range(60):
        o_cluster, from_edge = rng.choice(all_edges)
        d_cluster, to_edge = rng.choice(all_edges)
        if from_edge != to_edge:
            return o_cluster, d_cluster, from_edge, to_edge

    raise RuntimeError("Could not sample distinct random edges in cluster-edges mode")


def validate_args_and_inputs(
    args: argparse.Namespace,
    clusters: dict[str, list[str]],
    templates: dict[str, list[tuple[str, str, float]]],
    template_cycle: list[str],
) -> None:
    if args.wave_size_min > args.wave_size_max:
        raise ValueError("wave_size_min must be <= wave_size_max")
    if args.start_time >= args.end_time:
        raise ValueError("start_time must be < end_time")
    if not (0.0 <= args.random_trip_share <= 1.0):
        raise ValueError("random_trip_share must be in [0, 1]")
    if args.num_waves <= 0:
        raise ValueError("num_waves must be > 0")
    if args.wave_interval_min <= 0 or args.wave_interval_max <= 0:
        raise ValueError("wave intervals must be > 0")
    if args.wave_interval_min > args.wave_interval_max:
        raise ValueError("wave_interval_min must be <= wave_interval_max")

    for tname in template_cycle:
        if tname not in templates:
            raise ValueError(f"Unknown template in template-cycle: {tname}")

    needed_clusters: set[str] = set()
    for tname in template_cycle:
        for o, d, _w in templates[tname]:
            needed_clusters.add(o)
            needed_clusters.add(d)

    for c in sorted(needed_clusters):
        if c not in clusters:
            raise ValueError(f"Template references missing cluster '{c}'")
        if not clusters[c]:
            raise ValueError(f"Template references empty cluster '{c}'")

    possible_max = int((args.end_time - args.start_time) / max(args.wave_interval_min, 1e-9)) + 1
    if args.num_waves > possible_max:
        print(
            "[WARN] Requested num_waves may be unrealistic for time window and min interval: "
            f"requested={args.num_waves}, rough_max={possible_max}"
        )


def generate_waves(
    clusters: dict[str, list[str]],
    templates: dict[str, list[tuple[str, str, float]]],
    template_cycle: list[str],
    args: argparse.Namespace,
) -> list[Trip]:
    rng = random.Random(args.seed)

    trips: list[Trip] = []
    person_idx = 0
    t = float(args.start_time)
    wave_idx = 0

    while wave_idx < args.num_waves and t < args.end_time:
        template_name = template_cycle[wave_idx % len(template_cycle)]
        relations = templates[template_name]

        wave_size = rng.randint(args.wave_size_min, args.wave_size_max)
        num_random = int(round(wave_size * args.random_trip_share))
        num_random = max(0, min(wave_size, num_random))
        num_structured = wave_size - num_random

        relation_counts = allocate_counts(num_structured, relations)

        for (o_cluster, d_cluster, _w), cnt in zip(relations, relation_counts):
            for _ in range(cnt):
                from_edge, to_edge = sample_trip(clusters, o_cluster, d_cluster, rng)
                parking = f"{args.parking_area_prefix}{to_edge}" if args.include_parking_area else None
                trips.append(
                    Trip(
                        person_id=f"p{person_idx}",
                        wave_idx=wave_idx,
                        depart=t,
                        trip_type="structured",
                        template=template_name,
                        origin_cluster=o_cluster,
                        destination_cluster=d_cluster,
                        from_edge=from_edge,
                        to_edge=to_edge,
                        parking_area=parking,
                    )
                )
                person_idx += 1

        for _ in range(num_random):
            o_cluster, d_cluster, from_edge, to_edge = random_trip_relation(
                clusters, args.random_trip_mode, rng
            )
            parking = f"{args.parking_area_prefix}{to_edge}" if args.include_parking_area else None
            trips.append(
                Trip(
                    person_id=f"p{person_idx}",
                    wave_idx=wave_idx,
                    depart=t,
                    trip_type="random",
                    template=template_name,
                    origin_cluster=o_cluster,
                    destination_cluster=d_cluster,
                    from_edge=from_edge,
                    to_edge=to_edge,
                    parking_area=parking,
                )
            )
            person_idx += 1

        wave_idx += 1
        dt = rng.uniform(args.wave_interval_min, args.wave_interval_max)
        next_t = t + dt
        if next_t >= args.end_time:
            break
        t = next_t

    return trips


def write_xml(path: Path, trips: list[Trip], include_parking_area: bool) -> None:
    root = ET.Element(
        "routes",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    for tr in trips:
        person = ET.SubElement(root, "person", id=tr.person_id, depart=f"{tr.depart:.2f}")
        ride_attrib = {
            "lines": "taxi",
            "from": tr.from_edge,
            "to": tr.to_edge,
        }
        if include_parking_area and tr.parking_area is not None:
            ride_attrib["parkingArea"] = tr.parking_area
        ET.SubElement(person, "ride", attrib=ride_attrib)

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_csv(path: Path, trips: list[Trip]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "person_id",
                "wave_idx",
                "depart",
                "trip_type",
                "template",
                "origin_cluster",
                "destination_cluster",
                "from_edge",
                "to_edge",
                "parking_area",
            ]
        )
        for tr in trips:
            writer.writerow(
                [
                    tr.person_id,
                    tr.wave_idx,
                    f"{tr.depart:.2f}",
                    tr.trip_type,
                    tr.template,
                    tr.origin_cluster,
                    tr.destination_cluster,
                    tr.from_edge,
                    tr.to_edge,
                    tr.parking_area or "",
                ]
            )


def print_summary(trips: list[Trip]) -> None:
    if not trips:
        print("Generated 0 passengers (no wave fit within time bounds).")
        return

    wave_count = max(tr.wave_idx for tr in trips) + 1
    random_count = sum(1 for tr in trips if tr.trip_type == "random")
    by_template = Counter(tr.template for tr in trips)
    by_relation = Counter(f"{tr.origin_cluster}->{tr.destination_cluster}" for tr in trips)

    first_depart = min(tr.depart for tr in trips)
    last_depart = max(tr.depart for tr in trips)

    print(f"Waves generated: {wave_count}")
    print(f"Passengers generated: {len(trips)}")
    print(f"Random passengers: {random_count}")
    print(f"Structured passengers: {len(trips) - random_count}")
    print(f"First depart time: {first_depart:.2f}")
    print(f"Last depart time: {last_depart:.2f}")

    print("Counts per template:")
    for k, v in sorted(by_template.items()):
        print(f"  {k}: {v}")

    print("Counts per OD relation:")
    for k, v in sorted(by_relation.items()):
        print(f"  {k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate wave-based SUMO taxi/person demand from OD clusters")

    p.add_argument("--clusters-json", required=True, type=str)
    p.add_argument("--out-xml", required=True, type=str)
    p.add_argument("--out-csv", type=str, default=None)

    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--end-time", type=float, default=2400.0)
    p.add_argument("--num-waves", type=int, default=20)

    p.add_argument("--wave-size-min", type=int, default=6)
    p.add_argument("--wave-size-max", type=int, default=8)
    p.add_argument("--wave-interval-min", type=float, default=90.0)
    p.add_argument("--wave-interval-max", type=float, default=150.0)

    p.add_argument("--random-trip-share", type=float, default=0.10)
    p.add_argument(
        "--template-cycle",
        type=str,
        default="AB_AC,BA_CA,AB_BC,CB_BA,AC_BC,CA_CB",
    )
    p.add_argument(
        "--random-trip-mode",
        choices=["cluster-edges", "cluster-pairs"],
        default="cluster-edges",
    )
    p.add_argument("--seed", type=int, default=42)

    parking_group = p.add_mutually_exclusive_group()
    parking_group.add_argument("--include-parking-area", dest="include_parking_area", action="store_true")
    parking_group.add_argument("--no-include-parking-area", dest="include_parking_area", action="store_false")
    p.set_defaults(include_parking_area=True)

    p.add_argument("--parking-area-prefix", type=str, default="pa")
    return p


def main() -> None:
    args = build_parser().parse_args()

    clusters = load_clusters(Path(args.clusters_json))
    templates = get_builtin_templates()
    template_cycle = [x.strip() for x in args.template_cycle.split(",") if x.strip()]
    if not template_cycle:
        raise ValueError("template-cycle is empty")

    validate_args_and_inputs(args, clusters, templates, template_cycle)

    trips = generate_waves(clusters, templates, template_cycle, args)

    write_xml(Path(args.out_xml), trips, include_parking_area=args.include_parking_area)
    if args.out_csv:
        write_csv(Path(args.out_csv), trips)

    print_summary(trips)
    print(f"[OK] Wrote XML: {args.out_xml}")
    if args.out_csv:
        print(f"[OK] Wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
