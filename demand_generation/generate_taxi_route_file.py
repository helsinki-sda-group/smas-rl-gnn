from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass
class TaxiInit:
    taxi_id: str
    depart: float
    from_edge: str
    to_edge: str
    init_cluster: str
    capacity: int


def load_clusters_json(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"clusters-json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("clusters-json must be an object mapping cluster names to edge lists")

    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            raise ValueError("Cluster names must be strings")
        if not isinstance(v, list):
            raise ValueError(f"Cluster '{k}' must map to a list")
        out[k] = [str(x) for x in v if str(x).strip()]
    return out


def parse_edge_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def read_edges_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Edge list file not found: {path}")
    edges = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        edges.append(s)
    return edges


def load_passenger_persons(path: Path, warnings: list[str]) -> list[ET.Element]:
    if not path.exists():
        raise FileNotFoundError(f"passengers-xml not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()
    tag = root.tag.split("}")[-1]

    persons: list[ET.Element] = []
    if tag == "routes":
        warnings.append("Passenger XML root is <routes>; importing only child <person> elements")
        for child in root:
            if child.tag.split("}")[-1] == "person":
                persons.append(ET.fromstring(ET.tostring(child, encoding="unicode")))
    elif tag == "person":
        persons.append(ET.fromstring(ET.tostring(root, encoding="unicode")))
    else:
        for p in root.findall(".//person"):
            persons.append(ET.fromstring(ET.tostring(p, encoding="unicode")))

    nested_routes = root.findall(".//routes")
    if nested_routes:
        warnings.append("Nested <routes> detected in passenger XML; imported only person elements")

    return persons


def repair_parking_areas(
    person_elements: list[ET.Element],
    prefix: str,
    include_parking_area: bool,
    repair: bool,
    strict: bool,
    warnings: list[str],
) -> tuple[int, int]:
    rides_total = 0
    rides_with_parking = 0

    for person in person_elements:
        for ride in person.findall("ride"):
            rides_total += 1
            parking = ride.get("parkingArea")
            to_edge = ride.get("to")

            if parking is not None:
                rides_with_parking += 1
                continue

            if include_parking_area and repair:
                if not to_edge:
                    msg = "Cannot repair parkingArea: ride missing 'to'"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
                    continue
                ride.set("parkingArea", f"{prefix}{to_edge}")
                rides_with_parking += 1
            else:
                if include_parking_area and strict:
                    raise ValueError("Missing parkingArea in passenger ride (strict mode)")

    return rides_total, rides_with_parking


def require_clusters(clusters: dict[str, list[str]] | None, reason: str) -> dict[str, list[str]]:
    if clusters is None:
        raise ValueError(f"--clusters-json is required for {reason}")
    return clusters


def choose_cluster_subset(
    clusters: dict[str, list[str]],
    cluster_names: list[str],
    strict: bool,
    warnings: list[str],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in cluster_names:
        if c not in clusters:
            msg = f"Missing cluster '{c}'"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            continue
        if not clusters[c]:
            msg = f"Empty cluster '{c}'"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            continue
        out[c] = clusters[c]

    if not out:
        raise ValueError("No usable clusters after validation")
    return out


def cycle_list(items: list[str], n: int, strict: bool, warnings: list[str], label: str) -> list[str]:
    if not items:
        raise ValueError(f"No values provided for {label}")
    if len(items) >= n:
        return items[:n]
    if strict:
        raise ValueError(f"Not enough values for {label}: need {n}, got {len(items)}")
    warnings.append(f"Not enough values for {label}; cycling to match num-taxis")
    return [items[i % len(items)] for i in range(n)]


def taxi_from_edges(
    args: argparse.Namespace,
    clusters: dict[str, list[str]] | None,
    rng: random.Random,
    warnings: list[str],
) -> list[tuple[str, str]]:
    n = args.num_taxis

    if args.taxi_init_mode in {"cluster-balanced", "cluster-random"}:
        cdata = require_clusters(clusters, f"taxi-init-mode={args.taxi_init_mode}")
        names = [x.strip() for x in args.taxi_init_clusters.split(",") if x.strip()]
        if not names:
            raise ValueError("--taxi-init-clusters is empty")

        selected = choose_cluster_subset(cdata, names, args.strict, warnings)
        c_names = sorted(selected.keys())

        if args.taxi_init_mode == "cluster-random":
            out = []
            for _ in range(n):
                c = rng.choice(c_names)
                out.append((rng.choice(selected[c]), c))
            return out

        # cluster-balanced
        q, r = divmod(n, len(c_names))
        alloc = {c: q for c in c_names}
        for i in range(r):
            alloc[c_names[i]] += 1

        out = []
        for c in c_names:
            for _ in range(alloc[c]):
                out.append((rng.choice(selected[c]), c))
        return out

    if args.taxi_init_mode == "edge-list":
        if not args.taxi_from_edges_file:
            raise ValueError("--taxi-from-edges-file is required for taxi-init-mode=edge-list")
        edges = read_edges_file(Path(args.taxi_from_edges_file))
        use = cycle_list(edges, n, args.strict, warnings, "taxi from-edges")
        return [(e, "") for e in use]

    if args.taxi_init_mode == "manual":
        edges = parse_edge_csv(args.taxi_from_edges)
        if not edges:
            raise ValueError("--taxi-from-edges is required for taxi-init-mode=manual")
        use = cycle_list(edges, n, args.strict, warnings, "manual taxi from-edges")
        return [(e, "") for e in use]

    raise ValueError(f"Unsupported taxi-init-mode: {args.taxi_init_mode}")


def taxi_to_edges(
    args: argparse.Namespace,
    from_edges: list[tuple[str, str]],
    clusters: dict[str, list[str]] | None,
    rng: random.Random,
    warnings: list[str],
) -> list[str]:
    n = len(from_edges)

    if args.taxi_dest_mode == "same-as-from":
        warnings.append("taxi-dest-mode=same-as-from may produce from==to taxi trips")
        return [f for f, _c in from_edges]

    if args.taxi_dest_mode == "manual":
        edges = []
        if args.taxi_to_edges:
            edges = parse_edge_csv(args.taxi_to_edges)
        elif args.taxi_to_edges_file:
            edges = read_edges_file(Path(args.taxi_to_edges_file))
        if not edges:
            raise ValueError("manual taxi-dest-mode requires --taxi-to-edges or --taxi-to-edges-file")
        return cycle_list(edges, n, args.strict, warnings, "taxi to-edges")

    if args.taxi_dest_mode in {"random-valid", "cluster-random"}:
        cdata = require_clusters(clusters, f"taxi-dest-mode={args.taxi_dest_mode}")
        all_edges = [e for arr in cdata.values() for e in arr]
        if not all_edges:
            raise ValueError("No cluster edges available for taxi destination sampling")

        out: list[str] = []
        init_names = [x.strip() for x in args.taxi_init_clusters.split(",") if x.strip()]
        valid_init_names = [c for c in init_names if c in cdata and cdata[c]]
        if not valid_init_names:
            valid_init_names = [c for c in cdata if cdata[c]]

        for idx, (from_edge, init_cluster) in enumerate(from_edges):
            if args.taxi_dest_mode == "cluster-random":
                c = init_cluster if init_cluster in cdata and cdata[init_cluster] else rng.choice(valid_init_names)
                pool = cdata[c]
            else:
                pool = all_edges

            chosen = None
            for _ in range(60):
                cand = rng.choice(pool)
                if cand != from_edge:
                    chosen = cand
                    break

            if chosen is None:
                msg = f"Could not enforce from!=to for taxi index {idx}"
                if args.strict:
                    raise ValueError(msg)
                warnings.append(msg)
                chosen = rng.choice(pool)
            out.append(chosen)

        return out

    raise ValueError(f"Unsupported taxi-dest-mode: {args.taxi_dest_mode}")


def sample_taxi_edges(
    clusters: dict[str, list[str]] | None,
    args: argparse.Namespace,
    rng: random.Random,
    warnings: list[str],
) -> list[TaxiInit]:
    from_list = taxi_from_edges(args, clusters, rng, warnings)
    to_list = taxi_to_edges(args, from_list, clusters, rng, warnings)

    taxis: list[TaxiInit] = []
    for i, ((from_edge, init_cluster), to_edge) in enumerate(zip(from_list, to_list)):
        if not from_edge or not to_edge:
            raise ValueError("Sampled empty taxi from/to edge")
        if from_edge == to_edge:
            msg = f"Taxi {i} has from==to ({from_edge})"
            if args.strict and args.taxi_dest_mode != "same-as-from":
                raise ValueError(msg)
            warnings.append(msg)

        taxis.append(
            TaxiInit(
                taxi_id=f"{args.taxi_id_prefix}{i}",
                depart=float(args.taxi_depart),
                from_edge=from_edge,
                to_edge=to_edge,
                init_cluster=init_cluster,
                capacity=int(args.taxi_capacity),
            )
        )
    return taxis


def build_vtype(args: argparse.Namespace) -> ET.Element:
    vtype = ET.Element(
        "vType",
        id="taxi",
        vClass="taxi",
        color=str(args.taxi_color),
        personCapacity=str(int(args.taxi_capacity)),
    )
    ET.SubElement(vtype, "param", key="has.taxi.device", value="true")
    ET.SubElement(vtype, "param", key="device.taxi.pickUpDuration", value=str(int(args.pickup_duration)))
    ET.SubElement(vtype, "param", key="device.taxi.dropOffDuration", value=str(int(args.dropoff_duration)))
    ET.SubElement(vtype, "param", key="device.taxi.parking", value="true" if args.taxi_parking else "false")
    return vtype


def build_taxi_trips(taxis: list[TaxiInit]) -> list[ET.Element]:
    trips: list[ET.Element] = []
    for tx in taxis:
        depart_s = f"{tx.depart:.0f}" if tx.depart.is_integer() else f"{tx.depart:.2f}"
        trips.append(
            ET.Element(
                "trip",
                id=tx.taxi_id,
                type="taxi",
                depart=depart_s,
                **{"from": tx.from_edge, "to": tx.to_edge},
            )
        )
    return trips


def write_route_file(out_path: Path, vtype: ET.Element, taxi_trips: list[ET.Element], persons: list[ET.Element]) -> None:
    root = ET.Element(
        "routes",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    root.append(vtype)
    for trip in taxi_trips:
        root.append(trip)
    for p in persons:
        root.append(p)

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def write_taxi_csv(path: Path, taxis: list[TaxiInit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["taxi_id", "depart", "from_edge", "to_edge", "init_cluster", "capacity"])
        for t in taxis:
            w.writerow([t.taxi_id, f"{t.depart:.2f}", t.from_edge, t.to_edge, t.init_cluster, t.capacity])


def print_summary(
    out_path: Path,
    args: argparse.Namespace,
    taxis: list[TaxiInit],
    persons: list[ET.Element],
    rides_total: int,
    rides_with_parking: int,
    warnings: list[str],
) -> None:
    cluster_counts = Counter(t.init_cluster for t in taxis if t.init_cluster)

    print(f"Output file: {out_path}")
    print(f"Taxi capacity: {args.taxi_capacity}")
    print(f"Number of taxis: {len(taxis)}")

    if cluster_counts:
        print("Taxi initial clusters and counts:")
        for c, n in sorted(cluster_counts.items()):
            print(f"  {c}: {n}")

    print(f"Imported/generated passengers: {len(persons)}")
    print(f"Passenger rides with parkingArea: {rides_with_parking}/{rides_total}")

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create complete SUMO .rou.xml route file for ride-pooling")

    p.add_argument("--out-rou", required=True, type=str)
    p.add_argument("--num-taxis", required=True, type=int)

    p.add_argument("--clusters-json", type=str, default=None)
    p.add_argument("--passengers-xml", type=str, default=None)

    p.add_argument("--taxi-capacity", type=int, default=2)
    p.add_argument("--taxi-color", type=str, default="green")
    p.add_argument("--pickup-duration", type=int, default=0)
    p.add_argument("--dropoff-duration", type=int, default=10)

    taxi_parking = p.add_mutually_exclusive_group()
    taxi_parking.add_argument("--taxi-parking", dest="taxi_parking", action="store_true")
    taxi_parking.add_argument("--no-taxi-parking", dest="taxi_parking", action="store_false")
    p.set_defaults(taxi_parking=True)

    p.add_argument("--taxi-depart", type=float, default=0.0)
    p.add_argument("--taxi-id-prefix", type=str, default="t")

    p.add_argument(
        "--taxi-init-mode",
        choices=["cluster-balanced", "cluster-random", "edge-list", "manual"],
        default="cluster-balanced",
    )
    p.add_argument("--taxi-init-clusters", type=str, default="A,B,C")
    p.add_argument("--taxi-from-edges-file", type=str, default=None)
    p.add_argument("--taxi-from-edges", type=str, default=None)

    p.add_argument(
        "--taxi-dest-mode",
        choices=["random-valid", "same-as-from", "cluster-random", "manual"],
        default="random-valid",
    )
    p.add_argument("--taxi-to-edges-file", type=str, default=None)
    p.add_argument("--taxi-to-edges", type=str, default=None)

    include_pa = p.add_mutually_exclusive_group()
    include_pa.add_argument("--include-parking-area", dest="include_parking_area", action="store_true")
    include_pa.add_argument("--no-include-parking-area", dest="include_parking_area", action="store_false")
    p.set_defaults(include_parking_area=True)

    p.add_argument("--parking-area-prefix", type=str, default="pa")
    p.add_argument("--repair-parking-areas", action="store_true")

    p.add_argument("--out-taxi-csv", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.num_taxis <= 0:
        raise ValueError("num-taxis must be > 0")
    if args.taxi_capacity <= 0:
        raise ValueError("taxi-capacity must be > 0")

    warnings: list[str] = []

    clusters = load_clusters_json(Path(args.clusters_json)) if args.clusters_json else None

    rng = random.Random(args.seed)
    taxis = sample_taxi_edges(clusters, args, rng, warnings)

    persons: list[ET.Element] = []
    if args.passengers_xml:
        persons = load_passenger_persons(Path(args.passengers_xml), warnings)
        if not persons:
            msg = "No passenger <person> elements imported"
            if args.strict:
                raise ValueError(msg)
            warnings.append(msg)
    else:
        warnings.append("No passengers-xml provided; route file will contain only taxis")

    rides_total, rides_with_parking = repair_parking_areas(
        person_elements=persons,
        prefix=args.parking_area_prefix,
        include_parking_area=args.include_parking_area,
        repair=args.repair_parking_areas,
        strict=args.strict,
        warnings=warnings,
    )

    vtype = build_vtype(args)
    taxi_trips = build_taxi_trips(taxis)

    out_path = Path(args.out_rou)
    write_route_file(out_path, vtype, taxi_trips, persons)

    if args.out_taxi_csv:
        write_taxi_csv(Path(args.out_taxi_csv), taxis)

    print_summary(
        out_path=out_path,
        args=args,
        taxis=taxis,
        persons=persons,
        rides_total=rides_total,
        rides_with_parking=rides_with_parking,
        warnings=warnings,
    )
    if args.out_taxi_csv:
        print(f"Taxi CSV: {args.out_taxi_csv}")


if __name__ == "__main__":
    main()
