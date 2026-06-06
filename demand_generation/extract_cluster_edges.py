from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import sumolib  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "sumolib is required. Ensure SUMO tools are installed and PYTHONPATH includes SUMO_HOME/tools"
    ) from exc


Point = tuple[float, float]

DEFAULT_COLORS = [
    "#ff0055",
    "#00e5ff",
    "#39ff14",
    "#ffea00",
    "#ff6b00",
    "#b300ff",
    "#00ff9d",
    "#ff2bd6",
]


@dataclass(frozen=True)
class Polygon:
    name: str
    points: tuple[Point, ...]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class Stats:
    total_edges: int = 0
    filtered_edges: int = 0
    fallback_lane_shape_count: int = 0
    fallback_node_coords_count: int = 0
    ambiguous_permission_checks: int = 0


COORD_RE = re.compile(r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def parse_clusters(path: Path, strict: bool = False) -> dict[str, Polygon]:
    if not path.exists():
        raise FileNotFoundError(f"Cluster file not found: {path}")

    clusters: dict[str, Polygon] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                msg = f"Malformed cluster line {lineno}: missing ':'"
                if strict:
                    raise ValueError(msg)
                warn(msg)
                continue

            name_raw, coord_raw = line.split(":", 1)
            name = name_raw.strip()
            if not name:
                msg = f"Malformed cluster line {lineno}: empty cluster name"
                if strict:
                    raise ValueError(msg)
                warn(msg)
                continue

            matches = COORD_RE.findall(coord_raw)
            if len(matches) < 3:
                msg = f"Malformed cluster line {lineno}: expected at least 3 coordinate pairs"
                if strict:
                    raise ValueError(msg)
                warn(msg)
                continue

            points = tuple((float(x), float(y)) for x, y in matches)
            if name in clusters:
                msg = f"Duplicate cluster name '{name}' at line {lineno}"
                if strict:
                    raise ValueError(msg)
                warn(msg)
                continue

            clusters[name] = Polygon(name=name, points=points)

    if not clusters:
        raise ValueError(f"No valid clusters loaded from {path}")
    return clusters


def load_net(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"SUMO net file not found: {path}")
    return sumolib.net.readNet(str(path))


def get_all_edges(net) -> list:
    try:
        return list(net.getEdges(withInternal=True))
    except TypeError:
        return list(net.getEdges())


def edge_is_internal(edge) -> bool:
    edge_id = edge.getID()
    if edge_id.startswith(":"):
        return True
    get_function = getattr(edge, "getFunction", None)
    if callable(get_function):
        try:
            return get_function() == "internal"
        except Exception:
            return False
    return False


def edge_geometry(edge, stats: Stats) -> list[Point]:
    shape = []
    get_shape = getattr(edge, "getShape", None)
    if callable(get_shape):
        try:
            shape = list(get_shape())
        except Exception:
            shape = []
    if shape:
        return [(float(x), float(y)) for x, y in shape]

    lanes = list(edge.getLanes())
    if lanes:
        lane_shape = []
        lane_get_shape = getattr(lanes[0], "getShape", None)
        if callable(lane_get_shape):
            try:
                lane_shape = list(lane_get_shape())
            except Exception:
                lane_shape = []
        if lane_shape:
            stats.fallback_lane_shape_count += 1
            return [(float(x), float(y)) for x, y in lane_shape]

    fx, fy = edge.getFromNode().getCoord()
    tx, ty = edge.getToNode().getCoord()
    stats.fallback_node_coords_count += 1
    return [(float(fx), float(fy)), (float(tx), float(ty))]


def polyline_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def midpoint_on_polyline(points: Sequence[Point]) -> Point:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]

    total = polyline_length(points)
    if total <= 0.0:
        return points[len(points) // 2]

    target = total / 2.0
    walked = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg <= 0:
            continue
        if walked + seg >= target:
            r = (target - walked) / seg
            return (x1 + r * (x2 - x1), y1 + r * (y2 - y1))
        walked += seg
    return points[-1]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    x, y = point
    pts = polygon.points
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_on_edge = (x2 - x1) * (y - y1) / ((y2 - y1) if y2 != y1 else 1e-12) + x1
            if x < x_on_edge:
                inside = not inside
    return inside


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
    )


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if abs(o1) < 1e-9 and on_segment(a1, a2, b1):
        return True
    if abs(o2) < 1e-9 and on_segment(a1, a2, b2):
        return True
    if abs(o3) < 1e-9 and on_segment(b1, b2, a1):
        return True
    if abs(o4) < 1e-9 and on_segment(b1, b2, a2):
        return True

    return False


def segment_intersects_polygon(seg_a: Point, seg_b: Point, polygon: Polygon) -> bool:
    pts = polygon.points
    for i in range(len(pts)):
        pa = pts[i]
        pb = pts[(i + 1) % len(pts)]
        if segments_intersect(seg_a, seg_b, pa, pb):
            return True
    return False


def edge_in_cluster(points: Sequence[Point], polygon: Polygon, membership_mode: str) -> bool:
    if not points:
        return False

    if membership_mode == "midpoint":
        return point_in_polygon(midpoint_on_polyline(points), polygon)

    if membership_mode == "any-point":
        return any(point_in_polygon(p, polygon) for p in points)

    if membership_mode == "segment-intersects":
        if any(point_in_polygon(p, polygon) for p in points):
            return True
        for i in range(1, len(points)):
            if segment_intersects_polygon(points[i - 1], points[i], polygon):
                return True
        return False

    raise ValueError(f"Unsupported membership mode: {membership_mode}")


def edge_allows_vclass(edge, vclass: str) -> bool | None:
    allows = getattr(edge, "allows", None)
    if callable(allows):
        try:
            return bool(allows(vclass))
        except Exception:
            pass

    lanes = list(edge.getLanes())
    if not lanes:
        return None

    saw_decision = False
    for lane in lanes:
        lane_allows = getattr(lane, "allows", None)
        if callable(lane_allows):
            try:
                if lane_allows(vclass):
                    return True
                saw_decision = True
            except Exception:
                continue
    if saw_decision:
        return False
    return None


def edge_length(edge, points: Sequence[Point]) -> float:
    get_length = getattr(edge, "getLength", None)
    if callable(get_length):
        try:
            return float(get_length())
        except Exception:
            pass
    return polyline_length(points)


def edge_passes_filters(edge, points: Sequence[Point], args: argparse.Namespace, stats: Stats) -> bool:
    if args.exclude_internal and edge_is_internal(edge):
        return False

    length = edge_length(edge, points)
    if length < args.min_length:
        return False
    if args.max_length is not None and length > args.max_length:
        return False

    if args.allow_vclass:
        results = [edge_allows_vclass(edge, vc) for vc in args.allow_vclass]
        if any(r is True for r in results):
            pass
        elif all(r is False for r in results):
            return False
        else:
            stats.ambiguous_permission_checks += 1
            if args.strict:
                raise RuntimeError(
                    f"Ambiguous allow-vclass check for edge {edge.getID()} and {args.allow_vclass}"
                )

    if args.disallow_vclass:
        results = [edge_allows_vclass(edge, vc) for vc in args.disallow_vclass]
        if any(r is True for r in results):
            return False
        if any(r is None for r in results):
            stats.ambiguous_permission_checks += 1
            if args.strict:
                raise RuntimeError(
                    f"Ambiguous disallow-vclass check for edge {edge.getID()} and {args.disallow_vclass}"
                )

    if args.require_drivable:
        drivable_classes = args.allow_vclass if args.allow_vclass else ["passenger", "taxi"]
        results = [edge_allows_vclass(edge, vc) for vc in drivable_classes]
        if any(r is True for r in results):
            return True
        if all(r is False for r in results):
            return False
        stats.ambiguous_permission_checks += 1
        if args.strict:
            raise RuntimeError(
                f"Ambiguous require-drivable check for edge {edge.getID()} and {drivable_classes}"
            )

    return True


def parse_cluster_colors(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    out: dict[str, str] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid --cluster-colors mapping: {token}")
        name, color = token.split("=", 1)
        out[name.strip()] = color.strip()
    return out


def cluster_color(cluster_name: str, index: int, custom: dict[str, str]) -> str:
    return custom.get(cluster_name, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])


def write_json_output(path: Path, selected: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, sort_keys=True)


def write_txt_output(path: Path, selected: dict[str, list[str]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for cluster, edge_ids in selected.items():
        out = path / f"{cluster}.txt"
        with out.open("w", encoding="utf-8") as f:
            for edge_id in edge_ids:
                f.write(f"{edge_id}\n")


def write_summary_csv(
    path: Path,
    clusters: dict[str, Polygon],
    selected: dict[str, list[str]],
    membership_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster",
            "num_edges",
            "min_x",
            "min_y",
            "max_x",
            "max_y",
            "membership_mode",
        ])
        for name, polygon in clusters.items():
            min_x, min_y, max_x, max_y = polygon.bbox
            writer.writerow([
                name,
                len(selected[name]),
                f"{min_x:.6f}",
                f"{min_y:.6f}",
                f"{max_x:.6f}",
                f"{max_y:.6f}",
                membership_mode,
            ])


def write_selection_viz(viz_path: Path, selected: dict[str, list[str]]) -> None:
    if viz_path.suffix.lower() in {".xml", ".sel"}:
        out_dir = viz_path.parent / f"{viz_path.stem}_selection"
    else:
        out_dir = viz_path
    out_dir.mkdir(parents=True, exist_ok=True)

    for cluster, edge_ids in selected.items():
        root = ET.Element("selection")
        for edge_id in edge_ids:
            ET.SubElement(root, "edge", id=edge_id)
        out = out_dir / f"{cluster}.sel.xml"
        ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)


def write_add_viz(
    path: Path,
    selected: dict[str, list[str]],
    edge_shapes: dict[str, list[Point]],
    colors: dict[str, str],
    clusters_order: list[str],
    line_width: float,
    layer: int,
) -> None:
    root = ET.Element(
        "additional",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
        },
    )

    for i, cluster in enumerate(clusters_order):
        color = cluster_color(cluster, i, colors)
        for edge_id in selected[cluster]:
            pts = edge_shapes.get(edge_id, [])
            if len(pts) < 2:
                continue
            shape = " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
            ET.SubElement(
                root,
                "poly",
                id=f"cluster_{cluster}_{edge_id}",
                color=color,
                layer=str(layer),
                lineWidth=f"{line_width:.1f}",
                fill="false",
                shape=shape,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def print_summary(
    clusters: dict[str, Polygon],
    selected: dict[str, list[str]],
    stats: Stats,
    membership_mode: str,
    strict: bool,
) -> None:
    print(f"Clusters loaded: {len(clusters)}")
    print(f"Edges in network: {stats.total_edges}")
    print(f"Edges after filtering: {stats.filtered_edges}")
    print(f"Membership mode: {membership_mode}")

    for cluster in clusters:
        n = len(selected[cluster])
        print(f"Cluster {cluster}: {n} edges")
        if n == 0:
            msg = f"Cluster '{cluster}' is empty"
            if strict:
                raise RuntimeError(msg)
            warn(msg)

    print(
        "Geometry fallback usage: "
        f"lane_shape={stats.fallback_lane_shape_count}, "
        f"node_coords={stats.fallback_node_coords_count}"
    )
    if stats.fallback_node_coords_count >= 20:
        warn("Many edges required node-coordinate fallback; verify net geometry.")

    if stats.ambiguous_permission_checks > 0:
        warn(f"Ambiguous permission checks: {stats.ambiguous_permission_checks}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract SUMO edge IDs per cluster polygon from a .net.xml"
    )
    parser.add_argument("--net", required=True, type=str, help="Path to SUMO net.xml")
    parser.add_argument("--clusters", required=True, type=str, help="Path to clusters.txt")

    parser.add_argument(
        "--membership",
        choices=["midpoint", "any-point", "segment-intersects"],
        default="any-point",
    )

    internal_group = parser.add_mutually_exclusive_group()
    internal_group.add_argument("--exclude-internal", dest="exclude_internal", action="store_true")
    internal_group.add_argument("--include-internal", dest="exclude_internal", action="store_false")
    parser.set_defaults(exclude_internal=True)

    parser.add_argument("--min-length", type=float, default=0.0)
    parser.add_argument("--max-length", type=float, default=None)
    parser.add_argument("--allow-vclass", action="append", default=[])
    parser.add_argument("--disallow-vclass", action="append", default=[])

    drivable_group = parser.add_mutually_exclusive_group()
    drivable_group.add_argument("--require-drivable", dest="require_drivable", action="store_true")
    drivable_group.add_argument("--no-require-drivable", dest="require_drivable", action="store_false")
    parser.set_defaults(require_drivable=True)

    dedup_group = parser.add_mutually_exclusive_group()
    dedup_group.add_argument("--deduplicate", dest="deduplicate", action="store_true")
    dedup_group.add_argument("--no-deduplicate", dest="deduplicate", action="store_false")
    parser.set_defaults(deduplicate=True)

    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)

    parser.add_argument("--viz-add", type=str, default=None)
    parser.add_argument("--viz-format", choices=["add", "selection"], default="selection")
    parser.add_argument("--cluster-colors", type=str, default=None)
    parser.add_argument(
        "--viz-line-width",
        type=float,
        default=6.0,
        help="Line width for --viz-format add polylines (default: 6.0)",
    )
    parser.add_argument(
        "--viz-layer",
        type=int,
        default=120,
        help="Drawing layer for --viz-format add polylines (default: 120)",
    )
    parser.add_argument("--overlap-policy", choices=["first", "allow", "warn", "error"], default="warn")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    clusters = parse_clusters(Path(args.clusters), strict=args.strict)
    net = load_net(Path(args.net))
    all_edges = get_all_edges(net)

    stats = Stats(total_edges=len(all_edges))
    selected: dict[str, list[str]] = {name: [] for name in clusters}
    edge_shapes: dict[str, list[Point]] = {}

    for edge in all_edges:
        points = edge_geometry(edge, stats)
        if not edge_passes_filters(edge, points, args, stats):
            continue
        stats.filtered_edges += 1

        edge_id = edge.getID()
        matched: list[str] = []
        for cname, poly in clusters.items():
            if edge_in_cluster(points, poly, args.membership):
                matched.append(cname)

        if not matched:
            continue

        edge_shapes[edge_id] = points

        if len(matched) > 1:
            msg = f"Edge {edge_id} overlaps clusters {matched}"
            if args.overlap_policy == "error":
                raise RuntimeError(msg)
            if args.overlap_policy in {"warn", "first"}:
                warn(msg)

        if args.overlap_policy == "first":
            matched = [matched[0]]

        for cname in matched:
            selected[cname].append(edge_id)

    if args.deduplicate:
        for cname, edge_ids in selected.items():
            seen: set[str] = set()
            deduped: list[str] = []
            for eid in edge_ids:
                if eid in seen:
                    continue
                seen.add(eid)
                deduped.append(eid)
            selected[cname] = deduped

    print_summary(clusters, selected, stats, args.membership, args.strict)

    wrote_any = False

    if args.out_json:
        write_json_output(Path(args.out_json), selected)
        print(f"[OK] Wrote JSON: {args.out_json}")
        wrote_any = True

    if args.out_dir:
        write_txt_output(Path(args.out_dir), selected)
        print(f"[OK] Wrote per-cluster text files: {args.out_dir}")
        wrote_any = True

    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), clusters, selected, args.membership)
        print(f"[OK] Wrote summary CSV: {args.summary_csv}")
        wrote_any = True

    if args.viz_add:
        colors = parse_cluster_colors(args.cluster_colors)
        viz_path = Path(args.viz_add)
        if args.viz_format == "selection":
            write_selection_viz(viz_path, selected)
            print(f"[OK] Wrote selection visualization under: {viz_path}")
        else:
            write_add_viz(
                viz_path,
                selected,
                edge_shapes,
                colors,
                list(clusters.keys()),
                args.viz_line_width,
                args.viz_layer,
            )
            print(f"[OK] Wrote add-file visualization: {viz_path}")
        wrote_any = True

    if not wrote_any:
        # No explicit output path requested: print JSON mapping to stdout.
        print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

