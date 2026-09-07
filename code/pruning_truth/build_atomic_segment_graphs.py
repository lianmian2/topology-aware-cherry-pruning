from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_RE = re.compile(r"^(tree_\d{3})_before_(view_\d{2})$")
EPSILON = 1e-6
EVENT_MERGE_TOLERANCE_PX = 0.5
INTERSECTION_MERGE_TOLERANCE_PX = 1.0
GRAPH_SCHEMA_VERSION = "2.0"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_projection(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, np.ndarray, float]:
    delta = end - start
    length2 = float(np.dot(delta, delta))
    if length2 <= EPSILON:
        return 0.0, start.copy(), float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, delta) / length2, 0.0, 1.0))
    projection = start + fraction * delta
    return fraction, projection, float(np.linalg.norm(point - projection))


def project_polyline(point: np.ndarray, polyline: list[list[float]]) -> tuple[float, list[float], float]:
    best: tuple[float, list[float], float] | None = None
    offset = 0.0
    for index in range(1, len(polyline)):
        start = np.asarray(polyline[index - 1], dtype=float)
        end = np.asarray(polyline[index], dtype=float)
        length = float(np.linalg.norm(end - start))
        fraction, projected, distance = point_projection(point, start, end)
        candidate = (offset + fraction * length, projected.tolist(), distance)
        if best is None or candidate[2] < best[2]:
            best = candidate
        offset += length
    return best if best is not None else (0.0, list(point), float("inf"))


def polyline_length(polyline: list[list[float]]) -> float:
    return float(sum(math.dist(left, right) for left, right in zip(polyline[:-1], polyline[1:])))


def polyline_piece(polyline: list[list[float]], start_arc: float, end_arc: float) -> list[list[float]]:
    if end_arc - start_arc <= EPSILON:
        return []
    values: list[list[float]] = []
    offset = 0.0
    for index in range(1, len(polyline)):
        start = np.asarray(polyline[index - 1], dtype=float)
        end = np.asarray(polyline[index], dtype=float)
        length = float(np.linalg.norm(end - start))
        edge_start, edge_end = offset, offset + length
        overlap_start, overlap_end = max(start_arc, edge_start), min(end_arc, edge_end)
        if overlap_end + EPSILON >= overlap_start and length > EPSILON:
            first = start + ((overlap_start - edge_start) / length) * (end - start)
            second = start + ((overlap_end - edge_start) / length) * (end - start)
            for value in (first.tolist(), second.tolist()):
                if not values or math.dist(values[-1], value) > EPSILON:
                    values.append(value)
        offset = edge_end
    return values


def segment_intersection_point(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> list[float] | None:
    r = a1 - a0
    s = b1 - b0
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) <= EPSILON:
        return None
    offset = b0 - a0
    t = float((offset[0] * s[1] - offset[1] * s[0]) / denom)
    u = float((offset[0] * r[1] - offset[1] * r[0]) / denom)
    if -EPSILON <= t <= 1.0 + EPSILON and -EPSILON <= u <= 1.0 + EPSILON:
        return (a0 + t * r).tolist()
    return None


def cut_polyline_intersections(cut: dict[str, Any], polyline: list[list[float]]) -> list[list[float]]:
    start = np.asarray(cut["p0"], dtype=float)
    end = np.asarray(cut["p1"], dtype=float)
    points = []
    for left, right in zip(polyline[:-1], polyline[1:]):
        point = segment_intersection_point(start, end, np.asarray(left, dtype=float), np.asarray(right, dtype=float))
        if point is not None and not any(math.dist(point, existing) <= INTERSECTION_MERGE_TOLERANCE_PX for existing in points):
            points.append(point)
    return points


def cut_intersects_polyline(cut: dict[str, Any], polyline: list[list[float]]) -> bool:
    return bool(cut_polyline_intersections(cut, polyline))


def build_graph(groups: list[dict[str, Any]]) -> tuple[nx.Graph, dict[int, set[str]]]:
    graph = nx.Graph()
    point_ids: dict[tuple[int, int], int] = {}
    group_sets: dict[int, set[str]] = defaultdict(set)
    for group in groups:
        local_ids = []
        group_id = str(group.get("group_id", "unknown"))
        for raw in group.get("points", []):
            point = (int(round(raw[0])), int(round(raw[1])))
            if point not in point_ids:
                point_ids[point] = len(point_ids)
                graph.add_node(point_ids[point], point=[float(point[0]), float(point[1])])
            node_id = point_ids[point]
            local_ids.append(node_id)
            group_sets[node_id].add(group_id)
        for raw_edge in group.get("edges", []):
            left, right = int(raw_edge[0]), int(raw_edge[1])
            if not (0 <= left < len(local_ids) and 0 <= right < len(local_ids)):
                continue
            source, target = local_ids[left], local_ids[right]
            if source == target:
                continue
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, group_ids=set())
            graph.edges[source, target]["group_ids"].add(group_id)
    return graph, group_sets


def choose_roots(graph: nx.Graph, group_sets: dict[int, set[str]]) -> dict[int, int]:
    roots = {}
    for component_id, component in enumerate(nx.connected_components(graph)):
        candidates = [node for node in component if any(group == "trunk" for group in group_sets[node])]
        candidates = candidates or list(component)
        root = max(candidates, key=lambda node: graph.nodes[node]["point"][1])
        roots[component_id] = root
        for node in component:
            graph.nodes[node]["component_id"] = component_id
    return roots


def trace_topological_segments(graph: nx.Graph, roots: dict[int, int]) -> list[dict[str, Any]]:
    key_nodes = {node for node in graph.nodes if graph.degree[node] != 2} | set(roots.values())
    seen: set[frozenset[int]] = set()
    segments = []
    for start in sorted(key_nodes):
        for neighbor in graph.neighbors(start):
            edge = frozenset((start, neighbor))
            if edge in seen:
                continue
            path = [start, neighbor]
            seen.add(edge)
            previous, current = start, neighbor
            while current not in key_nodes:
                next_nodes = [node for node in graph.neighbors(current) if node != previous]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                seen.add(frozenset((current, next_node)))
                path.append(next_node)
                previous, current = current, next_node
            group_ids = set()
            for left, right in zip(path[:-1], path[1:]):
                group_ids.update(graph.edges[left, right]["group_ids"])
            segments.append(
                {
                    "start": path[0],
                    "end": path[-1],
                    "component_id": graph.nodes[path[0]]["component_id"],
                    "group_ids": sorted(group_ids),
                    "polyline": [graph.nodes[node]["point"] for node in path],
                }
            )
    return segments


def landmark_types(graph: nx.Graph, roots: dict[int, int]) -> dict[str, set[str]]:
    output = {}
    root_ids = set(roots.values())
    for node in graph.nodes:
        if graph.degree[node] == 2 and node not in root_ids:
            continue
        labels = set()
        if node in root_ids:
            labels.add("root")
        if graph.degree[node] >= 3:
            labels.add("junction")
        if graph.degree[node] <= 1:
            labels.add("endpoint")
        output[f"point:{node}"] = labels or {"anchor"}
    return output


def assign_buds(attachments: list[dict[str, Any]], topological: list[dict[str, Any]]) -> tuple[dict[int, list[dict[str, Any]]], list[int]]:
    assigned: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unattached = []
    for attachment in attachments:
        if attachment.get("skeleton_point") is None:
            unattached.append(int(attachment["bud_index"]))
            continue
        point = np.asarray(attachment["skeleton_point"], dtype=float)
        group_id = attachment.get("group_id")
        ranked = []
        for index, segment in enumerate(topological):
            arc, projection, distance = project_polyline(point, segment["polyline"])
            group_match = group_id in segment["group_ids"] if group_id else False
            ranked.append((0 if group_match else 1, distance, index, arc, projection))
        _, distance, index, arc, projection = min(ranked)
        item = dict(attachment)
        item["arc"] = arc
        item["projection"] = projection
        item["projection_distance"] = distance
        assigned[index].append(item)
    return assigned, unattached


def candidate_type(left: set[str], right: set[str]) -> str | None:
    if "bud" in left and "bud" in right:
        return "bud--bud"
    if "bud" in left and "junction" in right or "bud" in right and "junction" in left:
        return "junction--bud"
    if "bud" in left and "endpoint" in right or "bud" in right and "endpoint" in left:
        return "bud--endpoint"
    if "junction" in left and "endpoint" in right or "junction" in right and "endpoint" in left:
        return "junction--endpoint"
    if "junction" in left and "junction" in right:
        return "junction--junction"
    return None


def build_atomic_segments(graph: nx.Graph, roots: dict[int, int], topological: list[dict[str, Any]], attachments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, list[int]], list[int]]:
    types = landmark_types(graph, roots)
    landmark_bud_ids: dict[str, list[int]] = defaultdict(list)
    assigned, unattached = assign_buds(attachments, topological)
    atomic = []
    for topological_id, segment in enumerate(topological):
        total = polyline_length(segment["polyline"])
        start_key = f"point:{segment['start']}"
        end_key = f"point:{segment['end']}"
        events = [(0.0, start_key, None), (total, end_key, None)]
        for attachment in sorted(assigned.get(topological_id, []), key=lambda item: item["arc"]):
            arc = float(np.clip(attachment["arc"], 0.0, total))
            if arc <= EPSILON:
                key = start_key
            elif total - arc <= EPSILON:
                key = end_key
            else:
                key = f"bud:{attachment['bud_index']}"
                types.setdefault(key, set()).add("bud")
            types.setdefault(key, set()).add("bud")
            events.append((arc, key, int(attachment["bud_index"])))
        events.sort(key=lambda item: (item[0], 0 if item[1] == start_key else 2 if item[1] == end_key else 1, item[1]))
        collapsed: list[tuple[float, str]] = []
        for arc, key, bud_id in events:
            if collapsed and abs(arc - collapsed[-1][0]) <= EVENT_MERGE_TOLERANCE_PX:
                target_key = collapsed[-1][1]
                types[target_key].update(types.get(key, set()))
                if bud_id is not None:
                    landmark_bud_ids[target_key].append(bud_id)
                continue
            collapsed.append((arc, key))
            if bud_id is not None:
                landmark_bud_ids[key].append(bud_id)
        for (start_arc, start_key), (end_arc, end_key) in zip(collapsed[:-1], collapsed[1:]):
            polyline = polyline_piece(segment["polyline"], start_arc, end_arc)
            if len(polyline) < 2 or polyline_length(polyline) <= EPSILON:
                continue
            left_types, right_types = types[start_key], types[end_key]
            primary_type = candidate_type(left_types, right_types)
            pure_trunk = bool(segment["group_ids"]) and all(group_id == "trunk" for group_id in segment["group_ids"])
            atomic.append(
                {
                    "id": len(atomic),
                    "topological_segment_id": topological_id,
                    "component_id": segment["component_id"],
                    "start_landmark_id": start_key,
                    "end_landmark_id": end_key,
                    "start_types": sorted(left_types),
                    "end_types": sorted(right_types),
                    "candidate_type": primary_type,
                    "is_trunk_context": int(pure_trunk),
                    "is_candidate": int(primary_type is not None and not pure_trunk),
                    "group_ids": segment["group_ids"],
                    "polyline": polyline,
                    "length_px": polyline_length(polyline),
                    "start_arc_px": float(start_arc),
                    "end_arc_px": float(end_arc),
                    "start_bud_ids": sorted(set(landmark_bud_ids.get(start_key, []))),
                    "end_bud_ids": sorted(set(landmark_bud_ids.get(end_key, []))),
                    "is_cut_segment": 0,
                    "auto_suggested_cut": 0,
                    "label_mask": 0,
                    "cut_ids": [],
                }
            )
    return atomic, types, {key: sorted(set(value)) for key, value in landmark_bud_ids.items()}, unattached


def validate_atomic_partition(topological: list[dict[str, Any]], atomic: list[dict[str, Any]]) -> dict[str, Any]:
    issues = []
    rows = []
    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for segment in atomic:
        by_parent[int(segment["topological_segment_id"])].append(segment)
    for parent_id, parent in enumerate(topological):
        expected = polyline_length(parent["polyline"])
        pieces = sorted(by_parent.get(parent_id, []), key=lambda item: (item["start_arc_px"], item["end_arc_px"]))
        observed = sum(float(item["length_px"]) for item in pieces)
        cursor = 0.0
        for piece in pieces:
            start = float(piece["start_arc_px"])
            end = float(piece["end_arc_px"])
            if start < cursor - EVENT_MERGE_TOLERANCE_PX:
                issues.append({"kind": "overlap", "topological_segment_id": parent_id, "segment_id": piece["id"], "previous_end": cursor, "start": start})
            if start > cursor + EVENT_MERGE_TOLERANCE_PX:
                issues.append({"kind": "gap", "topological_segment_id": parent_id, "segment_id": piece["id"], "previous_end": cursor, "start": start})
            cursor = max(cursor, end)
        length_error = abs(observed - expected)
        if abs(cursor - expected) > 1.0 or length_error > 1.0:
            issues.append({"kind": "length_mismatch", "topological_segment_id": parent_id, "expected": expected, "observed": observed, "covered_to": cursor})
        rows.append({"topological_segment_id": parent_id, "expected_length_px": expected, "atomic_length_px": observed, "length_error_px": length_error, "atomic_segments": len(pieces)})
    return {"valid": not issues, "schema_version": GRAPH_SCHEMA_VERSION, "parents": rows, "issues": issues}


def label_segments(cut_lines: list[dict[str, Any]], atomic: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mappings = []
    for cut in cut_lines:
        hits = []
        for segment in atomic:
            for point in cut_polyline_intersections(cut, segment["polyline"]):
                hits.append({"segment_id": segment["id"], "point": point})
        intersections = sorted({int(hit["segment_id"]) for hit in hits})
        candidates = [segment_id for segment_id in intersections if atomic[segment_id]["is_candidate"]]
        clusters: list[dict[str, Any]] = []
        for hit in hits:
            cluster = next((item for item in clusters if math.dist(item["point"], hit["point"]) <= INTERSECTION_MERGE_TOLERANCE_PX), None)
            if cluster is None:
                clusters.append({"point": hit["point"], "segment_ids": [hit["segment_id"]]})
            elif hit["segment_id"] not in cluster["segment_ids"]:
                cluster["segment_ids"].append(hit["segment_id"])
        if not intersections:
            status = "non_intersection"
            suggestion = None
        elif not candidates:
            status = "context_only"
            suggestion = None
        elif len(clusters) == 1 and len(candidates) == 1:
            segment = atomic[candidates[0]]
            segment["auto_suggested_cut"] = 1
            segment["cut_ids"].append(str(cut["cut_id"]))
            status = "unique_candidate"
            suggestion = candidates[0]
        elif len(clusters) == 1:
            landmark_sets = [set((atomic[segment_id]["start_landmark_id"], atomic[segment_id]["end_landmark_id"])) for segment_id in candidates]
            shared = set.intersection(*landmark_sets) if landmark_sets else set()
            status = "shared_landmark_ambiguous" if shared else "multiple_physical_crossings"
            suggestion = None
        else:
            status = "multiple_physical_crossings"
            suggestion = None
        mappings.append({"cut_id": cut["cut_id"], "status": status, "intersected_segment_ids": intersections, "candidate_segment_ids": candidates, "intersection_clusters": clusters, "suggested_segment_id": suggestion})
    return mappings


def draw_overlay(image_path: Path, atomic: list[dict[str, Any]], landmarks: dict[str, set[str]], cuts: list[dict[str, Any]], output: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    colors = {
        "junction--bud": (70, 220, 90),
        "bud--bud": (40, 220, 245),
        "bud--endpoint": (255, 170, 50),
        "junction--endpoint": (30, 145, 255),
        "junction--junction": (220, 80, 205),
    }
    for segment in atomic:
        color = colors.get(segment.get("candidate_type"), (130, 130, 130)) if segment["is_candidate"] else (130, 130, 130)
        points = np.asarray(segment["polyline"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
    for cut in cuts:
        cv2.line(image, tuple(map(int, cut["p0"])), tuple(map(int, cut["p1"])), (0, 255, 255), 3, cv2.LINE_AA)
    ensure_dir(output.parent)
    cv2.imwrite(str(output), image)


def process_sample(cases_root: Path, output_root: Path, sample_id: str, gt_skeleton_root: Path | None) -> dict[str, Any]:
    match = SAMPLE_RE.match(sample_id)
    if match is None:
        raise ValueError(sample_id)
    tree_id, view = match.groups()
    case_root = cases_root / sample_id
    source_groups = case_root / "annotation_groups.json"
    if gt_skeleton_root is not None:
        source_groups = gt_skeleton_root / f"{sample_id}_skeleton.json"
    groups = load_json(source_groups)["groups"]
    attachments = load_json(case_root / "attachments.json")
    cut_path = PROJECT_ROOT / "04_results/pruning_truth/manual_cut_lines_20260720_raw" / f"{tree_id}_{view}" / "cut_lines.json"
    cuts = load_json(cut_path)["cut_lines"]
    image_path = PROJECT_ROOT / "01_data/01_raw/final_data" / tree_id / "before" / f"{view}.jpg"
    graph, group_sets = build_graph(groups)
    roots = choose_roots(graph, group_sets)
    topological = trace_topological_segments(graph, roots)
    atomic, types, landmark_bud_ids, unattached = build_atomic_segments(graph, roots, topological, attachments)
    geometry_audit = validate_atomic_partition(topological, atomic)
    mappings = label_segments(cuts, atomic)
    landmark_rows = [{"id": key, "types": sorted(value), "bud_ids": landmark_bud_ids.get(key, [])} for key, value in sorted(types.items())]
    relations = []
    for segment in atomic:
        relations.extend([
            {"type": "landmark_to_segment", "landmark_id": segment["start_landmark_id"], "segment_id": segment["id"]},
            {"type": "landmark_to_segment", "landmark_id": segment["end_landmark_id"], "segment_id": segment["id"]},
        ])
    output = ensure_dir(output_root / sample_id)
    payload = {
        "sample_id": sample_id,
        "schema_version": GRAPH_SCHEMA_VERSION,
        "representation": "landmark_segment_heterograph",
        "annotation_complete_by_user": True,
        "source": {
            "skeleton": str(source_groups),
            "skeleton_sha256": sha256_file(source_groups),
            "attachments": str(case_root / "attachments.json"),
            "attachments_sha256": sha256_file(case_root / "attachments.json"),
            "cut_truth": str(cut_path),
            "cut_truth_sha256": sha256_file(cut_path),
            "image": str(image_path),
        },
        "landmark_nodes": landmark_rows,
        "segment_nodes": atomic,
        "relations": relations,
        "cut_mappings": mappings,
    }
    save_json(output / "decision_graph.json", payload)
    save_json(output / "atomic_graph.json", payload)
    draw_overlay(image_path, atomic, types, cuts, output / "cut_label_overlay.jpg")
    draw_overlay(image_path, atomic, types, cuts, output / "segment_overlay.jpg")
    save_json(output / "geometry_audit.json", geometry_audit)
    if not geometry_audit["valid"]:
        raise RuntimeError(f"Atomic partition validation failed for {sample_id}: {geometry_audit['issues'][:3]}")
    stats = {
        "sample_id": sample_id,
        "topological_segments": len(topological),
        "atomic_segments": len(atomic),
        "candidate_segments": sum(segment["is_candidate"] == 1 for segment in atomic),
        "auto_suggested_segments": sum(segment["auto_suggested_cut"] == 1 for segment in atomic),
        "unattached_buds": len(unattached),
        "cuts": len(cuts),
        "unique_candidate": sum(item["status"] == "unique_candidate" for item in mappings),
        "shared_landmark_ambiguous": sum(item["status"] == "shared_landmark_ambiguous" for item in mappings),
        "multiple_physical_crossings": sum(item["status"] == "multiple_physical_crossings" for item in mappings),
        "non_intersection": sum(item["status"] == "non_intersection" for item in mappings),
        "context_only": sum(item["status"] == "context_only" for item in mappings),
        "geometry_valid": geometry_audit["valid"],
    }
    save_json(output / "graph_quality.json", stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bud-split atomic pruning graphs from current e2e outputs")
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--gt-skeleton-root", type=Path)
    args = parser.parse_args()
    cases_root = args.cases_root if args.cases_root.is_absolute() else PROJECT_ROOT / args.cases_root
    output_root = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    gt_root = args.gt_skeleton_root if args.gt_skeleton_root is None or args.gt_skeleton_root.is_absolute() else PROJECT_ROOT / args.gt_skeleton_root
    rows = [process_sample(cases_root, output_root, sample_id, gt_root) for sample_id in args.sample_id]
    with (ensure_dir(output_root) / "graph_quality_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    save_json(output_root / "summary.json", {"status": "completed", "samples": rows})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
