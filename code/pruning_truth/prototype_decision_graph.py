from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np


CASE_PATTERN = re.compile(r"^(tree_\d{3})_before_(view_\d{2})$")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def point_distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, np.ndarray]:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start)), start.copy()
    fraction = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    projection = start + fraction * delta
    return float(np.linalg.norm(point - projection)), projection


def segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    if segment_intersection(a0, a1, b0, b1) is not None:
        return 0.0
    distances = [
        point_distance_to_segment(a0, b0, b1)[0],
        point_distance_to_segment(a1, b0, b1)[0],
        point_distance_to_segment(b0, a0, a1)[0],
        point_distance_to_segment(b1, a0, a1)[0],
    ]
    return float(min(distances))


def segment_intersection(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray | None:
    r = a1 - a0
    s = b1 - b0
    denominator = float(r[0] * s[1] - r[1] * s[0])
    offset = b0 - a0
    if abs(denominator) <= 1e-9:
        return None
    t = float((offset[0] * s[1] - offset[1] * s[0]) / denominator)
    u = float((offset[0] * r[1] - offset[1] * r[0]) / denominator)
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return a0 + t * r
    return None


def polyline_length(points: list[tuple[int, int]]) -> float:
    return float(sum(math.dist(points[index - 1], points[index]) for index in range(1, len(points))))


def max_chord_deviation(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    start = np.asarray(points[0], dtype=float)
    end = np.asarray(points[-1], dtype=float)
    return float(max(point_distance_to_segment(np.asarray(point, dtype=float), start, end)[0] for point in points[1:-1]))


def turning_angles(points: list[tuple[int, int]]) -> list[float]:
    values: list[float] = []
    for index in range(1, len(points) - 1):
        incoming = np.asarray(points[index], dtype=float) - np.asarray(points[index - 1], dtype=float)
        outgoing = np.asarray(points[index + 1], dtype=float) - np.asarray(points[index], dtype=float)
        norms = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if norms <= 1e-9:
            continue
        cosine = float(np.clip(np.dot(incoming, outgoing) / norms, -1.0, 1.0))
        values.append(float(math.degrees(math.acos(cosine))))
    return values


def build_merged_graph(groups: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    coordinate_to_id: dict[tuple[int, int], int] = {}
    for group in groups:
        local_ids: list[int] = []
        group_id = str(group.get("group_id", "group"))
        group_type = str(group.get("group_type", "branch"))
        for raw_point in group.get("points", []):
            point = (int(raw_point[0]), int(raw_point[1]))
            if point not in coordinate_to_id:
                node_id = len(coordinate_to_id)
                coordinate_to_id[point] = node_id
                graph.add_node(node_id, point=point, group_ids=set(), group_types=set())
            node_id = coordinate_to_id[point]
            graph.nodes[node_id]["group_ids"].add(group_id)
            graph.nodes[node_id]["group_types"].add(group_type)
            local_ids.append(node_id)
        for raw_edge in group.get("edges", []):
            src, dst = int(raw_edge[0]), int(raw_edge[1])
            if not (0 <= src < len(local_ids) and 0 <= dst < len(local_ids)):
                continue
            src_id, dst_id = local_ids[src], local_ids[dst]
            if src_id == dst_id:
                continue
            if graph.has_edge(src_id, dst_id):
                graph.edges[src_id, dst_id]["group_ids"].add(group_id)
                graph.edges[src_id, dst_id]["group_types"].add(group_type)
            else:
                graph.add_edge(src_id, dst_id, group_ids={group_id}, group_types={group_type})
    return graph


def choose_component_roots(graph: nx.Graph) -> dict[int, int]:
    roots: dict[int, int] = {}
    for component_index, component in enumerate(nx.connected_components(graph)):
        nodes = list(component)
        trunk_nodes = [node for node in nodes if "trunk" in graph.nodes[node]["group_types"]]
        candidates = trunk_nodes or nodes
        roots[component_index] = max(candidates, key=lambda node: graph.nodes[node]["point"][1])
        for node in nodes:
            graph.nodes[node]["component_id"] = component_index
    return roots


def trace_segments(graph: nx.Graph, roots: dict[int, int]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    key_nodes = {node for node in graph.nodes if graph.degree[node] != 2}
    key_nodes.update(roots.values())
    for component in nx.connected_components(graph):
        if not key_nodes.intersection(component):
            key_nodes.add(next(iter(component)))

    visited: set[frozenset[int]] = set()
    segments: list[dict[str, Any]] = []
    incidence: dict[int, list[int]] = defaultdict(list)
    for start in sorted(key_nodes):
        for neighbor in graph.neighbors(start):
            edge_key = frozenset((start, neighbor))
            if edge_key in visited:
                continue
            path = [start, neighbor]
            visited.add(edge_key)
            previous, current = start, neighbor
            while current not in key_nodes:
                next_nodes = [node for node in graph.neighbors(current) if node != previous]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                visited.add(frozenset((current, next_node)))
                path.append(next_node)
                previous, current = current, next_node
            points = [tuple(graph.nodes[node]["point"]) for node in path]
            length = polyline_length(points)
            chord = float(math.dist(points[0], points[-1]))
            turn_values = turning_angles(points)
            group_ids = sorted({group_id for left, right in zip(path[:-1], path[1:]) for group_id in graph.edges[left, right]["group_ids"]})
            segment_id = len(segments)
            segment = {
                "id": segment_id,
                "start_point_id": int(path[0]),
                "end_point_id": int(path[-1]),
                "component_id": int(graph.nodes[path[0]]["component_id"]),
                "group_ids": group_ids,
                "polyline": [[int(x), int(y)] for x, y in points],
                "length_px": length,
                "chord_length_px": chord,
                "sinuosity": length / max(chord, 1e-6),
                "max_chord_deviation_px": max_chord_deviation(points),
                "total_turn_deg": float(sum(turn_values)),
                "max_local_turn_deg": float(max(turn_values, default=0.0)),
            }
            segments.append(segment)
            incidence[path[0]].append(segment_id)
            incidence[path[-1]].append(segment_id)

    point_nodes: dict[int, dict[str, Any]] = {}
    root_set = set(roots.values())
    for node in sorted(key_nodes):
        degree = int(graph.degree[node])
        if node in root_set:
            node_type = "root"
        elif degree >= 3:
            node_type = "junction"
        elif degree <= 1:
            node_type = "endpoint"
        else:
            node_type = "anchor"
        vectors: list[np.ndarray] = []
        origin = np.asarray(graph.nodes[node]["point"], dtype=float)
        for segment_id in incidence.get(node, []):
            points = np.asarray(segments[segment_id]["polyline"], dtype=float)
            neighbor = points[1] if segments[segment_id]["start_point_id"] == node else points[-2]
            vector = neighbor - origin
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                vectors.append(vector / norm)
        angles: list[float] = []
        for left_index in range(len(vectors)):
            for right_index in range(left_index + 1, len(vectors)):
                cosine = float(np.clip(np.dot(vectors[left_index], vectors[right_index]), -1.0, 1.0))
                angles.append(float(math.degrees(math.acos(cosine))))
        x, y = graph.nodes[node]["point"]
        point_nodes[node] = {
            "id": int(node),
            "type": node_type,
            "x": int(x),
            "y": int(y),
            "degree": degree,
            "component_id": int(graph.nodes[node]["component_id"]),
            "incident_segment_ids": sorted(incidence.get(node, [])),
            "incident_angles_deg": angles,
            "min_incident_angle_deg": float(min(angles)) if angles else None,
            "max_incident_angle_deg": float(max(angles)) if angles else None,
        }
    return segments, point_nodes


def orient_segment_graph(segments: list[dict[str, Any]], point_nodes: dict[int, dict[str, Any]], roots: dict[int, int]) -> nx.DiGraph:
    point_graph = nx.Graph()
    for segment in segments:
        point_graph.add_edge(segment["start_point_id"], segment["end_point_id"], segment_id=segment["id"], weight=segment["length_px"])
    segment_graph = nx.DiGraph()
    segment_graph.add_nodes_from(segment["id"] for segment in segments)
    for component_id, root in roots.items():
        if root not in point_graph:
            continue
        distances = nx.single_source_dijkstra_path_length(point_graph, root, weight="weight")
        for segment in segments:
            if segment["component_id"] != component_id:
                continue
            start = segment["start_point_id"]
            end = segment["end_point_id"]
            if distances.get(start, math.inf) <= distances.get(end, math.inf):
                proximal, distal = start, end
            else:
                proximal, distal = end, start
            segment["proximal_point_id"] = int(proximal)
            segment["distal_point_id"] = int(distal)
            segment["depth_px"] = float(min(distances.get(start, 0.0), distances.get(end, 0.0)))
        distal_to_segment = {segment["distal_point_id"]: segment["id"] for segment in segments if segment["component_id"] == component_id}
        for segment in segments:
            if segment["component_id"] != component_id:
                continue
            parent_id = distal_to_segment.get(segment["proximal_point_id"])
            segment["parent_segment_id"] = int(parent_id) if parent_id is not None else None
            if parent_id is not None and parent_id != segment["id"]:
                segment_graph.add_edge(parent_id, segment["id"])
    for segment in segments:
        segment_graph.add_node(segment["id"])
    for segment in segments:
        segment["child_segment_ids"] = sorted(int(child) for child in segment_graph.successors(segment["id"]))
    return segment_graph


def attach_buds(attachments: list[dict[str, Any]], segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bud_nodes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for attachment in attachments:
        skeleton_point = np.asarray(attachment.get("skeleton_point", attachment.get("bud_centroid", [0, 0])), dtype=float)
        best: tuple[float, int] | None = None
        for segment in segments:
            points = np.asarray(segment["polyline"], dtype=float)
            distance = min(
                point_distance_to_segment(skeleton_point, points[index - 1], points[index])[0]
                for index in range(1, len(points))
            )
            candidate = (float(distance), int(segment["id"]))
            if best is None or candidate < best:
                best = candidate
        centroid = attachment.get("bud_centroid", skeleton_point.tolist())
        bud_id = len(bud_nodes)
        bud_nodes.append(
            {
                "id": bud_id,
                "type": "bud",
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "bud_label": int(attachment.get("bud_label", -1)),
                "score": float(attachment.get("bud_score", 0.0)),
                "source_group_id": attachment.get("group_id"),
            }
        )
        if best is not None:
            relations.append({"type": "bud_to_segment", "bud_id": bud_id, "segment_id": best[1], "distance_px": best[0]})
    return bud_nodes, relations


def label_cuts(cut_lines: list[dict[str, Any]], segments: list[dict[str, Any]], segment_graph: nx.DiGraph, point_nodes: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    cut_segment_ids: set[int] = set()
    pruned_endpoints: dict[int, int] = {}
    for cut in cut_lines:
        cut_start = np.asarray(cut["p0"], dtype=float)
        cut_end = np.asarray(cut["p1"], dtype=float)
        candidates: dict[int, list[list[float]]] = defaultdict(list)
        nearest: tuple[float, int] | None = None
        for segment in segments:
            points = np.asarray(segment["polyline"], dtype=float)
            for index in range(1, len(points)):
                intersection = segment_intersection(cut_start, cut_end, points[index - 1], points[index])
                if intersection is not None:
                    candidates[int(segment["id"])].append([float(intersection[0]), float(intersection[1])])
                distance = segment_distance(cut_start, cut_end, points[index - 1], points[index])
                candidate = (float(distance), int(segment["id"]))
                if nearest is None or candidate < nearest:
                    nearest = candidate
        candidate_ids = sorted(candidates)
        status = "exact" if len(candidate_ids) == 1 else "ambiguous" if candidate_ids else "nonintersecting"
        if len(candidate_ids) == 1:
            cut_segment_ids.add(candidate_ids[0])
        side_checks: list[dict[str, Any]] = []
        normal = np.asarray(cut.get("normal_pruned_side", [0.0, 0.0]), dtype=float)
        center = np.asarray(cut.get("center", (cut_start + cut_end) / 2.0), dtype=float)
        for segment_id in candidate_ids:
            segment = segments[segment_id]
            endpoint_dots: dict[int, float] = {}
            for endpoint_id in (segment["start_point_id"], segment["end_point_id"]):
                endpoint = point_nodes[endpoint_id]
                vector = np.asarray([endpoint["x"], endpoint["y"]], dtype=float) - center
                endpoint_dots[int(endpoint_id)] = float(np.dot(normal, vector))
            pruned_endpoint_id = max(endpoint_dots, key=endpoint_dots.get)
            pruned_endpoints[segment_id] = pruned_endpoint_id
            side_checks.append(
                {
                    "segment_id": segment_id,
                    "endpoint_normal_dots": {str(key): value for key, value in endpoint_dots.items()},
                    "pruned_endpoint_id": int(pruned_endpoint_id),
                    "root_directed_distal_id": int(segment["distal_point_id"]),
                    "root_direction_consistent": bool(pruned_endpoint_id == segment["distal_point_id"]),
                }
            )
        matches.append(
            {
                "cut_id": cut.get("cut_id"),
                "status": status,
                "candidate_segment_ids": candidate_ids,
                "intersections": {str(key): value for key, value in candidates.items()},
                "nearest_segment_id": nearest[1] if nearest else None,
                "nearest_distance_px": nearest[0] if nearest else None,
                "pruned_side_checks": side_checks,
            }
        )

    pruned_segment_ids: set[int] = set(cut_segment_ids)
    propagation_graph = segment_graph.to_undirected()
    propagation_graph.remove_nodes_from(cut_segment_ids)
    endpoint_incidence: dict[int, list[int]] = defaultdict(list)
    for segment in segments:
        endpoint_incidence[int(segment["start_point_id"])].append(int(segment["id"]))
        endpoint_incidence[int(segment["end_point_id"])].append(int(segment["id"]))
    for cut_segment_id, pruned_endpoint_id in pruned_endpoints.items():
        for seed in endpoint_incidence[pruned_endpoint_id]:
            if seed in cut_segment_ids or seed not in propagation_graph:
                continue
            pruned_segment_ids.update(nx.node_connected_component(propagation_graph, seed))
    for segment in segments:
        segment_id = int(segment["id"])
        segment["cut_target"] = int(segment_id in cut_segment_ids)
        segment["fate_label"] = "pruned" if segment_id in pruned_segment_ids else "retained"
    return matches


def draw_existing_graph(image: np.ndarray, graph_data: dict[str, Any]) -> np.ndarray:
    canvas = image.copy()
    points = {int(node["id"]): (int(node["x"]), int(node["y"])) for node in graph_data.get("nodes", [])}
    for edge in graph_data.get("edges", []):
        src, dst = points.get(int(edge["src"])), points.get(int(edge["dst"]))
        if src is not None and dst is not None:
            cv2.line(canvas, src, dst, (170, 170, 170), 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(canvas, point, 3, (40, 170, 255), -1, cv2.LINE_AA)
    return canvas


def draw_decision_graph(image: np.ndarray, graph: dict[str, Any], cut_lines: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    segment_colors = {"retained": (70, 210, 70), "pruned": (50, 50, 230)}
    for segment in graph["segment_nodes"]:
        points = np.asarray(segment["polyline"], dtype=np.int32).reshape(-1, 1, 2)
        color = segment_colors.get(segment.get("fate_label"), (255, 180, 30))
        thickness = 5 if segment.get("cut_target") else 3
        cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
    point_colors = {"root": (255, 80, 40), "junction": (255, 0, 255), "endpoint": (0, 220, 255), "anchor": (220, 220, 220)}
    for node in graph["point_nodes"]:
        center = (int(node["x"]), int(node["y"]))
        cv2.circle(canvas, center, 8, point_colors[node["type"]], -1, cv2.LINE_AA)
    for bud in graph["bud_nodes"]:
        cv2.circle(canvas, (int(round(bud["x"])), int(round(bud["y"]))), 3, (255, 120, 0), -1, cv2.LINE_AA)
    for cut in cut_lines:
        cv2.line(canvas, tuple(map(int, cut["p0"])), tuple(map(int, cut["p1"])), (0, 255, 255), 5, cv2.LINE_AA)
    return canvas


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    header = np.full((70, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(header, text, (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([header, image])


def draw_cut_diagnostics(image: np.ndarray, graph: dict[str, Any], cut_lines: list[dict[str, Any]]) -> np.ndarray:
    overlay = draw_decision_graph(image, graph, cut_lines)
    tiles: list[np.ndarray] = []
    matches = {match["cut_id"]: match for match in graph["cut_matches"]}
    for cut in cut_lines:
        center = np.asarray(cut.get("center", [0, 0]), dtype=float)
        radius = 300
        left = max(int(center[0]) - radius, 0)
        top = max(int(center[1]) - radius, 0)
        right = min(int(center[0]) + radius, overlay.shape[1])
        bottom = min(int(center[1]) + radius, overlay.shape[0])
        crop = overlay[top:bottom, left:right]
        tile = np.zeros((680, 680, 3), dtype=np.uint8)
        scale = min(640 / max(crop.shape[1], 1), 600 / max(crop.shape[0], 1))
        resized = cv2.resize(crop, (max(int(crop.shape[1] * scale), 1), max(int(crop.shape[0] * scale), 1)))
        x_offset = (680 - resized.shape[1]) // 2
        y_offset = 70 + (600 - resized.shape[0]) // 2
        tile[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
        match = matches.get(cut.get("cut_id"), {})
        label = f"{cut.get('cut_id')}  {match.get('status')}  nearest={match.get('nearest_distance_px', 0.0):.1f}px"
        cv2.putText(tile, label, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    while len(tiles) < rows * columns:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)])


def junction_angle_error(graph: nx.Graph, point_nodes: dict[int, dict[str, Any]]) -> float:
    errors: list[float] = []
    for node_id, point_node in point_nodes.items():
        if point_node["degree"] < 3:
            continue
        origin = np.asarray(graph.nodes[node_id]["point"], dtype=float)
        vectors: list[np.ndarray] = []
        for neighbor in graph.neighbors(node_id):
            vector = np.asarray(graph.nodes[neighbor]["point"], dtype=float) - origin
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                vectors.append(vector / norm)
        original_angles: list[float] = []
        for left_index in range(len(vectors)):
            for right_index in range(left_index + 1, len(vectors)):
                cosine = float(np.clip(np.dot(vectors[left_index], vectors[right_index]), -1.0, 1.0))
                original_angles.append(float(math.degrees(math.acos(cosine))))
        reconstructed_angles = list(point_node["incident_angles_deg"])
        for original, reconstructed in zip(sorted(original_angles), sorted(reconstructed_angles)):
            errors.append(abs(original - reconstructed))
    return float(max(errors, default=0.0))


def process_case(project_root: Path, case_dir: Path, output_root: Path) -> dict[str, Any]:
    match = CASE_PATTERN.match(case_dir.name)
    if match is None:
        raise ValueError(f"Unsupported case name: {case_dir.name}")
    tree_id, view = match.groups()
    sample_id = f"{tree_id}_{view}"
    groups = load_json(case_dir / "annotation_groups.json")["groups"]
    old_graph = load_json(case_dir / "directed_graph.json")
    attachments = load_json(case_dir / "attachments.json")
    cut_data = load_json(project_root / "04_results/pruning_truth/manual_cut_lines_20260720_raw" / sample_id / "cut_lines.json")
    image_path = project_root / "01_data/01_raw/final_data" / tree_id / "before" / f"{view}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    merged = build_merged_graph(groups)
    roots = choose_component_roots(merged)
    segments, point_nodes = trace_segments(merged, roots)
    segment_graph = orient_segment_graph(segments, point_nodes, roots)
    bud_nodes, bud_relations = attach_buds(attachments, segments)
    cut_matches = label_cuts(cut_data.get("cut_lines", []), segments, segment_graph, point_nodes)
    point_segment_relations = [
        {"type": "point_to_segment", "point_id": point_id, "segment_id": segment_id}
        for point_id, node in point_nodes.items()
        for segment_id in node["incident_segment_ids"]
    ]
    decision_graph = {
        "sample_id": sample_id,
        "representation": "heterogeneous_point_segment_bud_graph",
        "point_nodes": list(point_nodes.values()),
        "segment_nodes": segments,
        "bud_nodes": bud_nodes,
        "relations": point_segment_relations + bud_relations,
        "cut_matches": cut_matches,
    }

    old_undirected = nx.Graph()
    old_undirected.add_nodes_from(int(node["id"]) for node in old_graph.get("nodes", []))
    old_undirected.add_edges_from((int(edge["src"]), int(edge["dst"])) for edge in old_graph.get("edges", []))
    covered_edges = sum(max(len(segment["polyline"]) - 1, 0) for segment in segments)
    metrics = {
        "sample_id": sample_id,
        "old_nodes": len(old_graph.get("nodes", [])),
        "old_edges": len(old_graph.get("edges", [])),
        "old_components": int(nx.number_connected_components(old_undirected)) if old_undirected.number_of_nodes() else 0,
        "old_roots": len(set(int(node["id"]) for node in old_graph.get("nodes", [])) - {int(edge["dst"]) for edge in old_graph.get("edges", [])}),
        "merged_skeleton_nodes": int(merged.number_of_nodes()),
        "merged_skeleton_edges": int(merged.number_of_edges()),
        "merged_components": int(nx.number_connected_components(merged)),
        "merged_cycle_rank": int(merged.number_of_edges() - merged.number_of_nodes() + nx.number_connected_components(merged)),
        "point_nodes": len(point_nodes),
        "segment_nodes": len(segments),
        "bud_nodes": len(bud_nodes),
        "relations": len(point_segment_relations) + len(bud_relations),
        "covered_skeleton_edges": int(covered_edges),
        "edge_coverage_ratio": float(covered_edges / max(merged.number_of_edges(), 1)),
        "junction_angle_max_error_deg": junction_angle_error(merged, point_nodes),
        "max_chord_deviation_px": float(max((segment["max_chord_deviation_px"] for segment in segments), default=0.0)),
        "mean_sinuosity": float(np.mean([segment["sinuosity"] for segment in segments])) if segments else 0.0,
        "cut_lines": len(cut_matches),
        "cut_exact": sum(match["status"] == "exact" for match in cut_matches),
        "cut_ambiguous": sum(match["status"] == "ambiguous" for match in cut_matches),
        "cut_nonintersecting": sum(match["status"] == "nonintersecting" for match in cut_matches),
        "pruned_segments": sum(segment["fate_label"] == "pruned" for segment in segments),
    }

    case_output = ensure_dir(output_root / case_dir.name)
    save_json(case_output / "decision_graph.json", decision_graph)
    save_json(case_output / "comparison_metrics.json", metrics)
    existing_overlay = add_header(draw_existing_graph(image, old_graph), "Existing directed graph")
    decision_overlay = add_header(draw_decision_graph(image, decision_graph, cut_data.get("cut_lines", [])), "Point-segment-bud decision graph")
    comparison = np.hstack([existing_overlay, decision_overlay])
    cv2.imwrite(str(case_output / "existing_directed_graph_overlay.jpg"), existing_overlay)
    cv2.imwrite(str(case_output / "decision_graph_overlay.jpg"), decision_overlay)
    cv2.imwrite(str(case_output / "comparison.jpg"), comparison)
    cv2.imwrite(str(case_output / "cut_diagnostics.jpg"), draw_cut_diagnostics(image, decision_graph, cut_data.get("cut_lines", [])))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype a geometry-preserving pruning decision graph")
    parser.add_argument("--cases-root", type=Path, default=Path("04_results/pruning_decision/e2e_regression_20260717_cases"))
    parser.add_argument("--output", type=Path, default=Path("04_results/pruning_decision/decision_graph_prototype_20260721"))
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["tree_003_before_view_01", "tree_130_before_view_04"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    cases_root = args.cases_root if args.cases_root.is_absolute() else project_root / args.cases_root
    output_root = ensure_dir(args.output if args.output.is_absolute() else project_root / args.output)
    rows = [process_case(project_root, cases_root / case_name, output_root) for case_name in args.cases]
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_root / "comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        output_root / "summary.json",
        {
            "status": "prototype",
            "result_class": "exploratory_data_preparation",
            "samples": len(rows),
            "metrics": rows,
            "prohibited_inference": "Not pruning accuracy and not evidence that the graph is botanically correct.",
        },
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
