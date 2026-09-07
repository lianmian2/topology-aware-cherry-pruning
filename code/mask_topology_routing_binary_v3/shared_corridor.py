from __future__ import annotations

import math
from collections import defaultdict
from itertools import permutations
from typing import Any, Sequence

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from mask_topology_routing.utils import render_topology_groups_to_mask

from .config import BinaryV3Config


def _graph(group: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(group.get("points", []))))
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = map(int, edge)
        if src != dst and src in graph and dst in graph:
            graph.add_edge(src, dst)
    return graph


def _family(group: dict[str, Any]) -> str:
    return str(group.get("group_id", "")).split("_root_", 1)[0]


def _distance(left: Sequence[int], right: Sequence[int]) -> float:
    return float(math.dist(tuple(map(float, left)), tuple(map(float, right))))


def _nearest_nodes(
    left: dict[str, Any], right: dict[str, Any], left_nodes: Sequence[int] | None = None
) -> tuple[float, int, int]:
    left_points = left.get("points", [])
    right_points = right.get("points", [])
    candidates = left_nodes if left_nodes is not None else range(len(left_points))
    return min(
        (_distance(left_points[left_index], right_point), int(left_index), int(right_index))
        for left_index in candidates
        for right_index, right_point in enumerate(right_points)
    )


def _line_supported(start: Sequence[int], end: Sequence[int], mask: np.ndarray, max_blank: int) -> bool:
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    cv2.line(canvas, tuple(map(int, start)), tuple(map(int, end)), 1, 1, cv2.LINE_8)
    values = (mask[canvas > 0] > 0).tolist()
    longest = current = 0
    for supported in values:
        current = 0 if supported else current + 1
        longest = max(longest, current)
    return longest <= int(max_blank)


def _path_length(points: Sequence[Sequence[int]], path: Sequence[int]) -> float:
    return float(sum(_distance(points[src], points[dst]) for src, dst in zip(path[:-1], path[1:])))


def _root_endpoints(
    group: dict[str, Any], graph: nx.Graph, trunk_distance: np.ndarray, tolerance: float
) -> list[int]:
    result = []
    for node in graph:
        if graph.degree[node] != 1:
            continue
        x, y = map(int, group["points"][node])
        if 0 <= y < trunk_distance.shape[0] and 0 <= x < trunk_distance.shape[1]:
            if float(trunk_distance[y, x]) <= tolerance:
                result.append(int(node))
    return result


def find_shared_corridor_candidates(
    groups: Sequence[dict[str, Any]], source_mask: np.ndarray, config: BinaryV3Config
) -> list[dict[str, Any]]:
    trunk = next((group for group in groups if group.get("group_type") == "trunk"), None)
    if trunk is None:
        return []
    trunk_mask = render_topology_groups_to_mask([trunk], source_mask.shape).astype(np.uint8)
    trunk_distance = distance_transform_edt(trunk_mask == 0)
    root_tolerance = max(config.explicit_root_tolerance_px, config.strict_tolerance_px)
    corridor_stats = config.rulebook["summary"]["h_corridor_length_px"]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if group.get("group_type") != "trunk" and "_root_" in str(group.get("group_id", "")):
            families[_family(group)].append(group)
    candidates: list[dict[str, Any]] = []
    for family, members in sorted(families.items()):
        if len(members) < 3:
            continue
        graphs = {str(group["group_id"]): _graph(group) for group in members}
        for continuation, owner, rooted_tail in permutations(members, 3):
            continuation_id = str(continuation["group_id"])
            owner_id = str(owner["group_id"])
            tail_id = str(rooted_tail["group_id"])
            continuation_graph = graphs[continuation_id]
            owner_graph = graphs[owner_id]
            tail_graph = graphs[tail_id]
            if not all(nx.is_tree(graph) for graph in (continuation_graph, owner_graph, tail_graph)):
                continue
            tail_endpoints = [node for node in tail_graph if tail_graph.degree[node] == 1]
            if not tail_endpoints:
                continue
            first_gap, continuation_node, owner_start = _nearest_nodes(continuation, owner)
            second_gap, tail_node, owner_end = _nearest_nodes(rooted_tail, owner, tail_endpoints)
            if first_gap > config.structural_tolerance_px or second_gap > config.structural_tolerance_px:
                continue
            if owner_start == owner_end:
                continue
            owner_path = nx.shortest_path(owner_graph, owner_start, owner_end)
            corridor_length = _path_length(owner["points"], owner_path)
            if not float(corridor_stats["p05"]) <= corridor_length <= float(corridor_stats["p95"]):
                continue
            continuation_roots = _root_endpoints(
                continuation, continuation_graph, trunk_distance, root_tolerance
            )
            owner_roots = _root_endpoints(owner, owner_graph, trunk_distance, root_tolerance)
            tail_roots = _root_endpoints(rooted_tail, tail_graph, trunk_distance, root_tolerance)
            if len(continuation_roots) != 1 or len(owner_roots) != 1 or len(tail_roots) != 1:
                continue
            continuation_root = continuation_roots[0]
            if continuation_root == continuation_node:
                continue
            root_arm = nx.shortest_path(continuation_graph, continuation_node, continuation_root)
            if not _line_supported(
                continuation["points"][continuation_node], owner["points"][owner_start],
                source_mask, int(config.repair_max_blank_px),
            ):
                continue
            if not _line_supported(
                rooted_tail["points"][tail_node], owner["points"][owner_end],
                source_mask, int(config.repair_max_blank_px),
            ):
                continue
            candidates.append({
                "family": family,
                "continuation_group_id": continuation_id,
                "corridor_owner_group_id": owner_id,
                "rooted_tail_group_id": tail_id,
                "continuation_node": int(continuation_node),
                "continuation_root_node_to_reassign": int(continuation_root),
                "continuation_root_arm": [int(node) for node in root_arm],
                "owner_start_node": int(owner_start),
                "owner_end_node": int(owner_end),
                "owner_corridor_path": [int(node) for node in owner_path],
                "tail_node": int(tail_node),
                "first_gap_px": float(first_gap),
                "second_gap_px": float(second_gap),
                "corridor_length_px": float(corridor_length),
                "corridor_rule_source": {
                    "rulebook": str(config.rulebook_path),
                    "field": "summary.h_corridor_length_px.p05_to_p95",
                },
            })
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            candidate["family"], candidate["continuation_group_id"],
            candidate["corridor_owner_group_id"], candidate["rooted_tail_group_id"],
        )
        previous = unique.get(key)
        gap = candidate["first_gap_px"] + candidate["second_gap_px"]
        if previous is None or gap < previous["first_gap_px"] + previous["second_gap_px"]:
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda item: (item["first_gap_px"] + item["second_gap_px"], item["corridor_length_px"]),
    )


def _append_group(
    target: dict[str, Any], source: dict[str, Any], selected_edges: Sequence[Sequence[int]] | None = None
) -> tuple[dict[str, Any], dict[int, int]]:
    result = {
        **target,
        "points": [list(map(int, point)) for point in target.get("points", [])],
        "edges": [list(map(int, edge)) for edge in target.get("edges", [])],
    }
    lookup = {tuple(point): index for index, point in enumerate(result["points"])}
    remap: dict[int, int] = {}
    selected_nodes = None
    if selected_edges is not None:
        selected_nodes = {int(node) for edge in selected_edges for node in edge if len(edge) == 2}
    for source_index, point in enumerate(source.get("points", [])):
        if selected_nodes is not None and source_index not in selected_nodes:
            continue
        key = tuple(map(int, point))
        if key not in lookup:
            lookup[key] = len(result["points"])
            result["points"].append(list(key))
        remap[source_index] = lookup[key]
    existing = {tuple(sorted(map(int, edge))) for edge in result["edges"] if len(edge) == 2}
    for edge in selected_edges if selected_edges is not None else source.get("edges", []):
        if len(edge) != 2:
            continue
        if int(edge[0]) not in remap or int(edge[1]) not in remap:
            continue
        mapped = tuple(sorted((remap[int(edge[0])], remap[int(edge[1])])))
        if mapped[0] != mapped[1]:
            existing.add(mapped)
    result["edges"] = [list(edge) for edge in sorted(existing)]
    return result, remap


def _add_edge(group: dict[str, Any], left: int, right: int) -> None:
    edge = tuple(sorted((int(left), int(right))))
    existing = {tuple(sorted(map(int, item))) for item in group.get("edges", []) if len(item) == 2}
    if edge[0] != edge[1] and edge not in existing:
        group.setdefault("edges", []).append(list(edge))


def apply_unique_shared_corridors(
    groups: Sequence[dict[str, Any]], source_mask: np.ndarray, config: BinaryV3Config
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refined = [
        {
            **group,
            "points": [list(map(int, point)) for point in group.get("points", [])],
            "edges": [list(map(int, edge)) for edge in group.get("edges", [])],
        }
        for group in groups
    ]
    candidates = find_shared_corridor_candidates(refined, source_mask, config)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_family[candidate["family"]].append(candidate)
    selected = [items[0] for items in by_family.values() if len(items) == 1]
    applied: list[dict[str, Any]] = []
    for candidate in selected:
        by_id = {str(group.get("group_id")): index for index, group in enumerate(refined)}
        continuation_index = by_id.get(candidate["continuation_group_id"])
        owner_index = by_id.get(candidate["corridor_owner_group_id"])
        tail_index = by_id.get(candidate["rooted_tail_group_id"])
        if continuation_index is None or owner_index is None or tail_index is None:
            continue
        continuation = refined[continuation_index]
        owner = refined[owner_index]
        tail = refined[tail_index]
        owner_path = candidate["owner_corridor_path"]
        owner_edges = [[src, dst] for src, dst in zip(owner_path[:-1], owner_path[1:])]
        merged, tail_remap = _append_group(continuation, tail)
        merged, owner_remap = _append_group(merged, owner, selected_edges=owner_edges)
        continuation_node = int(candidate["continuation_node"])
        tail_node = tail_remap[int(candidate["tail_node"])]
        owner_start = owner_remap[int(candidate["owner_start_node"])]
        owner_end = owner_remap[int(candidate["owner_end_node"])]
        _add_edge(merged, continuation_node, owner_start)
        _add_edge(merged, tail_node, owner_end)
        corridor_id = (
            f"shared_corridor::{candidate['family']}::"
            f"{candidate['corridor_owner_group_id']}::{candidate['owner_start_node']}-{candidate['owner_end_node']}"
        )
        corridor_keys = [
            [owner["points"][src], owner["points"][dst]] for src, dst in owner_edges
        ]
        crossing = {
            "xy": list(map(int, continuation["points"][int(candidate["continuation_root_node_to_reassign"])])),
            "parent_group_id": "trunk",
            "reason": "projection_crossing_not_botanical_root",
        }
        merged["group_id"] = str(continuation.get("group_id"))
        merged["v31_shared_corridor_ids"] = sorted(
            set(merged.get("v31_shared_corridor_ids", [])) | {corridor_id}
        )
        merged["v31_shared_corridor_edge_keys"] = [
            *merged.get("v31_shared_corridor_edge_keys", []),
            {"corridor_id": corridor_id, "edge_xy": corridor_keys, "canonical_owner_group_id": owner["group_id"]},
        ]
        merged["non_topological_parent_crossings"] = [
            *merged.get("non_topological_parent_crossings", []), crossing,
        ]
        merged["v31_merged_group_ids"] = sorted(
            set(merged.get("v31_merged_group_ids", [str(continuation.get("group_id"))]))
            | {str(tail.get("group_id"))}
        )
        owner["v31_shared_corridor_ids"] = sorted(
            set(owner.get("v31_shared_corridor_ids", [])) | {corridor_id}
        )
        owner["v31_shared_corridor_edge_keys"] = [
            *owner.get("v31_shared_corridor_edge_keys", []),
            {"corridor_id": corridor_id, "edge_xy": corridor_keys, "canonical_owner_group_id": owner["group_id"]},
        ]
        refined[continuation_index] = merged
        refined[owner_index] = owner
        refined.pop(tail_index)
        applied.append({**candidate, "corridor_id": corridor_id, "status": "applied_unique_family_candidate"})
    return refined, {
        "candidate_count": len(candidates),
        "unique_family_candidate_count": len(selected),
        "ambiguous_families": sorted(family for family, items in by_family.items() if len(items) != 1),
        "applied": applied,
    }
