from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from mask_topology_routing.utils import (
    _group_contact_count,
    finalize_clipped_annotation_groups,
    render_topology_groups_to_mask,
)

from .config import BinaryV3Config


def _copy_groups(groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **group,
            "points": [list(map(int, point)) for point in group.get("points", [])],
            "edges": [list(map(int, edge)) for edge in group.get("edges", [])],
        }
        for group in groups
    ]


def _group_graph(group: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(group.get("points", []))))
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = map(int, edge)
        if src != dst and src in graph and dst in graph:
            graph.add_edge(src, dst)
    return graph


def _ordered_path(group: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]] | None:
    graph = _group_graph(group)
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1 or not nx.is_connected(graph):
        return None
    endpoints = [node for node in graph if graph.degree[node] == 1]
    if len(endpoints) != 2 or any(graph.degree[node] > 2 for node in graph):
        return None
    points = group["points"]
    start = max(endpoints, key=lambda node: (points[node][1], -points[node][0]))
    terminal = endpoints[0] if endpoints[1] == start else endpoints[1]
    path = nx.shortest_path(graph, start, terminal)
    ordered = [list(map(int, points[node])) for node in path]
    return ordered, [[index, index + 1] for index in range(len(ordered) - 1)]


def _arc(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((len(points),), dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(points[1:] - points[:-1], axis=1))])


def _project(point: np.ndarray, points: np.ndarray, arc: np.ndarray) -> tuple[float, float, int]:
    best = (float("inf"), 0.0, 0)
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        vector = end - start
        denominator = float(np.dot(vector, vector))
        ratio = 0.0 if denominator < 1e-9 else float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
        projected = start + ratio * vector
        distance = float(np.linalg.norm(point - projected))
        if distance < best[0]:
            best = (distance, float(arc[index] + ratio * (arc[index + 1] - arc[index])), index)
    return best


def _path_tortuosity(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    length = float(_arc(points)[-1])
    chord = float(np.linalg.norm(points[-1] - points[0]))
    return length / max(chord, 1e-6)


def _group_root_projection(group: dict[str, Any], trunk_points: np.ndarray, trunk_arc: np.ndarray) -> tuple[float, float]:
    candidates = []
    for point in group.get("points", []):
        candidates.append(_project(np.asarray(point, dtype=np.float64), trunk_points, trunk_arc)[:2])
    return min(candidates, default=(float("inf"), 0.0), key=lambda item: item[0])


def _merge_group_payload(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    result = {**target, "points": [list(point) for point in target["points"]], "edges": [list(edge) for edge in target["edges"]]}
    lookup = {tuple(point): index for index, point in enumerate(result["points"])}
    remap: dict[int, int] = {}
    for old_index, point in enumerate(source.get("points", [])):
        key = tuple(map(int, point))
        if key not in lookup:
            lookup[key] = len(result["points"])
            result["points"].append(list(key))
        remap[old_index] = lookup[key]
    edges = {tuple(sorted(map(int, edge))) for edge in result["edges"] if len(edge) == 2}
    for edge in source.get("edges", []):
        if len(edge) == 2:
            mapped = tuple(sorted((remap[int(edge[0])], remap[int(edge[1])])))
            if mapped[0] != mapped[1]:
                edges.add(mapped)
    result["edges"] = [list(edge) for edge in sorted(edges)]
    result["v3_merged_group_ids"] = sorted(
        set(result.get("v3_merged_group_ids", [str(target.get("group_id"))]))
        | set(source.get("v3_merged_group_ids", [str(source.get("group_id"))]))
    )
    return result


def _merge_supported_root_families(
    groups: list[dict[str, Any]], source_mask: np.ndarray, config: BinaryV3Config
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trunk = next((group for group in groups if group.get("group_type") == "trunk"), None)
    if trunk is None:
        return groups, []
    trunk_mask = render_topology_groups_to_mask([trunk], source_mask.shape).astype(np.uint8)
    trunk_distance = distance_transform_edt(trunk_mask == 0)
    families: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        if group.get("group_type") == "trunk":
            continue
        families[str(group.get("group_id", "")).split("_root_", 1)[0]].append(index)
    removed: set[int] = set()
    changes: list[dict[str, Any]] = []
    for family, indices in sorted(families.items()):
        active = [index for index in indices if index not in removed]
        changed = True
        while changed and len(active) > 1:
            changed = False
            for offset, left_index in enumerate(active):
                for right_index in active[offset + 1 :]:
                    left_points = {tuple(point) for point in groups[left_index].get("points", [])}
                    right_points = {tuple(point) for point in groups[right_index].get("points", [])}
                    if not left_points.intersection(right_points):
                        continue
                    merged = _merge_group_payload(groups[left_index], groups[right_index])
                    graph = _group_graph(merged)
                    root_count = _group_contact_count(merged, trunk_distance, config.explicit_root_tolerance_px)
                    if graph.number_of_edges() and nx.is_tree(graph) and root_count == 1:
                        groups[left_index] = merged
                        removed.add(right_index)
                        changes.append({"family": family, "kept": groups[left_index]["group_id"], "merged": groups[right_index]["group_id"]})
                        active = [index for index in active if index != right_index]
                        changed = True
                        break
                if changed:
                    break
    return [group for index, group in enumerate(groups) if index not in removed], changes


def _refine_internal_trunk_terminal(
    groups: list[dict[str, Any]], source_mask: np.ndarray, config: BinaryV3Config
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trunk_index = next((index for index, group in enumerate(groups) if group.get("group_type") == "trunk"), None)
    if trunk_index is None:
        return groups, {"status": "not_applicable_no_trunk"}
    ordered_payload = _ordered_path(groups[trunk_index])
    if ordered_payload is None:
        return groups, {"status": "not_applicable_non_path_trunk"}
    ordered, ordered_edges = ordered_payload
    trunk_points = np.asarray(ordered, dtype=np.float64)
    trunk_arc = _arc(trunk_points)
    total = float(trunk_arc[-1])
    if total <= 0:
        return groups, {"status": "not_applicable_empty_trunk"}
    projections = []
    for index, group in enumerate(groups):
        if index == trunk_index:
            continue
        distance, arc_value = _group_root_projection(group, trunk_points, trunk_arc)
        if distance <= config.explicit_root_tolerance_px:
            projections.append((arc_value, index, distance))
    if not projections:
        return groups, {"status": "not_applicable_no_explicit_roots"}
    last_root_arc = max(item[0] for item in projections)
    extension_ratio = (total - last_root_arc) / total
    extension_distribution = config.distribution(
        "trunk_distributions", "terminal_extension_after_last_root_ratio"
    )
    candidate_extension = extension_distribution["p99"]
    maximum_extension = extension_distribution["max"]
    observed_tortuosity = _path_tortuosity(trunk_points)
    maximum_tortuosity = config.distribution("trunk_distributions", "tortuosity")["max"]
    audit = {
        "status": "unchanged",
        "observed_extension_ratio": extension_ratio,
        "candidate_extension_trigger": candidate_extension,
        "maximum_extension_limit": maximum_extension,
        "extension_rule_source": config.source_ref(
            "trunk_distributions", "terminal_extension_after_last_root_ratio", "p99"
        ),
        "observed_tortuosity": observed_tortuosity,
        "tortuosity_limit": maximum_tortuosity,
        "tortuosity_rule_source": config.source_ref("trunk_distributions", "tortuosity", "max"),
    }
    if extension_ratio <= candidate_extension or observed_tortuosity <= maximum_tortuosity:
        return groups, audit
    cut_index = int(np.argmin(np.abs(trunk_arc - last_root_arc)))
    if cut_index <= 0 or cut_index >= len(ordered) - 1:
        return groups, {**audit, "status": "rejected_invalid_cut_index"}
    prefix = trunk_points[: cut_index + 1]
    tail = ordered[cut_index:]
    minimum_trunk_length = config.distribution("trunk_distributions", "length_px")["min"]
    minimum_branch_length = config.distribution("branch_distributions", "length_px")["p05"]
    prefix_length = float(_arc(prefix)[-1])
    tail_length = float(_arc(np.asarray(tail, dtype=np.float64))[-1])
    if prefix_length < minimum_trunk_length or tail_length < minimum_branch_length:
        return groups, {**audit, "status": "rejected_rulebook_length_guard"}
    if _path_tortuosity(prefix) > maximum_tortuosity:
        return groups, {**audit, "status": "rejected_prefix_still_implausible"}
    existing_ids = {str(group.get("group_id")) for group in groups}
    tail_id = "branch_v3_trunk_tail"
    suffix = 1
    while tail_id in existing_ids:
        suffix += 1
        tail_id = f"branch_v3_trunk_tail_{suffix:02d}"
    old_trunk_id = str(groups[trunk_index].get("group_id", "trunk"))
    groups[trunk_index] = {
        **groups[trunk_index],
        "points": [list(map(int, point)) for point in prefix.tolist()],
        "edges": [[index, index + 1] for index in range(len(prefix) - 1)],
        "v3_internal_terminal_refined": True,
    }
    tail_group = {
        "group_id": tail_id,
        "group_type": "branch",
        "color_hex": "#9C27B0",
        "points": [list(map(int, point)) for point in tail],
        "edges": [[index, index + 1] for index in range(len(tail) - 1)],
        "fork_origin_group": old_trunk_id,
        "parent_group_id": old_trunk_id,
        "v3_reclassified_from_trunk": True,
    }
    groups.append(tail_group)
    tail_mask = render_topology_groups_to_mask([tail_group], source_mask.shape).astype(np.uint8)
    tail_distance = distance_transform_edt(tail_mask == 0)
    new_trunk_mask = render_topology_groups_to_mask([groups[trunk_index]], source_mask.shape).astype(np.uint8)
    new_trunk_distance = distance_transform_edt(new_trunk_mask == 0)
    promoted = []
    for index, group in enumerate(groups[:-1]):
        if index == trunk_index:
            continue
        trunk_contacts = _group_contact_count(group, new_trunk_distance, config.explicit_root_tolerance_px)
        tail_contacts = _group_contact_count(group, tail_distance, config.explicit_root_tolerance_px)
        if trunk_contacts == 0 and tail_contacts == 1:
            group["group_type"] = "secondary_branch"
            group["parent_group_id"] = tail_id
            promoted.append(str(group.get("group_id")))
    return groups, {
        **audit,
        "status": "refined",
        "cut_index": cut_index,
        "prefix_length_px": prefix_length,
        "reclassified_tail_length_px": tail_length,
        "tail_group_id": tail_id,
        "secondary_groups_reparented": promoted,
    }


def refine_clipped_groups_v3(
    groups: Sequence[dict[str, Any]], source_mask: np.ndarray, config: BinaryV3Config
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply only binary-supported, auditable v3 refinements after mask_clip."""
    refined = _copy_groups(groups)
    repair_stats: dict[str, Any] = {}
    if config.enable_postclip_connectivity_repair:
        refined, repair_stats = finalize_clipped_annotation_groups(
            refined,
            source_mask,
            structural_tolerance_px=config.structural_tolerance_px,
            enable_postclip_crossing_reassignment=True,
        )
    merge_changes: list[dict[str, Any]] = []
    if config.enable_supported_root_family_merge:
        refined, merge_changes = _merge_supported_root_families(refined, source_mask, config)
    trunk_refinement = {"status": "disabled"}
    if config.enable_internal_trunk_termination:
        refined, trunk_refinement = _refine_internal_trunk_terminal(refined, source_mask, config)
    return refined, {
        "postclip_connectivity_repair": repair_stats,
        "supported_root_family_merges": merge_changes,
        "internal_trunk_terminal": trunk_refinement,
        "groups_before": len(groups),
        "groups_after": len(refined),
    }
