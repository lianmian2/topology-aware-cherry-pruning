from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from mask_topology_routing.utils import render_topology_groups_to_mask

from .config import BinaryV3Config


def _sample_group(group: dict[str, Any], spacing: float = 1.0) -> np.ndarray:
    points = np.asarray(group.get("points", []), dtype=np.float64)
    samples: list[np.ndarray] = []
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = map(int, edge)
        if not (0 <= src < len(points) and 0 <= dst < len(points)):
            continue
        start, end = points[src], points[dst]
        count = max(int(math.ceil(float(np.linalg.norm(end - start)) / spacing)), 1)
        ratio = np.linspace(0.0, 1.0, count + 1)[:, None]
        samples.append(start[None, :] * (1.0 - ratio) + end[None, :] * ratio)
    if not samples:
        return points
    return np.unique(np.rint(np.concatenate(samples, axis=0)).astype(np.int32), axis=0).astype(np.float64)


def _geometry_contact_count(group: dict[str, Any], parent: dict[str, Any], tolerance: float) -> int:
    child_points = _sample_group(group)
    parent_points = _sample_group(parent)
    if len(child_points) == 0 or len(parent_points) == 0:
        return 0
    contact = child_points[cKDTree(parent_points).query(child_points, k=1)[0] <= tolerance]
    if len(contact) == 0:
        return 0
    integer_points = {tuple(map(int, point)) for point in contact}
    clusters: list[set[tuple[int, int]]] = []
    while integer_points:
        cluster = {integer_points.pop()}
        stack = list(cluster)
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (x + dx, y + dy)
                    if neighbor in integer_points:
                        integer_points.remove(neighbor)
                        cluster.add(neighbor)
                        stack.append(neighbor)
        clusters.append(cluster)
    ignored = [
        np.asarray(item.get("xy"), dtype=np.float64)
        for item in group.get("non_topological_parent_crossings", [])
        if item.get("parent_group_id") in {None, str(parent.get("group_id"))} and len(item.get("xy", [])) == 2
    ]
    if ignored:
        clusters = [
            cluster for cluster in clusters
            if all(
                min(np.linalg.norm(np.asarray(point, dtype=np.float64) - center) for point in cluster)
                > max(24.0, tolerance)
                for center in ignored
            )
        ]
    return len(clusters)


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


def _length(group: dict[str, Any]) -> float:
    points = group.get("points", [])
    total = 0.0
    for edge in group.get("edges", []):
        if len(edge) == 2 and all(0 <= int(node) < len(points) for node in edge):
            total += math.dist(points[int(edge[0])], points[int(edge[1])])
    return total


def _trunk_path_metrics(trunk: dict[str, Any], image_shape: tuple[int, int]) -> dict[str, float | None]:
    graph = _graph(trunk)
    endpoints = [node for node in graph if graph.degree[node] == 1]
    if len(endpoints) != 2 or not nx.is_connected(graph):
        return {"length_px": _length(trunk), "tortuosity": None, "lateral_drift_norm_width": None}
    points = trunk.get("points", [])
    root = max(endpoints, key=lambda node: points[node][1])
    terminal = endpoints[0] if endpoints[1] == root else endpoints[1]
    path = nx.shortest_path(graph, root, terminal)
    ordered = np.asarray([points[node] for node in path], dtype=np.float64)
    segment_lengths = np.linalg.norm(ordered[1:] - ordered[:-1], axis=1)
    length = float(segment_lengths.sum())
    chord = float(np.linalg.norm(ordered[-1] - ordered[0]))
    width = max(int(image_shape[1]), 1)
    return {
        "length_px": length,
        "tortuosity": length / max(chord, 1e-6),
        "lateral_drift_norm_width": float(np.ptp(ordered[:, 0])) / width,
        "root_y": float(ordered[0, 1]),
        "root_x": float(ordered[0, 0]),
        "terminal_y": float(ordered[-1, 1]),
    }


def _physical_edge_owners(groups: Sequence[dict[str, Any]]) -> dict[tuple[tuple[int, int], tuple[int, int]], set[str]]:
    owners: dict[tuple[tuple[int, int], tuple[int, int]], set[str]] = defaultdict(set)
    for group in groups:
        points = [tuple(map(int, point)) for point in group.get("points", [])]
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = map(int, edge)
            if src != dst and 0 <= src < len(points) and 0 <= dst < len(points):
                owners[tuple(sorted((points[src], points[dst])))].add(str(group.get("group_id")))
    return owners


def _declared_shared_edge_ids(groups: Sequence[dict[str, Any]]) -> dict[tuple[tuple[int, int], tuple[int, int]], set[str]]:
    declared: dict[tuple[tuple[int, int], tuple[int, int]], set[str]] = defaultdict(set)
    for group in groups:
        for item in group.get("v31_shared_corridor_edge_keys", []):
            corridor_id = str(item.get("corridor_id", ""))
            for edge in item.get("edge_xy", []):
                if corridor_id and len(edge) == 2:
                    declared[tuple(sorted((tuple(map(int, edge[0])), tuple(map(int, edge[1])))))].add(corridor_id)
    return declared


def _long_channel_coverage(
    groups: Sequence[dict[str, Any]], source_mask: np.ndarray, config: BinaryV3Config
) -> dict[str, Any]:
    source_skeleton = skeletonize(source_mask > 0).astype(np.uint8)
    rendered = render_topology_groups_to_mask(groups, source_mask.shape).astype(np.uint8)
    distance = distance_transform_edt(rendered == 0)
    total = int(source_skeleton.sum())
    coverage_24 = float(np.mean(distance[source_skeleton > 0] <= config.structural_tolerance_px)) if total else 1.0
    coverage_5 = float(np.mean(distance[source_skeleton > 0] <= config.strict_tolerance_px)) if total else 1.0
    covered_pixels_24 = int(np.sum(distance[source_skeleton > 0] <= config.structural_tolerance_px)) if total else 0
    count, labels = cv2.connectedComponents(source_skeleton, connectivity=8)
    minimum_length = config.distribution("branch_distributions", "length_px")["p05"]
    long_components = []
    for component_id in range(1, count):
        component = labels == component_id
        pixels = int(component.sum())
        if pixels < minimum_length:
            continue
        component_coverage = float(np.mean(distance[component] <= config.structural_tolerance_px))
        long_components.append({"component_id": component_id, "skeleton_pixels": pixels, "coverage_24px": component_coverage})
    return {
        "source_skeleton_pixels": total,
        "coverage_24px": coverage_24,
        "coverage_5px": coverage_5,
        "covered_pixels_24px": covered_pixels_24,
        "long_component_threshold_px": minimum_length,
        "threshold_rule_source": config.source_ref("branch_distributions", "length_px", "p05"),
        "long_components": long_components,
        "long_components_below_90pct": sum(item["coverage_24px"] < 0.9 for item in long_components),
    }


def audit_binary_topology_v3(
    groups: Sequence[dict[str, Any]], source_mask: np.ndarray, config: BinaryV3Config,
    baseline_groups: Sequence[dict[str, Any]] | None = None,
    include_global_checks: bool = True,
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    trunk = next((group for group in groups if group.get("group_type") == "trunk"), None)
    if trunk is None:
        return {"status": "excluded", "exclusion_reasons": ["missing_trunk"]}
    shape = source_mask.shape[:2]
    parent_groups = {str(group.get("group_id")): group for group in groups}
    rows = []
    zero_roots = multiple_roots = disconnected = cyclic = isolated_nodes = 0
    for group in groups:
        if group is trunk:
            continue
        graph = _graph(group)
        components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
        cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + components
        isolated = sum(graph.degree[node] == 0 for node in graph)
        parent_id = str(group.get("parent_group_id") or trunk.get("group_id"))
        parent = parent_groups.get(parent_id, trunk)
        root_count = _geometry_contact_count(group, parent, config.explicit_root_tolerance_px)
        zero_roots += root_count == 0
        multiple_roots += root_count > 1
        disconnected += components != 1
        cyclic += cycle_rank > 0
        isolated_nodes += isolated
        rows.append({
            "group_id": str(group.get("group_id")),
            "group_type": str(group.get("group_type")),
            "parent_group_id": parent_id,
            "explicit_root_count": int(root_count),
            "connected_components": int(components),
            "cycle_rank": int(cycle_rank),
            "isolated_nodes": int(isolated),
            "length_px": _length(group),
        })
    if disconnected:
        reasons.append("disconnected_graph_groups")
    if isolated_nodes:
        reasons.append("isolated_graph_nodes")
    if cyclic:
        reasons.append("cyclic_branch_groups")
    if zero_roots:
        reasons.append("zero_explicit_root_groups")
    if multiple_roots:
        reasons.append("multiple_explicit_root_groups")
    shared_edge_ids = _declared_shared_edge_ids(groups)
    duplicate_edges = {
        edge: owners for edge, owners in _physical_edge_owners(groups).items()
        if len(owners) > 1 and len(shared_edge_ids.get(edge, set())) != 1
    }
    if duplicate_edges:
        reasons.append("duplicate_physical_edges")
    trunk_metrics = _trunk_path_metrics(trunk, shape)
    tort_limit = config.distribution("trunk_distributions", "tortuosity")["max"]
    lateral_limit = config.distribution("trunk_distributions", "lateral_drift_norm_width")["max"]
    if include_global_checks and trunk_metrics.get("tortuosity") is None:
        reasons.append("trunk_is_not_single_path")
    elif include_global_checks and float(trunk_metrics["tortuosity"]) > tort_limit:
        reasons.append("trunk_tortuosity_outside_rulebook")
    if include_global_checks and trunk_metrics.get("lateral_drift_norm_width") is not None and float(trunk_metrics["lateral_drift_norm_width"]) > lateral_limit:
        warnings.append("trunk_lateral_drift_outside_rulebook")
    if include_global_checks:
        source_binary = (source_mask > 0).astype(np.uint8)
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            source_binary, connectivity=8
        )
    else:
        component_count = 0
        component_labels = np.zeros((1, 1), dtype=np.int32)
        component_stats = np.zeros((1, 5), dtype=np.int32)
    root_gap = None
    aligned_lower_components: list[dict[str, Any]] = []
    root_gap_limit = config.distribution("trunk_distributions", "root_to_tree_bottom_norm_height")["max"]
    if include_global_checks and component_count > 1 and trunk_metrics.get("root_y") is not None:
        root_x = int(round(float(trunk_metrics["root_x"])))
        root_y = int(round(float(trunk_metrics["root_y"])))
        root_x = int(np.clip(root_x, 0, shape[1] - 1))
        root_y = int(np.clip(root_y, 0, shape[0] - 1))
        root_component = int(component_labels[root_y, root_x])
        if root_component <= 0:
            search_radius = int(math.ceil(config.explicit_root_tolerance_px))
            y0, y1 = max(0, root_y - search_radius), min(shape[0], root_y + search_radius + 1)
            x0, x1 = max(0, root_x - search_radius), min(shape[1], root_x + search_radius + 1)
            local = component_labels[y0:y1, x0:x1]
            positive = local[local > 0]
            if positive.size:
                root_component = int(Counter(map(int, positive)).most_common(1)[0][0])
        if root_component > 0:
            stat = component_stats[root_component]
            component_bottom = int(stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT] - 1)
            root_gap = max(float(component_bottom - root_y), 0.0) / max(shape[0], 1)
            lateral_limit_px = config.distribution("trunk_distributions", "lateral_drift_norm_width")["max"] * shape[1]
            vertical_search_px = config.distribution("branch_distributions", "length_px")["p05"]
            for component_id in range(1, component_count):
                if component_id == root_component:
                    continue
                other = component_stats[component_id]
                other_top = int(other[cv2.CC_STAT_TOP])
                other_left = int(other[cv2.CC_STAT_LEFT])
                other_right = other_left + int(other[cv2.CC_STAT_WIDTH]) - 1
                vertical_gap = other_top - root_y
                horizontal_gap = 0.0 if other_left <= root_x <= other_right else min(abs(root_x - other_left), abs(root_x - other_right))
                if 0 < vertical_gap <= vertical_search_px and horizontal_gap <= lateral_limit_px:
                    aligned_lower_components.append({
                        "component_id": component_id,
                        "area_px": int(other[cv2.CC_STAT_AREA]),
                        "vertical_gap_px": int(vertical_gap),
                        "horizontal_gap_px": float(horizontal_gap),
                    })
        if root_gap is not None and root_gap > root_gap_limit:
            reasons.append("trunk_root_missing_supported_bottom")
        if aligned_lower_components:
            reasons.append("supported_aligned_lower_component_unrouted")
    channel = _long_channel_coverage(groups, source_mask, config) if include_global_checks else None
    baseline_channel = (
        _long_channel_coverage(baseline_groups, source_mask, config)
        if include_global_checks and baseline_groups is not None else None
    )
    allowed_short_spur_loss = config.distribution("branch_distributions", "length_px")["p05"]
    lost_supported_pixels = 0
    if baseline_channel is not None and channel is not None:
        lost_supported_pixels = max(
            int(baseline_channel["covered_pixels_24px"]) - int(channel["covered_pixels_24px"]), 0
        )
    if baseline_channel is not None and lost_supported_pixels >= allowed_short_spur_loss:
        reasons.append("v3_reduced_supported_long_channel_coverage")
    elif lost_supported_pixels > 0:
        warnings.append("short_spur_coverage_removed_below_rulebook_p05")
    elif channel is not None and channel["long_components_below_90pct"]:
        warnings.append("incomplete_long_binary_channels")
    return {
        "status": "accepted" if not reasons else "excluded",
        "exclusion_reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
        "counts": {
            "groups": len(groups),
            "branch_groups": len(rows),
            "zero_explicit_root_groups": zero_roots,
            "multiple_explicit_root_groups": multiple_roots,
            "disconnected_graph_groups": disconnected,
            "cyclic_branch_groups": cyclic,
            "isolated_graph_nodes": isolated_nodes,
            "duplicate_physical_edges": len(duplicate_edges),
        },
        "trunk": {
            **trunk_metrics,
            "root_to_supported_bottom_norm_height": root_gap,
            "tortuosity_limit": tort_limit,
            "tortuosity_rule_source": config.source_ref("trunk_distributions", "tortuosity", "max"),
            "lateral_drift_limit": lateral_limit,
            "lateral_drift_rule_source": config.source_ref("trunk_distributions", "lateral_drift_norm_width", "max"),
            "root_gap_limit": root_gap_limit,
            "root_gap_rule_source": config.source_ref("trunk_distributions", "root_to_tree_bottom_norm_height", "max"),
            "aligned_lower_components": aligned_lower_components,
            "aligned_component_lateral_rule_source": config.source_ref("trunk_distributions", "lateral_drift_norm_width", "max"),
            "aligned_component_vertical_rule_source": config.source_ref("branch_distributions", "length_px", "p05"),
        },
        "branches": rows,
        "long_channel_coverage": channel,
        "baseline_long_channel_coverage": baseline_channel,
        "lost_supported_pixels_vs_baseline": lost_supported_pixels,
        "long_channel_loss_threshold_px": allowed_short_spur_loss,
        "long_channel_loss_rule_source": config.source_ref("branch_distributions", "length_px", "p05"),
        "duplicate_edge_examples": [
            {"edge": [list(edge[0]), list(edge[1])], "owners": sorted(owners)}
            for edge, owners in list(duplicate_edges.items())[:20]
        ],
    }
