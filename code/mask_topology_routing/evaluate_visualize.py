"""纯几何 Mask 拓扑恢复评估脚本."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

try:
    from .data import load_manifest
    from .utils import (
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_RESULT_ROOT,
        build_prediction_result,
        decode_prediction_to_annotation,
        draw_prediction_overlay,
        ensure_dir,
        load_image_rgb,
        load_json,
        save_json,
    )
except ImportError:
    from data import load_manifest
    from utils import (
        DEFAULT_PROCESSED_ROOT,
        DEFAULT_RESULT_ROOT,
        build_prediction_result,
        decode_prediction_to_annotation,
        draw_prediction_overlay,
        ensure_dir,
        load_image_rgb,
        load_json,
        save_json,
    )


def merge_groups_to_global_graph(annotation: Dict, image_shape: Tuple[int, int], max_keypoints: int = 512) -> Dict:
    height, width = image_shape
    point_to_gid: Dict[Tuple[int, int], int] = {}
    unique_points: List[Tuple[int, int]] = []
    trunk_labels: List[int] = []
    adjacency = np.zeros((max_keypoints, max_keypoints), dtype=np.float32)

    def clip_point(point_xy: Sequence[float]) -> Tuple[int, int]:
        x = int(np.clip(round(float(point_xy[0])), 0, width - 1))
        y = int(np.clip(round(float(point_xy[1])), 0, height - 1))
        return x, y

    def register_point(point_xy: Sequence[float], is_trunk: bool) -> int:
        key = clip_point(point_xy)
        if key not in point_to_gid:
            if len(unique_points) >= max_keypoints:
                return -1
            point_to_gid[key] = len(unique_points)
            unique_points.append(key)
            trunk_labels.append(int(is_trunk))
        else:
            gid = point_to_gid[key]
            trunk_labels[gid] = max(trunk_labels[gid], int(is_trunk))
        return point_to_gid[key]

    for group in annotation.get("groups", []):
        local_to_global: Dict[int, int] = {}
        is_trunk = group.get("group_type") == "trunk"
        for local_idx, point in enumerate(group.get("points", [])):
            gid = register_point(point, is_trunk=is_trunk)
            if gid >= 0:
                local_to_global[local_idx] = gid
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src = local_to_global.get(int(edge[0]), -1)
            dst = local_to_global.get(int(edge[1]), -1)
            if src >= 0 and dst >= 0 and src != dst:
                adjacency[src, dst] = 1.0
                adjacency[dst, src] = 1.0

    num_points = len(unique_points)
    points = np.zeros((max_keypoints, 2), dtype=np.float32)
    trunk_arr = np.zeros((max_keypoints,), dtype=np.float32)
    valid_mask = np.zeros((max_keypoints,), dtype=np.float32)
    if num_points > 0:
        points[:num_points] = np.asarray(unique_points, dtype=np.float32)
        trunk_arr[:num_points] = np.asarray(trunk_labels, dtype=np.float32)
        valid_mask[:num_points] = 1.0
    return {"num_points": num_points, "points": points, "adjacency": adjacency, "trunk_labels": trunk_arr, "valid_mask": valid_mask}


def extract_annotation_points(annotation: Dict) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for group in annotation.get("groups", []):
        points.extend([(int(round(p[0])), int(round(p[1]))) for p in group.get("points", [])])
    return points


def render_annotation_groups(annotation: Dict, shape: Tuple[int, int], group_type: str | None = None, thickness: int = 2) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for group in annotation.get("groups", []):
        if group_type is not None and group.get("group_type") != group_type:
            continue
        points = [(int(round(p[0])), int(round(p[1]))) for p in group.get("points", [])]
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(points) and 0 <= dst < len(points):
                cv2.line(canvas, points[src], points[dst], color=1, thickness=thickness, lineType=cv2.LINE_8)
    return canvas


def _vivid_palette(n: int) -> List[Tuple[int, int, int]]:
    base = [
        (64, 220, 255),
        (64, 255, 120),
        (255, 210, 64),
        (200, 80, 255),
        (255, 128, 32),
        (32, 200, 255),
        (180, 255, 64),
        (255, 64, 180),
        (128, 160, 255),
        (255, 255, 64),
        (64, 255, 255),
        (96, 128, 255),
    ]
    if n <= len(base):
        return base[:n]
    palette = list(base)
    for idx in range(len(base), n):
        hue = 18 + int(144 * idx / max(n, 1))
        rgb = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
        palette.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return palette


def draw_group_overlay(image_rgb: np.ndarray, annotation: Dict, line_thickness: int = 5) -> np.ndarray:
    canvas = image_rgb.copy()
    branch_groups = [group for group in annotation.get("groups", []) if group.get("group_type") == "branch"]
    branch_palette = _vivid_palette(max(len(branch_groups), 1))
    branch_color_map = {group.get("group_id", f"branch_{idx:02d}"): branch_palette[idx] for idx, group in enumerate(branch_groups)}

    for group in annotation.get("groups", []):
        points = [tuple(map(int, (round(p[0]), round(p[1])))) for p in group.get("points", [])]
        if group.get("group_type") == "trunk":
            color = (255, 64, 64)
            thickness = line_thickness + 2
        else:
            color = branch_color_map.get(group.get("group_id"), (64, 255, 64))
            thickness = line_thickness
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(points) and 0 <= dst < len(points):
                cv2.line(canvas, points[src], points[dst], color=(0, 0, 0), thickness=thickness + 3, lineType=cv2.LINE_AA)
                cv2.line(canvas, points[src], points[dst], color=color, thickness=thickness, lineType=cv2.LINE_AA)
        if group.get("group_type") != "trunk" and points:
            center = tuple(map(int, np.mean(np.asarray(points, dtype=np.float32), axis=0)))
            cv2.putText(canvas, group.get("group_id", ""), center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, group.get("group_id", ""), center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    header = np.zeros((36, image_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, image_bgr])


def heatmap_to_color(prob: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    prob = prob - prob.min()
    prob = prob / max(float(prob.max()), 1e-6)
    return cv2.applyColorMap((prob * 255).astype(np.uint8), cmap)


def draw_gt_overlay(image_rgb: np.ndarray, annotation: Dict) -> np.ndarray:
    return draw_group_overlay(image_rgb, annotation, line_thickness=4)


def _group_masks(annotation: Dict, shape: Tuple[int, int], group_type: str = "branch", thickness: int = 2) -> List[Tuple[str, np.ndarray]]:
    masks: List[Tuple[str, np.ndarray]] = []
    for group in annotation.get("groups", []):
        if group.get("group_type") != group_type:
            continue
        single = {"groups": [group]}
        masks.append((group.get("group_id", f"{group_type}_{len(masks):02d}"), render_annotation_groups(single, shape, group_type=group_type, thickness=thickness)))
    return masks


def compute_group_consistency_metrics(pred_annotation: Dict, gt_annotation: Dict, shape: Tuple[int, int], thickness: int = 2) -> Dict[str, float]:
    pred_groups = _group_masks(pred_annotation, shape, group_type="branch", thickness=thickness)
    gt_groups = _group_masks(gt_annotation, shape, group_type="branch", thickness=thickness)
    if not pred_groups and not gt_groups:
        return {
            "group_precision": 1.0,
            "group_recall": 1.0,
            "group_f1": 1.0,
            "group_mean_iou": 1.0,
            "group_overlap_ratio": 0.0,
        }
    if not pred_groups or not gt_groups:
        return {
            "group_precision": 0.0,
            "group_recall": 0.0,
            "group_f1": 0.0,
            "group_mean_iou": 0.0,
            "group_overlap_ratio": 0.0,
        }

    pred_masks = [(gid, mask > 0) for gid, mask in pred_groups]
    gt_masks = [(gid, mask > 0) for gid, mask in gt_groups]
    rows, cols = len(pred_masks), len(gt_masks)
    precision_mat = np.zeros((rows, cols), dtype=np.float32)
    recall_mat = np.zeros((rows, cols), dtype=np.float32)
    f1_mat = np.zeros((rows, cols), dtype=np.float32)
    iou_mat = np.zeros((rows, cols), dtype=np.float32)
    for i, (_, pred_mask) in enumerate(pred_masks):
        pred_area = float(pred_mask.sum())
        for j, (_, gt_mask) in enumerate(gt_masks):
            inter = float(np.logical_and(pred_mask, gt_mask).sum())
            gt_area = float(gt_mask.sum())
            union = float(np.logical_or(pred_mask, gt_mask).sum())
            precision = inter / max(pred_area, 1.0)
            recall = inter / max(gt_area, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
            iou = inter / max(union, 1.0)
            precision_mat[i, j] = precision
            recall_mat[i, j] = recall
            f1_mat[i, j] = f1
            iou_mat[i, j] = iou

    row_ind, col_ind = linear_sum_assignment(1.0 - f1_mat)
    matched_f1 = f1_mat[row_ind, col_ind] if len(row_ind) else np.asarray([], dtype=np.float32)
    matched_iou = iou_mat[row_ind, col_ind] if len(row_ind) else np.asarray([], dtype=np.float32)

    pred_best_precision = precision_mat.max(axis=1) if precision_mat.size else np.asarray([], dtype=np.float32)
    gt_best_recall = recall_mat.max(axis=0) if recall_mat.size else np.asarray([], dtype=np.float32)
    group_precision = float(pred_best_precision.mean()) if pred_best_precision.size else 0.0
    group_recall = float(gt_best_recall.mean()) if gt_best_recall.size else 0.0
    group_f1 = 2.0 * group_precision * group_recall / max(group_precision + group_recall, 1e-6)
    group_mean_iou = float(matched_iou.mean()) if matched_iou.size else 0.0

    pred_stack = np.stack([mask.astype(np.uint8) for _, mask in pred_masks], axis=0)
    overlap_pixels = float((pred_stack.sum(axis=0) > 1).sum())
    pred_union = float((pred_stack.sum(axis=0) > 0).sum())
    overlap_ratio = overlap_pixels / max(pred_union, 1.0)
    return {
        "group_precision": float(group_precision),
        "group_recall": float(group_recall),
        "group_f1": float(group_f1),
        "group_mean_iou": float(group_mean_iou),
        "group_overlap_ratio": float(overlap_ratio),
    }


def compute_group_topology_metrics(
    pred_annotation: Dict,
    gt_annotation: Dict,
    shape: Tuple[int, int],
    tolerance: float = 24.0,
    gt_cache: Sequence[Tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> Dict[str, float]:
    pred_groups = [(group_id, mask > 0) for group_id, mask in _group_masks(pred_annotation, shape, "branch", 1)]
    if gt_cache is None:
        gt_groups = [(group_id, mask > 0) for group_id, mask in _group_masks(gt_annotation, shape, "branch", 1)]
        gt_distances = [distance_transform_edt(~mask) for _, mask in gt_groups]
    else:
        gt_groups = [(group_id, mask) for group_id, mask, _ in gt_cache]
        gt_distances = [distance for _, _, distance in gt_cache]
    if not pred_groups and not gt_groups:
        return {"branch_group_topology_precision": 1.0, "branch_group_topology_recall": 1.0, "branch_group_topology_f1": 1.0}
    if not pred_groups or not gt_groups:
        return {"branch_group_topology_precision": 0.0, "branch_group_topology_recall": 0.0, "branch_group_topology_f1": 0.0}

    pred_distances = [distance_transform_edt(~mask) for _, mask in pred_groups]
    pred_matched = 0.0
    pred_total = 0.0
    for _, pred_mask in pred_groups:
        area = float(pred_mask.sum())
        pred_total += area
        pred_matched += max(float((distance[pred_mask] <= tolerance).sum()) for distance in gt_distances)
    gt_matched = 0.0
    gt_total = 0.0
    for _, gt_mask in gt_groups:
        area = float(gt_mask.sum())
        gt_total += area
        gt_matched += max(float((distance[gt_mask] <= tolerance).sum()) for distance in pred_distances)
    precision = pred_matched / max(pred_total, 1.0)
    recall = gt_matched / max(gt_total, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
    return {
        "branch_group_topology_precision": float(precision),
        "branch_group_topology_recall": float(recall),
        "branch_group_topology_f1": float(f1),
    }


def _group_graph_validity(group: Dict) -> Tuple[bool, bool, int]:
    points = group.get("points", [])
    adjacency = [set() for _ in points]
    edge_count = 0
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if 0 <= src < len(points) and 0 <= dst < len(points) and src != dst and dst not in adjacency[src]:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
            edge_count += 1
    if not points:
        return False, False, 0
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency[node] - visited)
    connected = len(visited) == len(points)
    is_tree = connected and edge_count == len(points) - 1
    max_degree = max((len(neighbors) for neighbors in adjacency), default=0)
    return connected, is_tree, max_degree


def compute_botanical_topology_validity(annotation: Dict, shape: Tuple[int, int], contact_radius: int = 12) -> Dict[str, float]:
    groups = annotation.get("groups", [])
    trunks = [group for group in groups if group.get("group_type") == "trunk"]
    branches = [group for group in groups if group.get("group_type") == "branch"]
    trunk_single_path = False
    trunk_mask = np.zeros(shape, dtype=np.uint8)
    if len(trunks) == 1:
        connected, is_tree, max_degree = _group_graph_validity(trunks[0])
        trunk_single_path = bool(connected and is_tree and max_degree <= 2)
        trunk_mask = render_annotation_groups({"groups": trunks}, shape, group_type="trunk", thickness=1)
    trunk_distance = distance_transform_edt(trunk_mask == 0)
    branch_tree_valid = True
    exactly_one_root_contact = True
    contact_counts = []
    parent_edges = []

    def point_to_group_distance(point_xy: Sequence[float], group: Dict) -> float:
        point = np.asarray(point_xy, dtype=np.float32)
        group_points = np.asarray(group.get("points", []), dtype=np.float32)
        best = float("inf")
        for edge in group.get("edges", []):
            if len(edge) != 2 or max(map(int, edge)) >= len(group_points):
                continue
            start = group_points[int(edge[0])]
            segment = group_points[int(edge[1])] - start
            length_sq = float(np.dot(segment, segment))
            ratio = 0.0 if length_sq < 1e-9 else float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(point - (start + ratio * segment))))
        return best

    for branch_index, branch in enumerate(branches):
        connected, is_tree, _ = _group_graph_validity(branch)
        branch_tree_valid = branch_tree_valid and connected and is_tree
        points = branch.get("points", [])
        degrees = [0] * len(points)
        for edge in branch.get("edges", []):
            if len(edge) == 2 and 0 <= int(edge[0]) < len(points) and 0 <= int(edge[1]) < len(points):
                degrees[int(edge[0])] += 1
                degrees[int(edge[1])] += 1
        parent_index = None
        parent_distance = float("inf")
        if points:
            for candidate_index, candidate in enumerate(groups):
                if candidate is branch:
                    continue
                distance = point_to_group_distance(points[0], candidate)
                if distance < parent_distance:
                    parent_distance = distance
                    parent_index = candidate_index
        primary_contact = int(parent_index is not None and parent_distance <= contact_radius)
        if primary_contact:
            parent_edges.append((int(parent_index), int(groups.index(branch))))
        extra_trunk_contacts = 0
        for index, degree in enumerate(degrees):
            if degree != 1 or index == 0:
                continue
            x = int(np.clip(round(points[index][0]), 0, shape[1] - 1))
            y = int(np.clip(round(points[index][1]), 0, shape[0] - 1))
            extra_trunk_contacts += int(trunk_distance[y, x] <= 2.0)
        root_contacts = primary_contact + extra_trunk_contacts
        contact_counts.append(root_contacts)
        exactly_one_root_contact = exactly_one_root_contact and root_contacts == 1

    adjacency = [set() for _ in groups]
    indegree = [0] * len(groups)
    for parent, child in parent_edges:
        if child not in adjacency[parent]:
            adjacency[parent].add(child)
            indegree[child] += 1
    queue = [index for index, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    group_dag = visited == len(groups)
    final_dag = bool(trunk_single_path and branch_tree_valid and exactly_one_root_contact and group_dag)
    return {
        "trunk_single_path": float(trunk_single_path),
        "no_trunk_fork": float(trunk_single_path),
        "every_branch_one_root_contact": float(exactly_one_root_contact),
        "final_graph_dag": float(final_dag),
        "botanical_topology_valid": float(final_dag),
        "multi_contact_branch_groups": float(sum(count > 1 for count in contact_counts)),
    }


def compute_junction_pairing_accuracy(
    pairing_debug: Sequence[Dict],
    gt_annotation: Dict,
    shape: Tuple[int, int],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    tolerance: float = 24.0,
    gt_cache: Sequence[Tuple[str, np.ndarray]] | None = None,
) -> Dict[str, float]:
    gt_groups = list(gt_cache) if gt_cache is not None else []
    if gt_cache is None:
        for group in gt_annotation.get("groups", []):
            mask = render_annotation_groups({"groups": [group]}, shape, group_type=group.get("group_type"), thickness=1) > 0
            if np.any(mask):
                gt_groups.append((group.get("group_id", group.get("group_type", "group")), distance_transform_edt(~mask)))
    correct = 0
    trials = 0
    unmatched = 0
    for decision in pairing_debug or []:
        labels = []
        for arm in decision.get("arms", []):
            polyline = arm.get("polyline_xy", [])
            tail = polyline[max(1, len(polyline) // 3):] if len(polyline) >= 2 else polyline
            coords = np.asarray([[point[0] * scale_x, point[1] * scale_y] for point in tail], dtype=np.float32)
            if len(coords) == 0 or not gt_groups:
                labels.append(None)
                continue
            xs = np.clip(np.round(coords[:, 0]).astype(int), 0, shape[1] - 1)
            ys = np.clip(np.round(coords[:, 1]).astype(int), 0, shape[0] - 1)
            scores = [(group_id, float((distance[ys, xs] <= tolerance).mean())) for group_id, distance in gt_groups]
            group_id, score = max(scores, key=lambda item: item[1])
            labels.append(group_id if score >= 0.5 else None)
        pairs = list(decision.get("pairing", []))
        if decision.get("trunk_pair") is not None:
            pairs.append(decision["trunk_pair"])
        for pair in pairs:
            if len(pair) != 2 or max(map(int, pair)) >= len(labels):
                continue
            label_a, label_b = labels[int(pair[0])], labels[int(pair[1])]
            if label_a is None or label_b is None:
                unmatched += 1
                continue
            trials += 1
            correct += int(label_a == label_b)
    return {
        "junction_pairing_correct": float(correct),
        "junction_pairing_trials": float(trials),
        "junction_pairing_unmatched": float(unmatched),
        "junction_pairing_accuracy": float(correct / trials) if trials else 1.0,
    }


def greedy_match_points(pred_points: Sequence[Tuple[int, int]], gt_points: Sequence[Tuple[int, int]], distance_threshold: float) -> Tuple[List[Tuple[int, int]], List[float]]:
    pairs = [
        (float(np.linalg.norm(np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32))), pi, gi)
        for pi, pred in enumerate(pred_points)
        for gi, gt in enumerate(gt_points)
    ]
    pairs = [pair for pair in pairs if pair[0] <= distance_threshold]
    pairs.sort(key=lambda x: x[0])
    matched_pred, matched_gt = set(), set()
    matches, distances = [], []
    for dist, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi))
        distances.append(dist)
    return matches, distances


def compute_point_metrics(pred_points: Sequence[Tuple[int, int]], gt_points: Sequence[Tuple[int, int]], threshold: float) -> Dict[str, float]:
    matches, distances = greedy_match_points(pred_points, gt_points, distance_threshold=threshold)
    tp = len(matches)
    fp = len(pred_points) - tp
    fn = len(gt_points) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_match_distance": float(np.mean(distances)) if distances else 0.0,
    }


def compute_line_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: float) -> Dict[str, float]:
    pred = pred_mask > 0
    gt = gt_mask > 0
    if not np.any(pred) and not np.any(gt):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "chamfer": 0.0}
    if not np.any(pred) or not np.any(gt):
        chamfer = float(max(pred_mask.shape))
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "chamfer": chamfer}

    dist_to_gt = distance_transform_edt(~gt)
    dist_to_pred = distance_transform_edt(~pred)
    precision = float((dist_to_gt[pred] <= tolerance).mean()) if np.any(pred) else 0.0
    recall = float((dist_to_pred[gt] <= tolerance).mean()) if np.any(gt) else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    chamfer = float((dist_to_gt[pred].mean() + dist_to_pred[gt].mean()) * 0.5)
    return {"precision": precision, "recall": recall, "f1": float(f1), "chamfer": chamfer}


def compute_line_metrics_multi_tolerance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerances: Sequence[float] = (1.0, 3.0, 5.0, 8.0, 12.0, 16.0, 24.0),
) -> Dict[str, float]:
    """多 tolerance F1 扫描 + PR 曲线 AUC."""
    pred = pred_mask > 0
    gt = gt_mask > 0

    if not np.any(pred) and not np.any(gt):
        base = {}
        for t in tolerances:
            base[f"f1_{int(t)}px"] = 1.0
            base[f"precision_{int(t)}px"] = 1.0
            base[f"recall_{int(t)}px"] = 1.0
        base["pr_auc"] = 1.0
        return base

    if not np.any(pred) or not np.any(gt):
        base = {}
        for t in tolerances:
            base[f"f1_{int(t)}px"] = 0.0
            base[f"precision_{int(t)}px"] = 0.0
            base[f"recall_{int(t)}px"] = 0.0
        base["pr_auc"] = 0.0
        return base

    dist_to_gt = distance_transform_edt(~gt)
    dist_to_pred = distance_transform_edt(~pred)
    pred_dists = dist_to_gt[pred]
    gt_dists = dist_to_pred[gt]

    precisions = []
    recalls = []
    result = {}

    for t in tolerances:
        p = float((pred_dists <= t).mean())
        r = float((gt_dists <= t).mean())
        f1 = 2 * p * r / (p + r + 1e-6)
        precisions.append(p)
        recalls.append(r)
        t_int = int(t)
        result[f"f1_{t_int}px"] = float(f1)
        result[f"precision_{t_int}px"] = float(p)
        result[f"recall_{t_int}px"] = float(r)

    # PR AUC: trapezoidal integration over recall range
    sorted_pairs = sorted(zip(recalls, precisions))
    auc = 0.0
    for i in range(1, len(sorted_pairs)):
        r0, p0 = sorted_pairs[i - 1]
        r1, p1 = sorted_pairs[i]
        auc += (r1 - r0) * (p0 + p1) * 0.5
    result["pr_auc"] = float(auc)

    return result


def compute_hausdorff(pred_mask: np.ndarray, gt_mask: np.ndarray, percentile: float = 95.0) -> Dict[str, float]:
    """Hausdorff 距离 (第 percentile 分位)."""
    pred = pred_mask > 0
    gt = gt_mask > 0

    if not np.any(pred) or not np.any(gt):
        return {"hausdorff_95": -1.0, "hausdorff_max": -1.0}

    dist_to_gt = distance_transform_edt(~gt)
    dist_to_pred = distance_transform_edt(~pred)

    all_dists = np.concatenate([dist_to_gt[pred], dist_to_pred[gt]])
    return {
        f"hausdorff_{int(percentile)}": float(np.percentile(all_dists, percentile)),
        "hausdorff_max": float(all_dists.max()),
    }


def compute_branch_detection_rate(
    pred_annotation: Dict,
    gt_annotation: Dict,
    shape: Tuple[int, int],
    tolerance: float = 24.0,
    thickness: int = 2,
) -> Dict[str, float]:
    """每条预测 branch 是否在 GT branch 的 tolerance 范围内."""
    pred_branches = [g for g in pred_annotation.get("groups", []) if g.get("group_type") == "branch"]
    gt_branches = [g for g in gt_annotation.get("groups", []) if g.get("group_type") == "branch"]

    if not pred_branches and not gt_branches:
        return {"branch_detection_rate": 1.0, "branch_detection_tp": 0.0, "branch_detection_fp": 0.0, "branch_detection_fn": 0.0}

    if not pred_branches:
        return {"branch_detection_rate": 0.0, "branch_detection_tp": 0.0, "branch_detection_fp": 0.0, "branch_detection_fn": float(len(gt_branches))}

    if not gt_branches:
        return {"branch_detection_rate": 0.0, "branch_detection_tp": 0.0, "branch_detection_fp": float(len(pred_branches)), "branch_detection_fn": 0.0}

    # Render each branch as line mask
    def render_branch(g, shape_wh):
        single = {"groups": [g]}
        return render_annotation_groups(single, shape_wh, thickness=2) > 0

    gt_masks = [render_branch(g, shape) for g in gt_branches]
    pred_masks = [render_branch(g, shape) for g in pred_branches]

    # For each GT branch, compute distance transform; check if any pred branch overlaps within tolerance
    tp = 0
    matched_gt = set()
    matched_pred = set()

    for pi, pm in enumerate(pred_masks):
        for gi, gm in enumerate(gt_masks):
            if gi in matched_gt:
                continue
            # Check if pred branch centerline is within tolerance of GT branch
            dt = distance_transform_edt(~gm)
            min_dist = float(dt[pm].min()) if np.any(pm) else float('inf')
            if min_dist <= tolerance:
                tp += 1
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    fp = len(pred_branches) - len(matched_pred)
    fn = len(gt_branches) - len(matched_gt)
    rate = tp / max(tp + fp + fn, 1)

    return {
        "branch_detection_rate": float(rate),
        "branch_detection_tp": float(tp),
        "branch_detection_fp": float(fp),
        "branch_detection_fn": float(fn),
    }


def compose_6panel_diagnostics(image_rgb: np.ndarray, prediction, gt_overlay: np.ndarray) -> np.ndarray:
    def to_bgr(image_rgb_like: np.ndarray, title: str) -> np.ndarray:
        return add_title(cv2.cvtColor(image_rgb_like, cv2.COLOR_RGB2BGR), title)

    mask = (prediction.mask > 0).astype(np.uint8) * 255
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    if prediction.root_point != (-1, -1):
        cv2.circle(mask_rgb, prediction.root_point, 7, (255, 255, 0), thickness=-1)
        cv2.circle(mask_rgb, prediction.root_point, 7, (0, 0, 0), thickness=2)
    p1 = to_bgr(gt_overlay, "Panel1: GT Skeleton")
    p2 = to_bgr(mask_rgb, "Panel2: Rooted Combined Mask + Root")

    dt_overlay = cv2.cvtColor(heatmap_to_color(prediction.dt_map), cv2.COLOR_BGR2RGB)
    for p0, p1_line in zip(prediction.trunk_line[:-1], prediction.trunk_line[1:]):
        cv2.line(dt_overlay, p0, p1_line, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    p3 = to_bgr(dt_overlay, "Panel3: Distance Transform + Trunk")

    skeleton_rgb = np.repeat((prediction.skeleton_map > 0).astype(np.uint8)[..., None] * 255, 3, axis=2)
    for x, y in prediction.raw_endpoints:
        cv2.circle(skeleton_rgb, (x, y), 3, (255, 255, 0), thickness=-1)
    for x, y in prediction.filtered_endpoints:
        cv2.circle(skeleton_rgb, (x, y), 4, (64, 255, 64), thickness=-1)
    p4 = to_bgr(skeleton_rgb, "Panel4: Skeleton + Endpoints")

    cost_norm = prediction.cost_matrix / max(float(prediction.cost_matrix.max()), 1e-6)
    route_rgb = cv2.cvtColor(heatmap_to_color(cost_norm, cmap=cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    route_rgb = draw_prediction_overlay(route_rgb, prediction, line_thickness=5)
    p5 = to_bgr(route_rgb, "Panel5: Skeleton Cost + Topology")

    pred_overlay = getattr(prediction, "group_group_overlay", None)
    if pred_overlay is None:
        pred_overlay = draw_prediction_overlay(image_rgb, prediction)
    stats_text = (
        f"RootedCC={prediction.routing_stats.get('selected_label', 0)} | "
        f"Endpoints={prediction.routing_stats.get('endpoints_filtered', 0)} | "
        f"Routed={prediction.routing_stats.get('routed', 0)}"
    )
    p6 = add_title(cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR), f"Panel6: Grouped Prediction | {stats_text}")

    panels = [p1, p2, p3, p4, p5, p6]
    ref_h = 480
    resized = []
    for panel in panels:
        h, w = panel.shape[:2]
        new_w = int(ref_h * w / h)
        resized.append(cv2.resize(panel, (new_w, ref_h), interpolation=cv2.INTER_LINEAR))
    max_w = max(panel.shape[1] for panel in resized)
    padded = []
    for panel in resized:
        if panel.shape[1] < max_w:
            pad = np.zeros((panel.shape[0], max_w - panel.shape[1], 3), dtype=np.uint8)
            panel = np.hstack([panel, pad])
        padded.append(panel)
    return np.vstack([np.hstack(padded[:3]), np.hstack(padded[3:6])])


def evaluate_one_sample(sample: Dict, args: argparse.Namespace):
    image_rgb = load_image_rgb(sample["image_path"])
    mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码: {sample['mask_path']}")
    annotation = load_json(Path(sample["annotation_path"]))

    prediction = build_prediction_result(
        combined_mask=mask,
        image_rgb=image_rgb,
        root_band_height=args.root_band_height,
        trunk_exclusion_radius=args.trunk_exclusion_radius,
        min_branch_length=args.min_branch_length,
        prune_spur_length=args.prune_spur_length,
        trunk_rdp_epsilon_ratio=args.trunk_rdp_epsilon_ratio,
        branch_rdp_epsilon=args.branch_rdp_epsilon,
        min_endpoint_branch_length=args.min_endpoint_branch_length,
        max_endpoints_to_route=args.max_endpoints_to_route,
        max_processing_dim=args.max_processing_dim,
        enable_vertical_bridge=args.enable_vertical_bridge,
        bridge_max_gap=args.bridge_max_gap,
        bridge_max_dx=args.bridge_max_dx,
        bridge_min_component_area=args.bridge_min_component_area,
        bridge_min_component_height=args.bridge_min_component_height,
        enable_junction_pairing=args.enable_junction_pairing,
        junction_cluster_radius=args.junction_cluster_radius,
        junction_prior_path=args.junction_prior_path,
    )
    prediction_annotation = decode_prediction_to_annotation(sample["image_path"], image_rgb.shape[:2], prediction)
    clip_stats: Dict[str, float] = {}
    if getattr(args, "apply_mask_clip", False):
        from .mask_clip import clip_annotation_groups

        clipped_groups, raw_clip_stats = clip_annotation_groups(
            prediction_annotation.get("groups", []),
            prediction.mask,
        )
        prediction_annotation = {**prediction_annotation, "groups": clipped_groups}
        clip_stats = {
            "clip_edges_removed": float(raw_clip_stats.get("edges_removed", 0)),
            "clip_edges_grazing_snapped": float(raw_clip_stats.get("edges_grazing_snapped", 0)),
            "clip_edges_gap_bridged": float(raw_clip_stats.get("edges_gap_bridged", 0)),
            "clip_points_orphaned": float(raw_clip_stats.get("points_orphaned", 0)),
        }

    gt_graph = merge_groups_to_global_graph(annotation, image_rgb.shape[:2])
    gt_points = [(float(gt_graph["points"][i, 0]), float(gt_graph["points"][i, 1])) for i in range(gt_graph["num_points"])]
    pred_points = [(float(x), float(y)) for x, y in prediction.points]

    pred_annotation = prediction_annotation

    trunk_gt_mask = render_annotation_groups(annotation, image_rgb.shape[:2], group_type="trunk", thickness=2)
    branch_gt_mask = render_annotation_groups(annotation, image_rgb.shape[:2], group_type="branch", thickness=2)
    trunk_pred_mask = render_annotation_groups(pred_annotation, image_rgb.shape[:2], group_type="trunk", thickness=2)
    branch_pred_mask = render_annotation_groups(pred_annotation, image_rgb.shape[:2], group_type="branch", thickness=2)

    metrics: Dict[str, float] = {
        "pred_num_points": float(len(pred_points)),
        "gt_num_points": float(len(gt_points)),
        "pred_num_branches": float(sum(1 for g in pred_annotation.get("groups", []) if g.get("group_type") == "branch")),
        "gt_num_branches": float(sum(1 for g in annotation.get("groups", []) if g.get("group_type") == "branch")),
        **clip_stats,
    }

    for threshold in args.point_thresholds:
        point_metrics = compute_point_metrics(pred_points, gt_points, threshold=threshold)
        suffix = f"thr{int(threshold)}"
        metrics[f"point_precision_{suffix}"] = point_metrics["precision"]
        metrics[f"point_recall_{suffix}"] = point_metrics["recall"]
        metrics[f"point_f1_{suffix}"] = point_metrics["f1"]
        metrics[f"point_mean_match_distance_{suffix}"] = point_metrics["mean_match_distance"]

    trunk_metrics = compute_line_metrics(trunk_pred_mask, trunk_gt_mask, tolerance=args.line_tolerance)
    branch_metrics = compute_line_metrics(branch_pred_mask, branch_gt_mask, tolerance=args.line_tolerance)
    group_metrics = compute_group_consistency_metrics(pred_annotation, annotation, image_rgb.shape[:2], thickness=2)
    group_topology_metrics = compute_group_topology_metrics(pred_annotation, annotation, image_rgb.shape[:2], tolerance=24.0)
    topology_validity = compute_botanical_topology_validity(pred_annotation, image_rgb.shape[:2])
    pairing_metrics = compute_junction_pairing_accuracy(
        prediction.routing_stats.get("junction_pairing_debug", []),
        annotation,
        image_rgb.shape[:2],
        scale_x=float(prediction.routing_stats.get("scale_x", 1.0)),
        scale_y=float(prediction.routing_stats.get("scale_y", 1.0)),
    )
    for key, value in trunk_metrics.items():
        metrics[f"trunk_{key}"] = value
    for key, value in branch_metrics.items():
        metrics[f"branch_{key}"] = value
    for key, value in group_metrics.items():
        metrics[f"branch_{key}"] = value
    metrics.update(group_topology_metrics)
    metrics.update(topology_validity)
    metrics.update(pairing_metrics)

    # 新增: 多 tolerance F1 + PR AUC + Hausdorff + branch detection rate
    if getattr(args, "multi_tolerance", False):
        branch_mt = compute_line_metrics_multi_tolerance(branch_pred_mask, branch_gt_mask)
        for key, value in branch_mt.items():
            metrics[f"branch_mt_{key}"] = value
        trunk_mt = compute_line_metrics_multi_tolerance(trunk_pred_mask, trunk_gt_mask)
        for key, value in trunk_mt.items():
            metrics[f"trunk_mt_{key}"] = value

    if getattr(args, "compute_hausdorff", False):
        branch_hd = compute_hausdorff(branch_pred_mask, branch_gt_mask)
        for key, value in branch_hd.items():
            metrics[f"branch_mt_{key}"] = value
        trunk_hd = compute_hausdorff(trunk_pred_mask, trunk_gt_mask)
        for key, value in trunk_hd.items():
            metrics[f"trunk_mt_{key}"] = value

    if getattr(args, "compute_branch_detection", False):
        bd = compute_branch_detection_rate(pred_annotation, annotation, image_rgb.shape[:2])
        for key, value in bd.items():
            metrics[f"branch_mt_{key}"] = value

    rs = prediction.routing_stats
    route_total = max(int(rs.get("routed", 0)) + int(rs.get("skipped_short", 0)) + int(rs.get("skipped_no_path", 0)), 1)
    metrics.update(
        {
            "routing_endpoints_total": float(rs.get("endpoints_total", 0)),
            "routing_endpoints_filtered": float(rs.get("endpoints_filtered", 0)),
            "routing_routed": float(rs.get("routed", 0)),
            "routing_success_rate": float(rs.get("routed", 0) / route_total),
            "routing_skipped_short": float(rs.get("skipped_short", 0)),
            "routing_skipped_no_path": float(rs.get("skipped_no_path", 0)),
            "routing_bridge_segments": float(rs.get("bridge_segments", 0)),
            "routing_junction_clusters": float(rs.get("junction_clusters", 0)),
            "routing_crossing_clusters": float(rs.get("crossing_clusters", 0)),
            "routing_crossing_pairs": float(rs.get("crossing_pairs", 0)),
            "dt_max": float(rs.get("max_dt", 0.0)),
            "trunk_score": float(rs.get("trunk_score", 0.0)),
            "routing_trunk_cycles_detected": float(rs.get("trunk_cycles_detected", 0)),
            "routing_trunk_cycles_junction_related": float(rs.get("trunk_cycles_junction_related", 0)),
            "routing_multi_contact_groups": float(len(rs.get("junction_multi_contact_groups", []))),
            "routing_multi_contact_groups_split": float(rs.get("multi_contact_groups_split", 0)),
            "routing_tape_root_contacts_pruned": float(rs.get("tape_root_contacts_pruned", 0)),
        }
    )

    trunk_fork_nodes = 0
    for group in pred_annotation.get("groups", []):
        if group.get("group_type") != "trunk":
            continue
        degrees = [0] * len(group.get("points", []))
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(degrees) and 0 <= dst < len(degrees):
                degrees[src] += 1
                degrees[dst] += 1
        trunk_fork_nodes += sum(degree > 2 for degree in degrees)
    metrics["routing_trunk_fork_nodes"] = float(trunk_fork_nodes)

    gt_overlay = draw_gt_overlay(image_rgb, annotation)
    prediction_group_overlay = draw_group_overlay(image_rgb, pred_annotation, line_thickness=5)
    prediction.group_group_overlay = prediction_group_overlay
    vis = compose_6panel_diagnostics(image_rgb, prediction, gt_overlay)
    return metrics, vis, prediction_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="纯几何 Mask 拓扑恢复评估")
    parser.add_argument("--processed-root", type=str, default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--result-root", type=str, default=str(DEFAULT_RESULT_ROOT / "evaluation"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-thresholds", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--line-tolerance", type=float, default=5.0)
    parser.add_argument("--root-band-height", type=int, default=16)
    parser.add_argument("--trunk-exclusion-radius", type=float, default=10.0)
    parser.add_argument("--min-branch-length", type=float, default=12.0)
    parser.add_argument("--prune-spur-length", type=int, default=8)
    parser.add_argument("--trunk-rdp-epsilon-ratio", type=float, default=0.01)
    parser.add_argument("--branch-rdp-epsilon", type=float, default=2.0)
    parser.add_argument("--min-endpoint-branch-length", type=int, default=16)
    parser.add_argument("--max-endpoints-to-route", type=int, default=24)
    parser.add_argument("--max-processing-dim", type=int, default=1280)
    parser.add_argument("--enable-vertical-bridge", action="store_true", default=True)
    parser.add_argument("--disable-vertical-bridge", action="store_false", dest="enable_vertical_bridge")
    parser.add_argument("--bridge-max-gap", type=int, default=96)
    parser.add_argument("--bridge-max-dx", type=int, default=28)
    parser.add_argument("--bridge-min-component-area", type=int, default=80)
    parser.add_argument("--bridge-min-component-height", type=int, default=40)
    parser.add_argument("--enable-junction-pairing", action="store_true", default=True)
    parser.add_argument("--disable-junction-pairing", action="store_false", dest="enable_junction_pairing")
    parser.add_argument("--junction-cluster-radius", type=float, default=12.0)
    parser.add_argument("--junction-prior-path", type=str, default=str((Path(__file__).resolve().parent / "gt_junction_priors.json")))
    parser.add_argument("--multi-tolerance", action="store_true", default=False,
                        help="Enable multi-tolerance F1 scan + PR AUC")
    parser.add_argument("--compute-hausdorff", action="store_true", default=False,
                        help="Compute 95th percentile Hausdorff distance")
    parser.add_argument("--compute-branch-detection", action="store_true", default=False,
                        help="Compute branch detection rate (tolerance=24px)")
    parser.add_argument("--apply-mask-clip", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    processed_root = Path(args.processed_root)
    manifest_path = processed_root / "annotations" / "skeleton_prediction" / "manifest_test.json"
    if not manifest_path.exists():
        manifest_path = processed_root / "annotations" / "skeleton_prediction" / "manifest_train.json"
    all_samples = load_manifest(manifest_path)
    if args.num_samples <= 0 or args.num_samples >= len(all_samples):
        selected = list(all_samples)
    else:
        selected = random.sample(all_samples, args.num_samples)

    run_name = f"mask_topology_routing_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_root = ensure_dir(Path(args.result_root) / run_name)
    pred_dir = ensure_dir(eval_root / "predictions")
    vis_dir = ensure_dir(eval_root / "visualizations")

    all_metrics = []
    for sample in selected:
        print(f"评估: {sample['sample_name']}")
        metrics, vis, prediction_annotation = evaluate_one_sample(sample, args)
        metrics["sample_name"] = sample["sample_name"]
        all_metrics.append(metrics)
        save_json(metrics, pred_dir / f"{sample['sample_name']}_metrics.json")
        save_json(prediction_annotation, pred_dir / f"{sample['sample_name']}_prediction.json")
        cv2.imwrite(str(vis_dir / f"{sample['sample_name']}_6panel.png"), vis)

    numeric_keys = [key for key in all_metrics[0].keys() if key != "sample_name"]
    summary = {key: float(np.mean([metric[key] for metric in all_metrics])) for key in numeric_keys}
    save_json({"run_name": run_name, "summary": summary, "samples": all_metrics}, eval_root / "metrics_summary.json")

    print("\n=== 点匹配指标 ===")
    for key in [k for k in numeric_keys if k.startswith("point_")]:
        print(f"  {key}: {summary[key]:.4f}")

    print("\n=== 主干/侧枝解耦指标 ===")
    for key in [k for k in numeric_keys if k.startswith("trunk_") or k.startswith("branch_")]:
        print(f"  {key}: {summary[key]:.4f}")

    print("\n=== 路由健康度 ===")
    for key in [k for k in numeric_keys if k.startswith("routing_") or k in ("dt_max", "trunk_score")]:
        print(f"  {key}: {summary[key]:.4f}")

    print(f"\n结果已保存: {eval_root}")


if __name__ == "__main__":
    main()
