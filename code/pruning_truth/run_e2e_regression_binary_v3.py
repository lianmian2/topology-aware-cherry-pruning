from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_SIGNATURE_VERSION = "binary-v3.1.0-ablation"
MODEL_DIR = PROJECT_ROOT / "02_code" / "02_models"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (MODEL_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_e2e_regression as v2
from bud_skeleton_fusion import extract_bud_orientations, extract_directed_bud_orientations
from bud_skeleton_fusion.bud_skeleton_attachment import attach_buds_to_skeleton, attachment_stats
from bud_skeleton_fusion.evaluate_bud_topology_cv import rerank_variant
from logic_branch import run_branch_segmentation
from logic_bud import run_bud_detection_pipeline
from logic_models import ModelManager
from logic_roi import apply_roi_filter
from mask_topology_routing.mask_clip import clip_annotation_groups
from mask_topology_routing.evaluate_visualize import compute_junction_pairing_accuracy
from mask_topology_routing import utils as routing_utils
from mask_topology_routing.utils import _build_directed_topology, build_prediction_result, prepare_processed_router_mask
from mask_topology_routing_binary_v3 import (
    audit_binary_topology_v3,
    load_binary_v3_config,
    refine_clipped_groups_v3,
)
from mask_topology_routing_binary_v3.thickness_routing import make_router_selector


_ACTIVE_THICKNESS_SELECTOR = None
_THICKNESS_RUN_CONTEXT: dict[str, Any] | None = None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_signature(
    sample: dict[str, Any], defaults: dict[str, Any], config: Any,
    botanical_ablation: bool = False,
) -> dict[str, Any]:
    image_path = absolute_path(sample["image_path"])
    truth_path = absolute_path(sample["truth_path"]) if sample.get("truth_path") else None
    code_paths = [
        Path(__file__),
        MODEL_DIR / "mask_topology_routing_binary_v3" / "config.py",
        MODEL_DIR / "mask_topology_routing_binary_v3" / "refinement.py",
        MODEL_DIR / "mask_topology_routing_binary_v3" / "quality.py",
        MODEL_DIR / "bud_skeleton_fusion" / "evaluate_bud_topology_cv.py",
    ]
    if _THICKNESS_RUN_CONTEXT is not None:
        code_paths.append(MODEL_DIR / "mask_topology_routing_binary_v3" / "thickness_routing.py")
    payload = {
        "pipeline_signature_version": PIPELINE_SIGNATURE_VERSION,
        "schema_version": config.schema_version,
        "sample_id": sample["sample_id"],
        "image_sha256": sha256_path(image_path),
        "truth_sha256": sha256_path(truth_path) if truth_path and truth_path.exists() else None,
        "rulebook_sha256": config.rulebook_sha256,
        "weight_sha256": {stage: defaults[f"{stage}_sha256"] for stage in ("roi", "branch", "bud")},
        "code_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256_path(path) for path in code_paths},
        "router_config": v2.ROUTER_CONFIG,
        "mask_clip": defaults["mask_clip"],
        "thickness_trunk": _THICKNESS_RUN_CONTEXT,
        "botanical_ablation": bool(botanical_ablation),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["signature_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def absolute_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def numbered_overlay(image_rgb: np.ndarray, groups: list[dict[str, Any]]) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    palette = [(0, 255, 0), (255, 180, 0), (255, 0, 255), (0, 165, 255), (0, 255, 255)]
    for group_index, group in enumerate(groups):
        points = [tuple(map(int, point)) for point in group.get("points", [])]
        color = (20, 20, 220) if group.get("group_type") == "trunk" else palette[group_index % len(palette)]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(points)))
        for edge_index, edge in enumerate(group.get("edges", [])):
            if len(edge) != 2:
                continue
            src, dst = map(int, edge)
            if not (0 <= src < len(points) and 0 <= dst < len(points)):
                continue
            graph.add_edge(src, dst)
            cv2.line(canvas, points[src], points[dst], color, 5, cv2.LINE_AA)
            middle = ((points[src][0] + points[dst][0]) // 2, (points[src][1] + points[dst][1]) // 2)
            cv2.putText(canvas, f"E{edge_index:03d}", middle, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"E{edge_index:03d}", middle, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for node in graph:
            if graph.degree[node] < 3:
                continue
            cv2.circle(canvas, points[node], 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, points[node], 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"J{node:03d}", (points[node][0] + 7, points[node][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"J{node:03d}", (points[node][0] + 7, points[node][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        if points:
            centroid = np.asarray(points, dtype=np.float64).mean(axis=0).astype(int)
            label = "TRUNK" if group.get("group_type") == "trunk" else str(group.get("group_id"))
            cv2.putText(canvas, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return canvas


def sampled_group_points(group: dict[str, Any], spacing_px: float = 3.0) -> np.ndarray:
    points = np.asarray(group.get("points", []), dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    points = points.reshape((-1, 2))
    samples: list[np.ndarray] = []
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = map(int, edge)
        if not (0 <= src < len(points) and 0 <= dst < len(points)):
            continue
        start, end = points[src], points[dst]
        count = max(int(math.ceil(float(np.linalg.norm(end - start)) / spacing_px)), 1)
        ratios = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)[:, None]
        samples.append(start[None, :] * (1.0 - ratios) + end[None, :] * ratios)
    if not samples:
        return points.copy()
    merged = np.concatenate(samples, axis=0)
    return np.unique(np.rint(merged).astype(np.int32), axis=0).astype(np.float64)


def sampled_group_hierarchy(
    group: dict[str, Any], spacing_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(group.get("points", []), dtype=np.float64).reshape((-1, 2))
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = map(int, edge)
        if 0 <= src < len(points) and 0 <= dst < len(points) and src != dst:
            graph.add_edge(src, dst, length=float(np.linalg.norm(points[src] - points[dst])))
    samples: list[np.ndarray] = []
    levels: list[np.ndarray] = []
    for component_nodes in nx.connected_components(graph):
        component = graph.subgraph(component_nodes).copy()
        if component.number_of_edges() == 0:
            continue
        tree = nx.minimum_spanning_tree(component, weight="length")
        root = 0 if 0 in component_nodes else min(component_nodes)
        parent = {root: None}
        depth = {root: 0}
        for node in nx.bfs_tree(tree, root):
            if node == root:
                continue
            ancestor = next(neighbor for neighbor in tree.neighbors(node) if neighbor in parent)
            parent[node] = ancestor
            depth[node] = min(depth[ancestor] + int(tree.degree[ancestor] >= 3), 3)
        for src, dst in tree.edges():
            if parent.get(dst) == src:
                child = dst
            elif parent.get(src) == dst:
                child = src
            else:
                child = dst
            start, end = points[src], points[dst]
            count = max(int(math.ceil(float(np.linalg.norm(end - start)) / spacing_px)), 1)
            ratios = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)[:, None]
            samples.append(start[None, :] * (1.0 - ratios) + end[None, :] * ratios)
            levels.append(np.full(count + 1, int(depth.get(child, 0)), dtype=np.int16))
    if not samples:
        return points.copy(), np.zeros(len(points), dtype=np.int16)
    return np.concatenate(samples, axis=0), np.concatenate(levels, axis=0)


def branch_hierarchy_metrics(
    pred_groups: list[dict[str, Any]], gt_groups: list[dict[str, Any]], tolerance: float = 24.0,
) -> dict[str, float]:
    pred = [sampled_group_hierarchy(group) for group in pred_groups]
    truth = [sampled_group_hierarchy(group) for group in gt_groups]
    pred = [item for item in pred if len(item[0])]
    truth = [item for item in truth if len(item[0])]
    if not pred or not truth:
        value = 1.0 if not pred and not truth else 0.0
        return {
            "precision": value, "recall": value, "f1": value,
            "spatially_matched_label_accuracy": value,
        }
    geometry_f1 = np.zeros((len(pred), len(truth)), dtype=np.float64)
    for pred_index, (pred_points, _) in enumerate(pred):
        pred_tree = cKDTree(pred_points)
        for truth_index, (truth_points, _) in enumerate(truth):
            truth_tree = cKDTree(truth_points)
            precision = float(np.mean(truth_tree.query(pred_points, k=1)[0] <= tolerance))
            recall = float(np.mean(pred_tree.query(truth_points, k=1)[0] <= tolerance))
            geometry_f1[pred_index, truth_index] = (
                2.0 * precision * recall / max(precision + recall, 1e-12)
            )
    matched_pred, matched_truth = linear_sum_assignment(1.0 - geometry_f1)
    correct_pred = 0
    correct_truth = 0
    spatial_matches = 0
    spatial_label_matches = 0
    for pred_index, truth_index in zip(matched_pred, matched_truth):
        pred_points, pred_levels = pred[pred_index]
        truth_points, truth_levels = truth[truth_index]
        truth_distance, truth_nearest = cKDTree(truth_points).query(pred_points, k=1)
        pred_distance, pred_nearest = cKDTree(pred_points).query(truth_points, k=1)
        pred_spatial = truth_distance <= tolerance
        truth_spatial = pred_distance <= tolerance
        pred_correct = pred_spatial & (pred_levels == truth_levels[truth_nearest])
        truth_correct = truth_spatial & (truth_levels == pred_levels[pred_nearest])
        correct_pred += int(pred_correct.sum())
        correct_truth += int(truth_correct.sum())
        spatial_matches += int(pred_spatial.sum()) + int(truth_spatial.sum())
        spatial_label_matches += int(pred_correct.sum()) + int(truth_correct.sum())
    pred_total = sum(len(points) for points, _ in pred)
    truth_total = sum(len(points) for points, _ in truth)
    precision = float(correct_pred / max(pred_total, 1))
    recall = float(correct_truth / max(truth_total, 1))
    return {
        "precision": precision,
        "recall": recall,
        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "spatially_matched_label_accuracy": float(
            spatial_label_matches / max(spatial_matches, 1)
        ),
    }


def nearest_metrics(pred_points: np.ndarray, gt_points: np.ndarray, tolerance: float) -> dict[str, float]:
    if len(pred_points) == 0 and len(gt_points) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "chamfer": 0.0}
    if len(pred_points) == 0 or len(gt_points) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "chamfer": float("inf")}
    pred_to_gt = cKDTree(gt_points).query(pred_points, k=1)[0]
    gt_to_pred = cKDTree(pred_points).query(gt_points, k=1)[0]
    precision = float(np.mean(pred_to_gt <= tolerance))
    recall = float(np.mean(gt_to_pred <= tolerance))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "chamfer": float((pred_to_gt.mean() + gt_to_pred.mean()) * 0.5),
    }


def topology_group_metrics(
    pred_groups: list[np.ndarray], gt_groups: list[np.ndarray], tolerance: float
) -> dict[str, float]:
    if not pred_groups or not gt_groups:
        value = 1.0 if not pred_groups and not gt_groups else 0.0
        return {
            "branch_group_topology_precision": value,
            "branch_group_topology_recall": value,
            "branch_group_topology_f1": value,
            "branch_group_mean_iou": value,
        }
    precision = np.zeros((len(pred_groups), len(gt_groups)), dtype=np.float64)
    recall = np.zeros_like(precision)
    f1 = np.zeros_like(precision)
    iou = np.zeros_like(precision)
    gt_trees = [cKDTree(points) for points in gt_groups]
    pred_trees = [cKDTree(points) for points in pred_groups]
    for pred_index, pred_points in enumerate(pred_groups):
        for gt_index, gt_points in enumerate(gt_groups):
            p = float(np.mean(gt_trees[gt_index].query(pred_points, k=1)[0] <= tolerance))
            r = float(np.mean(pred_trees[pred_index].query(gt_points, k=1)[0] <= tolerance))
            precision[pred_index, gt_index] = p
            recall[pred_index, gt_index] = r
            f1[pred_index, gt_index] = 2.0 * p * r / max(p + r, 1e-12)
            iou[pred_index, gt_index] = p * r / max(p + r - p * r, 1e-12)
    row, column = linear_sum_assignment(1.0 - f1)
    group_precision = float(precision.max(axis=1).mean())
    group_recall = float(recall.max(axis=0).mean())
    group_f1 = 2.0 * group_precision * group_recall / max(group_precision + group_recall, 1e-12)
    return {
        "branch_group_topology_precision": group_precision,
        "branch_group_topology_recall": group_recall,
        "branch_group_topology_f1": group_f1,
        "branch_group_mean_iou": float(iou[row, column].mean()) if len(row) else 0.0,
    }


def truth_metrics(groups: list[dict[str, Any]], truth_path: Path | None, shape: tuple[int, int]) -> dict[str, Any] | None:
    if truth_path is None or not truth_path.exists():
        return None
    truth = read_json(truth_path)
    result: dict[str, Any] = {
        "truth_path": str(truth_path),
        "metric_engine": "polyline_sampling_3px_kdtree",
        "primary_tolerance_px": 24.0,
        "supplementary_tolerance_px": 5.0,
    }
    for group_type in ("trunk", "branch"):
        if group_type == "trunk":
            pred_selected = [item for item in groups if item.get("group_type") == "trunk"]
            gt_selected = [item for item in truth.get("groups", []) if item.get("group_type") == "trunk"]
        else:
            pred_selected = [item for item in groups if item.get("group_type") != "trunk"]
            gt_selected = [item for item in truth.get("groups", []) if item.get("group_type") != "trunk"]
        pred_points = np.concatenate([sampled_group_points(item) for item in pred_selected], axis=0) if pred_selected else np.empty((0, 2))
        gt_points = np.concatenate([sampled_group_points(item) for item in gt_selected], axis=0) if gt_selected else np.empty((0, 2))
        result[group_type] = {
            "24px": nearest_metrics(pred_points, gt_points, 24.0),
            "5px": nearest_metrics(pred_points, gt_points, 5.0),
        }
    pred_group_points = [sampled_group_points(item) for item in groups if item.get("group_type") != "trunk"]
    gt_group_points = [sampled_group_points(item) for item in truth.get("groups", []) if item.get("group_type") != "trunk"]
    pred_group_points = [points for points in pred_group_points if len(points) > 0]
    gt_group_points = [points for points in gt_group_points if len(points) > 0]
    result["group_topology_24px"] = topology_group_metrics(pred_group_points, gt_group_points, 24.0)
    result["group_topology_5px"] = topology_group_metrics(pred_group_points, gt_group_points, 5.0)
    result["branch_hierarchy_24px"] = branch_hierarchy_metrics(
        [item for item in groups if item.get("group_type") != "trunk"],
        [item for item in truth.get("groups", []) if item.get("group_type") != "trunk"],
        tolerance=24.0,
    )
    result["group_consistency"] = {
        "group_f1": result["group_topology_24px"]["branch_group_topology_f1"],
        "group_mean_iou": result["group_topology_24px"]["branch_group_mean_iou"],
        "note": "Tolerance-aware geometric grouping; not raster area IoU.",
    }
    return result


def metric_delta(current: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, float] | None:
    if current is None or baseline is None:
        return None
    return {
        "trunk_f1_24px": float(current["trunk"]["24px"]["f1"] - baseline["trunk"]["24px"]["f1"]),
        "trunk_f1_5px": float(current["trunk"]["5px"]["f1"] - baseline["trunk"]["5px"]["f1"]),
        "branch_f1_24px": float(current["branch"]["24px"]["f1"] - baseline["branch"]["24px"]["f1"]),
        "branch_f1_5px": float(current["branch"]["5px"]["f1"] - baseline["branch"]["5px"]["f1"]),
        "group_topology_f1_24px": float(
            current["group_topology_24px"]["branch_group_topology_f1"]
            - baseline["group_topology_24px"]["branch_group_topology_f1"]
        ),
        "group_topology_f1_5px": float(
            current["group_topology_5px"]["branch_group_topology_f1"]
            - baseline["group_topology_5px"]["branch_group_topology_f1"]
        ),
    }


def rooted_branch_integrity(
    quality: dict[str, Any], truth_result: dict[str, Any] | None,
) -> dict[str, float | int | None]:
    branches = list(quality.get("branches", []))
    valid = [
        branch for branch in branches
        if int(branch.get("explicit_root_count", 0)) == 1
        and int(branch.get("connected_components", 0)) == 1
        and int(branch.get("cycle_rank", 0)) == 0
        and int(branch.get("isolated_nodes", 0)) == 0
    ]
    validity = float(len(valid) / len(branches)) if branches else 1.0
    group_f1 = None
    score = None
    if truth_result is not None:
        group_f1 = float(
            truth_result["group_topology_24px"]["branch_group_topology_f1"]
        )
        score = float(2.0 * validity * group_f1 / max(validity + group_f1, 1e-12))
    return {
        "branch_groups": len(branches),
        "valid_single_root_branch_groups": len(valid),
        "valid_single_root_branch_rate": validity,
        "branch_identity_f1_24px": group_f1,
        "rooted_branch_integrity_score_24px": score,
        "definition": "harmonic_mean(branch_identity_f1_24px, valid_single_root_branch_rate)",
    }


def run_sample_v3(
    sample: dict[str, Any], defaults: dict[str, Any], output_root: Path, manager: ModelManager, config: Any,
    botanical_ablation: bool = False,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    sample_output = v2.ensure_dir(output_root / sample_id)
    image_path = absolute_path(sample["image_path"])
    image_rgb = v2.load_image_rgb(image_path)
    device = manager.device
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    timings: dict[str, float] = {}

    started = time.perf_counter()
    roi_mask, roi_filtered = apply_roi_filter(manager.load_roi_model(), image_rgb)
    timings["roi_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    branch_mask = run_branch_segmentation(manager.load_branch_model(), roi_filtered, device=device)
    timings["branch_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    bud_result = run_bud_detection_pipeline(
        str(image_path), bud_model=manager.load_bud_global_model(), roi_model=manager.load_roi_model(),
        use_roi_filter=True, score_thr=float(defaults["bud_score_threshold"]), device=device,
    )
    timings["bud_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    processed_mask, tape_mask = prepare_processed_router_mask(branch_mask, image_rgb)
    initial_config = {
        **v2.ROUTER_CONFIG,
        "enable_junction_pairing": False,
        "enable_bud_density_prior": False,
        "enable_bud_direction_flow": False,
        "enable_bud_root_split": False,
    }
    initial_prediction = build_prediction_result(
        combined_mask=branch_mask, image_rgb=image_rgb, processed_mask=processed_mask,
        protected_tape_mask=tape_mask, **initial_config,
    )
    thickness_audit = None
    if _ACTIVE_THICKNESS_SELECTOR is not None:
        ranked = list(getattr(_ACTIVE_THICKNESS_SELECTOR, "last_ranked_candidates", []))
        thickness_audit = {
            "enabled": True,
            "selected_candidate": ranked[0] if ranked else None,
            "top_candidates": ranked[:12],
        }
    timings["routing_geometry_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    bud_orientations = extract_bud_orientations(bud_result["masks_info"])
    bud_directions = extract_directed_bud_orientations(
        bud_result["masks_info"], initial_prediction.skeleton_map, bud_result["scores"],
        min_confidence=v2.BUD_DIRECTION_MIN_CONFIDENCE,
    )
    prediction = rerank_variant(
        "full_flow", initial_prediction, tape_mask, {"boxes": bud_result["boxes"]},
        bud_directions, bud_orientations,
    )
    ablation_predictions = {}
    if botanical_ablation:
        ablation_predictions = {
            "geometry_only": rerank_variant(
                "geometry", initial_prediction, tape_mask, {"boxes": bud_result["boxes"]},
                bud_directions, bud_orientations,
                enable_hierarchy=False, source_mask=branch_mask,
            ),
            "hierarchy_only": rerank_variant(
                "geometry", initial_prediction, tape_mask, {"boxes": bud_result["boxes"]},
                bud_directions, bud_orientations,
                enable_hierarchy=True, source_mask=branch_mask,
            ),
            "bud_direction_only": prediction,
            "hierarchy_plus_bud_direction": rerank_variant(
                "full_flow", initial_prediction, tape_mask, {"boxes": bud_result["boxes"]},
                bud_directions, bud_orientations,
                enable_hierarchy=True, source_mask=branch_mask,
            ),
        }
    timings["bud_flow_rerank_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    clip = defaults["mask_clip"]
    clipped, clip_stats = clip_annotation_groups(
        prediction.annotation_groups, prediction.mask,
        max_exit_distance=float(clip["max_exit_distance"]), max_gap=float(clip["max_gap"]),
        snap_radius=int(clip["snap_radius"]),
    )
    variant_configs = {
        "v2_clipped_baseline": replace(
            config, enable_postclip_connectivity_repair=False,
            enable_supported_root_family_merge=False, enable_internal_trunk_termination=False,
        ),
        "postclip_connectivity": replace(
            config, enable_postclip_connectivity_repair=True,
            enable_supported_root_family_merge=False, enable_internal_trunk_termination=False,
        ),
        "supported_family_merge": replace(
            config, enable_postclip_connectivity_repair=True,
            enable_supported_root_family_merge=True, enable_internal_trunk_termination=False,
        ),
        "binary_v3_full": config,
    }
    variant_groups: dict[str, list[dict[str, Any]]] = {}
    variant_results: dict[str, dict[str, Any]] = {}
    truth_path = absolute_path(sample["truth_path"]) if sample.get("truth_path") else None
    for variant_name, variant_config in variant_configs.items():
        selected_groups, selected_refinement = refine_clipped_groups_v3(clipped, processed_mask, variant_config)
        selected_quality = audit_binary_topology_v3(
            selected_groups, processed_mask, variant_config, baseline_groups=clipped,
            include_global_checks=variant_name == "binary_v3_full",
        )
        selected_metrics = truth_metrics(selected_groups, truth_path, image_rgb.shape[:2])
        variant_groups[variant_name] = selected_groups
        variant_results[variant_name] = {
            "groups": len(selected_groups),
            "refinement": selected_refinement,
            "quality": selected_quality,
            "truth_metrics": selected_metrics,
        }
    groups = variant_groups["binary_v3_full"]
    refinement = variant_results["binary_v3_full"]["refinement"]
    quality = variant_results["binary_v3_full"]["quality"]
    botanical_ablation_results: dict[str, Any] = {}
    for ablation_name, ablation_prediction in ablation_predictions.items():
        ablation_clipped, ablation_clip_stats = clip_annotation_groups(
            ablation_prediction.annotation_groups, ablation_prediction.mask,
            max_exit_distance=float(clip["max_exit_distance"]), max_gap=float(clip["max_gap"]),
            snap_radius=int(clip["snap_radius"]),
        )
        ablation_groups, ablation_refinement = refine_clipped_groups_v3(
            ablation_clipped, processed_mask, config,
        )
        ablation_quality = audit_binary_topology_v3(
            ablation_groups, processed_mask, config, baseline_groups=ablation_clipped,
            include_global_checks=True,
        )
        ablation_truth = truth_metrics(ablation_groups, truth_path, image_rgb.shape[:2])
        pairing = {
            "junction_pairing_accuracy": None,
            "junction_pairing_correct": 0.0,
            "junction_pairing_trials": 0.0,
        }
        if truth_path is not None and truth_path.exists():
            pairing = compute_junction_pairing_accuracy(
                ablation_prediction.routing_stats.get("junction_pairing_debug", []),
                read_json(truth_path), image_rgb.shape[:2],
                float(ablation_prediction.routing_stats.get("scale_x", 1.0)),
                float(ablation_prediction.routing_stats.get("scale_y", 1.0)),
            )
        botanical_ablation_results[ablation_name] = {
            "truth_metrics": ablation_truth,
            "quality": ablation_quality,
            "refinement": ablation_refinement,
            "clip": ablation_clip_stats,
            "rooted_branch_integrity": rooted_branch_integrity(ablation_quality, ablation_truth),
            "crossing_port_pairing": pairing,
            "activations": {
                "hierarchy_enabled": bool(
                    ablation_prediction.routing_stats.get("hierarchy_ablation_enabled", False)
                ),
                "directed_buds_reliable": int(
                    ablation_prediction.routing_stats.get("directed_buds_reliable", 0)
                ),
                "bud_flow_evidence_clusters": int(
                    ablation_prediction.routing_stats.get("bud_flow_evidence_clusters", 0)
                ),
                "partition_x_clusters": int(
                    ablation_prediction.routing_stats.get("partition_x_clusters", 0)
                ),
                "hierarchy_violations": int(
                    ablation_prediction.routing_stats.get("hierarchy_violations", 0)
                ),
                "bud_veto_candidates": int(
                    ablation_prediction.routing_stats.get("bud_veto_candidates", 0)
                ),
            },
        }
        v2.save_json(
            sample_output / "botanical_ablation" / f"{ablation_name}_annotation_groups.json",
            {"groups": ablation_groups},
        )
    timings["clip_refine_audit_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    attachments = attach_buds_to_skeleton(
        bud_result["boxes"], bud_result["masks_info"], prediction.skeleton_map, prediction.mask,
        annotation_groups=groups,
    )
    graph = _build_directed_topology(groups)
    timings["fusion_graph_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = sum(timings.values())

    metrics = variant_results["binary_v3_full"]["truth_metrics"]
    baseline_metrics = variant_results["v2_clipped_baseline"]["truth_metrics"]
    for variant in variant_results.values():
        variant["delta_vs_v2_clipped"] = metric_delta(variant["truth_metrics"], baseline_metrics)
    summary = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "truth_path": str(truth_path) if truth_path else None,
        "result_class": "binary_topology_rule_fit_internal_consistency",
        "quality": quality,
        "refinement": refinement,
        "truth_metrics": metrics,
        "baseline_truth_metrics": baseline_metrics,
        "paired_delta_vs_v2_clipped": metric_delta(metrics, baseline_metrics),
        "variants": variant_results,
        "bud_voting": {
            "minimum_direction_confidence": v2.BUD_DIRECTION_MIN_CONFIDENCE,
            "directions_total": len(bud_directions),
            "directions_reliable": sum(item.is_reliable for item in bud_directions),
            "evidence_clusters": int(prediction.routing_stats.get("bud_flow_evidence_clusters", 0)),
            "root_splits": int(prediction.routing_stats.get("bud_flow_root_splits", 0)),
        },
        "botanical_ablation": botanical_ablation_results,
        "thickness_trunk": thickness_audit or {"enabled": False},
        "clip": clip_stats,
        "attachment": attachment_stats(attachments),
        "dag": graph["stats"],
        "groups": len(groups),
        "timings": timings,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
    }
    v2.save_image(sample_output / "roi_mask.png", roi_mask)
    v2.save_image(sample_output / "branch_mask.png", branch_mask)
    v2.save_image(sample_output / "processed_router_mask.png", processed_mask.astype(np.uint8) * 255)
    v2.save_image(sample_output / "routing_mask.png", prediction.mask.astype(np.uint8) * 255)
    v2.save_image(sample_output / "skeleton_map.png", prediction.skeleton_map.astype(np.uint8) * 255)
    if tape_mask is not None:
        v2.save_image(sample_output / "tape_mask.png", (tape_mask > 0).astype(np.uint8) * 255)
    v2.save_image(sample_output / "buds.png", bud_result["annotated_image"])
    v2.save_image(sample_output / "numbered_overlay.jpg", numbered_overlay(image_rgb, groups))
    v2.save_json(sample_output / "annotation_groups.json", {"groups": groups})
    for variant_name, selected_groups in variant_groups.items():
        v2.save_json(sample_output / "variants" / f"{variant_name}_annotation_groups.json", {"groups": selected_groups})
    v2.save_json(sample_output / "directed_graph.json", graph)
    v2.save_json(sample_output / "bud_directions.json", [asdict(item) for item in bud_directions])
    v2.save_json(sample_output / "attachments.json", [asdict(item) for item in attachments])
    v2.save_json(sample_output / "v3_refinement.json", refinement)
    v2.save_json(sample_output / "v3_quality_gate.json", quality)
    v2.save_json(sample_output / "summary.json", summary)
    return summary


def samples_from_rulebook(rulebook: dict[str, Any]) -> list[dict[str, Any]]:
    samples = []
    for item in rulebook["samples"]:
        samples.append({
            "sample_id": item["sample_id"],
            "image_path": item["image_path"],
            "truth_path": item["truth_path"],
            "category": "gt_binary_topology_rulebook",
            "selection_evidence": "all_100_skeleton_truth",
        })
    return samples


def aggregate_variants(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "trunk_f1_24px", "trunk_f1_5px", "branch_f1_24px", "branch_f1_5px",
        "group_topology_f1_24px", "group_topology_f1_5px",
    )
    aggregate: dict[str, Any] = {}
    for variant_name in ("v2_clipped_baseline", "postclip_connectivity", "supported_family_merge", "binary_v3_full"):
        rows = [
            item["variants"][variant_name]
            for item in summaries
            if item.get("variants", {}).get(variant_name, {}).get("truth_metrics") is not None
        ]
        metric_summary: dict[str, Any] = {}
        for key in keys:
            deltas = [float(row["delta_vs_v2_clipped"][key]) for row in rows]
            metric_summary[key] = {
                "mean_delta": float(np.mean(deltas)) if deltas else None,
                "improved": sum(value > 1e-9 for value in deltas),
                "degraded": sum(value < -1e-9 for value in deltas),
                "unchanged": sum(abs(value) <= 1e-9 for value in deltas),
            }
        aggregate[variant_name] = {
            "samples_with_truth": len(rows),
            "accepted": sum(row["quality"]["status"] == "accepted" for row in rows),
            "root_counts": {
                "zero": sum(row["quality"].get("counts", {}).get("zero_explicit_root_groups", 0) for row in rows),
                "multiple": sum(row["quality"].get("counts", {}).get("multiple_explicit_root_groups", 0) for row in rows),
                "disconnected": sum(row["quality"].get("counts", {}).get("disconnected_graph_groups", 0) for row in rows),
                "isolated_nodes": sum(row["quality"].get("counts", {}).get("isolated_graph_nodes", 0) for row in rows),
                "duplicate_edges": sum(row["quality"].get("counts", {}).get("duplicate_physical_edges", 0) for row in rows),
            },
            "paired_changes": metric_summary,
        }
    return aggregate


def aggregate_botanical_ablation(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "geometry_only", "hierarchy_only", "bud_direction_only",
        "hierarchy_plus_bud_direction",
    )
    aggregate: dict[str, Any] = {}
    for name in names:
        rows = [
            item["botanical_ablation"][name]
            for item in summaries
            if name in item.get("botanical_ablation", {})
            and item["botanical_ablation"][name].get("truth_metrics") is not None
        ]
        scores = [
            float(row["rooted_branch_integrity"]["rooted_branch_integrity_score_24px"])
            for row in rows
        ]
        validity = [
            float(row["rooted_branch_integrity"]["valid_single_root_branch_rate"])
            for row in rows
        ]
        group_f1 = [
            float(row["rooted_branch_integrity"]["branch_identity_f1_24px"])
            for row in rows
        ]
        hierarchy_f1 = [
            float(row["truth_metrics"]["branch_hierarchy_24px"]["f1"])
            for row in rows
        ]
        hierarchy_label_accuracy = [
            float(row["truth_metrics"]["branch_hierarchy_24px"]["spatially_matched_label_accuracy"])
            for row in rows
        ]
        pairing_correct = sum(
            float(row["crossing_port_pairing"].get("junction_pairing_correct", 0.0)) for row in rows
        )
        pairing_trials = sum(
            float(row["crossing_port_pairing"].get("junction_pairing_trials", 0.0)) for row in rows
        )
        aggregate[name] = {
            "samples": len(rows),
            "rooted_branch_integrity_score_24px": float(np.mean(scores)) if scores else None,
            "valid_single_root_branch_rate": float(np.mean(validity)) if validity else None,
            "branch_identity_f1_24px": float(np.mean(group_f1)) if group_f1 else None,
            "branch_hierarchy_f1_24px": float(np.mean(hierarchy_f1)) if hierarchy_f1 else None,
            "branch_hierarchy_label_accuracy_24px": (
                float(np.mean(hierarchy_label_accuracy)) if hierarchy_label_accuracy else None
            ),
            "crossing_port_pairing_accuracy": (
                float(pairing_correct / pairing_trials) if pairing_trials else None
            ),
            "crossing_port_pairing_trials": int(pairing_trials),
            "activations": {
                key: int(sum(row["activations"].get(key, 0) for row in rows))
                for key in (
                    "directed_buds_reliable", "bud_flow_evidence_clusters",
                    "partition_x_clusters", "hierarchy_violations", "bud_veto_candidates",
                )
            },
        }
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent binary-mask topology routing v3")
    parser.add_argument("--manifest", type=Path, default=SCRIPT_DIR / "e2e_regression_manifest.json")
    parser.add_argument("--rulebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument(
        "--sample-spec", action="append", metavar="SAMPLE_ID=IMAGE_PATH",
        help="Add an image without skeleton truth (repeatable).",
    )
    parser.add_argument("--all-gt", action="store_true")
    parser.add_argument("--random-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--require-closeout", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enable-thickness-trunk", action="store_true")
    parser.add_argument(
        "--botanical-ablation", action="store_true",
        help="Run geometry/hierarchy/bud-direction/combined variants from one perception pass.",
    )
    parser.add_argument(
        "--thickness-rulebook", type=Path,
        default=PROJECT_ROOT / "04_results/mask_topology_routing/gt_mask_thickness_rulebook_20260731/gt_mask_thickness_rulebook.json",
    )
    return parser.parse_args()


def main() -> int:
    global _ACTIVE_THICKNESS_SELECTOR, _THICKNESS_RUN_CONTEXT
    args = parse_args()
    manifest = read_json(args.manifest)
    config = load_binary_v3_config(args.rulebook)
    if args.enable_thickness_trunk:
        thickness_path = absolute_path(args.thickness_rulebook)
        if not thickness_path.is_file():
            raise FileNotFoundError(thickness_path)
        _ACTIVE_THICKNESS_SELECTOR = make_router_selector(thickness_path, config.rulebook_path)
        routing_utils._choose_trunk_path_on_skeleton = _ACTIVE_THICKNESS_SELECTOR
        _THICKNESS_RUN_CONTEXT = {
            "enabled": True,
            "rulebook_path": str(thickness_path.resolve()),
            "rulebook_sha256": sha256_path(thickness_path),
            "selector": "evidence_gated_sustained_width_v1",
        }
    output = v2.ensure_dir(args.output)
    weight_checks = []
    for stage in ("roi", "branch", "bud"):
        weight_path = absolute_path(manifest["defaults"][f"{stage}_weight"])
        observed = sha256_path(weight_path) if weight_path.exists() else "missing"
        expected = manifest["defaults"][f"{stage}_sha256"]
        weight_checks.append({
            "stage": stage, "path": str(weight_path), "expected_sha256": expected,
            "observed_sha256": observed, "valid": observed == expected,
        })
    if not all(item["valid"] for item in weight_checks):
        v2.save_json(output / "weight_validation.json", {"valid": False, "checks": weight_checks})
        raise RuntimeError("Frozen model weight validation failed")
    v2.save_json(output / "weight_validation.json", {"valid": True, "checks": weight_checks})
    if args.require_closeout:
        closeout = read_json(args.require_closeout)
        if not closeout.get("release_ready_for_random_20_visual_audit"):
            raise RuntimeError("The full-100 closeout has not released random visual auditing")
    samples = samples_from_rulebook(config.rulebook) if args.all_gt else list(manifest["samples"])
    for specification in args.sample_spec or []:
        if "=" not in specification:
            raise ValueError(f"Invalid --sample-spec: {specification!r}")
        sample_id, image_path = specification.split("=", 1)
        samples.append({
            "sample_id": sample_id.strip(), "image_path": image_path.strip(), "truth_path": None,
            "category": "fixed_visual_case", "selection_evidence": "user_named_case",
        })
    samples = list({str(sample["sample_id"]): sample for sample in samples}.values())
    if args.random_samples:
        by_tree: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            match = re.match(r"(tree_\d+)_", str(sample["sample_id"]))
            tree_id = match.group(1) if match else str(sample["sample_id"])
            by_tree.setdefault(tree_id, []).append(sample)
        generator = random.Random(args.seed)
        tree_ids = sorted(by_tree)
        generator.shuffle(tree_ids)
        chosen_trees = tree_ids[: args.random_samples]
        samples = [generator.choice(sorted(by_tree[tree_id], key=lambda item: item["sample_id"])) for tree_id in chosen_trees]
    selected = set(args.sample_id or [])
    if selected:
        samples = [sample for sample in samples if sample["sample_id"] in selected]
        missing = selected - {sample["sample_id"] for sample in samples}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
    v2.save_json(output / "run_config.json", {
        "schema_version": config.schema_version,
        "result_class": "binary_topology_rule_fit_internal_consistency",
        "independent_generalization_claim": False,
        "rulebook_path": str(config.rulebook_path),
        "rulebook_sha256": config.rulebook_sha256,
        "device": args.device,
        "defaults": manifest["defaults"],
        "router_config": v2.ROUTER_CONFIG,
        "sample_count": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "random_selection": {"count": args.random_samples, "seed": args.seed} if args.random_samples else None,
        "thickness_trunk": _THICKNESS_RUN_CONTEXT or {"enabled": False},
        "botanical_ablation": bool(args.botanical_ablation),
        "weight_validation": weight_checks,
    })
    manager = ModelManager()
    manager.device = args.device
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started_all = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        summary_path = output / sample_id / "summary.json"
        completion_path = output / sample_id / "completion.json"
        signature = sample_signature(
            sample, manifest["defaults"], config,
            botanical_ablation=args.botanical_ablation,
        )
        completion = read_json(completion_path) if completion_path.exists() else None
        required_artifacts = [
            summary_path, output / sample_id / "annotation_groups.json",
            output / sample_id / "numbered_overlay.jpg", output / sample_id / "v3_quality_gate.json",
        ]
        if (
            args.resume and completion is not None
            and completion.get("signature_sha256") == signature["signature_sha256"]
            and all(path.exists() for path in required_artifacts)
        ):
            summaries.append(read_json(summary_path))
            print(f"[{index}/{len(samples)}] {sample_id} 已完成，跳过", flush=True)
            continue
        print(f"[{index}/{len(samples)}] {sample_id} 开始", flush=True)
        sample_started = time.perf_counter()
        try:
            summary = run_sample_v3(
                sample, manifest["defaults"], output, manager, config,
                botanical_ablation=args.botanical_ablation,
            )
            summaries.append(summary)
            v2.save_json(completion_path, signature)
            elapsed = time.perf_counter() - sample_started
            average = (time.perf_counter() - started_all) / index
            eta = average * (len(samples) - index)
            print(
                f"[{index}/{len(samples)}] {sample_id} 完成 {elapsed:.1f}s | "
                f"质量={summary['quality']['status']} | 预计剩余 {eta / 60:.1f}min",
                flush=True,
            )
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": repr(exc)})
            print(f"[{index}/{len(samples)}] {sample_id} 失败: {exc!r}", flush=True)
        v2.save_json(output / "progress.json", {
            "requested": len(samples), "completed": len(summaries), "failed": len(failures),
            "last_sample": sample_id, "failures": failures,
        })
    accepted = sum(item["quality"]["status"] == "accepted" for item in summaries)
    result = {
        "status": "completed" if not failures else "completed_with_failures",
        "result_class": "binary_topology_rule_fit_internal_consistency",
        "samples_requested": len(samples),
        "samples_completed": len(summaries),
        "accepted": accepted,
        "excluded": len(summaries) - accepted,
        "failures": failures,
        "exclusion_reasons": dict(sorted(Counter(
            reason for item in summaries for reason in item["quality"].get("exclusion_reasons", [])
        ).items())),
        "variant_paired_summary": aggregate_variants(summaries),
        "botanical_ablation_summary": aggregate_botanical_ablation(summaries),
        "summaries": summaries,
    }
    v2.save_json(output / "regression_summary.json", result)
    print(json.dumps(v2.json_safe({key: value for key, value in result.items() if key != "summaries"}), ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
