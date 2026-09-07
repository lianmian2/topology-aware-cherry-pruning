from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "02_code" / "02_models"
GUI_DIR = PROJECT_ROOT / "07_graphical_interface" / "unified_system"
for path in (MODEL_DIR, GUI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bud_skeleton_fusion import extract_bud_orientations, extract_directed_bud_orientations
from bud_skeleton_fusion.bud_skeleton_attachment import attach_buds_to_skeleton, attachment_stats
from bud_skeleton_fusion.evaluate_bud_topology_cv import rerank_variant
from logic_branch import run_branch_segmentation
from logic_bud import run_bud_detection_pipeline
from logic_models import ModelManager
from logic_roi import apply_roi_filter
from mask_topology_routing.mask_clip import clip_annotation_groups
from mask_topology_routing.utils import (
    _build_directed_topology,
    build_prediction_result,
    prepare_processed_router_mask,
)


ROUTER_CONFIG = {
    "root_band_height": 16,
    "trunk_exclusion_radius": 10.0,
    "min_branch_length": 12.0,
    "prune_spur_length": 8,
    "trunk_rdp_epsilon_ratio": 0.01,
    "branch_rdp_epsilon": 2.0,
    "min_endpoint_branch_length": 16,
    "max_endpoints_to_route": 24,
    "max_processing_dim": 1280,
    "enable_vertical_bridge": True,
    "bridge_max_gap": 96,
    "bridge_max_dx": 28,
    "bridge_min_component_area": 80,
    "bridge_min_component_height": 40,
    "enable_junction_pairing": True,
    "enable_bud_density_prior": True,
    "enable_bud_direction_flow": True,
    "enable_bud_root_split": True,
    "enable_partition_constraints": False,
    "structural_tolerance_px": 24.0,
    "junction_cluster_radius": 12.0,
}

BUD_DIRECTION_MIN_CONFIDENCE = 0.18


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def group_length(points: list[list[int]], edges: list[list[int]]) -> float:
    total = 0.0
    for edge in edges:
        if len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if 0 <= src < len(points) and 0 <= dst < len(points):
            total += math.dist(points[src], points[dst])
    return total


def group_centroid(group: dict[str, Any]) -> tuple[float, float]:
    points = group.get("points", [])
    if not points:
        return (0.0, 0.0)
    array = np.asarray(points, dtype=float)
    return (float(array[:, 0].mean()), float(array[:, 1].mean()))


def extract_features(
    sample_id: str,
    groups: list[dict[str, Any]],
    attachments: list[Any],
    labels: np.ndarray,
    image_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    height, width = image_shape
    diagonal = math.hypot(width, height)
    bud_by_group: dict[str, list[int]] = {}
    for attachment in attachments:
        if attachment.group_id is not None:
            bud_by_group.setdefault(str(attachment.group_id), []).append(int(attachment.bud_index))
    branches = [group for group in groups if group.get("group_type") != "trunk"]
    centroids = {str(group.get("group_id")): group_centroid(group) for group in branches}
    rows: list[dict[str, Any]] = []
    for group in branches:
        group_id = str(group.get("group_id"))
        points = group.get("points", [])
        edges = group.get("edges", [])
        length = group_length(points, edges)
        centroid = centroids[group_id]
        other_distances = [math.dist(centroid, value) for key, value in centroids.items() if key != group_id]
        bud_indices = [index for index in bud_by_group.get(group_id, []) if 0 <= index < len(labels)]
        flower_count = sum(int(labels[index]) == 0 for index in bud_indices)
        rows.append(
            {
                "sample_id": sample_id,
                "branch_id": group_id,
                "branch_length_norm": length / max(diagonal, 1.0),
                "relative_height": centroid[1] / max(height, 1),
                "bud_count": len(bud_indices),
                "bud_density": len(bud_indices) / max(length, 1.0),
                "flower_bud_ratio": flower_count / len(bud_indices) if bud_indices else None,
                "local_crowding": 1.0 / max(min(other_distances), 1.0) if other_distances else 0.0,
            }
        )
    return rows


def write_feature_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "sample_id",
        "branch_id",
        "branch_length_norm",
        "relative_height",
        "bud_count",
        "bud_density",
        "flower_bud_ratio",
        "local_crowding",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hierarchy_audit(groups: list[dict[str, Any]], routing_stats: dict[str, Any]) -> dict[str, Any]:
    branches = [group for group in groups if group.get("group_type") != "trunk"]
    fork_depths = [int(group["fork_depth"]) for group in branches if "fork_depth" in group]
    root_contacts = [int(group["root_contact_count"]) for group in branches if "root_contact_count" in group]
    return {
        "branch_order_validation": "topological_constraints" if routing_stats.get("partition_constraints_enabled") else "not_available",
        "branch_groups": len(branches),
        "primary_branch_groups": sum(group.get("group_type") == "branch" for group in branches),
        "secondary_branch_groups": sum(group.get("group_type") == "secondary_branch" for group in branches),
        "max_fork_depth": max(fork_depths) if fork_depths else None,
        "groups_without_direct_trunk_contact": sum(value == 0 for value in root_contacts) if root_contacts else None,
        "hierarchy_splits": int(routing_stats.get("hierarchy_splits", 0)),
        "hierarchy_violations": int(routing_stats.get("hierarchy_violations", 0)),
        "angle_splits": int(routing_stats.get("angle_splits", 0)),
        "return_splits": int(routing_stats.get("return_splits", 0)),
        "bud_vetoes": int(routing_stats.get("bud_vetoes", 0)),
    }


def routing_quality_audit(groups: list[dict[str, Any]]) -> dict[str, Any]:
    edge_owners: dict[tuple[tuple[int, int], tuple[int, int]], set[str]] = {}
    point_owners: dict[tuple[int, int], set[str]] = {}
    root_family_members: dict[str, set[str]] = {}
    for group in groups:
        group_id = str(group.get("group_id", "unknown"))
        if group.get("group_type") != "trunk":
            family_id = group_id.split("_root_", 1)[0]
            root_family_members.setdefault(family_id, set()).add(group_id)
        points = [tuple(map(int, point)) for point in group.get("points", [])]
        for point in set(points):
            point_owners.setdefault(point, set()).add(group_id)
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if not (0 <= src < len(points) and 0 <= dst < len(points)) or src == dst:
                continue
            edge_key = tuple(sorted((points[src], points[dst])))
            edge_owners.setdefault(edge_key, set()).add(group_id)
    duplicated_edges = {edge: owners for edge, owners in edge_owners.items() if len(owners) > 1}
    shared_points = {point: owners for point, owners in point_owners.items() if len(owners) > 1}
    max_root_family_members = max((len(members) for members in root_family_members.values()), default=0)
    if duplicated_edges:
        status = "reject_duplicate_routed_edges"
    elif max_root_family_members >= 5:
        status = "reject_excessive_root_family_fragmentation"
    else:
        status = "pass"
    return {
        "status": status,
        "duplicate_physical_edges": len(duplicated_edges),
        "duplicate_edge_group_memberships": sum(len(owners) - 1 for owners in duplicated_edges.values()),
        "shared_landmark_points": len(shared_points),
        "max_groups_per_point": max((len(owners) for owners in point_owners.values()), default=0),
        "max_root_family_members": max_root_family_members,
        "root_families_with_multiple_routes": {
            family_id: sorted(members)
            for family_id, members in root_family_members.items()
            if len(members) > 1
        },
        "duplicate_edge_examples": [
            {"edge": [list(edge[0]), list(edge[1])], "group_ids": sorted(owners)}
            for edge, owners in list(duplicated_edges.items())[:20]
        ],
    }


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    defaults = manifest["defaults"]
    checks: list[dict[str, Any]] = []
    for stage in ("roi", "branch", "bud"):
        path = PROJECT_ROOT / defaults[f"{stage}_weight"]
        observed = sha256_file(path) if path.exists() else "missing"
        expected = defaults[f"{stage}_sha256"]
        checks.append(
            {
                "kind": "weight",
                "stage": stage,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "valid": observed == expected,
            }
        )
    for sample in manifest["samples"]:
        path = PROJECT_ROOT / sample["image_path"]
        checks.append(
            {
                "kind": "sample",
                "sample_id": sample["sample_id"],
                "path": str(path),
                "valid": path.exists(),
            }
        )
    return {"valid": all(check["valid"] for check in checks), "checks": checks}


def run_sample(
    sample: dict[str, Any], defaults: dict[str, Any], output_root: Path, manager: ModelManager,
    postclip_refiner=None, postclip_quality_auditor=None,
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    sample_output = ensure_dir(output_root / sample_id)
    image_path = PROJECT_ROOT / sample["image_path"]
    image_rgb = load_image_rgb(image_path)
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
        str(image_path),
        bud_model=manager.load_bud_global_model(),
        roi_model=manager.load_roi_model(),
        use_roi_filter=True,
        score_thr=float(defaults["bud_score_threshold"]),
        device=device,
    )
    timings["bud_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    processed_mask, tape_mask = prepare_processed_router_mask(branch_mask, image_rgb)
    initial_router_config = {
        **ROUTER_CONFIG,
        "enable_junction_pairing": False,
        "enable_bud_density_prior": False,
        "enable_bud_direction_flow": False,
        "enable_bud_root_split": False,
    }
    initial_prediction = build_prediction_result(
        combined_mask=branch_mask,
        image_rgb=image_rgb,
        processed_mask=processed_mask,
        protected_tape_mask=tape_mask,
        **initial_router_config,
    )
    timings["routing_geometry_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    bud_orientations = extract_bud_orientations(bud_result["masks_info"])
    bud_directions = extract_directed_bud_orientations(
        bud_result["masks_info"],
        initial_prediction.skeleton_map,
        bud_result["scores"],
        min_confidence=BUD_DIRECTION_MIN_CONFIDENCE,
    )
    prediction = rerank_variant(
        "full_flow",
        initial_prediction,
        tape_mask,
        {"boxes": bud_result["boxes"]},
        bud_directions,
        bud_orientations,
    )
    timings["bud_flow_rerank_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    clip_config = defaults["mask_clip"]
    groups, clip_stats = clip_annotation_groups(
        prediction.annotation_groups,
        prediction.mask,
        max_exit_distance=float(clip_config["max_exit_distance"]),
        max_gap=float(clip_config["max_gap"]),
        snap_radius=int(clip_config["snap_radius"]),
    )
    clipped_groups = groups
    postclip_refinement = None
    if postclip_refiner is not None:
        groups, postclip_refinement = postclip_refiner(groups, processed_mask)
    postclip_quality = None
    if postclip_quality_auditor is not None:
        postclip_quality = postclip_quality_auditor(groups, processed_mask, clipped_groups)
    timings["clip_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    attachments = attach_buds_to_skeleton(
        bud_result["boxes"],
        bud_result["masks_info"],
        prediction.skeleton_map,
        prediction.mask,
        annotation_groups=groups,
    )
    graph = _build_directed_topology(groups)
    features = extract_features(sample_id, groups, attachments, bud_result["labels"], image_rgb.shape[:2])
    timings["fusion_graph_feature_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = sum(timings.values())

    attachment_rows = []
    for attachment in attachments:
        row = asdict(attachment)
        row["bud_label"] = int(bud_result["labels"][attachment.bud_index])
        row["bud_score"] = float(bud_result["scores"][attachment.bud_index])
        attachment_rows.append(row)
    summary = {
        "sample_id": sample_id,
        "category": sample["category"],
        "selection_evidence": sample["selection_evidence"],
        "image_path": str(image_path),
        "image_shape": list(image_rgb.shape),
        "roi_coverage": float(np.mean(roi_mask > 0)),
        "branch_pixels": int(np.sum(branch_mask > 0)),
        "groups": len(groups),
        "buds": int(bud_result["total_count"]),
        "bud_voting": {
            "minimum_direction_confidence": BUD_DIRECTION_MIN_CONFIDENCE,
            "directions_total": len(bud_directions),
            "directions_reliable": sum(direction.is_reliable for direction in bud_directions),
            "evidence_clusters": int(prediction.routing_stats.get("bud_flow_evidence_clusters", 0)),
            "root_splits": int(prediction.routing_stats.get("bud_flow_root_splits", 0)),
            "cycle_repairs": int(prediction.routing_stats.get("bud_consistency_cycles_repaired", 0)),
        },
        "hierarchy": hierarchy_audit(groups, prediction.routing_stats),
        "routing_quality": routing_quality_audit(groups),
        "routing_stats": prediction.routing_stats,
        "attachment": attachment_stats(attachments),
        "clip": clip_stats,
        "postclip_refinement": postclip_refinement,
        "postclip_quality": postclip_quality,
        "dag": graph["stats"],
        "feature_rows": len(features),
        "timings": timings,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
    }

    save_image(sample_output / "roi_mask.png", roi_mask)
    save_image(sample_output / "roi_filtered.png", cv2.cvtColor(roi_filtered, cv2.COLOR_RGB2BGR))
    save_image(sample_output / "branch_mask.png", branch_mask)
    save_image(sample_output / "processed_router_mask.png", processed_mask.astype(np.uint8) * 255)
    if tape_mask is not None:
        save_image(sample_output / "tape_mask.png", (tape_mask > 0).astype(np.uint8) * 255)
    save_image(sample_output / "skeleton_map_geometry.png", initial_prediction.skeleton_map.astype(np.uint8) * 255)
    save_image(sample_output / "skeleton_map.png", prediction.skeleton_map.astype(np.uint8) * 255)
    save_image(sample_output / "buds.png", bud_result["annotated_image"])
    save_json(sample_output / "annotation_groups.json", {"groups": groups})
    save_json(sample_output / "attachments.json", attachment_rows)
    save_json(sample_output / "bud_directions.json", [asdict(direction) for direction in bud_directions])
    save_json(sample_output / "directed_graph.json", graph)
    write_feature_csv(sample_output / "branch_features.csv", features)
    save_json(sample_output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen RGB-to-DAG regression")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("e2e_regression_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation = validate_manifest(manifest)
    output = ensure_dir(args.output)
    save_json(output / "manifest_validation.json", validation)
    if not validation["valid"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 2
    save_json(
        output / "run_config.json",
        {
            "manifest": str(args.manifest.resolve()),
            "git_commit": git_commit(),
            "device": args.device,
            "router_config": ROUTER_CONFIG,
            "defaults": manifest["defaults"],
        },
    )
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0

    selected_ids = set(args.sample_id or [])
    samples = [sample for sample in manifest["samples"] if not selected_ids or sample["sample_id"] in selected_ids]
    unknown = selected_ids - {sample["sample_id"] for sample in samples}
    if unknown:
        raise ValueError(f"Unknown sample IDs: {sorted(unknown)}")
    manager = ModelManager()
    manager.device = args.device
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample in samples:
        try:
            summaries.append(run_sample(sample, manifest["defaults"], output, manager))
        except Exception as exc:
            failures.append({"sample_id": sample["sample_id"], "error": repr(exc)})
    result = {
        "status": "completed" if not failures else "failed",
        "samples_requested": len(samples),
        "samples_completed": len(summaries),
        "failures": failures,
        "dag_rate": float(np.mean([item["dag"]["is_dag"] for item in summaries])) if summaries else 0.0,
        "category_counts": dict(sorted(Counter(item["category"] for item in summaries).items())),
        "summaries": summaries,
    }
    save_json(output / "regression_summary.json", result)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
