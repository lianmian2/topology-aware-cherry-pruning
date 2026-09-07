from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "02_code" / "02_models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from bud_skeleton_fusion import DirectedBudOrientation, extract_directed_bud_orientations
from bud_skeleton_fusion.bud_detection_cache import load_bud_detection_cache
from mask_topology_routing.data import load_manifest
from mask_topology_routing.utils import DEFAULT_PROCESSED_ROOT, ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在GT骨架上验证芽基到芽尖的有向生长先验")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "04_results" / "bud_skeleton_fusion" / "full_image_bud_cache_20260714" / "per_sample"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "04_results" / "bud_skeleton_fusion" / "gt_bud_direction_20260714"))
    parser.add_argument("--association-radius", type=float, default=36.0)
    parser.add_argument("--min-tangent-projection", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_all_samples() -> List[Dict]:
    manifest_root = DEFAULT_PROCESSED_ROOT / "annotations" / "skeleton_prediction"
    samples = []
    for name in ("manifest_train.json", "manifest_test.json"):
        path = manifest_root / name
        if path.exists():
            samples.extend(load_manifest(path))
    unique = {sample["sample_name"]: sample for sample in samples}
    annotation_dir = PROJECT_ROOT / "01_data" / "02_annotated" / "skeleton_annotation"
    for annotation_path in annotation_dir.glob("*_skeleton.json"):
        sample_name = annotation_path.name[:-len("_skeleton.json")]
        if sample_name in unique:
            continue
        for split in ("train", "test"):
            base = PROJECT_ROOT / "01_data" / "03_processed" / split / "skeleton_prediction"
            image_path = base / "images" / f"{sample_name}.jpg"
            mask_path = base / "branch_masks" / f"{sample_name}.png"
            if image_path.exists() and mask_path.exists():
                unique[sample_name] = {
                    "sample_name": sample_name,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "annotation_path": str(annotation_path),
                }
                break
    return [unique[name] for name in sorted(unique)]


def render_groups(groups: List[Dict], shape: Tuple[int, int], group_type: Optional[str] = None) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for group in groups:
        if group_type is not None and group.get("group_type") != group_type:
            continue
        points = [tuple(map(int, point)) for point in group.get("points", [])]
        for edge in group.get("edges", []):
            if len(edge) == 2 and 0 <= int(edge[0]) < len(points) and 0 <= int(edge[1]) < len(points):
                cv2.line(canvas, points[int(edge[0])], points[int(edge[1])], 1, 2, cv2.LINE_AA)
    return canvas


def build_rooted_group(group: Dict, trunk_distance: np.ndarray) -> Optional[Dict]:
    points = np.asarray(group.get("points", []), dtype=np.float32)
    if len(points) < 2:
        return None
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    valid_edges = []
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if 0 <= src < len(points) and 0 <= dst < len(points) and src != dst:
            length = float(np.linalg.norm(points[dst] - points[src]))
            graph.add_edge(src, dst, weight=max(length, 1e-6))
            valid_edges.append((src, dst))
    if not valid_edges:
        return None
    height, width = trunk_distance.shape
    root = min(
        graph.nodes,
        key=lambda index: float(trunk_distance[
            int(np.clip(round(points[index, 1]), 0, height - 1)),
            int(np.clip(round(points[index, 0]), 0, width - 1)),
        ]),
    )
    path_distance = nx.single_source_dijkstra_path_length(graph, root, weight="weight")
    return {"group": group, "points": points, "edges": valid_edges, "root": int(root), "path_distance": path_distance}


def nearest_oriented_edge(point_xy: np.ndarray, rooted_groups: List[Dict]) -> Tuple[Optional[Dict], float, np.ndarray]:
    best_group = None
    best_distance = float("inf")
    best_tangent = np.zeros((2,), dtype=np.float32)
    for rooted in rooted_groups:
        points = rooted["points"]
        for src, dst in rooted["edges"]:
            start = points[src]
            end = points[dst]
            segment = end - start
            segment_sq = float(np.dot(segment, segment))
            ratio = 0.0 if segment_sq < 1e-9 else float(np.clip(np.dot(point_xy - start, segment) / segment_sq, 0.0, 1.0))
            projection = start + ratio * segment
            distance = float(np.linalg.norm(point_xy - projection))
            if distance >= best_distance:
                continue
            if rooted["path_distance"].get(dst, 0.0) >= rooted["path_distance"].get(src, 0.0):
                tangent = segment
            else:
                tangent = -segment
            tangent_norm = float(np.linalg.norm(tangent))
            best_group = rooted
            best_distance = distance
            best_tangent = tangent / max(tangent_norm, 1e-6)
    return best_group, best_distance, best_tangent


def local_mask_tangent(mask_skeleton: np.ndarray, point_xy: Tuple[float, float], radius: int = 24) -> Optional[np.ndarray]:
    x, y = map(int, map(round, point_xy))
    y1, y2 = max(0, y - radius), min(mask_skeleton.shape[0], y + radius + 1)
    x1, x2 = max(0, x - radius), min(mask_skeleton.shape[1], x + radius + 1)
    ys, xs = np.where(mask_skeleton[y1:y2, x1:x2] > 0)
    if len(xs) < 4:
        return None
    points = np.column_stack([xs + x1, ys + y1]).astype(np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    tangent = vectors[:, int(np.argmax(values))].astype(np.float32)
    return tangent / max(float(np.linalg.norm(tangent)), 1e-6)


def analyze_sample(sample: Dict, cache_path: Path, association_radius: float, min_projection: float) -> Dict:
    cache = load_bud_detection_cache(cache_path)
    annotation = load_json(Path(sample["annotation_path"]))
    height, width = cache["image_shape"]
    trunk_mask = render_groups(annotation.get("groups", []), (height, width), "trunk")
    full_skeleton = render_groups(annotation.get("groups", []), (height, width))
    trunk_distance = distance_transform_edt(trunk_mask == 0)
    rooted_groups = [
        rooted for rooted in (
            build_rooted_group(group, trunk_distance)
            for group in annotation.get("groups", [])
            if group.get("group_type") == "branch"
        ) if rooted is not None
    ]
    directions = extract_directed_bud_orientations(cache["masks_info"], full_skeleton, cache["scores"])
    mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is not None and mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask_skeleton = skeletonize(mask > 0).astype(np.uint8) if mask is not None else np.zeros((height, width), dtype=np.uint8)
    votes_by_group = defaultdict(list)
    records = []
    for direction in directions:
        rooted, distance, tangent = nearest_oriented_edge(np.asarray(direction.centroid_global, dtype=np.float32), rooted_groups)
        if rooted is None or distance > association_radius:
            continue
        vector = np.asarray(direction.vector_xy, dtype=np.float32)
        signed_projection = float(np.dot(vector, tangent))
        local_tangent = local_mask_tangent(mask_skeleton, direction.centroid_global)
        gt_alignment = abs(signed_projection)
        local_alignment = abs(float(np.dot(vector, local_tangent))) if local_tangent is not None else 0.0
        latent_spur = bool(gt_alignment < 0.35 and local_alignment > 0.75)
        direction.is_latent_spur = latent_spur
        eligible = bool(direction.is_reliable and not latent_spur and gt_alignment >= min_projection)
        weight = float(direction.confidence * gt_alignment) if eligible else 0.0
        group_id = rooted["group"].get("group_id", "")
        if eligible:
            votes_by_group[group_id].append((signed_projection, weight))
        records.append({
            "bud_index": int(direction.bud_index),
            "group_id": group_id,
            "distance": float(distance),
            "confidence": float(direction.confidence),
            "signed_projection": signed_projection,
            "latent_spur": latent_spur,
            "eligible": eligible,
            "failure_reason": direction.failure_reason,
        })
    individual_votes = [record for record in records if record["eligible"]]
    branch_votes = []
    for group_id, votes in votes_by_group.items():
        if len(votes) < 2:
            continue
        signed_score = sum(np.sign(value) * weight for value, weight in votes) / max(sum(weight for _, weight in votes), 1e-6)
        branch_votes.append({"group_id": group_id, "n_votes": len(votes), "signed_score": float(signed_score), "correct": bool(signed_score > 0.0)})
    return {
        "sample_name": sample["sample_name"],
        "total_buds": int(len(directions)),
        "reliable_buds": int(sum(direction.is_reliable for direction in directions)),
        "associated_buds": int(len(records)),
        "eligible_direction_votes": int(len(individual_votes)),
        "individual_direction_accuracy": float(np.mean([record["signed_projection"] > 0.0 for record in individual_votes])) if individual_votes else 0.0,
        "latent_spur_count": int(sum(record["latent_spur"] for record in records)),
        "branch_direction_trials": int(len(branch_votes)),
        "branch_direction_accuracy": float(np.mean([vote["correct"] for vote in branch_votes])) if branch_votes else 0.0,
        "branch_votes": branch_votes,
        "records": records,
    }


def grouped_folds(sample_names: List[str], num_folds: int = 5) -> Dict[str, int]:
    tree_samples = defaultdict(list)
    for sample_name in sample_names:
        match = re.match(r"(tree_\d+)_", sample_name)
        if match:
            tree_samples[match.group(1)].append(sample_name)
    fold_sizes = [0] * num_folds
    assignment = {}
    for tree_id, names in sorted(tree_samples.items(), key=lambda item: (-len(item[1]), item[0])):
        fold = int(np.argmin(fold_sizes))
        fold_sizes[fold] += len(names)
        for name in names:
            assignment[name] = fold
    return assignment


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    per_sample_dir = ensure_dir(output_dir / "per_sample")
    samples = load_all_samples()
    if args.limit > 0:
        samples = samples[:args.limit]
    results = []
    for index, sample in enumerate(samples, start=1):
        cache_path = Path(args.cache_dir) / f"{sample['sample_name']}.npz"
        if not cache_path.exists():
            continue
        result_path = per_sample_dir / f"{sample['sample_name']}.json"
        if result_path.exists() and not args.overwrite:
            result = load_json(result_path)
        else:
            result = analyze_sample(sample, cache_path, args.association_radius, args.min_tangent_projection)
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
        print(f"[{index}/{len(samples)}] {sample['sample_name']}: branch direction={result['branch_direction_accuracy']:.3f}", flush=True)
    fold_assignment = grouped_folds([result["sample_name"] for result in results])
    total_individual = sum(result["eligible_direction_votes"] for result in results)
    total_branch = sum(result["branch_direction_trials"] for result in results)
    summary = {
        "num_samples": len(results),
        "total_buds": sum(result["total_buds"] for result in results),
        "reliable_buds": sum(result["reliable_buds"] for result in results),
        "associated_buds": sum(result["associated_buds"] for result in results),
        "eligible_direction_votes": total_individual,
        "individual_direction_accuracy": sum(result["individual_direction_accuracy"] * result["eligible_direction_votes"] for result in results) / max(total_individual, 1),
        "latent_spur_count": sum(result["latent_spur_count"] for result in results),
        "branch_direction_trials": total_branch,
        "branch_direction_accuracy": sum(result["branch_direction_accuracy"] * result["branch_direction_trials"] for result in results) / max(total_branch, 1),
        "failure_reasons": dict(Counter(
            record["failure_reason"] or "reliable"
            for result in results
            for record in result.get("records", [])
        )),
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "fold_manifest.json").write_text(json.dumps(fold_assignment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
