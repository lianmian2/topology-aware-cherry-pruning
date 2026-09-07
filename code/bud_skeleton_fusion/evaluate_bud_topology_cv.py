from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "02_code" / "02_models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from bud_skeleton_fusion import extract_bud_orientations, extract_directed_bud_orientations
from bud_skeleton_fusion.bud_detection_cache import load_bud_detection_cache
from mask_topology_routing.data import load_manifest
from mask_topology_routing.evaluate_visualize import (
    compute_botanical_topology_validity,
    compute_group_topology_metrics,
    compute_junction_pairing_accuracy,
    compute_line_metrics,
    draw_group_overlay,
    render_annotation_groups,
)
from mask_topology_routing.mask_clip import clip_annotation_groups
from mask_topology_routing.utils import (
    DEFAULT_PROCESSED_ROOT,
    _apply_partition_constraints,
    build_prediction_result,
    decode_prediction_to_annotation,
    ensure_dir,
    load_image_rgb,
    load_json,
    prepare_processed_router_mask,
    reconstruct_branch_groups_with_junction_pairing,
    render_topology_groups_to_mask,
)


VARIANTS = ("geometry", "density", "unsigned_axis", "directed_axis", "full_flow")
KEY_METRICS = (
    "junction_pairing_accuracy",
    "botanical_topology_valid",
    "branch_group_topology_f1",
    "branch_f1_5px",
    "branch_f1_24px",
    "trunk_f1_5px",
    "trunk_f1_24px",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="芽点方向流与骨架拓扑路由五折消融")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "04_results" / "bud_skeleton_fusion" / "full_image_bud_cache_20260714" / "per_sample"))
    parser.add_argument("--direction-analysis-dir", default=str(PROJECT_ROOT / "04_results" / "bud_skeleton_fusion" / "gt_bud_direction_20260714"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "04_results" / "bud_skeleton_fusion" / "bud_topology_cv_20260714"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_samples() -> List[Dict]:
    root = DEFAULT_PROCESSED_ROOT / "annotations" / "skeleton_prediction"
    samples = []
    for filename in ("manifest_train.json", "manifest_test.json"):
        path = root / filename
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


def assign_grouped_folds(samples: Sequence[Dict], num_folds: int = 5) -> Dict[str, int]:
    by_tree = defaultdict(list)
    for sample in samples:
        match = re.match(r"(tree_\d+)_", sample["sample_name"])
        by_tree[match.group(1) if match else sample["sample_name"]].append(sample["sample_name"])
    sizes = [0] * num_folds
    assignment = {}
    for _, names in sorted(by_tree.items(), key=lambda item: (-len(item[1]), item[0])):
        fold = int(np.argmin(sizes))
        sizes[fold] += len(names)
        assignment.update({name: fold for name in names})
    return assignment


def select_fold_thresholds(samples: Sequence[Dict], folds: Dict[str, int], analysis_dir: Path) -> Dict[int, float]:
    records_by_sample = {}
    for sample in samples:
        path = analysis_dir / "per_sample" / f"{sample['sample_name']}.json"
        if path.exists():
            records_by_sample[sample["sample_name"]] = load_json(path).get("records", [])
    thresholds = {}
    candidates = (0.18, 0.25, 0.35, 0.45, 0.55)
    for held_out in range(5):
        train_records = [
            record
            for sample_name, records in records_by_sample.items()
            if folds.get(sample_name) != held_out
            for record in records
            if not record.get("latent_spur", False) and abs(float(record.get("signed_projection", 0.0))) >= 0.2
        ]
        base_count = sum(float(record.get("confidence", 0.0)) >= candidates[0] for record in train_records)
        scored = []
        for threshold in candidates:
            selected = [record for record in train_records if float(record.get("confidence", 0.0)) >= threshold]
            accuracy = float(np.mean([float(record["signed_projection"]) > 0.0 for record in selected])) if selected else 0.0
            coverage = len(selected) / max(base_count, 1)
            score = accuracy if coverage >= 0.5 else accuracy - (0.5 - coverage)
            scored.append((score, accuracy, coverage, -threshold, threshold))
        thresholds[held_out] = float(max(scored)[-1]) if scored else 0.18
    return thresholds


def clip_prediction(sample: Dict, image_shape: Tuple[int, int], prediction) -> Dict:
    annotation = decode_prediction_to_annotation(sample["image_path"], image_shape, prediction)
    groups, _ = clip_annotation_groups(annotation.get("groups", []), prediction.mask)
    return {**annotation, "groups": groups}


def build_gt_metric_cache(gt_annotation: Dict, shape: Tuple[int, int]) -> Dict:
    branch_groups = []
    all_groups = []
    for group in gt_annotation.get("groups", []):
        mask = render_annotation_groups({"groups": [group]}, shape, group.get("group_type"), 1) > 0
        if not np.any(mask):
            continue
        item = (group.get("group_id", group.get("group_type", "group")), mask, distance_transform_edt(~mask))
        all_groups.append((item[0], item[2]))
        if group.get("group_type") == "branch":
            branch_groups.append(item)
    return {
        "trunk_mask": render_annotation_groups(gt_annotation, shape, "trunk", 2),
        "branch_mask": render_annotation_groups(gt_annotation, shape, "branch", 2),
        "branch_groups": branch_groups,
        "all_group_distances": all_groups,
    }


def evaluate_annotation(prediction, pred_annotation: Dict, gt_annotation: Dict, shape: Tuple[int, int], gt_cache: Dict) -> Dict[str, float]:
    trunk_pred = render_annotation_groups(pred_annotation, shape, "trunk", 2)
    branch_pred = render_annotation_groups(pred_annotation, shape, "branch", 2)
    trunk_gt = gt_cache["trunk_mask"]
    branch_gt = gt_cache["branch_mask"]
    metrics = {}
    for tolerance in (5.0, 24.0):
        metrics[f"branch_f1_{int(tolerance)}px"] = compute_line_metrics(branch_pred, branch_gt, tolerance)["f1"]
        metrics[f"trunk_f1_{int(tolerance)}px"] = compute_line_metrics(trunk_pred, trunk_gt, tolerance)["f1"]
    metrics.update(compute_group_topology_metrics(pred_annotation, gt_annotation, shape, 24.0, gt_cache["branch_groups"]))
    metrics.update(compute_botanical_topology_validity(pred_annotation, shape))
    metrics.update(compute_junction_pairing_accuracy(
        prediction.routing_stats.get("junction_pairing_debug", []),
        gt_annotation,
        shape,
        float(prediction.routing_stats.get("scale_x", 1.0)),
        float(prediction.routing_stats.get("scale_y", 1.0)),
        gt_cache=gt_cache["all_group_distances"],
    ))
    metrics["directed_buds_reliable"] = float(prediction.routing_stats.get("directed_buds_reliable", 0.0))
    metrics["bud_flow_evidence_clusters"] = float(prediction.routing_stats.get("bud_flow_evidence_clusters", 0.0))
    return metrics


def rerank_variant(
    variant: str,
    candidate_prediction,
    tape_mask: np.ndarray,
    cache: Dict,
    directions: Sequence,
    unsigned_orientations: Sequence,
    enable_hierarchy: bool = False,
    source_mask: np.ndarray | None = None,
):
    scale_x = float(candidate_prediction.routing_stats.get("scale_x", 1.0))
    scale_y = float(candidate_prediction.routing_stats.get("scale_y", 1.0))
    processing_width, processing_height = candidate_prediction.routing_stats.get(
        "processing_shape", [candidate_prediction.mask.shape[1], candidate_prediction.mask.shape[0]]
    )

    def scale_groups(groups: Sequence[Dict], sx: float, sy: float) -> List[Dict]:
        return [
            {
                **deepcopy(group),
                "points": [[int(round(point[0] * sx)), int(round(point[1] * sy))] for point in group.get("points", [])],
            }
            for group in groups
        ]

    boxes = np.asarray(cache["boxes"], dtype=np.float32)
    centers = [
        (float((box[0] + box[2]) * 0.5 / scale_x), float((box[1] + box[3]) * 0.5 / scale_y))
        for box in boxes
    ]
    scaled_directions = []
    for direction in directions:
        base = np.asarray([direction.base_global[0] / scale_x, direction.base_global[1] / scale_y], dtype=np.float32)
        tip = np.asarray([direction.tip_global[0] / scale_x, direction.tip_global[1] / scale_y], dtype=np.float32)
        vector = tip - base
        vector /= max(float(np.linalg.norm(vector)), 1e-6)
        scaled_directions.append({
            "bud_index": int(direction.bud_index),
            "base_global": tuple(map(float, base)),
            "tip_global": tuple(map(float, tip)),
            "centroid_global": (float(direction.centroid_global[0] / scale_x), float(direction.centroid_global[1] / scale_y)),
            "vector_xy": tuple(map(float, vector)),
            "confidence": float(direction.confidence),
            "is_reliable": bool(direction.is_reliable),
            "is_latent_spur": bool(direction.is_latent_spur),
        })
    scaled_unsigned = []
    for orientation in unsigned_orientations:
        scaled_unsigned.append({
            "axis_angle": float(orientation.axis_angle),
            "is_elongated": bool(orientation.is_elongated),
            "centroid_global": (float(orientation.centroid_global[0] / scale_x), float(orientation.centroid_global[1] / scale_y)),
        })
    density = variant in {"density", "full_flow"}
    flow = variant in {"directed_axis", "full_flow"}
    orientations = scaled_unsigned if variant == "unsigned_axis" else None
    variant_centers = centers if variant != "geometry" else None
    groups_processing, stats = reconstruct_branch_groups_with_junction_pairing(
        scale_groups(candidate_prediction.annotation_groups, 1.0 / scale_x, 1.0 / scale_y),
        dt_map=cv2.resize(candidate_prediction.dt_map, (int(processing_width), int(processing_height)), interpolation=cv2.INTER_LINEAR),
        min_branch_length=12.0,
        cluster_radius=12.0,
        root_point_xy=(int(round(candidate_prediction.root_point[0] / scale_x)), int(round(candidate_prediction.root_point[1] / scale_y))),
        bud_centers=variant_centers,
        bud_orientations=orientations,
        bud_directions=scaled_directions if flow else None,
        protected_tape_mask=cv2.resize((tape_mask > 0).astype(np.uint8), (int(processing_width), int(processing_height)), interpolation=cv2.INTER_NEAREST) if tape_mask is not None else None,
        enable_bud_density_prior=density,
        enable_bud_direction_flow=flow,
    )
    if enable_hierarchy:
        if source_mask is None:
            raise ValueError("source_mask is required when enable_hierarchy=True")
        processing_shape = (int(processing_width), int(processing_height))
        source_processing = cv2.resize(
            (source_mask > 0).astype(np.uint8), processing_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        tape_processing = None if tape_mask is None else cv2.resize(
            (tape_mask > 0).astype(np.uint8), processing_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        trunk_mask = render_topology_groups_to_mask(
            groups_processing[:1], source_processing.shape,
        ).astype(np.uint8)
        groups_processing, hierarchy_stats = _apply_partition_constraints(
            groups_processing,
            trunk_mask=trunk_mask,
            source_mask=source_processing,
            tape_mask=tape_processing,
            tolerance=max(2.0, 24.0 / max(scale_x, scale_y)),
            bud_directions=scaled_directions if flow else None,
        )
        stats = {**stats, **hierarchy_stats, "hierarchy_ablation_enabled": True}
    else:
        stats["hierarchy_ablation_enabled"] = False
    groups = scale_groups(groups_processing, scale_x, scale_y)
    routing_stats = {**candidate_prediction.routing_stats, **stats}
    routing_stats["directed_buds_total"] = float(len(directions) if flow else 0)
    routing_stats["directed_buds_reliable"] = float(sum(direction.is_reliable for direction in directions) if flow else 0)
    return replace(candidate_prediction, annotation_groups=groups, routing_stats=routing_stats)


def bootstrap_delta(values_a: np.ndarray, values_b: np.ndarray, repeats: int, rng: np.random.Generator) -> Dict[str, float]:
    delta = values_b - values_a
    if len(delta) == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    indices = rng.integers(0, len(delta), size=(repeats, len(delta)))
    boot = delta[indices].mean(axis=1)
    try:
        p_value = float(wilcoxon(delta).pvalue) if np.any(np.abs(delta) > 1e-12) else 1.0
    except ValueError:
        p_value = 1.0
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": p_value,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    per_sample_dir = ensure_dir(output_dir / "per_sample")
    visual_dir = ensure_dir(output_dir / "visualizations")
    all_samples = [sample for sample in load_samples() if (Path(args.cache_dir) / f"{sample['sample_name']}.npz").exists()]
    folds = assign_grouped_folds(all_samples)
    thresholds = select_fold_thresholds(all_samples, folds, Path(args.direction_analysis_dir))
    samples = list(all_samples)
    if args.sample_names:
        requested = set(args.sample_names)
        samples = [sample for sample in samples if sample["sample_name"] in requested]
    if args.end_index > 0:
        samples = samples[args.start_index:args.end_index]
    elif args.start_index > 0:
        samples = samples[args.start_index:]
    if args.limit > 0:
        samples = samples[:args.limit]
    rows = []
    focus_samples = {"tree_007_before_view_01", "tree_009_before_view_01", "tree_009_before_view_02"}
    for sample_index, sample in enumerate(samples, start=1):
        sample_name = sample["sample_name"]
        existing_paths = [per_sample_dir / f"{sample_name}_{variant}.json" for variant in args.variants]
        if args.resume and all(path.exists() for path in existing_paths):
            rows.extend([load_json(path)["metrics"] for path in existing_paths])
            print(f"[{sample_index}/{len(samples)}] {sample_name}: resumed", flush=True)
            continue
        image_rgb = load_image_rgb(sample["image_path"])
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(sample["mask_path"])
        gt = load_json(Path(sample["annotation_path"]))
        gt_metric_cache = build_gt_metric_cache(gt, image_rgb.shape[:2])
        cache = load_bud_detection_cache(Path(args.cache_dir) / f"{sample_name}.npz")
        processed_mask, tape_mask = prepare_processed_router_mask(mask, image_rgb)
        candidate_prediction = build_prediction_result(
            combined_mask=processed_mask,
            processed_mask=processed_mask,
            protected_tape_mask=tape_mask,
            enable_junction_pairing=False,
            enable_bud_density_prior=False,
            enable_bud_direction_flow=False,
        )
        threshold = thresholds.get(folds.get(sample_name, 0), 0.18)
        directions = extract_directed_bud_orientations(cache["masks_info"], candidate_prediction.skeleton_map, cache["scores"], min_confidence=threshold)
        unsigned = extract_bud_orientations(cache["masks_info"])
        predictions = {
            variant: rerank_variant(variant, candidate_prediction, tape_mask, cache, directions, unsigned)
            for variant in args.variants
        }
        panels = []
        annotation_metric_cache = {}
        for variant in args.variants:
            prediction = predictions[variant]
            annotation = clip_prediction(sample, image_rgb.shape[:2], prediction)
            annotation_key = json.dumps(annotation.get("groups", []), sort_keys=True, separators=(",", ":"))
            if annotation_key in annotation_metric_cache:
                metrics = dict(annotation_metric_cache[annotation_key])
                metrics.update(compute_junction_pairing_accuracy(
                    prediction.routing_stats.get("junction_pairing_debug", []),
                    gt,
                    image_rgb.shape[:2],
                    float(prediction.routing_stats.get("scale_x", 1.0)),
                    float(prediction.routing_stats.get("scale_y", 1.0)),
                    gt_cache=gt_metric_cache["all_group_distances"],
                ))
                metrics["directed_buds_reliable"] = float(prediction.routing_stats.get("directed_buds_reliable", 0.0))
                metrics["bud_flow_evidence_clusters"] = float(prediction.routing_stats.get("bud_flow_evidence_clusters", 0.0))
            else:
                metrics = evaluate_annotation(prediction, annotation, gt, image_rgb.shape[:2], gt_metric_cache)
                annotation_metric_cache[annotation_key] = dict(metrics)
            row = {"sample_name": sample_name, "tree_id": re.match(r"(tree_\d+)_", sample_name).group(1), "fold": folds[sample_name], "variant": variant, "direction_threshold": threshold, **metrics}
            rows.append(row)
            debug_stats = {
                "junction_pairing_debug": prediction.routing_stats.get("junction_pairing_debug", []),
                "junction_clusters": prediction.routing_stats.get("junction_clusters", 0),
                "crossing_clusters": prediction.routing_stats.get("crossing_clusters", 0),
                "bud_flow_evidence_clusters": prediction.routing_stats.get("bud_flow_evidence_clusters", 0),
            }
            (per_sample_dir / f"{sample_name}_{variant}.json").write_text(
                json.dumps({"metrics": row, "routing_debug": debug_stats, "annotation": annotation}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if sample_name in focus_samples:
                panels.append(draw_group_overlay(image_rgb, annotation, line_thickness=5))
        if panels:
            gt_panel = draw_group_overlay(image_rgb, gt, line_thickness=5)
            all_panels = [gt_panel] + panels
            target_h = 700
            resized = [cv2.resize(panel, (int(panel.shape[1] * target_h / panel.shape[0]), target_h)) for panel in all_panels]
            cv2.imwrite(str(visual_dir / f"{sample_name}_gt_ablation.jpg"), cv2.cvtColor(np.hstack(resized), cv2.COLOR_RGB2BGR))
        print(f"[{sample_index}/{len(samples)}] {sample_name}", flush=True)
        (output_dir / "progress.json").write_text(
            json.dumps({"completed": sample_index, "total": len(samples), "latest": sample_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for variant in args.variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary[variant] = {}
        for key in KEY_METRICS:
            if key == "junction_pairing_accuracy":
                correct = sum(row["junction_pairing_correct"] for row in variant_rows)
                trials = sum(row["junction_pairing_trials"] for row in variant_rows)
                summary[variant][key] = float(correct / trials) if trials else 1.0
            else:
                summary[variant][key] = float(np.mean([row[key] for row in variant_rows]))
    rng = np.random.default_rng(args.seed)
    paired = {}
    if "geometry" in args.variants:
        for variant in args.variants:
            if variant == "geometry":
                continue
            paired[variant] = {}
            for key in KEY_METRICS:
                base_rows = [row for row in rows if row["variant"] == "geometry"]
                changed_rows = [row for row in rows if row["variant"] == variant]
                if key == "junction_pairing_accuracy":
                    keep = [
                        index for index, (base_row, changed_row) in enumerate(zip(base_rows, changed_rows))
                        if base_row["junction_pairing_trials"] > 0 or changed_row["junction_pairing_trials"] > 0
                    ]
                    base = np.asarray([
                        base_rows[index]["junction_pairing_correct"] / max(base_rows[index]["junction_pairing_trials"], 1.0)
                        for index in keep
                    ], dtype=np.float64)
                    changed = np.asarray([
                        changed_rows[index]["junction_pairing_correct"] / max(changed_rows[index]["junction_pairing_trials"], 1.0)
                        for index in keep
                    ], dtype=np.float64)
                else:
                    base = np.asarray([row[key] for row in base_rows], dtype=np.float64)
                    changed = np.asarray([row[key] for row in changed_rows], dtype=np.float64)
                paired[variant][key] = bootstrap_delta(base, changed, args.bootstrap_repeats, rng)
    report = {"num_samples": len(samples), "fold_assignment": folds, "fold_direction_thresholds": thresholds, "summary": summary, "paired_vs_geometry": paired}
    (output_dir / "cv_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
