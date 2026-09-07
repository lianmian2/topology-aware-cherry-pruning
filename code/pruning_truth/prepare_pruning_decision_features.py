from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def split_map(dataset: Path) -> dict[str, str]:
    result = {}
    for name in ("train", "val", "test"):
        for tree_id in read_json(dataset / "splits" / f"{name}_trees.json")["tree_ids"]:
            if tree_id in result:
                raise ValueError(f"Tree appears in multiple splits: {tree_id}")
            result[tree_id] = name
    return result


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def point_polyline_distance(point: tuple[float, float], polyline: np.ndarray) -> float:
    if len(polyline) == 0:
        return float("inf")
    if len(polyline) == 1:
        return float(np.linalg.norm(polyline[0] - np.asarray(point, dtype=float)))
    p = np.asarray(point, dtype=float)
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    lengths = np.einsum("ij,ij->i", vectors, vectors)
    ratios = np.divide(
        np.einsum("ij,ij->i", p - starts, vectors),
        lengths,
        out=np.zeros_like(lengths),
        where=lengths > 0,
    )
    projections = starts + np.clip(ratios, 0.0, 1.0)[:, None] * vectors
    return float(np.linalg.norm(projections - p, axis=1).min())


def polyline_curvature(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 3:
        return 0.0, 0.0
    vectors = np.diff(points, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    valid = (norms[:-1] > 0) & (norms[1:] > 0)
    if not bool(valid.any()):
        return 0.0, 0.0
    left = vectors[:-1][valid] / norms[:-1][valid, None]
    right = vectors[1:][valid] / norms[1:][valid, None]
    angles = np.arccos(np.clip(np.einsum("ij,ij->i", left, right), -1.0, 1.0))
    return float(angles.mean()), float(angles.max())


def tangent(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(2, dtype=float)
    vector = points[-1] - points[0]
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros(2, dtype=float)


def segment_adjacency(segments: list[dict[str, Any]]) -> tuple[dict[int, set[int]], dict[str, list[int]]]:
    incidence: dict[str, list[int]] = defaultdict(list)
    for segment in segments:
        segment_id = int(segment["id"])
        incidence[str(segment["start_landmark_id"])].append(segment_id)
        incidence[str(segment["end_landmark_id"])].append(segment_id)
    adjacency: dict[int, set[int]] = {int(segment["id"]): set() for segment in segments}
    for ids in incidence.values():
        for source in ids:
            adjacency[source].update(target for target in ids if target != source)
    return adjacency, incidence


def root_depths(segments: list[dict[str, Any]], adjacency: dict[int, set[int]]) -> dict[int, int]:
    roots = [
        int(segment["id"])
        for segment in segments
        if "root" in set(segment.get("start_types", [])) | set(segment.get("end_types", []))
    ]
    depths = {segment_id: 0 for segment_id in roots}
    queue = deque(roots)
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            if target not in depths:
                depths[target] = depths[source] + 1
                queue.append(target)
    fallback = max(depths.values(), default=0) + 1
    return {int(segment["id"]): depths.get(int(segment["id"]), fallback) for segment in segments}


def load_branch_features(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["branch_id"]): {
                key: float(value)
                for key, value in row.items()
                if key not in {"sample_id", "branch_id"} and value not in {None, ""}
            }
            for row in csv.DictReader(handle)
        }


def assign_buds(
    segments: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
    directions: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        for group_id in segment.get("group_ids", []):
            by_group[str(group_id)].append(segment)
    direction_map = {int(item["bud_index"]): item for item in directions}
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for attachment in attachments:
        point = attachment.get("skeleton_point")
        group_id = attachment.get("group_id")
        if point is None or group_id is None or str(group_id) not in by_group:
            continue
        candidates = by_group[str(group_id)]
        selected = min(
            candidates,
            key=lambda item: point_polyline_distance(
                (float(point[0]), float(point[1])), np.asarray(item.get("polyline", []), dtype=float)
            ),
        )
        item = dict(attachment)
        item["direction"] = direction_map.get(int(attachment["bud_index"]))
        result[int(selected["id"])].append(item)
    return result


def roi_distance_map(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return roi_distance_map_from_mask(mask)


def roi_distance_map_from_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    return cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)


FEATURE_COLUMNS = [
    "length_norm",
    "chord_norm",
    "tortuosity",
    "curvature_mean",
    "curvature_max",
    "mid_x_norm",
    "relative_height",
    "orientation_x",
    "orientation_y",
    "image_edge_distance_norm",
    "roi_edge_distance_norm",
    "root_depth_norm",
    "neighbor_count",
    "candidate_neighbor_count",
    "neighbor_length_mean_norm",
    "same_group_segment_count_norm",
    "start_degree",
    "end_degree",
    "start_bud",
    "end_bud",
    "start_junction",
    "end_junction",
    "start_endpoint",
    "end_endpoint",
    "candidate_bud_bud",
    "candidate_junction_bud",
    "candidate_bud_endpoint",
    "candidate_junction_endpoint",
    "candidate_junction_junction",
    "bud_count_norm",
    "bud_density_norm",
    "flower_bud_ratio",
    "bud_score_mean",
    "bud_attachment_distance_norm",
    "reliable_direction_ratio",
    "bud_direction_alignment",
    "local_crowding",
    "group_relative_height",
    "group_bud_density",
]


def process_graph(graph_path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]]:
    graph = read_json(graph_path)
    image = cv2.imread(str(graph["source"]["image"]))
    if image is None:
        raise FileNotFoundError(graph["source"]["image"])
    source_dir = Path(graph["source"]["attachments"]).parent
    attachments = read_json(Path(graph["source"]["attachments"]))
    directions_path = source_dir / "bud_directions.json"
    directions = read_json(directions_path) if directions_path.exists() else []
    branch_features = load_branch_features(source_dir / "branch_features.csv")
    roi_mask = cv2.imread(str(source_dir / "roi_mask.png"), cv2.IMREAD_GRAYSCALE)
    return process_graph_payload(graph, image.shape[:2], attachments, directions,
                                 branch_features, roi_mask, split, sha256_file(graph_path))


def process_graph_payload(
    graph: dict[str, Any], image_shape: tuple[int, int],
    attachments: list[dict[str, Any]], directions: list[dict[str, Any]],
    branch_features: dict[str, dict[str, float]], roi_mask: np.ndarray | None,
    split: str, source_graph_sha256: str = "in_memory",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]]:
    segments = graph["segment_nodes"]
    sample_id = str(graph["sample_id"])
    tree_id = sample_id.split("_before_")[0]
    view = sample_id.rsplit("_", 2)[-2] + "_" + sample_id.rsplit("_", 1)[-1]
    height, width = image_shape
    diagonal = max(math.hypot(width, height), 1.0)
    bud_map = assign_buds(segments, attachments, directions)
    roi_dist = roi_distance_map_from_mask(roi_mask)
    adjacency, incidence = segment_adjacency(segments)
    depths = root_depths(segments, adjacency)
    max_depth = max(depths.values(), default=1) or 1
    segment_map = {int(segment["id"]): segment for segment in segments}
    group_counts = Counter(str(group_id) for segment in segments for group_id in segment.get("group_ids", []))
    rows = []
    features = []
    labels = []
    masks = []
    for segment in segments:
        segment_id = int(segment["id"])
        points = np.asarray(segment.get("polyline", []), dtype=float)
        if len(points) == 0:
            points = np.zeros((1, 2), dtype=float)
        arc_length = float(segment.get("length_px", polyline_length(points)))
        chord_length = float(np.linalg.norm(points[-1] - points[0]))
        curvature_mean, curvature_max = polyline_curvature(points)
        direction = tangent(points)
        midpoint = points.mean(axis=0)
        image_edge_distance = min(
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(width - 1 - points[:, 0].max()),
            float(height - 1 - points[:, 1].max()),
        )
        roi_edge_distance = 0.0
        if roi_dist is not None:
            xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
            roi_edge_distance = float(roi_dist[ys, xs].min())
        neighbors = adjacency[segment_id]
        neighbor_lengths = [float(segment_map[item].get("length_px", 0.0)) for item in neighbors]
        start_types = set(map(str, segment.get("start_types", [])))
        end_types = set(map(str, segment.get("end_types", [])))
        candidate_type = str(segment.get("candidate_type") or "")
        group_ids = [str(item) for item in segment.get("group_ids", [])]
        group_values = [branch_features[item] for item in group_ids if item in branch_features]
        buds = bud_map.get(segment_id, [])
        bud_scores = [float(item.get("bud_score", 0.0)) for item in buds]
        bud_distances = [float(item.get("distance", 0.0) or 0.0) for item in buds]
        reliable = [item["direction"] for item in buds if item.get("direction") and item["direction"].get("is_reliable")]
        alignments = []
        for item in reliable:
            vector = np.asarray(item.get("vector_xy", [0.0, 0.0]), dtype=float)
            alignments.append(abs(float(np.dot(direction, vector))))
        flower_count = sum(int(item.get("bud_label", -1)) == 0 for item in buds)
        feature = {
            "length_norm": arc_length / diagonal,
            "chord_norm": chord_length / diagonal,
            "tortuosity": arc_length / max(chord_length, 1.0),
            "curvature_mean": curvature_mean,
            "curvature_max": curvature_max,
            "mid_x_norm": float(midpoint[0]) / max(width, 1),
            "relative_height": 1.0 - float(midpoint[1]) / max(height, 1),
            "orientation_x": float(direction[0]),
            "orientation_y": float(direction[1]),
            "image_edge_distance_norm": max(image_edge_distance, 0.0) / diagonal,
            "roi_edge_distance_norm": max(roi_edge_distance, 0.0) / diagonal,
            "root_depth_norm": float(depths[segment_id]) / max_depth,
            "neighbor_count": float(len(neighbors)),
            "candidate_neighbor_count": float(sum(bool(segment_map[item].get("label_mask")) for item in neighbors)),
            "neighbor_length_mean_norm": (float(np.mean(neighbor_lengths)) if neighbor_lengths else 0.0) / diagonal,
            "same_group_segment_count_norm": float(max((group_counts[item] for item in group_ids), default=0)) / max(len(segments), 1),
            "start_degree": float(len(incidence[str(segment["start_landmark_id"])])),
            "end_degree": float(len(incidence[str(segment["end_landmark_id"])])),
            "start_bud": float("bud" in start_types),
            "end_bud": float("bud" in end_types),
            "start_junction": float("junction" in start_types or "root" in start_types),
            "end_junction": float("junction" in end_types or "root" in end_types),
            "start_endpoint": float("endpoint" in start_types),
            "end_endpoint": float("endpoint" in end_types),
            "candidate_bud_bud": float(candidate_type == "bud--bud"),
            "candidate_junction_bud": float(candidate_type == "junction--bud"),
            "candidate_bud_endpoint": float(candidate_type == "bud--endpoint"),
            "candidate_junction_endpoint": float(candidate_type == "junction--endpoint"),
            "candidate_junction_junction": float(candidate_type == "junction--junction"),
            "bud_count_norm": float(len(buds)) / 10.0,
            "bud_density_norm": float(len(buds)) / max(arc_length, 1.0) * 100.0,
            "flower_bud_ratio": float(flower_count) / max(len(buds), 1),
            "bud_score_mean": float(np.mean(bud_scores)) if bud_scores else 0.0,
            "bud_attachment_distance_norm": (float(np.mean(bud_distances)) if bud_distances else 0.0) / diagonal,
            "reliable_direction_ratio": float(len(reliable)) / max(len(buds), 1),
            "bud_direction_alignment": float(np.mean(alignments)) if alignments else 0.0,
            "local_crowding": float(np.mean([item.get("local_crowding", 0.0) for item in group_values])) if group_values else 0.0,
            "group_relative_height": float(np.mean([item.get("relative_height", 0.0) for item in group_values])) if group_values else 0.0,
            "group_bud_density": float(np.mean([item.get("bud_density", 0.0) for item in group_values])) if group_values else 0.0,
        }
        label = int(segment.get("is_cut_segment", 0))
        label_mask = int(bool(segment.get("label_mask")) and bool(segment.get("is_candidate")))
        row = {
            "sample_id": sample_id,
            "tree_id": tree_id,
            "view": view,
            "split": split,
            "segment_id": segment_id,
            "candidate_type": candidate_type,
            "group_ids": ";".join(group_ids),
            "is_cut_segment": label,
            "label_mask": label_mask,
            **feature,
        }
        rows.append(row)
        features.append([feature[column] for column in FEATURE_COLUMNS])
        labels.append(label)
        masks.append(bool(label_mask))
    edges = sorted((source, target) for source, targets in adjacency.items() for target in targets)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    tensor = {
        "sample_id": sample_id,
        "tree_id": tree_id,
        "x": torch.tensor(features, dtype=torch.float32),
        "edge_index": edge_index,
        "y": torch.tensor(labels, dtype=torch.long),
        "label_mask": torch.tensor(masks, dtype=torch.bool),
    }
    sample_audit = {
        "sample_id": sample_id,
        "tree_id": tree_id,
        "split": split,
        "segments": len(segments),
        "candidates": int(sum(masks)),
        "positive_segments": int(sum(labels[index] for index, value in enumerate(masks) if value)),
        "source_graph_sha256": source_graph_sha256,
        "feature_tensor_shape": list(tensor["x"].shape),
    }
    return rows, sample_audit, tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe pruning decision features")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    splits = split_map(dataset)
    rows = []
    audits = []
    for graph_path in sorted((dataset / "graphs").glob("*/decision_graph.json")):
        sample_id = graph_path.parent.name
        tree_id = sample_id.split("_before_")[0]
        if tree_id not in splits:
            raise ValueError(f"Accepted sample tree missing from splits: {sample_id}")
        sample_rows, audit, tensor = process_graph(graph_path, splits[tree_id])
        rows.extend(sample_rows)
        audits.append(audit)
        graph_output = ensure_dir(output / "graphs" / sample_id)
        torch.save(tensor, graph_output / "graph.pt")
    if not rows:
        raise RuntimeError("No accepted decision graphs found")
    ensure_dir(output / "audit")
    with (output / "features.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "audit" / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)
    eligible = [row for row in rows if row["label_mask"]]
    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "data_preparation",
        "source_dataset": str(dataset),
        "source_audit_sha256": sha256_file(dataset / "audit" / "summary.json"),
        "accepted_views": len(audits),
        "trees": len({row["tree_id"] for row in rows}),
        "all_segments": len(rows),
        "candidate_segments": len(eligible),
        "positive_segments": sum(int(row["is_cut_segment"]) for row in eligible),
        "feature_columns": FEATURE_COLUMNS,
        "split_counts": {
            name: {
                "trees": len({row["tree_id"] for row in eligible if row["split"] == name}),
                "views": len({row["sample_id"] for row in eligible if row["split"] == name}),
                "candidates": sum(row["split"] == name for row in eligible),
                "positives": sum(int(row["is_cut_segment"]) for row in eligible if row["split"] == name),
            }
            for name in ("train", "val", "test")
        },
        "label_leakage_fields_excluded": [
            "normal_pruned_side",
            "auto_suggested_cut",
            "manual_reconstruction_remap",
            "deleted_cut_ids",
            "added_cut_ids",
            "after_image",
        ],
    }
    atomic_json(output / "audit" / "summary.json", summary)
    atomic_json(output / "feature_schema.json", {"schema_version": "1.0", "feature_columns": FEATURE_COLUMNS})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
