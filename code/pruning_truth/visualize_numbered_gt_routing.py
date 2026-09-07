from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "02_code" / "02_models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from mask_topology_routing.evaluate_visualize import compute_line_metrics, render_annotation_groups


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def group_color(index: int, total: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_text(image: np.ndarray, text: str, xy: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def junction_node_indices(annotation: dict[str, Any]) -> set[tuple[int, int]]:
    junctions: set[tuple[int, int]] = set()
    for group_index, group in enumerate(annotation.get("groups", [])):
        points = group.get("points", [])
        if points and group.get("group_type") != "trunk":
            junctions.add((group_index, 0))
        degrees = [0] * len(points)
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(points) and 0 <= dst < len(points):
                degrees[src] += 1
                degrees[dst] += 1
        junctions.update((group_index, node_index) for node_index, degree in enumerate(degrees) if degree >= 3)
    return junctions


def numbered_overlay(image: np.ndarray, annotation: dict[str, Any], prefix: str) -> tuple[np.ndarray, dict[str, Any]]:
    canvas = image.copy()
    groups = annotation.get("groups", [])
    junctions = junction_node_indices(annotation)
    mapping: dict[str, Any] = {"prefix": prefix, "groups": []}
    for group_index, group in enumerate(groups):
        group_code = f"{prefix}G{group_index:02d}"
        color = group_color(group_index, max(len(groups), 1))
        points = [tuple(map(int, point)) for point in group.get("points", [])]
        edges = [edge for edge in group.get("edges", []) if len(edge) == 2]
        edge_rows = []
        for edge_index, edge in enumerate(edges):
            src, dst = int(edge[0]), int(edge[1])
            if not (0 <= src < len(points) and 0 <= dst < len(points)):
                continue
            point_a, point_b = points[src], points[dst]
            cv2.line(canvas, point_a, point_b, color, 5, cv2.LINE_AA)
            midpoint = ((point_a[0] + point_b[0]) // 2, (point_a[1] + point_b[1]) // 2)
            edge_code = f"{group_code}-E{edge_index:03d}"
            draw_text(canvas, edge_code, midpoint, 0.36, color, 1)
            edge_rows.append({"edge_code": edge_code, "local_edge": [src, dst]})
        node_rows = []
        for node_index, point in enumerate(points):
            node_code = f"{group_code}-N{node_index:03d}"
            is_junction = (group_index, node_index) in junctions
            cv2.circle(canvas, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 5, color, -1, cv2.LINE_AA)
            if is_junction:
                cv2.circle(canvas, point, 11, color, 2, cv2.LINE_AA)
                draw_text(canvas, node_code, (point[0] + 9, point[1] - 9), 0.45, color, 2)
            node_rows.append({"node_code": node_code, "xy": [point[0], point[1]], "is_junction": is_junction})
        if points:
            centroid = tuple(np.asarray(points, dtype=np.int32).mean(axis=0).astype(int))
            label = f"{group_code} {group.get('group_id', '')} [{group.get('group_type', '')}]"
            draw_text(canvas, label, centroid, 0.72, color, 2)
        mapping["groups"].append({
            "group_code": group_code,
            "source_group_id": group.get("group_id"),
            "group_type": group.get("group_type"),
            "color_bgr": list(color),
            "nodes": node_rows,
            "edges": edge_rows,
        })
    return canvas, mapping


def combined_branch_mask(annotation: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    nontrunk = {"groups": [group for group in annotation.get("groups", []) if group.get("group_type") != "trunk"]}
    return render_annotation_groups(nontrunk, shape, thickness=2)


def metrics(prediction: dict[str, Any], truth: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    prediction_trunk = render_annotation_groups(prediction, shape, group_type="trunk", thickness=2)
    truth_trunk = render_annotation_groups(truth, shape, group_type="trunk", thickness=2)
    prediction_branch = combined_branch_mask(prediction, shape)
    truth_branch = combined_branch_mask(truth, shape)
    return {
        f"{kind}_{tolerance}px": compute_line_metrics(predicted, target, float(tolerance))
        for kind, predicted, target in (
            ("trunk", prediction_trunk, truth_trunk),
            ("branch", prediction_branch, truth_branch),
        )
        for tolerance in (5, 24)
    }


def comparison_canvas(truth_image: np.ndarray, prediction_image: np.ndarray, sample_id: str, metric_rows: dict[str, Any], excluded: bool, reason: str) -> np.ndarray:
    height, width = truth_image.shape[:2]
    header_height = 180
    canvas = np.full((height + header_height, width * 2, 3), 245, dtype=np.uint8)
    canvas[header_height:, :width] = truth_image
    canvas[header_height:, width:] = prediction_image
    draw_text(canvas, f"{sample_id} | LEFT: GROUND TRUTH | RIGHT: PREDICTION", (24, 42), 1.0, (20, 20, 20), 2)
    metric_text = " | ".join(
        f"{key} F1={value['f1']:.3f}" for key, value in metric_rows.items()
    )
    draw_text(canvas, metric_text, (24, 90), 0.7, (20, 20, 20), 2)
    status = f"AUTO EXCLUDED: {reason}" if excluded else "QUALITY GATE: PASS"
    draw_text(canvas, status, (24, 140), 0.9, (0, 0, 255) if excluded else (0, 120, 0), 2)
    cv2.line(canvas, (width, header_height), (width, height + header_height), (255, 255, 255), 5)
    return canvas


def process_sample(dataset: Path, output: Path, sample_id: str) -> dict[str, Any]:
    auto_root = dataset / "auto_perception" / sample_id
    summary = read_json(auto_root / "summary.json")
    image_path = Path(summary["image_path"])
    image = load_bgr(image_path)
    truth_path = PROJECT_ROOT / "01_data" / "02_annotated" / "skeleton_annotation" / f"{sample_id}_skeleton.json"
    truth = read_json(truth_path)
    prediction = read_json(auto_root / "annotation_groups.json")
    marker_path = dataset / "atomic_graphs" / sample_id / ".complete.json"
    marker = read_json(marker_path)
    excluded = marker.get("status") == "auto_excluded"
    reason = marker.get("exclusion", {}).get("reason", "")
    truth_overlay, truth_mapping = numbered_overlay(image, truth, "T")
    prediction_overlay, prediction_mapping = numbered_overlay(image, prediction, "P")
    metric_rows = metrics(prediction, truth, image.shape[:2])
    sample_output = ensure_dir(output / sample_id)
    save_image(sample_output / "ground_truth_numbered.png", truth_overlay)
    save_image(sample_output / "prediction_numbered.png", prediction_overlay)
    save_image(
        sample_output / "comparison_numbered.jpg",
        comparison_canvas(truth_overlay, prediction_overlay, sample_id, metric_rows, excluded, reason),
    )
    save_json(sample_output / "numbering.json", {
        "sample_id": sample_id,
        "image": str(image_path),
        "ground_truth": str(truth_path),
        "prediction": str(auto_root / "annotation_groups.json"),
        "status": "auto_excluded" if excluded else "pass",
        "exclusion_reason": reason or None,
        "metrics": metric_rows,
        "ground_truth_numbering": truth_mapping,
        "prediction_numbering": prediction_mapping,
    })
    return {
        "sample_id": sample_id,
        "status": "auto_excluded" if excluded else "pass",
        "exclusion_reason": reason or None,
        "truth_groups": len(truth.get("groups", [])),
        "prediction_groups": len(prediction.get("groups", [])),
        "metrics": metric_rows,
    }


def process_prediction_only(dataset: Path, output: Path, sample_id: str) -> dict[str, Any]:
    auto_root = dataset / "auto_perception" / sample_id
    summary = read_json(auto_root / "summary.json")
    image_path = Path(summary["image_path"])
    image = load_bgr(image_path)
    prediction_path = auto_root / "annotation_groups.json"
    prediction = read_json(prediction_path)
    marker = read_json(dataset / "atomic_graphs" / sample_id / ".complete.json")
    excluded = marker.get("status") == "auto_excluded"
    reason = marker.get("exclusion", {}).get("reason", "")
    prediction_overlay, prediction_mapping = numbered_overlay(image, prediction, "P")
    sample_output = ensure_dir(output / sample_id)
    save_image(sample_output / "prediction_numbered.png", prediction_overlay)
    save_json(sample_output / "numbering.json", {
        "sample_id": sample_id,
        "image": str(image_path),
        "prediction": str(prediction_path),
        "status": "auto_excluded" if excluded else "pass",
        "exclusion_reason": reason or None,
        "prediction_numbering": prediction_mapping,
    })
    return {
        "sample_id": sample_id,
        "status": "auto_excluded" if excluded else "pass",
        "exclusion_reason": reason or None,
        "prediction_groups": len(prediction.get("groups", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Numbered GT/prediction routing visualization")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--prediction-only-sample-id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    output = ensure_dir(args.output if args.output.is_absolute() else PROJECT_ROOT / args.output)
    if not args.sample_id and not args.prediction_only_sample_id:
        parser.error("at least one --sample-id or --prediction-only-sample-id is required")
    rows = [process_sample(dataset, output, sample_id) for sample_id in args.sample_id]
    rows.extend(process_prediction_only(dataset, output, sample_id) for sample_id in args.prediction_only_sample_id)
    summary = {
        "result_class": "data_preparation_visual_audit",
        "random_seed": args.seed,
        "sample_ids": args.sample_id + args.prediction_only_sample_id,
        "numbering": "TGxx/PGxx=truth/prediction group; Exxx=edge within group; Nxxx labels are displayed only for topological junction nodes",
        "samples": rows,
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
