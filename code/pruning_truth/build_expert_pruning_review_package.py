from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RANK_COLORS = [(35, 35, 230), (0, 150, 255), (230, 190, 20)]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_panel(image: np.ndarray, max_width: int = 1100, max_height: int = 850) -> np.ndarray:
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def draw_recommendations(
    image: np.ndarray,
    segments: dict[int, dict],
    prediction: pd.DataFrame,
    panel_name: str,
) -> np.ndarray:
    canvas = image.copy()
    for rank, (_, row) in enumerate(prediction.head(3).iterrows(), start=1):
        points = np.rint(np.asarray(segments[int(row.segment_id)]["polyline"], dtype=float)).astype(np.int32)
        color = RANK_COLORS[rank - 1]
        cv2.polylines(canvas, [points], False, (255, 255, 255), thickness=11, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [points], False, color, thickness=7, lineType=cv2.LINE_AA)
        center = tuple(np.rint(points.mean(0)).astype(int))
        cv2.circle(canvas, center, 18, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, 15, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(rank),
            (center[0] - 6, center[1] + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (250, 58), (245, 245, 245), -1)
    cv2.putText(canvas, panel_name, (18, 41), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3, cv2.LINE_AA)
    return resize_panel(canvas)


def adjacency(feature_dir: Path, sample_id: str) -> dict[int, set[int]]:
    tensor = torch.load(feature_dir / "graphs" / sample_id / "graph.pt", map_location="cpu")
    result = {index: set() for index in range(len(tensor["y"]))}
    for source, target in tensor["edge_index"].t().tolist():
        result[int(source)].add(int(target))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build blinded expert pruning review package")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions if args.predictions.is_absolute() else PROJECT_ROOT / args.predictions
    feature_dir = args.features if args.features.is_absolute() else PROJECT_ROOT / args.features
    source_dataset = args.source_dataset if args.source_dataset.is_absolute() else PROJECT_ROOT / args.source_dataset
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    panels_dir = ensure_dir(output / "blinded_panels")
    predictions = pd.read_csv(predictions_path)
    models = sorted(predictions.model.unique())
    if len(models) != 2:
        raise ValueError(f"Expected exactly two models, found: {models}")
    rng = np.random.default_rng(args.seed)
    mapping_rows = []
    scoring_rows = []
    truth_rows = []
    for sample_id in sorted(predictions.sample_id.unique()):
        graph_path = source_dataset / "graphs" / sample_id / "decision_graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        image = cv2.imread(graph["source"]["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(graph["source"]["image"])
        segments = {int(item["id"]): item for item in graph["segment_nodes"]}
        sample_predictions = predictions.loc[predictions.sample_id == sample_id]
        assignment = models if rng.random() < 0.5 else list(reversed(models))
        panel_images = []
        local_adjacency = adjacency(feature_dir, sample_id)
        truths = set(sample_predictions.loc[sample_predictions.is_cut_segment == 1, "segment_id"].astype(int))
        for panel_name, model in zip(("Panel A", "Panel B"), assignment):
            ranked = sample_predictions.loc[sample_predictions.model == model].sort_values("score", ascending=False).head(3)
            panel_images.append(draw_recommendations(image, segments, ranked, panel_name))
            mapping_rows.append({"sample_id": sample_id, "panel": panel_name, "model": model})
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                scoring_rows.append(
                    {
                        "sample_id": sample_id,
                        "panel": panel_name,
                        "rank": rank,
                        "acceptable_recommendation_0_1": "",
                        "unnecessary_but_harmless_0_1": "",
                        "harmful_or_unacceptable_0_1": "",
                        "confidence_1_to_5": "",
                        "reason_or_note": "",
                    }
                )
                segment_id = int(row.segment_id)
                truth_rows.append(
                    {
                        "sample_id": sample_id,
                        "panel": panel_name,
                        "model": model,
                        "rank": rank,
                        "segment_id": segment_id,
                        "exact_truth": int(segment_id in truths),
                        "hop1_truth": int(bool(({segment_id} | local_adjacency.get(segment_id, set())) & truths)),
                        "truth_cut_count": len(truths),
                    }
                )
        target_height = min(panel.shape[0] for panel in panel_images)
        resized = [
            cv2.resize(panel, (round(panel.shape[1] * target_height / panel.shape[0]), target_height))
            for panel in panel_images
        ]
        combined = cv2.hconcat(resized)
        cv2.imwrite(str(panels_dir / f"{sample_id}.jpg"), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    pd.DataFrame(scoring_rows).to_csv(output / "expert_scoring_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mapping_rows).to_csv(output / "blind_model_key.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(truth_rows).to_csv(output / "truth_audit_key.csv", index=False, encoding="utf-8-sig")
    instructions = """# Expert pruning recommendation review\n\nEach image contains two blinded model panels. Ranks 1–3 indicate the model's recommended pruning segments. Review each ranked recommendation independently:\n\n1. `acceptable_recommendation_0_1`: this is an agronomically acceptable cut recommendation.\n2. `unnecessary_but_harmless_0_1`: it is not needed, but carrying it out would not materially harm tree structure or production.\n3. `harmful_or_unacceptable_0_1`: it should not be executed.\n4. `confidence_1_to_5`: confidence in the judgement.\n5. Add a short reason only when useful.\n\nThe three category columns should normally contain exactly one `1` per row. Complete `expert_scoring_sheet.csv` without opening the two key files. A single experienced reviewer is sufficient for the present application closeout; a second reviewer can be added later using a copied scoring sheet.\n"""
    (output / "README_expert_review.md").write_text(instructions, encoding="utf-8")
    summary = {
        "samples": int(predictions.sample_id.nunique()),
        "trees": int(predictions.tree_id.nunique()),
        "models": models,
        "panels": int(len(list(panels_dir.glob("*.jpg")))),
        "recommendations_to_score": len(scoring_rows),
        "blinded": True,
        "prediction_source": str(predictions_path),
        "independent_unit": "tree",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
