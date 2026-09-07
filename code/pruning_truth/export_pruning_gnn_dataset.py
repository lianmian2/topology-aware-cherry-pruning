from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import torch

from build_atomic_segment_graphs import PROJECT_ROOT, ensure_dir, process_sample, save_json


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_features(graph: dict, image_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2] if image is not None else (1, 1)
    diagonal = max((height * height + width * width) ** 0.5, 1.0)
    rows = []
    features = []
    labels = []
    masks = []
    incidence: dict[str, list[int]] = defaultdict(list)
    for segment in graph["segment_nodes"]:
        start_types = set(segment["start_types"])
        end_types = set(segment["end_types"])
        feature = [
            float(segment["length_px"]) / diagonal,
            float(segment["is_candidate"]),
            float("bud" in start_types),
            float("bud" in end_types),
            float("junction" in start_types),
            float("junction" in end_types),
            float("endpoint" in start_types),
            float("endpoint" in end_types),
            float("root" in start_types),
            float("root" in end_types),
        ]
        features.append(feature)
        labels.append(int(segment["is_cut_segment"]))
        # Context segments participate in message passing but are never direct
        # positive/negative supervision targets.
        masks.append(bool(segment["label_mask"]) and bool(segment["is_candidate"]))
        incidence[segment["start_landmark_id"]].append(int(segment["id"]))
        incidence[segment["end_landmark_id"]].append(int(segment["id"]))
        rows.append(
            {
                "sample_id": graph["sample_id"],
                "segment_id": int(segment["id"]),
                "candidate_role": "primary_semantic" if segment["is_candidate"] else "context_only",
                "candidate_type": segment["candidate_type"] or "",
                "length_px": float(segment["length_px"]),
                "is_cut_segment": int(segment["is_cut_segment"]),
                "label_mask": int(bool(segment["label_mask"])),
                "cut_ids": ";".join(segment["cut_ids"]),
            }
        )
    edges = set()
    for ids in incidence.values():
        for source in ids:
            for target in ids:
                if source != target:
                    edges.add((source, target))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(features, dtype=torch.float32), edge_index, torch.tensor(labels, dtype=torch.long), torch.tensor(masks, dtype=torch.bool), rows


def write_rows(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "segment_id", "candidate_role", "candidate_type", "length_px", "is_cut_segment", "label_mask", "cut_ids"])
        writer.writeheader()
        writer.writerows(rows)


def tree_split(tree_ids: list[str], seed: int = 20260727) -> dict[str, list[str]]:
    shuffled = sorted(tree_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total >= 3:
        train_end = min(total - 2, max(1, round(total * 0.70)))
        val_end = min(total - 1, max(train_end + 1, train_end + round(total * 0.15)))
        return {"train": sorted(shuffled[:train_end]), "val": sorted(shuffled[train_end:val_end]), "test": sorted(shuffled[val_end:])}
    train_end = round(total * 0.70)
    val_end = train_end + round(total * 0.15)
    return {"train": sorted(shuffled[:train_end]), "val": sorted(shuffled[train_end:val_end]), "test": sorted(shuffled[val_end:])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export current automatic pruning graphs as train-ready torch tensors")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    cases_root = args.cases_root if args.cases_root.is_absolute() else PROJECT_ROOT / args.cases_root
    output_root = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    manifest = load_manifest(manifest_path)
    rows = []
    missing = []
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        case = cases_root / sample_id
        if not (case / "summary.json").exists():
            missing.append(sample_id)
            continue
        graph_dir = output_root / "auto_graphs" / sample_id
        if not args.resume or not (graph_dir / "decision_graph.json").exists():
            process_sample(cases_root, output_root / "auto_graphs", sample_id, None)
        graph = load_manifest(graph_dir / "decision_graph.json")
        image_path = Path(graph["source"]["image"])
        x, edge_index, y, mask, feature_rows = segment_features(graph, image_path)
        torch.save({"sample_id": sample_id, "tree_id": sample_id.split("_before_")[0], "x": x, "edge_index": edge_index, "y": y, "label_mask": mask}, graph_dir / "graph.pt")
        write_rows(graph_dir / "segment_features.csv", feature_rows)
        quality = load_manifest(graph_dir / "graph_quality.json")
        quality["nodes"] = int(x.shape[0])
        quality["edges"] = int(edge_index.shape[1])
        rows.append(quality)
    ensure_dir(output_root / "qc")
    fields = sorted({key for row in rows for key in row})
    with (output_root / "qc" / "auto_graph_quality_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"expected_samples": len(manifest["samples"]), "exported_samples": len(rows), "missing_auto_perception": missing, "status": "complete" if not missing else "partial"}
    if args.finalize and missing:
        raise RuntimeError(f"Cannot finalize with {len(missing)} missing automatic-perception samples")
    if args.finalize:
        splits = tree_split(sorted({sample["sample_id"].split("_before_")[0] for sample in manifest["samples"]}))
        ensure_dir(output_root / "splits")
        for name, tree_ids in splits.items():
            save_json(output_root / "splits" / f"{name}_trees.json", {"tree_ids": tree_ids})
        summary["tree_splits"] = {name: len(values) for name, values in splits.items()}
    save_json(output_root / "qc" / "export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
