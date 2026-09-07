from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "02_code" / "02_models"
GUI_DIR = PROJECT_ROOT / "07_graphical_interface" / "unified_system"
for directory in (MODEL_DIR, GUI_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mask_topology_routing.data import load_manifest
from mask_topology_routing.utils import DEFAULT_PROCESSED_ROOT, ensure_dir
from bud_skeleton_fusion.bud_detection_cache import save_bud_detection_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="缓存100样本全图芽点检测实例")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "04_results" / "bud_skeleton_fusion" / "full_image_bud_cache_20260714"))
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_all_samples() -> list[dict]:
    annotation_root = DEFAULT_PROCESSED_ROOT / "annotations" / "skeleton_prediction"
    samples = []
    for manifest_name in ("manifest_train.json", "manifest_test.json"):
        manifest_path = annotation_root / manifest_name
        if manifest_path.exists():
            samples.extend(load_manifest(manifest_path))
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
                tree_id, timing, view_id = sample_name.rsplit("_", 3)[0], sample_name.rsplit("_", 3)[1], "_".join(sample_name.rsplit("_", 2)[-2:])
                source_path = PROJECT_ROOT / "01_data" / "01_raw" / "final_data" / tree_id / timing / f"{view_id}.jpg"
                unique[sample_name] = {
                    "sample_name": sample_name,
                    "image_path": str(image_path),
                    "source_image_path": str(source_path if source_path.exists() else image_path),
                    "mask_path": str(mask_path),
                    "annotation_path": str(annotation_path),
                }
                break
    return [unique[name] for name in sorted(unique)]


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(output_dir / "per_sample")
    samples = load_all_samples()
    if args.limit > 0:
        samples = samples[:args.limit]

    from logic_models import ModelManager
    from logic_bud import run_bud_detection_pipeline

    manager = ModelManager()
    bud_model = manager.load_bud_global_model()
    roi_model = manager.load_roi_model()
    summaries = []
    for index, sample in enumerate(samples, start=1):
        sample_name = sample["sample_name"]
        cache_path = cache_dir / f"{sample_name}.npz"
        if cache_path.exists() and not args.overwrite:
            print(f"[{index}/{len(samples)}] {sample_name}: cached", flush=True)
            continue
        image_path = Path(sample.get("source_image_path", sample["image_path"]))
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        result = run_bud_detection_pipeline(
            image,
            bud_model=bud_model,
            roi_model=roi_model,
            use_roi_filter=True,
            score_thr=args.score_thr,
            device=args.device,
        )
        save_bud_detection_cache(cache_path, result, image.shape[:2])
        summary = {
            "sample_name": sample_name,
            "total": int(result["total_count"]),
            "flower": int(result["flower_count"]),
            "leaf": int(result["leaf_count"]),
        }
        summaries.append(summary)
        print(f"[{index}/{len(samples)}] {sample_name}: {summary['total']} buds", flush=True)
        (output_dir / "progress.json").write_text(
            json.dumps({"completed": index, "total": len(samples), "latest": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_dir / "cache_summary.json").write_text(
        json.dumps({"num_samples": len(samples), "score_thr": args.score_thr, "new_samples": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
