#!/usr/bin/env python
"""Paired legacy-versus-cleaned CS-Net evaluation on the frozen cleaned test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent
ROI_DIR = ROOT / "02_code" / "02_models" / "roi_locator_V2"
sys.path.insert(0, str(MODEL_DIR))

from model import CSNet


TARGET_LABELS = {"Trunk", "Branch"}
MODEL_NAMES = ("legacy_old_labels", "cleaned_bce_dice", "cleaned_bce_dice_cldice")
METRICS = ("iou", "dice", "precision", "recall", "cldice", "pred_components", "floating_components", "floating_component_rate")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labelme_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    data = json.loads(path.read_text(encoding="utf-8"))
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in data.get("shapes", []):
        if item.get("label") not in TARGET_LABELS:
            continue
        points = np.asarray(item.get("points", []), dtype=np.float32)
        if len(points) < 3:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


def load_split(split_json: Path, image_dir: Path, cleaned_dir: Path, legacy_dir: Path) -> list[dict[str, Path | str]]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    samples = []
    missing = []
    for image in data.get("images", []):
        name = image["file_name"]
        stem = Path(name).stem
        paths = {
            "sample": stem,
            "image": image_dir / name,
            "cleaned": cleaned_dir / f"{stem}.json",
            "legacy": legacy_dir / f"{stem}.json",
        }
        absent = [str(path) for key, path in paths.items() if key != "sample" and not path.exists()]
        if absent:
            missing.append({"sample": stem, "missing": absent})
        else:
            samples.append(paths)
    if missing:
        raise FileNotFoundError(json.dumps({"split": str(split_json), "missing": missing}, ensure_ascii=False, indent=2))
    if not samples:
        raise ValueError(f"No samples in frozen split: {split_json}")
    return samples


def load_csnet(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(str(path), map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = CSNet(in_channels=3, n_classes=1)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def largest_roi_mask(result, score_threshold: float) -> tuple[np.ndarray | None, float]:
    instances = result.pred_instances
    if len(instances) == 0:
        return None, 0.0
    valid = instances.scores > score_threshold
    if not bool(valid.any()):
        return None, 0.0
    masks = instances.masks[valid].cpu().numpy()
    scores = instances.scores[valid].cpu().numpy()
    best_idx = int(np.argmax(masks.sum(axis=(1, 2))))
    roi = masks[best_idx].astype(np.uint8) * 255
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        roi = np.zeros_like(roi)
        cv2.drawContours(roi, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return roi, float(scores[best_idx])


def apply_roi(roi_model, image: np.ndarray, score_threshold: float) -> tuple[np.ndarray, np.ndarray, bool, float]:
    roi, score = largest_roi_mask(inference_detector(roi_model, image), score_threshold)
    if roi is None:
        return np.full(image.shape[:2], 255, dtype=np.uint8), image.copy(), False, score
    output = np.zeros_like(image)
    output[roi > 0] = image[roi > 0]
    return roi, output, True, score


def predict(model: torch.nn.Module, image: np.ndarray, transform: Compose, device: torch.device, threshold: float) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probability = torch.sigmoid(logits).squeeze().cpu().numpy()
    probability = cv2.resize(probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (probability >= threshold).astype(np.uint8)


def skeleton(mask: np.ndarray) -> np.ndarray:
    source = (mask > 0).astype(np.uint8) * 255
    output = np.zeros_like(source)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(source):
        eroded = cv2.erode(source, element)
        output = cv2.bitwise_or(output, cv2.subtract(source, cv2.dilate(eroded, element)))
        source = eroded
    return output > 0


def measure(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = prediction > 0
    gt = target > 0
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    skel_pred, skel_gt = skeleton(pred), skeleton(gt)
    tprec = float(np.logical_and(skel_pred, gt).sum()) / (float(skel_pred.sum()) + 1e-8)
    tsens = float(np.logical_and(skel_gt, pred).sum()) / (float(skel_gt.sum()) + 1e-8)
    cldice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
    count, labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    floating = sum(1 for label in range(1, count) if not np.any((labels == label) & gt))
    components = max(count - 1, 0)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "cldice": cldice,
            "pred_components": float(components), "floating_components": float(floating),
            "floating_component_rate": float(floating / max(components, 1))}


def overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    gt, pred = target > 0, prediction > 0
    canvas[np.logical_and(pred, ~gt)] = (0, 0, 255)
    canvas[np.logical_and(~pred, gt)] = (0, 255, 0)
    canvas[np.logical_and(pred, gt)] = (0, 255, 255)
    return cv2.addWeighted(image, 0.55, canvas, 0.45, 0)


def roi_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    tint = canvas.copy()
    tint[mask > 0] = (0, 165, 255)
    return cv2.addWeighted(canvas, 0.7, tint, 0.3, 0)


def bootstrap_delta(rows: list[dict[str, Any]], left: str, right: str, metric: str, seed: int, draws: int) -> dict[str, float]:
    values = np.asarray([row[f"{left}_{metric}"] - row[f"{right}_{metric}"] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(draws)])
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(means, 0.025)), "ci95_high": float(np.quantile(means, 0.975))}


def parse_args() -> argparse.Namespace:
    data_root = ROOT / "01_data" / "02_annotated" / "pre_lable_50"
    parser = argparse.ArgumentParser(description="Strict paired comparison of old and cleaned CS-Net models.")
    parser.add_argument("--split-json", type=Path, default=ROOT / "01_data/03_processed/annotations/mmdet/test/test.json")
    parser.add_argument("--image-dir", type=Path, default=data_root / "Photo")
    parser.add_argument("--cleaned-gt-dir", type=Path, default=data_root / "trunk_cleaned")
    parser.add_argument("--legacy-gt-dir", type=Path, default=data_root / "trunk")
    parser.add_argument("--legacy-checkpoint", type=Path, default=ROOT / "03_models/branch_segmentation/best_dice_model.pth")
    parser.add_argument("--cleaned-checkpoint", type=Path, default=ROOT / "03_models/compag/seg_csnet_roi_baseline/best.pth")
    parser.add_argument("--cldice-checkpoint", type=Path, default=ROOT / "03_models/compag/seg_csnet_roi_cldice/best.pth")
    parser.add_argument("--roi-config", type=Path, default=ROOT / "02_code/02_models/roi_locator_V2/configs/mask_rcnn_r50_fpn_roi_v2.py")
    parser.add_argument("--roi-checkpoint", type=Path, default=ROOT / "03_models/roi/epoch_12.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "04_results/branch_segmentation/cleaned_model_comparison_roi_v2_20260715")
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--roi-score-threshold", type=float, default=0.001)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing result directory: {args.output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    paths = [args.split_json, args.image_dir, args.cleaned_gt_dir, args.legacy_gt_dir, args.legacy_checkpoint,
             args.cleaned_checkpoint, args.cldice_checkpoint, args.roi_config, args.roi_checkpoint]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    samples = load_split(args.split_json, args.image_dir, args.cleaned_gt_dir, args.legacy_gt_dir)
    output = ensure_dir(args.output_dir)
    device = torch.device(args.device)
    print("=" * 72, flush=True)
    print("Cleaned-label CS-Net paired evaluation", flush=True)
    print(f"Frozen split: {args.split_json} ({len(samples)} original images)", flush=True)
    print(f"Device: {device} | ROI threshold: {args.roi_score_threshold} | Seg threshold: {args.threshold}", flush=True)
    print(f"Output: {output}", flush=True)
    print("=" * 72, flush=True)
    register_all_modules()
    print("Loading ROI model...", flush=True)
    roi_model = init_detector(str(args.roi_config), str(args.roi_checkpoint), device=str(device))
    print("Loading segmentation models: legacy / cleaned BCE+Dice / cleaned BCE+Dice+clDice...", flush=True)
    models = {
        "legacy_old_labels": load_csnet(args.legacy_checkpoint, device),
        "cleaned_bce_dice": load_csnet(args.cleaned_checkpoint, device),
        "cleaned_bce_dice_cldice": load_csnet(args.cldice_checkpoint, device),
    }
    transform = Compose([Resize(args.target_size, args.target_size), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    rows: list[dict[str, Any]] = []
    rendered: list[tuple[float, str, np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    for index, sample in enumerate(tqdm(samples, desc="paired segmentation evaluation"), start=1):
        print(f"[{index}/{len(samples)}] {sample['sample']} | ROI + 3-model inference...", flush=True)
        image = read_image(sample["image"])
        cleaned = load_labelme_mask(sample["cleaned"], image.shape[:2])
        legacy = load_labelme_mask(sample["legacy"], image.shape[:2])
        roi_mask, roi_image, roi_success, roi_score = apply_roi(roi_model, image, args.roi_score_threshold)
        predictions = {name: predict(model, roi_image, transform, device, args.threshold) for name, model in models.items()}
        roi_coverage = float((roi_mask > 0).mean())
        row: dict[str, Any] = {"sample": sample["sample"], "roi_success": roi_success, "roi_score": roi_score, "roi_coverage": roi_coverage}
        for name, prediction in predictions.items():
            row.update({f"{name}_{key}": value for key, value in measure(prediction, cleaned).items()})
        row.update({f"legacy_old_labels_legacygt_{key}": value for key, value in measure(predictions["legacy_old_labels"], legacy).items()})
        rows.append(row)
        rendered.append((row["cleaned_bce_dice_iou"] - row["legacy_old_labels_iou"], str(sample["sample"]), image, cleaned, predictions))
        write_image(output / "roi_inputs" / f"{sample['sample']}_mask.png", roi_mask)
        write_image(output / "roi_inputs" / f"{sample['sample']}_black_bg.jpg", roi_image)
        write_image(output / "roi_inputs" / f"{sample['sample']}_overlay.jpg", roi_overlay(image, roi_mask))
        print(
            f"  ROI={'ok' if roi_success else 'fallback'} score={roi_score:.3f} | "
            f"coverage={roi_coverage:.1%} | "
            f"IoU old={row['legacy_old_labels_iou']:.4f} clean={row['cleaned_bce_dice_iou']:.4f} "
            f"clDice={row['cleaned_bce_dice_cldice_iou']:.4f}",
            flush=True,
        )
    fields = list(rows[0])
    with (output / "per_sample_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    comparisons = {
        "cleaned_bce_dice_minus_legacy_old_labels": {metric: bootstrap_delta(rows, "cleaned_bce_dice", "legacy_old_labels", metric, args.seed, args.bootstrap_draws) for metric in METRICS},
        "cleaned_bce_dice_cldice_minus_cleaned_bce_dice": {metric: bootstrap_delta(rows, "cleaned_bce_dice_cldice", "cleaned_bce_dice", metric, args.seed + 1, args.bootstrap_draws) for metric in METRICS},
        "legacy_old_labels_cleanedgt_minus_legacygt": {metric: bootstrap_delta(rows, "legacy_old_labels", "legacy_old_labels_legacygt", metric, args.seed + 2, args.bootstrap_draws) for metric in METRICS},
    }
    means = {name: {metric: float(np.mean([row[f"{name}_{metric}"] for row in rows])) for metric in METRICS} for name in MODEL_NAMES}
    provenance = {"split_json": str(args.split_json), "split_sha256": sha256(args.split_json), "sample_count": len(samples),
                  "models": {"legacy_old_labels": str(args.legacy_checkpoint), "cleaned_bce_dice": str(args.cleaned_checkpoint), "cleaned_bce_dice_cldice": str(args.cldice_checkpoint), "roi": str(args.roi_checkpoint)},
                  "sha256": {"legacy_old_labels": sha256(args.legacy_checkpoint), "cleaned_bce_dice": sha256(args.cleaned_checkpoint),
                             "cleaned_bce_dice_cldice": sha256(args.cldice_checkpoint), "roi": sha256(args.roi_checkpoint)},
                  "target_size": args.target_size, "threshold": args.threshold, "roi_score_threshold": args.roi_score_threshold}
    summary = {"created_at": datetime.now().isoformat(timespec="seconds"), "status": "completed", "metric_gt": "trunk_cleaned", "means": means,
               "paired_deltas": comparisons, "roi": {"success_count": sum(row["roi_success"] for row in rows), "failure_count": sum(not row["roi_success"] for row in rows),
               "coverage_mean": float(np.mean([row["roi_coverage"] for row in rows])), "coverage_min": float(np.min([row["roi_coverage"] for row in rows])),
               "coverage_max": float(np.max([row["roi_coverage"] for row in rows]))},
               "provenance": provenance}
    write_json(output / "metrics_summary.json", summary)
    write_json(output / "configs" / "frozen_test_manifest.json", {"source_split": str(args.split_json), "samples": [row["sample"] for row in rows]})
    for rank, (_, name, image, target, predictions) in enumerate(sorted(rendered)[:5], start=1):
        for model_name, prediction in predictions.items():
            write_image(output / "visualizations" / "failure_cases" / f"{rank:02d}_{name}_{model_name}.jpg", overlay(image, target, prediction))
    report = ["# Cleaned-label segmentation comparison", "", f"- Frozen samples: {len(rows)}", "- GT: `trunk_cleaned`", "- Main contrast: `cleaned_bce_dice - legacy_old_labels`", "- clDice contrast: `cleaned_bce_dice_cldice - cleaned_bce_dice`", f"- Deployment ROI coverage: mean={summary['roi']['coverage_mean']:.1%}, min={summary['roi']['coverage_min']:.1%}, max={summary['roi']['coverage_max']:.1%}", "- ROI input audit: `roi_inputs/`", "", "| Model | IoU | Dice | clDice | Floating component rate |", "|---|---:|---:|---:|---:|"]
    report.extend(f"| {name} | {means[name]['iou']:.4f} | {means[name]['dice']:.4f} | {means[name]['cldice']:.4f} | {means[name]['floating_component_rate']:.4f} |" for name in MODEL_NAMES)
    ensure_dir(output / "report")
    (output / "report" / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("=" * 72, flush=True)
    print("Completed. Primary macro metrics:", flush=True)
    for name in MODEL_NAMES:
        print(f"  {name}: IoU={means[name]['iou']:.4f} Dice={means[name]['dice']:.4f} clDice={means[name]['cldice']:.4f}", flush=True)
    print(f"Summary: {output / 'metrics_summary.json'}", flush=True)
    print(f"Report:  {output / 'report' / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
