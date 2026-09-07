"""
Evaluate CS-Net V2 on the cleaned pre_lable_50 LabelMe annotations.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
sys.path.insert(0, str(ROI_DIR))

from model import CSNet  # noqa: E402


TARGET_LABELS = {"Trunk", "Branch"}
METRIC_KEYS = ("dice", "iou", "precision", "recall", "f1", "pred_area", "gt_area")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labelme_mask(json_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for item in data.get("shapes", []):
        if item.get("label") not in TARGET_LABELS:
            continue
        points = item.get("points") or []
        if len(points) < 3:
            continue
        poly = np.asarray(points, dtype=np.float32)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 255)
    return mask


def collect_samples(image_dir: Path, old_gt_dir: Path, new_gt_dir: Path, limit: int = 0) -> Tuple[List[Dict], List[Dict]]:
    samples: List[Dict] = []
    skipped: List[Dict] = []
    new_jsons = sorted(new_gt_dir.glob("*.json"))
    if limit > 0:
        new_jsons = new_jsons[:limit]

    for new_json in new_jsons:
        stem = new_json.stem
        image_path = image_dir / f"{stem}.jpg"
        old_json = old_gt_dir / f"{stem}.json"
        missing = []
        if not image_path.exists():
            missing.append(str(image_path))
        if not old_json.exists():
            missing.append(str(old_json))
        if missing:
            skipped.append({"sample": stem, "reason": "missing_file", "missing": missing})
            continue
        samples.append({"stem": stem, "image": image_path, "old_gt": old_json, "new_gt": new_json})
    return samples, skipped


def load_seg_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = CSNet(in_channels=3, n_classes=1)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_largest_roi_mask(result, score_thr: float) -> Tuple[np.ndarray, float]:
    instances = result.pred_instances
    if len(instances.masks) == 0:
        return None, 0.0
    masks = instances.masks
    scores = instances.scores
    valid = scores > score_thr
    if not bool(valid.any()):
        return None, 0.0
    masks = masks[valid]
    scores = scores[valid]
    if hasattr(masks, "cpu"):
        masks = masks.cpu()
    mask_array = masks.numpy()
    areas = np.sum(mask_array, axis=(1, 2))
    best_idx = int(np.argmax(areas))
    roi_mask = mask_array[best_idx].astype(np.uint8) * 255
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        refined = np.zeros_like(roi_mask)
        cv2.drawContours(refined, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
        roi_mask = refined
    return roi_mask, float(scores[best_idx].item())


def apply_roi(roi_model, image_bgr: np.ndarray, score_thr: float) -> Tuple[np.ndarray, bool, float]:
    result = inference_detector(roi_model, image_bgr)
    roi_mask, roi_score = get_largest_roi_mask(result, score_thr)
    if roi_mask is None:
        return image_bgr.copy(), False, roi_score
    filtered = image_bgr.copy()
    filtered[roi_mask == 0] = 0
    return filtered, True, roi_score


def predict_mask(model: torch.nn.Module, image_bgr: np.ndarray, transform: Compose, device: torch.device, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
    h, w = image_bgr.shape[:2]
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    pred = (prob > threshold).astype(np.uint8) * 255
    return pred, prob


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask > 0
    gt = gt_mask > 0
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    pred_area = float(pred.sum())
    gt_area = float(gt.sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    union = tp + fp + fn
    dice = (2.0 * tp) / (pred_area + gt_area + 1e-8)
    iou = tp / (union + 1e-8)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_area": pred_area,
        "gt_area": gt_area,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def render_overlay(image_bgr: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    gt = gt_mask > 0
    pred = pred_mask > 0
    pred_only = np.logical_and(pred, ~gt)
    gt_only = np.logical_and(gt, ~pred)
    overlap = np.logical_and(pred, gt)
    overlay[pred_only] = (0, 0, 255)
    overlay[gt_only] = (0, 255, 0)
    overlay[overlap] = (0, 255, 255)
    return cv2.addWeighted(image_bgr, 0.55, overlay, 0.45, 0)


def summarize_metrics(rows: List[Dict], skipped: List[Dict], args: argparse.Namespace) -> Dict:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(rows),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "config": config,
        "old": {},
        "new": {},
        "new_minus_old": {},
        "roi": {
            "success_count": int(sum(1 for row in rows if row["roi_success"])),
            "failure_count": int(sum(1 for row in rows if not row["roi_success"])),
        },
    }
    for scope in ("old", "new"):
        for metric in METRIC_KEYS:
            values = np.asarray([row[f"{scope}_{metric}"] for row in rows], dtype=np.float64)
            if values.size == 0:
                stats = {"mean": None, "std": None, "min": None, "max": None}
            else:
                stats = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            summary[scope][metric] = stats
    for metric in METRIC_KEYS:
        values = np.asarray([row[f"delta_{metric}"] for row in rows], dtype=np.float64)
        summary["new_minus_old"][metric] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }
    return summary


def write_csv(rows: List[Dict], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "sample",
        "roi_success",
        "roi_score",
        "image_height",
        "image_width",
        *[f"old_{key}" for key in METRIC_KEYS],
        *[f"new_{key}" for key in METRIC_KEYS],
        *[f"delta_{key}" for key in METRIC_KEYS],
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CS-Net V2 on old/new LabelMe annotations.")
    data_root = ROOT / "01_data" / "02_annotated" / "pre_lable_50"
    parser.add_argument("--image-dir", type=Path, default=data_root / "Photo")
    parser.add_argument("--old-gt-dir", type=Path, default=data_root / "trunk")
    parser.add_argument("--new-gt-dir", type=Path, default=data_root / "trunk_cleaned")
    parser.add_argument("--seg-checkpoint", type=Path, default=ROOT / "03_models" / "branch_segmentation" / "best_dice_model.pth")
    parser.add_argument("--roi-config", type=Path, default=ROI_DIR / "configs" / "mask_rcnn_r50_fpn_roi_v2.py")
    parser.add_argument("--roi-checkpoint", type=Path, default=ROOT / "03_models" / "roi" / "epoch_12.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "04_results" / "branch_segmentation" / "cleaned_labelme_eval_20260704")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--roi-score-thr", type=float, default=0.001)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--overlay-limit", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)
    per_sample_dir = ensure_dir(output_dir / "per_sample")
    overlay_dir = ensure_dir(output_dir / "overlays")

    samples, skipped = collect_samples(args.image_dir, args.old_gt_dir, args.new_gt_dir, args.limit)
    print(f"Samples: {len(samples)}, skipped before eval: {len(skipped)}")
    print(f"Device: {device}")

    register_all_modules()
    roi_model = init_detector(str(args.roi_config), str(args.roi_checkpoint), device=str(device))
    seg_model = load_seg_model(args.seg_checkpoint, device)
    transform = Compose([
        Resize(args.target_size, args.target_size, interpolation=cv2.INTER_LINEAR),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    rows: List[Dict] = []
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            image = read_image(sample["image"])
            h, w = image.shape[:2]
            old_gt = load_labelme_mask(sample["old_gt"], (h, w))
            new_gt = load_labelme_mask(sample["new_gt"], (h, w))
            roi_image, roi_success, roi_score = apply_roi(roi_model, image, args.roi_score_thr)
            pred_mask, _ = predict_mask(seg_model, roi_image, transform, device, args.threshold)
            old_metrics = compute_metrics(pred_mask, old_gt)
            new_metrics = compute_metrics(pred_mask, new_gt)
        except Exception as exc:
            skipped.append({"sample": sample["stem"], "reason": "eval_error", "error": repr(exc)})
            continue

        row = {
            "sample": sample["stem"],
            "roi_success": bool(roi_success),
            "roi_score": float(roi_score),
            "image_height": int(h),
            "image_width": int(w),
        }
        for key in METRIC_KEYS:
            row[f"old_{key}"] = old_metrics[key]
            row[f"new_{key}"] = new_metrics[key]
            row[f"delta_{key}"] = new_metrics[key] - old_metrics[key]
        rows.append(row)

        save_json({"sample": sample["stem"], **row, "old": old_metrics, "new": new_metrics}, per_sample_dir / f"{sample['stem']}.json")
        if idx < args.overlay_limit:
            write_image(overlay_dir / f"{sample['stem']}_old_overlay.jpg", render_overlay(image, old_gt, pred_mask))
            write_image(overlay_dir / f"{sample['stem']}_new_overlay.jpg", render_overlay(image, new_gt, pred_mask))
            write_image(overlay_dir / f"{sample['stem']}_roi_filtered.jpg", roi_image)

    write_csv(rows, output_dir / "per_sample_metrics.csv")
    summary = summarize_metrics(rows, skipped, args)
    save_json(summary, output_dir / "metrics_summary.json")

    print("\nSummary")
    print(f"  evaluated: {summary['sample_count']}")
    print(f"  skipped:   {summary['skipped_count']}")
    print(f"  ROI success: {summary['roi']['success_count']}/{summary['sample_count']}")
    for metric in ("dice", "iou", "precision", "recall", "f1"):
        old_mean = summary["old"][metric]["mean"]
        new_mean = summary["new"][metric]["mean"]
        delta = summary["new_minus_old"][metric]["mean"]
        print(f"  {metric}: old={old_mean:.4f} new={new_mean:.4f} delta={delta:+.4f}")
    print(f"  results: {output_dir}")


if __name__ == "__main__":
    main()
