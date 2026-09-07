import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE_DIR = BASE_DIR.parent / "mmdet_data" / "tape"
SOURCE_ANN_PATH = BASE_DIR.parent / "mmdet_data" / "annotations" / "Tape_merged_dedup.json"
OUTPUT_DIR = BASE_DIR / "mmdet_data"
SPLIT_RATIO = (0.8, 0.1, 0.1)
SEED = 42
GRABCUT_ITERATIONS = 5
VERTICAL_SHRINK = 0

IMAGE_PATTERN = re.compile(r"tree_(\d+)(?:_(before|after))?_view_(\d+)\.jpg$")


def setup_directories():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "img_dir" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "ann_dir" / split).mkdir(parents=True, exist_ok=True)


def parse_name(file_name: str):
    match = IMAGE_PATTERN.search(file_name)
    if not match:
        return None

    tree_id, stage, view_id = match.groups()
    return {
        "tree_id": tree_id.zfill(3),
        "stage": stage or "single",
        "view_id": view_id.zfill(2),
    }


def polygons_to_mask(image_info, annotations):
    mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)

    for ann in annotations:
        for seg in ann.get("segmentation", []):
            if not seg:
                continue
            points = np.array(seg, dtype=np.float32).reshape(-1, 2)
            points = np.round(points).astype(np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 255)

    return mask


def _make_vertical_kernel(kernel_size: int) -> Optional[np.ndarray]:
    if kernel_size <= 1:
        return None
    if kernel_size % 2 == 0:
        kernel_size += 1
    return np.ones((kernel_size, 1), np.uint8)


def refine_mask(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    grabcut_iterations: int = GRABCUT_ITERATIONS,
    vertical_shrink: int = VERTICAL_SHRINK,
) -> np.ndarray:
    """
    Use GrabCut to tighten coarse polygon annotations to closer image edges.
    """
    _, coarse_mask = cv2.threshold(coarse_mask, 127, 255, cv2.THRESH_BINARY)

    if np.all(coarse_mask == 0) or np.all(coarse_mask == 255):
        return coarse_mask

    gc_mask = np.zeros(image.shape[:2], np.uint8)
    gc_mask[coarse_mask == 0] = cv2.GC_BGD
    gc_mask[coarse_mask > 0] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if grabcut_iterations > 0:
        try:
            cv2.grabCut(
                image,
                gc_mask,
                None,
                bgd_model,
                fgd_model,
                grabcut_iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except Exception as exc:
            print(f"GrabCut failed, fallback to coarse mask: {exc}")
            return coarse_mask

        refined_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    else:
        refined_mask = coarse_mask.copy()

    vertical_kernel = _make_vertical_kernel(vertical_shrink)
    if vertical_kernel is not None:
        refined_mask = cv2.erode(refined_mask, vertical_kernel, iterations=1)

    if np.sum(refined_mask) == 0:
        return coarse_mask

    return refined_mask


def build_split(tree_groups):
    random.seed(SEED)
    tree_ids = sorted(tree_groups.keys())
    random.shuffle(tree_ids)

    num_trees = len(tree_ids)
    train_end = int(num_trees * SPLIT_RATIO[0])
    val_end = train_end + int(num_trees * SPLIT_RATIO[1])

    train_trees = set(tree_ids[:train_end])
    val_trees = set(tree_ids[train_end:val_end])
    test_trees = set(tree_ids[val_end:])

    return train_trees, val_trees, test_trees


def print_progress(current: int, total: int, split: str, file_name: str):
    if total <= 0:
        return
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    message = (
        f"\rProcessing [{bar}] {current}/{total} "
        f"split={split:<5} file={file_name}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def main():
    print(f"Using merged annotations: {SOURCE_ANN_PATH}")
    print(f"Using source images: {SOURCE_IMAGE_DIR}")

    if not SOURCE_ANN_PATH.exists():
        raise FileNotFoundError(f"Annotation file not found: {SOURCE_ANN_PATH}")
    if not SOURCE_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {SOURCE_IMAGE_DIR}")

    with SOURCE_ANN_PATH.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    anns_by_image_id = defaultdict(list)
    for ann in annotations:
        anns_by_image_id[ann["image_id"]].append(ann)

    valid_items = []
    missing_images = []
    skipped_names = []
    tree_groups = defaultdict(list)

    for image_info in images:
        file_name = image_info["file_name"]
        parsed = parse_name(file_name)
        if parsed is None:
            skipped_names.append(file_name)
            continue

        image_path = SOURCE_IMAGE_DIR / file_name
        if not image_path.exists():
            missing_images.append(file_name)
            continue

        item = {
            "image_info": image_info,
            "annotations": anns_by_image_id.get(image_info["id"], []),
            "image_path": image_path,
            "tree_id": parsed["tree_id"],
            "stage": parsed["stage"],
            "view_id": parsed["view_id"],
        }
        valid_items.append(item)
        tree_groups[parsed["tree_id"]].append(item)

    if not valid_items:
        raise RuntimeError("No valid image-annotation pairs found.")

    setup_directories()
    train_trees, val_trees, test_trees = build_split(tree_groups)

    split_counts = {"train": 0, "val": 0, "test": 0}
    split_tree_counts = {
        "train": len(train_trees),
        "val": len(val_trees),
        "test": len(test_trees),
    }
    fallback_count = 0

    total_items = len(valid_items)
    for idx, item in enumerate(valid_items, start=1):
        tree_id = item["tree_id"]
        if tree_id in train_trees:
            split = "train"
        elif tree_id in val_trees:
            split = "val"
        else:
            split = "test"

        image_info = item["image_info"]
        coarse_mask = polygons_to_mask(image_info, item["annotations"])
        image = cv2.imread(str(item["image_path"]))
        if image is None:
            raise RuntimeError(f"Failed to read image: {item['image_path']}")

        mask = refine_mask(image, coarse_mask)
        if np.array_equal(mask, coarse_mask):
            fallback_count += 1

        dst_img = OUTPUT_DIR / "img_dir" / split / item["image_path"].name
        dst_mask = OUTPUT_DIR / "ann_dir" / split / f"{item['image_path'].stem}.png"

        shutil.copy2(item["image_path"], dst_img)
        cv2.imwrite(str(dst_mask), mask)
        split_counts[split] += 1
        print_progress(idx, total_items, split, item["image_path"].name)

    split_report = {
        "source_annotation": str(SOURCE_ANN_PATH),
        "source_image_dir": str(SOURCE_IMAGE_DIR),
        "seed": SEED,
        "mask_refinement": {
            "method": "grabcut_with_vertical_shrink",
            "grabcut_iterations": GRABCUT_ITERATIONS,
            "vertical_shrink": VERTICAL_SHRINK,
            "fallback_to_coarse_mask_count": fallback_count,
        },
        "split_ratio": {
            "train": SPLIT_RATIO[0],
            "val": SPLIT_RATIO[1],
            "test": SPLIT_RATIO[2],
        },
        "total_images_in_json": len(images),
        "valid_images_used": len(valid_items),
        "missing_images": missing_images,
        "skipped_unrecognized_names": skipped_names,
        "tree_count": len(tree_groups),
        "split_tree_counts": split_tree_counts,
        "split_image_counts": split_counts,
        "train_trees": sorted(train_trees),
        "val_trees": sorted(val_trees),
        "test_trees": sorted(test_trees),
    }

    with (OUTPUT_DIR / "split_report.json").open("w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=4, ensure_ascii=False)

    print("Data processing complete.")
    print(f"Tree count: {len(tree_groups)}")
    print(f"Image count: {len(valid_items)}")
    print(
        "Split image counts: "
        f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
    )


if __name__ == "__main__":
    main()
