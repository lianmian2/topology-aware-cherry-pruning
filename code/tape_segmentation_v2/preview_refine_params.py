import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data_processing import SOURCE_ANN_PATH, SOURCE_IMAGE_DIR, polygons_to_mask


WINDOW_NAME = "Tape Refine Preview"
MAX_PREVIEW_WIDTH = 1800


def parse_args():
    parser = argparse.ArgumentParser(description="交互式预览条带掩码收紧参数。")
    parser.add_argument("--image", help="图片文件名，例如 tree_007_before_view_01.jpg")
    return parser.parse_args()


def load_coco_annotations():
    with SOURCE_ANN_PATH.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_image_id = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image_id[ann["image_id"]].append(ann)

    images_by_name = {}
    for image_info in coco.get("images", []):
        images_by_name[image_info["file_name"]] = {
            "image_info": image_info,
            "annotations": anns_by_image_id.get(image_info["id"], []),
        }
    return images_by_name


def ensure_odd(value: int) -> int:
    if value <= 0:
        return 0
    if value % 2 == 0:
        value += 1
    return value


def make_vertical_kernel(kernel_size: int):
    if kernel_size <= 0:
        return None
    if kernel_size % 2 == 0:
        kernel_size += 1
    return np.ones((kernel_size, 1), np.uint8)


def directional_refine_mask(
    image_bgr: np.ndarray,
    coarse_mask: np.ndarray,
    grabcut_iterations: int,
    vertical_shrink: int,
) -> np.ndarray:
    _, coarse_mask = cv2.threshold(coarse_mask, 127, 255, cv2.THRESH_BINARY)

    if np.all(coarse_mask == 0) or np.all(coarse_mask == 255):
        return coarse_mask

    refined_mask = coarse_mask.copy()
    if grabcut_iterations > 0:
        gc_mask = np.zeros(image_bgr.shape[:2], np.uint8)
        gc_mask[coarse_mask == 0] = cv2.GC_BGD
        gc_mask[coarse_mask > 0] = cv2.GC_PR_FGD
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                image_bgr,
                gc_mask,
                None,
                bgd_model,
                fgd_model,
                grabcut_iterations,
                cv2.GC_INIT_WITH_MASK,
            )
            refined_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
        except Exception as exc:
            print(f"GrabCut 失败，回退到粗掩码: {exc}")
            refined_mask = coarse_mask.copy()

    vertical_kernel = make_vertical_kernel(vertical_shrink)
    if vertical_kernel is not None:
        refined_mask = cv2.erode(refined_mask, vertical_kernel, iterations=1)

    if np.sum(refined_mask) == 0:
        return coarse_mask

    return refined_mask


def create_trackbars():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("gc_iter", WINDOW_NAME, 1, 10, lambda _x: None)
    cv2.createTrackbar("v_shrink", WINDOW_NAME, 1, 12, lambda _x: None)


def get_params():
    grabcut_iterations = cv2.getTrackbarPos("gc_iter", WINDOW_NAME)
    vertical_shrink = cv2.getTrackbarPos("v_shrink", WINDOW_NAME)
    return {
        "grabcut_iterations": grabcut_iterations,
        "vertical_shrink": vertical_shrink,
    }


def to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask == 255] = color
    return cv2.addWeighted(overlay, 0.4, image_bgr, 0.6, 0)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.putText(output, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return output


def resize_for_preview(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= MAX_PREVIEW_WIDTH:
        return image
    scale = MAX_PREVIEW_WIDTH / w
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    images_by_name = load_coco_annotations()

    if args.image:
        file_name = args.image
    else:
        file_name = sorted(images_by_name.keys())[0]

    if file_name not in images_by_name:
        raise FileNotFoundError(f"Image not found in annotations: {file_name}")

    image_meta = images_by_name[file_name]
    image_path = SOURCE_IMAGE_DIR / file_name
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    coarse_mask = polygons_to_mask(image_meta["image_info"], image_meta["annotations"])
    if coarse_mask.shape[:2] != image_bgr.shape[:2]:
        coarse_mask = cv2.resize(
            coarse_mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    print(f"预览图片: {file_name}")
    print("操作说明:")
    print("  q / ESC : 退出")
    print("  gc_iter  = GrabCut迭代次数，用来轻微贴合真实边界；数值越大，收得通常越紧。")
    print("  v_shrink = 纵向收紧强度，只在上下方向变瘦，不处理横向连接。")
    print("建议先把 gc_iter 调到 1 或 2，再用 v_shrink 找到合适厚度。")

    create_trackbars()

    last_params = None
    while True:
        params = get_params()
        if params != last_params:
            refined_mask = directional_refine_mask(image_bgr, coarse_mask, **params)
            coarse_overlay = overlay_mask(image_bgr, coarse_mask, (0, 255, 255))
            refined_overlay = overlay_mask(image_bgr, refined_mask, (0, 0, 255))

            panels = [
                add_title(image_bgr, "Original"),
                add_title(to_bgr(coarse_mask), "Coarse Mask"),
                add_title(coarse_overlay, "Coarse Overlay"),
                add_title(to_bgr(refined_mask), "Refined Mask"),
                add_title(refined_overlay, "Refined Overlay"),
            ]
            preview = cv2.hconcat(panels)
            preview = resize_for_preview(preview)
            cv2.imshow(WINDOW_NAME, preview)

            print(
                "当前参数:",
                f"grabcut_iterations={params['grabcut_iterations']},",
                f"vertical_shrink={params['vertical_shrink']},",
            )
            last_params = params

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
