import os

import cv2
import numpy as np
import torch
import torchvision.ops as ops
from mmdet.apis import inference_detector

NMS_THR = 0.5
SCORE_THR = 0.5
PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = max(1, int(os.environ.get("CHERRY_BUD_BATCH_SIZE", "8")))


def process_intersection(boxes, scores, labels, masks):
    if len(boxes) == 0:
        return boxes, scores, labels, masks

    flower_idx = [i for i, label in enumerate(labels) if label == 0]
    leaf_idx = [i for i, label in enumerate(labels) if label == 1]

    def calc_intersection(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1: return 0.0
        return (x2 - x1) * (y2 - y1)

    def calc_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    to_delete = set()

    for f_idx in flower_idx:
        if f_idx in to_delete: continue
        for l_idx in leaf_idx:
            if l_idx in to_delete: continue

            f_box = boxes[f_idx]
            l_box = boxes[l_idx]
            inter_area = calc_intersection(f_box, l_box)
            f_area = calc_area(f_box)
            l_area = calc_area(l_box)
            inter_ratio = inter_area / min(f_area, l_area)

            if inter_ratio > 0.5:
                if scores[f_idx] >= scores[l_idx]:
                    to_delete.add(l_idx)
                else:
                    to_delete.add(f_idx)

    keep_idx = [i for i in range(len(boxes)) if i not in to_delete]
    return (
        boxes[keep_idx],
        scores[keep_idx],
        labels[keep_idx],
        [masks[i] for i in keep_idx]
    )


def _infer_patch(bud_model, patch_img, offset_x, offset_y, seed=0, score_thr=None):
    """单 patch 推理：固定种子 → 推理 → 映射到全局坐标"""
    if score_thr is None:
        score_thr = SCORE_THR

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    with torch.no_grad():
        result = inference_detector(bud_model, patch_img)

    pred = result.pred_instances
    if len(pred) == 0:
        return [], [], [], []

    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    bboxes = pred.bboxes.cpu().numpy()
    masks = pred.masks.cpu().numpy()

    boxes = []
    keep_scores = []
    keep_labels = []
    keep_masks = []

    for k in range(len(scores)):
        if scores[k] < score_thr:
            continue
        bx1, by1, bx2, by2 = bboxes[k]
        boxes.append([bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y])
        keep_scores.append(scores[k])
        keep_labels.append(labels[k])
        keep_masks.append((masks[k], offset_x, offset_y))

    return boxes, keep_scores, keep_labels, keep_masks


def _infer_patch_batch(bud_model, patch_items, seed=0, score_thr=None):
    if score_thr is None:
        score_thr = SCORE_THR

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    images = [item[2] for item in patch_items]
    with torch.no_grad():
        results = inference_detector(bud_model, images)
    if not isinstance(results, (list, tuple)):
        results = [results]

    batch_outputs = []
    for (offset_x, offset_y, _), result in zip(patch_items, results):
        pred = result.pred_instances
        if len(pred) == 0:
            batch_outputs.append(([], [], [], []))
            continue

        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        bboxes = pred.bboxes.cpu().numpy()
        masks = pred.masks.cpu().numpy()
        boxes = []
        keep_scores = []
        keep_labels = []
        keep_masks = []
        for index, score in enumerate(scores):
            if score < score_thr:
                continue
            bx1, by1, bx2, by2 = bboxes[index]
            boxes.append([bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y])
            keep_scores.append(score)
            keep_labels.append(labels[index])
            keep_masks.append((masks[index], offset_x, offset_y))
        batch_outputs.append((boxes, keep_scores, keep_labels, keep_masks))
    return batch_outputs


def run_bud_global_detection(bud_model, img_bgr, progress_callback=None, score_thr=None):
    """
    全图滑窗芽点检测：对每一张 patch 调用与局部检测完全相同的推理逻辑。
    输入: BGR 图像 (已经经过 ROI 过滤的)
    输出: boxes, scores, labels, masks_info
    """
    if score_thr is None:
        score_thr = SCORE_THR

    H, W = img_bgr.shape[:2]

    all_boxes = []
    all_scores = []
    all_labels = []
    all_masks = []

    # 1. 收集需要推理的 patch 坐标
    patch_coords = []
    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            y1, x1 = y, x
            y2, x2 = min(y + PATCH_SIZE, H), min(x + PATCH_SIZE, W)

            if y2 - y1 < PATCH_SIZE:
                y1 = max(0, y2 - PATCH_SIZE)
            if x2 - x1 < PATCH_SIZE:
                x1 = max(0, x2 - PATCH_SIZE)

            patch_img = img_bgr[y1:y1 + PATCH_SIZE, x1:x1 + PATCH_SIZE]

            gray = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
            if np.sum(gray == 0) / (PATCH_SIZE * PATCH_SIZE) > 0.8:
                continue

            patch_coords.append((x1, y1, patch_img))

    # 2. 逐 patch 调用与局部检测完全相同的推理逻辑
    total_patches = len(patch_coords)
    for start in range(0, total_patches, BATCH_SIZE):
        if progress_callback:
            progress_callback(start, total_patches)
        batch_items = patch_coords[start:start + BATCH_SIZE]
        for boxes, scores, labels, masks in _infer_patch_batch(
            bud_model, batch_items, seed=0, score_thr=score_thr
        ):
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_labels.extend(labels)
            all_masks.extend(masks)

    if len(all_boxes) == 0:
        return [], [], [], []

    # 3. 全局 NMS 去重
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=NMS_THR)

    final_boxes = boxes_tensor[keep_indices].numpy()
    final_scores = scores_tensor[keep_indices].numpy()
    final_labels = np.array(all_labels)[keep_indices.numpy()]
    final_masks = [all_masks[idx] for idx in keep_indices.numpy()]

    # 4. 花芽叶芽相交处理
    final_boxes, final_scores, final_labels, final_masks = process_intersection(
        final_boxes, final_scores, final_labels, final_masks
    )

    return final_boxes, final_scores, final_labels, final_masks


def run_bud_local_detection(bud_model, img_bgr, box_coords, seed=0):
    """
    局部芽点检测：对框选区域调用统一推理逻辑。
    输入: BGR 图像 (原图) 和 局部框 (x, y, w, h)
    输出: boxes, scores, labels, masks_info
    """
    x, y, w, h = box_coords
    patch_img = img_bgr[y:y + h, x:x + w]

    boxes, scores, labels, masks = _infer_patch(bud_model, patch_img, x, y, seed=seed)

    if len(boxes) == 0:
        return [], [], [], []

    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    labels_np = np.array(labels)

    boxes_np, scores_np, labels_np, masks = process_intersection(
        boxes_np, scores_np, labels_np, masks
    )

    return boxes_np, scores_np, labels_np, masks


# ---------------------------------------------------------------------------
# 高层 API：可直接从外部模块调用，无需依赖 GUI
# ---------------------------------------------------------------------------

BUD_CLASS_NAMES = {0: 'Flower_bud', 1: 'Leaf_bud'}
BUD_COLORS_BGR = {0: (255, 0, 0), 1: (0, 255, 0)}  # 花芽红, 叶芽绿


def draw_bud_annotations(image_bgr, boxes, scores, labels, masks_info=None,
                         draw_box=True, draw_score=True, draw_mask=True,
                         box_thickness=2, font_scale=0.5):
    """
    在图像上绘制芽点检测结果。

    Args:
        image_bgr: np.ndarray (H, W, 3) BGR 图像
        boxes: np.ndarray (N, 4) [x1, y1, x2, y2]
        scores: np.ndarray (N,)
        labels: np.ndarray (N,)  0=花芽, 1=叶芽
        masks_info: list of (mask, ox, oy), 可选
        draw_box: 是否画检测框
        draw_score: 是否在框上写置信度
        draw_mask: 是否叠加 mask
        box_thickness: 框线宽
        font_scale: 文字大小

    Returns:
        annotated: np.ndarray (H, W, 3) BGR 标注图
    """
    annotated = image_bgr.copy()

    if boxes is None or len(boxes) == 0:
        return annotated

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        label = int(labels[i])
        score = float(scores[i])
        color = BUD_COLORS_BGR.get(label, (0, 255, 255))

        if draw_box:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
        if draw_score:
            cv2.putText(annotated, f'{score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        if draw_mask and masks_info is not None and i < len(masks_info):
            mask_data = masks_info[i]
            local_mask, offset_x, offset_y = mask_data
            h_m, w_m = local_mask.shape
            H, W = annotated.shape[:2]
            end_y = min(offset_y + h_m, H)
            end_x = min(offset_x + w_m, W)
            valid_h = end_y - offset_y
            valid_w = end_x - offset_x
            if valid_h > 0 and valid_w > 0:
                global_bool = np.zeros((H, W), dtype=bool)
                global_bool[offset_y:end_y, offset_x:end_x] = local_mask[:valid_h, :valid_w]
                annotated[global_bool] = color

    return annotated


def run_bud_detection_pipeline(image, bud_model=None, roi_model=None,
                               use_roi_filter=True, score_thr=0.5,
                               progress_callback=None, device='cuda:0'):
    """
    全局芽点检测一键式 API。

    自动处理: 模型加载 → ROI 过滤(可选) → 滑窗检测 → NMS → 绘图。

    Args:
        image: str (图片路径) 或 np.ndarray (BGR 或 RGB, H x W x 3)
        bud_model: 预加载的 mmdet 模型, 为 None 则自动加载
        roi_model: 预加载的 ROI 模型, 为 None 则自动加载
        use_roi_filter: 是否先用 ROI 过滤背景
        score_thr: 置信度阈值
        progress_callback: callable(current, total) 进度回调
        device: 'cuda:0' 或 'cpu'

    Returns:
        dict:
            'boxes':      np.ndarray (N, 4) 全局坐标 [x1, y1, x2, y2]
            'scores':     np.ndarray (N,)
            'labels':     np.ndarray (N,)  0=Flower_bud, 1=Leaf_bud
            'class_names': ['Flower_bud' | 'Leaf_bud', ...]
            'masks_info': list of (mask_256x256, offset_x, offset_y)
            'annotated_image': np.ndarray (H, W, 3) BGR 标注图
            'flower_count': int
            'leaf_count': int
            'total_count': int
    """
    # --- 1. 加载图像 ---
    if isinstance(image, str):
        img_bgr = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image}")
    elif isinstance(image, np.ndarray):
        img_bgr = image.copy()
        # 如果是 RGB (3 通道但看起来不对), 统一转 BGR
    else:
        raise TypeError(f"image 必须是路径(str)或 np.ndarray, 收到: {type(image)}")

    H, W = img_bgr.shape[:2]

    # --- 2. 自动加载模型 ---
    if bud_model is None or roi_model is None:
        import sys
        import os as _os
        _current_dir = _os.path.dirname(_os.path.abspath(__file__))
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        from logic_models import ModelManager
        mm = ModelManager()

    if bud_model is None:
        bud_model = mm.load_bud_global_model()
    if roi_model is None and use_roi_filter:
        roi_model = mm.load_roi_model()

    # --- 3. ROI 过滤 ---
    if use_roi_filter and roi_model is not None:
        from logic_roi import apply_roi_filter
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _, roi_filtered_rgb = apply_roi_filter(roi_model, img_rgb)
        detection_input = cv2.cvtColor(roi_filtered_rgb, cv2.COLOR_RGB2BGR)
    else:
        detection_input = img_bgr

    # --- 4. 全局检测 ---
    boxes, scores, labels, masks_info = run_bud_global_detection(
        bud_model, detection_input, progress_callback=progress_callback, score_thr=score_thr
    )

    # --- 5. 构建返回结果 ---
    if len(boxes) == 0:
        return {
            'boxes': np.array([]).reshape(0, 4),
            'scores': np.array([]),
            'labels': np.array([]),
            'class_names': [],
            'masks_info': [],
            'annotated_image': img_bgr,
            'flower_count': 0,
            'leaf_count': 0,
            'total_count': 0,
        }

    class_names = [BUD_CLASS_NAMES.get(int(l), 'unknown') for l in labels]
    flower_count = int(np.sum(labels == 0))
    leaf_count = int(np.sum(labels == 1))

    annotated = draw_bud_annotations(img_bgr, boxes, scores, labels, masks_info)

    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels,
        'class_names': class_names,
        'masks_info': masks_info,
        'annotated_image': annotated,
        'flower_count': flower_count,
        'leaf_count': leaf_count,
        'total_count': len(boxes),
    }


def detect_and_save(image_path, output_path=None, **kwargs):
    """
    检测芽点并保存标注图到文件。

    Args:
        image_path: str, 输入图片路径
        output_path: str, 输出路径, 为 None 则自动生成 (原文件名_bud_detected.jpg)
        **kwargs: 传给 run_bud_detection_pipeline 的其他参数

    Returns:
        dict: 同 run_bud_detection_pipeline 的返回值
    """
    result = run_bud_detection_pipeline(image_path, **kwargs)

    if output_path is None:
        import os as _os
        base = _os.path.splitext(image_path)[0]
        output_path = f'{base}_bud_detected.jpg'

    cv2.imwrite(output_path, result['annotated_image'])
    print(f'标注图已保存至: {output_path}')
    print(f'检测到 {result["total_count"]} 个芽点 '
          f'(花芽: {result["flower_count"]}, 叶芽: {result["leaf_count"]})')

    return result
