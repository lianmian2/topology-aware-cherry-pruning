import numpy as np
from mmdet.apis import inference_detector

def get_largest_mask(result, score_thr=0.001):
    if len(result.pred_instances.masks) == 0:
        return None, 0.0
    
    masks = result.pred_instances.masks
    scores = result.pred_instances.scores
    
    valid_indices = scores > score_thr
    if not valid_indices.any():
        return None, 0.0
    
    masks = masks[valid_indices]
    scores = scores[valid_indices]
    
    if hasattr(masks, 'cpu'):
        masks = masks.cpu()
    mask_array = masks.numpy()
    
    areas = np.sum(mask_array, axis=(1, 2))
    best_idx = np.argmax(areas)
    
    best_mask = mask_array[best_idx].astype(np.uint8) * 255
    best_score = scores[best_idx].item()
    
    # 形态学后处理：只保留最大连通域
    import cv2
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        refined_mask = np.zeros_like(best_mask)
        cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        best_mask = refined_mask

    return best_mask, best_score

def apply_roi_filter(roi_model, img_rgb):
    """
    输入 RGB 图像，返回 ROI mask (255 表示前景，0 表示背景) 和 过滤后的 RGB 图像
    """
    import cv2
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result = inference_detector(roi_model, img_bgr)
    mask, score = get_largest_mask(result, score_thr=0.001)
    
    roi_filtered = img_rgb.copy()
    if mask is not None:
        roi_filtered[mask == 0] = [0, 0, 0]
    else:
        # 如果没有检测到 ROI，就返回全 255 的 mask
        mask = np.ones(img_rgb.shape[:2], dtype=np.uint8) * 255
        
    return mask, roi_filtered
