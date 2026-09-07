import cv2
import numpy as np
import torch
from PIL import Image

def filter_y_axis(pred_mask):
    """
    保留最大连通域所在的 Y 轴范围，过滤掉其他行的噪声/背景标定带。
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    
    if num_labels <= 1: # 只有背景
        return pred_mask
        
    # 找到面积最大的连通域（排除背景 label 0）
    max_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i
            
    y_start = stats[max_label, cv2.CC_STAT_TOP]
    height = stats[max_label, cv2.CC_STAT_HEIGHT]
    y_end = y_start + height
    
    filtered_mask = np.zeros_like(pred_mask)
    
    tolerance = 5
    y1 = max(0, y_start - tolerance)
    y2 = min(pred_mask.shape[0], y_end + tolerance)
    
    filtered_mask[y1:y2, :] = pred_mask[y1:y2, :]
    
    return filtered_mask

def run_tape_segmentation(tape_model, img_rgb, device='cuda:0'):
    """
    输入 RGB 原图
    输出 过滤后的标定带掩码
    """
    IMAGE_SIZE = (512, 512)
    original_image = Image.fromarray(img_rgb)
    resized_image = original_image.resize(IMAGE_SIZE, Image.BILINEAR)
    img_array = np.array(resized_image) / 255.0
    
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = tape_model(img_tensor)
        probs = torch.sigmoid(logits)
        pred_mask_512 = (probs > 0.3).float().cpu().numpy()[0, 0]
        pred_mask_512 = (pred_mask_512 * 255).astype(np.uint8)
        
    orig_h, orig_w = img_rgb.shape[:2]
    pred_mask_orig = cv2.resize(pred_mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    filtered_mask = filter_y_axis(pred_mask_orig)
    return filtered_mask
