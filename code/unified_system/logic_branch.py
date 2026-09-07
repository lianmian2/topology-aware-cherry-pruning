import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def run_branch_segmentation(seg_model, img_rgb, device='cuda:0'):
    """
    输入 RGB 图像（通常是经过 ROI 过滤后的图像）
    返回 二值化掩码图 (0, 255)
    """
    transform = A.Compose([
        A.Resize(1024, 1024, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = seg_model(img_tensor)
        if isinstance(output, tuple):
            output = output[0]
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        
    # 还原尺寸并二值化
    orig_h, orig_w = img_rgb.shape[:2]
    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    pred_binary = (pred_resized > 0.5).astype(np.uint8) * 255
    
    return pred_binary
