"""
测试集可视化脚本 V2
===================

对测试集进行预测并生成可视化结果
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from dataset import CherryBranchDataset
from model import CSNet


def load_model(checkpoint_path, device):
    model = CSNet(in_channels=3, n_classes=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def visualize_prediction(image, mask, pred, save_path, img_name):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')
    
    overlay = image.copy()
    pred_colored = np.zeros_like(overlay)
    pred_colored[pred > 0.5] = [255, 0, 0]
    gt_colored = np.zeros_like(overlay)
    gt_colored[mask > 0.5] = [0, 255, 0]
    
    blend = cv2.addWeighted(overlay, 0.6, pred_colored, 0.4, 0)
    blend = cv2.addWeighted(blend, 0.7, gt_colored, 0.3, 0)
    
    axes[3].imshow(blend)
    axes[3].set_title('Overlay (Green=GT, Red=Pred)', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, img_name), dpi=150, bbox_inches='tight')
    plt.close()


def calculate_metrics(pred, mask):
    pred_binary = (pred > 0.5).astype(np.float32)
    mask_binary = (mask > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_binary * mask_binary)
    union = np.sum(pred_binary) + np.sum(mask_binary) - intersection
    
    dice = (2 * intersection + 1e-5) / (np.sum(pred_binary) + np.sum(mask_binary) + 1e-5)
    iou = (intersection + 1e-5) / (union + 1e-5)
    
    return dice, iou


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_json = '/root/autodl-tmp/cherry/mmdet_data/annotations/test_roi_black_bg.json'
    val_img = '/root/autodl-tmp/cherry/mmdet_data/test_roi_black_bg'
    output_dir = '/root/autodl-tmp/cherry/branch_seg_csnet_v2/output/visualizations'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading validation dataset...")
    val_dataset = CherryBranchDataset(
        json_path=val_json,
        img_dir=val_img,
        target_size=1024,
        is_train=False
    )
    
    models = {
        'best_dice': '/root/autodl-tmp/cherry/branch_seg_csnet_v2/output/best_dice_model.pth',
        'best_iou': '/root/autodl-tmp/cherry/branch_seg_csnet_v2/output/best_iou_model.pth',
        'best_loss': '/root/autodl-tmp/cherry/branch_seg_csnet_v2/output/best_loss_model.pth'
    }
    
    all_results = {}
    
    for model_name, checkpoint in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print('='*60)
        
        model = load_model(checkpoint, device)
        
        all_dice = []
        all_iou = []
        
        vis_dir = os.path.join(output_dir, model_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        with torch.no_grad():
            for idx in tqdm(range(len(val_dataset)), desc=f"Processing {model_name}"):
                img_tensor, mask_tensor, img_name = val_dataset[idx]
                
                img_tensor_batch = img_tensor.unsqueeze(0).to(device)
                
                output = model(img_tensor_batch)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                mask = mask_tensor.squeeze().numpy()
                
                img_np = img_tensor.numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                dice, iou = calculate_metrics(pred, mask)
                all_dice.append(dice)
                all_iou.append(iou)
                
                save_name = f"sample_{idx:03d}_dice{dice:.3f}_iou{iou:.3f}.png"
                visualize_prediction(img_np, mask, pred, vis_dir, save_name)
        
        print(f"\n{model_name} Results:")
        print(f"  Mean Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
        print(f"  Mean IoU:  {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
        print(f"  Min Dice:  {np.min(all_dice):.4f}")
        print(f"  Max Dice:  {np.max(all_dice):.4f}")
        
        all_results[model_name] = {
            'mean_dice': float(np.mean(all_dice)),
            'mean_iou': float(np.mean(all_iou)),
            'std_dice': float(np.std(all_dice)),
            'std_iou': float(np.std(all_iou)),
            'min_dice': float(np.min(all_dice)),
            'max_dice': float(np.max(all_dice)),
            'min_iou': float(np.min(all_iou)),
            'max_iou': float(np.max(all_iou))
        }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for model_name, results in all_results.items():
        print(f"{model_name}: Dice={results['mean_dice']:.4f}, IoU={results['mean_iou']:.4f}")
    print("="*60)
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
