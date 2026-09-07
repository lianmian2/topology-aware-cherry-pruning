"""
损失函数模块 V2 - 高效clDice实现
==================================

V2版本变更：
- 使用高效的max_pool2d实现骨架化，显存友好
- 减少骨架化迭代次数（3次 vs 10次）
- 移除复杂的torch.min/max操作

核心组件：
1. BCE Loss - 基础像素级分类损失
2. Dice Loss - 解决类别不平衡问题
3. clDice Loss (Centerline Dice) - 拓扑保持损失

作者：Cherry Branch Segmentation Project V2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss
    
    用于解决前景背景极度不平衡的问题
    
    公式：Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigmoid: bool = True
    ) -> torch.Tensor:
        if sigmoid:
            pred = torch.sigmoid(pred)
        
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        
        intersection = (pred_flat * target_flat).sum(1)
        cardinality = pred_flat.sum(1) + target_flat.sum(1)
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1.0 - dice.mean()


class SoftSkeletonization(nn.Module):
    """
    软骨架化模块 - 高效实现
    
    使用max_pool2d实现形态学操作，显存效率高
    
    原理：
    - 腐蚀：-max_pool(-x) 等价于取邻域最小值
    - 膨胀：max_pool(x) 等价于取邻域最大值
    - 骨架化：迭代腐蚀并保留边界信息
    """
    
    def __init__(self, iterations: int = 3):
        """
        Args:
            iterations: 骨架化迭代次数 (默认3次，足够提取细枝条骨架)
        """
        super().__init__()
        self.iterations = iterations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        高效骨架化实现
        
        Args:
            x: 输入二值掩码 (B, 1, H, W)，值域[0, 1]
        
        Returns:
            skeleton: 骨架 (B, 1, H, W)
        """
        skeleton = x.clone()
        
        for _ in range(self.iterations):
            eroded = -F.max_pool2d(-skeleton, 3, stride=1, padding=1)
            dilated = F.max_pool2d(skeleton, 3, stride=1, padding=1)
            boundary = dilated - eroded
            skeleton = torch.where(boundary > 0.5, eroded, skeleton)
        
        return skeleton


class clDiceLoss(nn.Module):
    """
    clDice Loss (Centerline Dice) - 高效实现
    
    拓扑保持损失函数，专门用于细长曲线结构分割
    
    原理：
    1. 对预测和真实mask进行骨架化
    2. 计算骨架之间的Dice系数
    3. 强制预测保持拓扑连贯性
    
    高效实现：
    - 使用max_pool2d代替复杂的torch.min/max
    - 减少迭代次数（3次）
    - 显存占用大幅降低
    """
    
    def __init__(
        self,
        iterations: int = 3,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth
        self.skeletonize = SoftSkeletonization(iterations=iterations)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigmoid: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算clDice损失
        
        Args:
            pred: 预测值 (B, 1, H, W)
            target: 目标值 (B, 1, H, W)
            sigmoid: 是否对预测应用sigmoid
        
        Returns:
            loss: clDice损失值
            metrics: 包含tprec和tsens的字典
        """
        if sigmoid:
            pred = torch.sigmoid(pred)
        
        target = target.float()
        
        pred_skeleton = self.skeletonize(pred)
        target_skeleton = self.skeletonize(target)
        
        tprec_numerator = (pred_skeleton * target).sum()
        tprec_denominator = pred_skeleton.sum() + self.smooth
        tprec = tprec_numerator / tprec_denominator
        
        tsens_numerator = (target_skeleton * pred).sum()
        tsens_denominator = target_skeleton.sum() + self.smooth
        tsens = tsens_numerator / tsens_denominator
        
        cl_dice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)
        
        loss = 1.0 - cl_dice
        
        metrics = {
            'tprec': tprec.item(),
            'tsens': tsens.item(),
            'cl_dice': cl_dice.item()
        }
        
        return loss, metrics


class HybridLoss(nn.Module):
    """
    混合损失函数 V2
    
    Loss = λ1 * BCE + λ2 * Dice + λ3 * clDice
    
    各损失的作用：
    - BCE: 像素级分类，提供基础监督
    - Dice: 区域匹配，解决类别不平衡
    - clDice: 拓扑保持，确保连续性
    
    V2改进：
    - 高效骨架化实现（max_pool2d）
    - 减少迭代次数（3次）
    - 显存占用大幅降低
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        cldice_weight: float = 0.5,
        skeleton_iterations: int = 3,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.cldice = clDiceLoss(
            iterations=skeleton_iterations,
            smooth=smooth
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target, sigmoid=True)
        cldice_loss, cldice_metrics = self.cldice(pred, target, sigmoid=True)
        
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.cldice_weight * cldice_loss
        )
        
        loss_dict = {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'cldice': cldice_loss.item(),
            'tprec': cldice_metrics['tprec'],
            'tsens': cldice_metrics['tsens'],
            'cl_dice_score': cldice_metrics['cl_dice'],
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice 混合损失（不含clDice）
    
    用于对比实验
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target, sigmoid=True)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    print("="*60)
    print("损失函数 V2 测试")
    print("="*60)
    
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    print("\n1. 高效clDice Loss:")
    cldice = clDiceLoss(iterations=3)
    loss, metrics = cldice(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Metrics: {metrics}")
    
    print("\n2. Hybrid Loss (BCE + Dice + clDice):")
    hybrid = HybridLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        cldice_weight=0.5,
        skeleton_iterations=3
    )
    loss, loss_dict = hybrid(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Details: {loss_dict}")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
