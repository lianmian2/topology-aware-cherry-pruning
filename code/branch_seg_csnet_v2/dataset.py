"""
数据处理模块 - Cherry Branch Segmentation Dataset
=====================================================

功能：
1. 加载COCO格式的实例分割标注
2. 将Branch和Trunk类别的多边形合并为二值掩码
3. 高分辨率输入 (1024x1024)
4. 极限数据增强 (albumentations)

作者：Cherry Branch Segmentation Project V2
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CherryBranchDataset(Dataset):
    """
    樱桃树枝条语义分割数据集
    
    特点：
    - 从COCO实例分割格式提取Branch和Trunk类别
    - 合并为单通道二值掩码 (0=背景, 1=枝条/树干)
    - 支持1024x1024高分辨率输入
    - 极限数据增强策略
    """
    
    def __init__(
        self,
        json_path: str,
        img_dir: str,
        target_size: int = 1024,
        is_train: bool = True,
        target_categories: list = None
    ):
        """
        初始化数据集
        
        Args:
            json_path: COCO JSON标注文件路径
            img_dir: 图片目录路径
            target_size: 目标尺寸 (默认1024x1024)
            is_train: 是否为训练集 (决定是否应用数据增强)
            target_categories: 目标类别名称列表 (默认为['Branch', 'Trunk'])
        """
        self.json_path = json_path
        self.img_dir = img_dir
        self.target_size = target_size
        self.is_train = is_train
        
        if target_categories is None:
            target_categories = ['Branch', 'Trunk']
        self.target_categories = target_categories
        
        print(f"\n{'='*60}")
        print(f"初始化数据集: {'训练集' if is_train else '验证集'}")
        print(f"{'='*60}")
        print(f"标注文件: {json_path}")
        print(f"图片目录: {img_dir}")
        print(f"目标尺寸: {target_size}x{target_size}")
        print(f"目标类别: {target_categories}")
        
        self.coco = COCO(json_path)
        
        self.cat_ids = []
        for cat_name in target_categories:
            cat_ids = self.coco.getCatIds(catNms=[cat_name])
            if cat_ids:
                self.cat_ids.extend(cat_ids)
                print(f"  找到类别 '{cat_name}': ID={cat_ids}")
            else:
                print(f"  警告: 未找到类别 '{cat_name}'")
        
        self.img_ids = self.coco.getImgIds()
        print(f"共加载 {len(self.img_ids)} 张图片")
        
        self.transform = self._get_transforms()
        
        self._print_class_statistics()
    
    def _print_class_statistics(self):
        """打印类别统计信息"""
        print(f"\n类别统计:")
        total_anns = 0
        for cat_name in self.target_categories:
            cat_ids = self.coco.getCatIds(catNms=[cat_name])
            if cat_ids:
                ann_ids = self.coco.getAnnIds(catIds=cat_ids)
                print(f"  {cat_name}: {len(ann_ids)} 个标注")
                total_anns += len(ann_ids)
        print(f"  总计: {total_anns} 个标注")
    
    def _get_transforms(self):
        """
        获取数据增强变换
        
        训练集：极限数据增强
        - 随机旋转、翻转 (细长物体对角度敏感)
        - 弹性形变 (模拟风吹树枝弯曲)
        - 网格畸变 (增加形变多样性)
        - 高斯模糊 (模拟对焦不准)
        - 色彩抖动 (光照变化)
        
        验证集：仅基础预处理
        """
        if self.is_train:
            return A.Compose([
                A.Resize(self.target_size, self.target_size, interpolation=cv2.INTER_LINEAR),
                
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.3
                ),
                
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.3
                ),
                
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.3
                ),
                
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.target_size, self.target_size, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _create_binary_mask(self, img_info: dict) -> np.ndarray:
        """
        从COCO标注创建二值掩码
        
        将所有Branch和Trunk类别的多边形合并为单通道二值掩码
        
        Args:
            img_info: COCO图像信息字典
        
        Returns:
            binary_mask: 二值掩码 (H, W), 0=背景, 255=前景
        """
        height = img_info['height']
        width = img_info['width']
        
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
        for cat_id in self.cat_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=[cat_id])
            annotations = self.coco.loadAnns(ann_ids)
            
            for ann in annotations:
                if 'segmentation' not in ann:
                    continue
                
                if isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        if len(seg) < 6:
                            continue
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(binary_mask, [poly], 255)
                
                elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                    rle_mask = self.coco.annToMask(ann)
                    binary_mask = np.maximum(binary_mask, rle_mask * 255)
        
        return binary_mask
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            image: 图像张量 (3, H, W)
            mask: 掩码张量 (1, H, W)
            img_name: 图片文件名
        """
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        actual_height, actual_width = image.shape[:2]
        
        img_info_for_mask = img_info.copy()
        img_info_for_mask['height'] = actual_height
        img_info_for_mask['width'] = actual_width
        
        mask = self._create_binary_mask(img_info_for_mask)
        
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float() / 255.0
        
        return image, mask, img_info['file_name']


def create_dataloaders(
    train_json: str,
    train_img_dir: str,
    val_json: str,
    val_img_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    target_size: int = 1024
) -> tuple:
    """
    创建训练和验证数据加载器
    
    Args:
        train_json: 训练集JSON路径
        train_img_dir: 训练集图片目录
        val_json: 验证集JSON路径
        val_img_dir: 验证集图片目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        target_size: 目标尺寸
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    train_dataset = CherryBranchDataset(
        json_path=train_json,
        img_dir=train_img_dir,
        target_size=target_size,
        is_train=True
    )
    
    val_dataset = CherryBranchDataset(
        json_path=val_json,
        img_dir=val_img_dir,
        target_size=target_size,
        is_train=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
