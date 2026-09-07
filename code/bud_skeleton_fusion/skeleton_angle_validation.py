"""
芽点-骨架夹角验证 — 验证芽点主轴方向与骨架切线方向的一致性.

两条子分析:
2a. 芽轴 vs 骨架切线 → 芽是否沿枝条生长方向排列
    夹角 < 30° → 一致
    夹角 > 60° → 吸附可能有误
2b. 芽轴 vs 分割边界法向量 → 芽是否贴在枝条表面
    夹角接近 0° → 芽轴平行于边界 (正常)
    夹角 > 45° → 可能此处不应有骨架

用于:
- 评估吸附质量 (辅助模块 5)
- 辅助汇接判断 (如果芽点方向都指向同一臂方向 → 该臂更可能是"真实"分支, 辅助模块 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .bud_orientation import BudOrientation
from .bud_skeleton_attachment import BudAttachment


@dataclass
class BudAngleValidation:
    """单个芽点的夹角验证结果."""

    bud_index: int
    skeleton_tangent_angle: float        # 骨架局部切线角度 (弧度, [0, π))
    bud_skeleton_angle: float            # 芽轴与骨架切线夹角 (弧度, full_range=True 时 [0, π], 否则 [0, π/2])
    boundary_normal_angle: float         # 分割边界法向量角度 (弧度)
    bud_boundary_angle: float            # 芽轴与边界法线夹角 (弧度)
    consistency_score: float             # 综合一致性评分 [0, 1]
    is_consistent: bool                  # 是否通过一致性检查


def _extract_skeleton_tangent(
    skeleton_map: np.ndarray,
    point_xy: Tuple[int, int],
    window_size: int = 15,
) -> Optional[Tuple[float, float, float]]:
    """
    在骨架吸附点附近提取局部切线方向 (PCA).

    Args:
        skeleton_map: (H, W) uint8 骨架图
        point_xy: (x, y) 吸附点坐标
        window_size: 前后搜索窗口大小 (像素数)

    Returns:
        (tangent_angle_rad, dx, dy) 或 None
    """
    H, W = skeleton_map.shape
    px, py = point_xy

    # 收集窗口内的骨架像素
    ys, xs = np.where(skeleton_map)
    if len(xs) < 2:
        return None

    # 只取吸附点附近的像素
    dists = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
    nearby = dists <= window_size
    nearby_xs = xs[nearby]
    nearby_ys = ys[nearby]

    if len(nearby_xs) < 3:
        return None

    # PCA 提取主方向
    points = np.column_stack([nearby_xs, nearby_ys]).astype(np.float32)
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, -1]   # 最大特征值对应的特征向量

    angle = float(np.arctan2(principal[1], principal[0]))
    if angle < 0:
        angle += np.pi

    return (angle, float(principal[0]), float(principal[1]))


def _compute_boundary_normal(
    mask: np.ndarray,
    point_xy: Tuple[int, int],
    window_size: int = 10,
) -> Optional[float]:
    """
    在分割边界附近计算法向量 (Sobel 梯度).

    Args:
        mask: (H, W) uint8 分割 mask
        point_xy: (x, y) 吸附点
        window_size: Sobel 窗口大小

    Returns:
        normal_angle_rad 或 None
    """
    H, W = mask.shape
    px, py = point_xy

    y1 = max(0, py - window_size)
    y2 = min(H, py + window_size)
    x1 = max(0, px - window_size)
    x2 = min(W, px + window_size)

    if y2 - y1 < 5 or x2 - x1 < 5:
        return None

    patch = mask[y1:y2, x1:x2].astype(np.float32)

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    if np.max(grad_mag) < 1e-6:
        return None

    # 取梯度最大的点作为边界法向量
    max_idx = np.argmax(grad_mag)
    max_y = max_idx // (x2 - x1)
    max_x = max_idx % (x2 - x1)

    nx = float(gx[max_y, max_x])
    ny = float(gy[max_y, max_x])
    norm = np.sqrt(nx ** 2 + ny ** 2)
    if norm < 1e-6:
        return None

    angle = float(np.arctan2(ny / norm, nx / norm))
    if angle < 0:
        angle += 2 * np.pi

    return angle


def validate_bud_angles(
    attachments: List[BudAttachment],
    orientations: List[BudOrientation],
    skeleton_map: np.ndarray,
    combined_mask: np.ndarray,
    full_range: bool = False,
) -> List[BudAngleValidation]:
    """
    对每个已吸附的芽点做夹角验证.

    Args:
        attachments: 芽点吸附结果
        orientations: 芽点方向信息 (与 attachments 按 bud_index 对应)
        skeleton_map: (H, W) uint8 骨架图
        combined_mask: (H, W) uint8 组合分割 mask
        full_range: False (默认) → 夹角截断到 [0, π/2]; True → 完整 [0, π] 可检测反向芽点

    Returns:
        List[BudAngleValidation]
    """
    results: List[BudAngleValidation] = []

    for att in attachments:
        bud_idx = att.bud_index
        if bud_idx >= len(orientations):
            continue

        orient = orientations[bud_idx]
        bud_angle = orient.axis_angle

        skeleton_angle = 0.0
        bud_skel_angle = np.pi / 2
        boundary_angle = 0.0
        bud_boundary_angle = np.pi / 2

        if att.skeleton_point is not None:
            # 2a: 骨架切线
            tangent_result = _extract_skeleton_tangent(skeleton_map, att.skeleton_point)
            if tangent_result is not None:
                skeleton_angle = tangent_result[0]
                raw_diff = abs(bud_angle - skeleton_angle)
                if full_range:
                    bud_skel_angle = raw_diff
                else:
                    bud_skel_angle = min(raw_diff, np.pi - raw_diff)

            # 2b: 分割边界法向量
            normal = _compute_boundary_normal(combined_mask, att.skeleton_point)
            if normal is not None:
                boundary_angle = normal
                raw_diff = abs(bud_angle - normal)
                if full_range:
                    bud_boundary_angle = raw_diff
                else:
                    bud_boundary_angle = min(raw_diff, np.pi - raw_diff)

        # 综合评分 (始终用截断到 [0, π/2] 的值计算)
        skel_score = max(0.0, 1.0 - min(bud_skel_angle, np.pi - bud_skel_angle) / (np.pi / 3))   # 60° → 0
        boundary_score = max(0.0, 1.0 - min(bud_boundary_angle, np.pi - bud_boundary_angle) / (np.pi / 4))  # 45° → 0

        if orient.is_elongated:
            consistency_score = float(0.5 * skel_score + 0.5 * boundary_score)
        else:
            consistency_score = float(skel_score * 0.3 + boundary_score * 0.3)  # 近圆形芽点权重减半

        is_consistent = consistency_score >= 0.5

        results.append(BudAngleValidation(
            bud_index=bud_idx,
            skeleton_tangent_angle=skeleton_angle,
            bud_skeleton_angle=bud_skel_angle,
            boundary_normal_angle=boundary_angle,
            bud_boundary_angle=bud_boundary_angle,
            consistency_score=consistency_score,
            is_consistent=is_consistent,
        ))

    return results


def angle_validation_summary(validations: List[BudAngleValidation]) -> dict:
    """汇总夹角验证统计."""
    total = len(validations)
    consistent = sum(1 for v in validations if v.is_consistent)
    mean_skel_angle = np.mean([v.bud_skeleton_angle for v in validations]) if total > 0 else 0
    mean_boundary_angle = np.mean([v.bud_boundary_angle for v in validations if v.boundary_normal_angle != 0]) if total > 0 else 0
    mean_consistency = np.mean([v.consistency_score for v in validations]) if total > 0 else 0

    return {
        'total_validated': total,
        'consistent_count': consistent,
        'consistency_rate': consistent / total if total > 0 else 0,
        'mean_bud_skeleton_angle_deg': float(np.rad2deg(mean_skel_angle)),
        'mean_bud_boundary_angle_deg': float(np.rad2deg(mean_boundary_angle)),
        'mean_consistency_score': float(mean_consistency),
    }
