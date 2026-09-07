"""
芽点辅助分割消歧 — 在分割模糊区域利用芽点增强/降低骨架存在置信度.

场景:
- 细枝、遮挡、分割模型不确定的区域
- 有芽点 → 增强骨架存在置信度 (降低 Dijkstra 路由代价)
- 无芽点 → 降低骨架存在置信度

输出: 区域置信度 heatmap [0, 1], 可与模块 3 (无芽剪枝) 和模块 4 (分叉判断) 联合使用.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label as connected_label

from .bud_skeleton_attachment import BudAttachment


def compute_bud_confidence_heatmap(
    skeleton_map: np.ndarray,
    combined_mask: np.ndarray,
    attachments: List[BudAttachment],
    bud_influence_radius: int = 40,
    skeleton_decay_sigma: float = 20.0,
) -> np.ndarray:
    """
    生成芽点增强的区域置信度 heatmap.

    逻辑:
    - 骨架附近有芽点 → 高置信度 (接近 1.0)
    - 骨架附近无芽点 → 中置信度 (0.5)
    - mask 内部远离骨架的区域 → 低置信度但非零 (可能是漏检)
    - mask 外部 → 0

    Args:
        skeleton_map: (H, W) uint8 骨架二值图
        combined_mask: (H, W) uint8 组合分割 mask
        attachments: 芽点吸附结果
        bud_influence_radius: 芽点影响半径 (px)
        skeleton_decay_sigma: 骨架距离衰减 σ

    Returns:
        (H, W) float32 heatmap in [0, 1]
    """
    H, W = skeleton_map.shape

    # 1. 基础骨架置信度: 离骨架越远, 越不可能是枝条
    skel_dist = distance_transform_edt(1 - (skeleton_map > 0).astype(np.uint8))
    skeleton_confidence = np.exp(-skel_dist ** 2 / (2 * skeleton_decay_sigma ** 2))

    # 2. 芽点增强: 芽点附近的区域提高置信度
    bud_boost = np.zeros((H, W), dtype=np.float32)
    for att in attachments:
        if att.skeleton_point is not None:
            sx, sy = att.skeleton_point
        else:
            bx, by = att.bud_centroid
            sx, sy = int(round(bx)), int(round(by))

        if 0 <= sx < W and 0 <= sy < H:
            # 在芽点吸附点周围画高斯圆
            y1 = max(0, sy - bud_influence_radius)
            y2 = min(H, sy + bud_influence_radius)
            x1 = max(0, sx - bud_influence_radius)
            x2 = min(W, sx + bud_influence_radius)

            yy, xx = np.ogrid[y1:y2, x1:x2]
            dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
            gaussian = np.exp(-dist ** 2 / (2 * (bud_influence_radius / 3) ** 2))
            bud_boost[y1:y2, x1:x2] = np.maximum(bud_boost[y1:y2, x1:x2], gaussian)

    # 3. 组合: 骨架置信度 + 芽点增强
    heatmap = skeleton_confidence + bud_boost * 0.5
    heatmap = np.clip(heatmap, 0.0, 1.0)

    # 4. mask 外部设为 0
    mask_binary = combined_mask.astype(bool)
    heatmap[~mask_binary] = 0.0

    return heatmap.astype(np.float32)


def identify_ambiguous_regions(
    heatmap: np.ndarray,
    combined_mask: np.ndarray,
    low_threshold: float = 0.3,
    high_threshold: float = 0.6,
    min_region_size: int = 100,
) -> List[dict]:
    """
    从置信度 heatmap 中识别模糊区域.

    模糊区域定义: skeleton_confidence 在 [low, high] 之间, 且在 mask 内部.

    Args:
        heatmap: (H, W) float32 置信度 heatmap
        combined_mask: (H, W) uint8 组合 mask
        low_threshold: 低阈值 (低于此 → 不太可能有骨架)
        high_threshold: 高阈值 (高于此 → 很可能有骨架)
        min_region_size: 最小区域像素数

    Returns:
        List[dict]: 每个模糊区域 {bbox, area, mean_confidence, has_buds_nearby}
    """
    mask_binary = combined_mask.astype(bool)

    # 模糊区域: 置信度在 [low, high] 之间
    ambiguous = (heatmap >= low_threshold) & (heatmap <= high_threshold) & mask_binary
    ambiguous_u8 = ambiguous.astype(np.uint8)

    labeled, num_features = connected_label(ambiguous_u8)

    regions: List[dict] = []
    for label_id in range(1, num_features + 1):
        region = (labeled == label_id)
        area = int(np.sum(region))
        if area < min_region_size:
            continue

        ys, xs = np.where(region)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        mean_conf = float(np.mean(heatmap[region]))

        regions.append({
            'bbox': (x1, y1, x2, y2),
            'area': area,
            'centroid': (float(np.mean(xs)), float(np.mean(ys))),
            'mean_confidence': mean_conf,
        })

    return regions


def compute_cost_adjustment(
    heatmap: np.ndarray,
    base_cost: np.ndarray,
    max_adjustment: float = 0.3,
) -> np.ndarray:
    """
    基于芽点置信度 heatmap 调整 Dijkstra 代价图.

    置信度高的区域降低代价 (鼓励路由经过), 置信度低的区域保持原代价.

    Args:
        heatmap: (H, W) float32 [0, 1]
        base_cost: (H, W) float32 原始代价图
        max_adjustment: 最大代价降低比例

    Returns:
        (H, W) float32 调整后的代价图
    """
    adjustment = 1.0 - heatmap * max_adjustment
    return (base_cost * adjustment).astype(np.float32)


def adjust_cost_from_bud_centers(
    cost: np.ndarray,
    mask: np.ndarray,
    bud_centers: List[Tuple[float, float]],
    dt_norm: np.ndarray,
    influence_radius: float = 35.0,
    max_reduction: float = 0.25,
) -> np.ndarray:
    """从芽点中心直接调整 Dijkstra cost map (无需 skeleton/attachments).

    对 mask 内芽点附近区域降低 cost, 引导骨架路由经过有芽区域.
    细枝区域 (低 dt_norm) 额外加权, 因为细枝芽点信号更重要.

    Args:
        cost: (H, W) float32 原始 cost map
        mask: (H, W) bool 二值 mask
        bud_centers: [(x, y), ...] 芽点中心坐标 (processing 坐标系)
        dt_norm: (H, W) float32 归一化距离变换 [0, 1]
        influence_radius: 芽点影响半径 (px)
        max_reduction: 最大 cost 降低比例 [0, 1]

    Returns:
        (H, W) float32 调整后的 cost map
    """
    if not bud_centers:
        return cost

    H, W = cost.shape
    bud_boost = np.zeros((H, W), dtype=np.float32)
    sigma = influence_radius / 3.0

    for cx, cy in bud_centers:
        sx, sy = int(round(cx)), int(round(cy))
        if not (0 <= sx < W and 0 <= sy < H):
            continue
        if not mask[sy, sx]:
            continue

        r = int(np.ceil(influence_radius))
        y1, y2 = max(0, sy - r), min(H, sy + r + 1)
        x1, x2 = max(0, sx - r), min(W, sx + r + 1)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        bud_boost[y1:y2, x1:x2] = np.maximum(bud_boost[y1:y2, x1:x2], gaussian)

    # mask 外不调整; 细枝区域 (低 dt) 增强 boost
    thin_penalty = np.maximum(0.0, 0.5 - dt_norm) * 2.0
    adjustment = 1.0 - bud_boost * max_reduction * (1.0 + thin_penalty)
    adjustment = np.clip(adjustment, 1.0 - max_reduction * 2.0, 1.0)
    adjustment[~mask] = 1.0

    return (cost * adjustment).astype(np.float32)
