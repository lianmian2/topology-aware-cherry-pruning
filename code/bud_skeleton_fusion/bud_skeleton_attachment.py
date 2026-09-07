"""
芽点→骨架吸附 — 将芽点检测结果吸附到 DijkstraSkeletonRouter 的骨架上.

核心改进 (相比 TreeTopologyBuilder2D 的简单最近邻):
1. 在 combined_mask 的连通分量内搜索最近骨架像素 (避免穿越分割边界)
2. 将芽点归属到对应的 annotation_group (通过 polyline 距离)
3. 统计修正了多少芽点的吸附关系

集成位置: DijkstraSkeletonRouter.__call__ 的 Stage 5 (分支路由) 之后,
           Stage 6 (汇接配对重建) 之前.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import label as connected_label
from skimage.morphology import skeletonize


@dataclass
class BudAttachment:
    """单个芽点的吸附结果."""

    bud_index: int
    bud_centroid: Tuple[float, float]           # 芽点 bbox 中心 (全局坐标)
    skeleton_point: Optional[Tuple[int, int]]    # 吸附到的骨架像素 (全局坐标), None = 未吸附
    distance: float                              # 到骨架像素的距离 (px)
    component_label: int                         # 所属连通分量标签 (-1 = 不在 mask 内)
    group_id: Optional[str]                      # 归属的 annotation_group ID
    attachment_method: str                       # "component" / "fallback" / "none"


def _compute_bud_centroids(boxes: np.ndarray) -> List[Tuple[float, float]]:
    """从 bbox [x1, y1, x2, y2] 计算中心点."""
    if len(boxes) == 0:
        return []
    return [(float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)) for b in boxes]


def _point_to_polyline_distance(
    px: float, py: float,
    points: List[List[int]],
    edges: List[List[int]],
) -> float:
    """计算点到折线的最短距离."""
    if not points or not edges:
        return float('inf')
    min_dist = float('inf')
    for e in edges:
        if e[0] >= len(points) or e[1] >= len(points):
            continue
        p0 = points[e[0]]
        p1 = points[e[1]]
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])

        dx = x1 - x0
        dy = y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-10:
            dist = np.sqrt((px - x0) ** 2 + (py - y0) ** 2)
        else:
            t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg_len2))
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def _find_nearest_skeleton_in_mask(
    bud_xy: Tuple[float, float],
    skeleton_map: np.ndarray,
    mask: np.ndarray,
    max_radius: float = 80.0,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """
    在 mask 限定的区域内搜索最近的骨架像素.

    Args:
        bud_xy: (x, y) 芽点中心
        skeleton_map: (H, W) uint8, 1 = 骨架
        mask: (H, W) bool, 搜索范围 (连通分量)
        max_radius: 最大搜索半径 (px)

    Returns:
        (nearest_point, distance) — nearest_point 为 None 表示未找到
    """
    H, W = skeleton_map.shape
    bx, by = int(round(bud_xy[0])), int(round(bud_xy[1]))

    skeleton_in_component = skeleton_map & mask.astype(np.uint8)
    ys, xs = np.where(skeleton_in_component)

    if len(ys) == 0:
        return None, float('inf')

    dists = np.sqrt((xs - bx) ** 2 + (ys - by) ** 2)
    min_idx = np.argmin(dists)
    min_dist = float(dists[min_idx])

    if min_dist > max_radius:
        return None, min_dist

    return (int(xs[min_idx]), int(ys[min_idx])), min_dist


def _find_nearest_skeleton_global(
    bud_xy: Tuple[float, float],
    skeleton_map: np.ndarray,
    max_radius: float = 80.0,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """全图搜索最近骨架像素 (退化 fallback)."""
    ys, xs = np.where(skeleton_map)
    if len(ys) == 0:
        return None, float('inf')

    bx, by = bud_xy
    dists = np.sqrt((xs - bx) ** 2 + (ys - by) ** 2)
    min_idx = np.argmin(dists)
    min_dist = float(dists[min_idx])

    if min_dist > max_radius:
        return None, min_dist

    return (int(xs[min_idx]), int(ys[min_idx])), min_dist


def attach_buds_to_skeleton(
    boxes: np.ndarray,
    masks_info: List[Tuple[np.ndarray, int, int]],
    skeleton_map: np.ndarray,
    combined_mask: np.ndarray,
    annotation_groups: Optional[List[Dict]] = None,
    bud_snap_radius: float = 12.0,
    bud_spur_radius: float = 80.0,
    component_min_pixels: int = 64,
    support_states: Optional[Sequence[str]] = None,
    allow_global_fallback: bool = True,
    assign_fallback_group: bool = True,
) -> List[BudAttachment]:
    """
    将芽点吸附到骨架, 支持分割感知搜索.

    Args:
        boxes: (N, 4) [x1, y1, x2, y2] 全局坐标
        masks_info: list of (mask, offset_x, offset_y), 来自 logic_bud.py
        skeleton_map: (H, W) uint8, 骨架二值图 (来自 PredictionResult.skeleton_map)
        combined_mask: (H, W) uint8, 组合 mask (trunk + branch, DijkstraSkeletonRouter 输入)
        annotation_groups: DijkstraSkeletonRouter 的 annotation_groups, 用于将芽点归属到 group
        bud_snap_radius: 近距离直接 snap 的阈值 (px)
        bud_spur_radius: 最大吸附半径 (px)
        component_min_pixels: 连通分量最小像素数 (过滤碎片)
        support_states: 可选的每芽一致性状态；`background_reject` 不参与吸附。
        allow_global_fallback: 是否允许 mask 外芽点作全局最近邻回退。
        assign_fallback_group: 回退芽点是否允许写入枝条 group。

    Returns:
        List[BudAttachment]: 每个芽点的吸附结果
    """
    H, W = skeleton_map.shape
    centroids = _compute_bud_centroids(boxes)

    # 1. 标记 combined_mask 的连通分量
    mask_binary = combined_mask.astype(bool)
    labeled, num_components = connected_label(mask_binary)
    component_sizes = np.bincount(labeled.ravel())[1:]  # 跳过背景 (0)

    valid_components = set()
    for i, size in enumerate(component_sizes, 1):
        if size >= component_min_pixels:
            valid_components.add(i)

    attachments: List[BudAttachment] = []

    for idx, (bud_xy, (mask, offset_x, offset_y)) in enumerate(zip(centroids, masks_info)):
        bx, by = int(round(bud_xy[0])), int(round(bud_xy[1]))

        # 2. 确定芽点所属的连通分量
        if 0 <= by < H and 0 <= bx < W:
            component_label = int(labeled[by, bx])
        else:
            component_label = -1

        if component_label > 0 and component_label not in valid_components:
            component_label = -1

        skeleton_point: Optional[Tuple[int, int]] = None
        distance: float = float('inf')
        attachment_method: str = "none"
        support_state = support_states[idx] if support_states is not None and idx < len(support_states) else None

        if support_state == "background_reject":
            attachments.append(BudAttachment(
                bud_index=idx,
                bud_centroid=bud_xy,
                skeleton_point=None,
                distance=float("inf"),
                component_label=-1,
                group_id=None,
                attachment_method="rejected",
            ))
            continue

        # 3. 在所属连通分量内搜索最近骨架像素
        if component_label > 0:
            component_mask = (labeled == component_label)
            skel_pt, dist = _find_nearest_skeleton_in_mask(
                bud_xy, skeleton_map, component_mask, max_radius=bud_spur_radius
            )
            if skel_pt is not None:
                if dist <= bud_snap_radius:
                    attachment_method = "component_snap"
                else:
                    attachment_method = "component"
                skeleton_point = skel_pt
                distance = dist

        # 4. Fallback: 连通分量内未找到骨架, 全图搜索
        if skeleton_point is None and allow_global_fallback:
            skel_pt, dist = _find_nearest_skeleton_global(
                bud_xy, skeleton_map, max_radius=bud_spur_radius
            )
            if skel_pt is not None:
                attachment_method = "fallback"
                skeleton_point = skel_pt
                distance = dist

        # 5. 归属到 annotation_group
        group_id: Optional[str] = None
        if skeleton_point is not None and annotation_groups and (attachment_method != "fallback" or assign_fallback_group):
            sx, sy = skeleton_point
            best_group = None
            best_dist = float('inf')
            for group in annotation_groups:
                pts = group.get("points", [])
                eds = group.get("edges", [])
                d = _point_to_polyline_distance(float(sx), float(sy), pts, eds)
                if d < best_dist:
                    best_dist = d
                    best_group = group.get("group_id")
            group_id = best_group

        attachments.append(BudAttachment(
            bud_index=idx,
            bud_centroid=bud_xy,
            skeleton_point=skeleton_point,
            distance=distance,
            component_label=component_label,
            group_id=group_id,
            attachment_method=attachment_method,
        ))

    return attachments


def attachment_stats(attachments: List[BudAttachment]) -> dict:
    """汇总吸附统计."""
    total = len(attachments)
    attached = sum(1 for a in attachments if a.skeleton_point is not None)
    component_attached = sum(1 for a in attachments if a.attachment_method in ("component", "component_snap"))
    fallback = sum(1 for a in attachments if a.attachment_method == "fallback")
    snapped = sum(1 for a in attachments if a.attachment_method == "component_snap")
    grouped = sum(1 for a in attachments if a.group_id is not None)

    mean_dist = np.mean([a.distance for a in attachments if a.skeleton_point is not None]) if attached > 0 else float('inf')

    return {
        'total_buds': total,
        'attached': attached,
        'attachment_rate': attached / total if total > 0 else 0,
        'component_aware': component_attached,
        'fallback': fallback,
        'snapped': snapped,
        'grouped': grouped,
        'mean_distance': mean_dist,
    }


def build_group_bud_map(
    attachments: List[BudAttachment],
    annotation_groups: List[Dict],
) -> Dict[str, List[BudAttachment]]:
    """
    构建 group_id → 附着芽点列表 的映射.

    Args:
        attachments: 芽点吸附结果
        annotation_groups: 拓扑分组列表

    Returns:
        dict: group_id → List[BudAttachment]
    """
    group_map: Dict[str, List[BudAttachment]] = {
        group.get("group_id", ""): [] for group in annotation_groups
    }
    group_map["unattached"] = []

    for att in attachments:
        if att.group_id and att.group_id in group_map:
            group_map[att.group_id].append(att)
        else:
            group_map["unattached"].append(att)

    return group_map
