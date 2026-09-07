"""
芽点尖轴提取 — 从芽点分割 mask 中提取形态方向信息.

输入: logic_bud.py 的 masks_info — list of (mask_256x256, offset_x, offset_y)
输出: BudOrientation 列表 — 每个芽点的主轴角度、长宽比、全局质心、是否 elongated.

方法:
- cv2.fitEllipse 拟合椭圆 → 主轴角度 (优先, 代码最简单)
- 备选: cv2.moments 二阶中心矩 → PCA 主轴
- 输出无符号方向角 [0, π), 正负方向需结合枝条 traversal 推断.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class BudOrientation:
    """单个芽点的方向信息."""

    bud_index: int
    axis_angle: float          # 无符号主轴角度 [0, π), 弧度
    aspect_ratio: float        # 长轴/短轴, >= 1.0
    centroid_global: Tuple[float, float]  # 全局图像坐标 (x, y)
    is_elongated: bool         # aspect_ratio >= 1.3 视为有方向性
    ellipse_center: Tuple[float, float]   # fitEllipse 中心 (局部坐标)
    ellipse_axes: Tuple[float, float]     # (major_axis_len/2, minor_axis_len/2)


@dataclass
class DirectedBudOrientation:
    bud_index: int
    base_global: Tuple[float, float]
    tip_global: Tuple[float, float]
    centroid_global: Tuple[float, float]
    vector_xy: Tuple[float, float]
    confidence: float
    aspect_ratio: float
    attachment_distance: float
    endpoint_distance_margin: float
    method: str
    is_reliable: bool
    is_latent_spur: bool
    failure_reason: str


def _fit_ellipse_orientation(mask: np.ndarray) -> Optional[Tuple[float, float, Tuple[float, float], Tuple[float, float]]]:
    """
    cv2.fitEllipse 提取主轴角度.

    Returns:
        (angle_deg, aspect_ratio, center, axes) 或 None (拟合失败时)
        angle_deg: [0, 180) OpenCV 风格 (x 轴正方向逆时针)
        aspect_ratio: major/minor >= 1.0
        center: (cx, cy)
        axes: (major_half, minor_half)
    """
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_points = np.vstack(contours)
    if len(all_points) < 5:
        return None

    try:
        ellipse = cv2.fitEllipse(all_points)
        center, axes, angle = ellipse  # angle: [0, 180), cv2.fitEllipse 从 Y 轴测量
        major = max(axes) / 2.0
        minor = min(axes) / 2.0
        if minor < 1e-6:
            return None
        aspect_ratio = major / minor
        # cv2.fitEllipse 返回的 angle 从垂轴(Y)测量, 转为从水平轴(X)测量以匹配 arctan2
        angle = (angle + 90.0) % 180.0
        return (angle, aspect_ratio, center, (major, minor))
    except cv2.error:
        return None


def _moments_orientation(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    二阶中心矩 PCA 提取主轴角度 (备选方案).

    Returns:
        (angle_rad, aspect_ratio) 或 None
        angle_rad: [0, π)
    """
    moments = cv2.moments(mask.astype(np.float32))
    if moments['mu20'] == 0 and moments['mu02'] == 0:
        return None

    mu20 = moments['mu20']
    mu02 = moments['mu02']
    mu11 = moments['mu11']

    # 协方差矩阵的特征值/特征向量
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 * mu11
    if det <= 0 or trace <= 0:
        return None

    discriminant = max(trace * trace - 4 * det, 0)
    lambda1 = (trace + np.sqrt(discriminant)) / 2
    lambda2 = (trace - np.sqrt(discriminant)) / 2
    if lambda2 < 1e-10:
        return None

    aspect_ratio = np.sqrt(lambda1 / lambda2)

    # 主轴方向 (特征向量)
    if abs(mu11) < 1e-10:
        angle = 0.0 if mu20 >= mu02 else np.pi / 2
    else:
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    if angle < 0:
        angle += np.pi
    return (angle, aspect_ratio)


def extract_bud_orientations(
    masks_info: List[Tuple[np.ndarray, int, int]],
    min_aspect_ratio: float = 1.3,
    use_fit_ellipse: bool = True,
) -> List[BudOrientation]:
    """
    从芽点 mask 列表提取方向信息.

    Args:
        masks_info: list of (mask, offset_x, offset_y)
            mask: (H, W) bool 或 uint8 二值 mask (256x256 局部 patch)
            offset_x, offset_y: 局部 patch 在全局图像中的偏移
        min_aspect_ratio: 低于此值的芽点标记为 is_elongated=False
        use_fit_ellipse: True 用 cv2.fitEllipse, False 用二阶矩

    Returns:
        List[BudOrientation]: 与输入顺序一一对应, 提取失败的芽点 is_elongated=False
    """
    orientations: List[BudOrientation] = []

    for idx, (mask, offset_x, offset_y) in enumerate(masks_info):
        axis_angle = 0.0
        aspect_ratio = 1.0
        is_elongated = False
        ellipse_center = (0.0, 0.0)
        ellipse_axes = (0.0, 0.0)

        if use_fit_ellipse:
            result = _fit_ellipse_orientation(mask)
            if result is not None:
                angle_deg, aspect_ratio, center, axes = result
                axis_angle = np.deg2rad(angle_deg)
                # 标准化到 [0, π)
                if axis_angle >= np.pi:
                    axis_angle -= np.pi
                if axis_angle < 0:
                    axis_angle += np.pi
                is_elongated = aspect_ratio >= min_aspect_ratio
                ellipse_center = center
                ellipse_axes = axes
        else:
            result = _moments_orientation(mask)
            if result is not None:
                axis_angle, aspect_ratio = result
                is_elongated = aspect_ratio >= min_aspect_ratio

        centroid_local_x = np.mean(np.where(mask)[1]) if mask.any() else mask.shape[1] / 2
        centroid_local_y = np.mean(np.where(mask)[0]) if mask.any() else mask.shape[0] / 2
        centroid_global = (float(centroid_local_x + offset_x), float(centroid_local_y + offset_y))

        orientations.append(BudOrientation(
            bud_index=idx,
            axis_angle=axis_angle,
            aspect_ratio=aspect_ratio,
            centroid_global=centroid_global,
            is_elongated=is_elongated,
            ellipse_center=ellipse_center,
            ellipse_axes=ellipse_axes,
        ))

    return orientations


def orientation_stats(orientations: List[BudOrientation]) -> dict:
    """汇总芽点方向统计."""
    total = len(orientations)
    elongated = sum(1 for o in orientations if o.is_elongated)
    angles = [o.axis_angle for o in orientations if o.is_elongated]

    return {
        'total_buds': total,
        'elongated_count': elongated,
        'elongated_ratio': elongated / total if total > 0 else 0,
        'mean_aspect_ratio': np.mean([o.aspect_ratio for o in orientations]) if total > 0 else 0,
        'mean_angle_rad': np.mean(angles) if angles else 0,
        'mean_angle_deg': np.rad2deg(np.mean(angles)) if angles else 0,
    }


def _mask_axis_endpoints(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    binary = mask.astype(bool)
    if not np.any(binary):
        return None, None, "empty_mask"
    skeleton = skeletonize(binary).astype(np.uint8)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), dtype=np.int16)) - skeleton
    endpoints_yx = np.column_stack(np.where((skeleton > 0) & (degree == 1)))
    if len(endpoints_yx) >= 2:
        delta = endpoints_yx[:, None, :] - endpoints_yx[None, :, :]
        pair = np.unravel_index(int(np.argmax(np.sum(delta * delta, axis=2))), (len(endpoints_yx), len(endpoints_yx)))
        return endpoints_yx[pair[0]][::-1].astype(np.float32), endpoints_yx[pair[1]][::-1].astype(np.float32), "skeleton_endpoints"
    ys, xs = np.where(binary)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) < 2:
        return None, None, "insufficient_mask"
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    projection = centered @ axis
    return points[int(np.argmin(projection))], points[int(np.argmax(projection))], "pca_extrema"


def extract_directed_bud_orientations(
    masks_info: List[Tuple[np.ndarray, int, int]],
    skeleton_map: np.ndarray,
    scores: Optional[Sequence[float]] = None,
    min_confidence: float = 0.18,
    max_attachment_distance: float = 60.0,
) -> List[DirectedBudOrientation]:
    skeleton = (skeleton_map > 0).astype(np.uint8)
    if np.any(skeleton):
        distance_to_skeleton = cv2.distanceTransform((skeleton == 0).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        distance_to_skeleton = np.full(skeleton.shape, float(max(skeleton.shape)), dtype=np.float32)
    unsigned = extract_bud_orientations(masks_info)
    height, width = skeleton.shape
    directed: List[DirectedBudOrientation] = []
    for index, ((mask, offset_x, offset_y), orientation) in enumerate(zip(masks_info, unsigned)):
        endpoint_a, endpoint_b, method = _mask_axis_endpoints(mask)
        if endpoint_a is None or endpoint_b is None:
            directed.append(DirectedBudOrientation(
                bud_index=index,
                base_global=orientation.centroid_global,
                tip_global=orientation.centroid_global,
                centroid_global=orientation.centroid_global,
                vector_xy=(0.0, 0.0),
                confidence=0.0,
                aspect_ratio=float(orientation.aspect_ratio),
                attachment_distance=float("inf"),
                endpoint_distance_margin=0.0,
                method=method,
                is_reliable=False,
                is_latent_spur=False,
                failure_reason=method,
            ))
            continue
        endpoint_a = endpoint_a + np.asarray([offset_x, offset_y], dtype=np.float32)
        endpoint_b = endpoint_b + np.asarray([offset_x, offset_y], dtype=np.float32)

        def endpoint_distance(point: np.ndarray) -> float:
            x = int(np.clip(round(float(point[0])), 0, width - 1))
            y = int(np.clip(round(float(point[1])), 0, height - 1))
            return float(distance_to_skeleton[y, x])

        distance_a = endpoint_distance(endpoint_a)
        distance_b = endpoint_distance(endpoint_b)
        if distance_a <= distance_b:
            base, tip = endpoint_a, endpoint_b
        else:
            base, tip = endpoint_b, endpoint_a
        vector = tip - base
        vector_norm = float(np.linalg.norm(vector))
        unit_vector = vector / max(vector_norm, 1e-6)
        attachment_distance = min(distance_a, distance_b)
        distance_margin = abs(distance_a - distance_b)
        elongation_confidence = float(np.clip((orientation.aspect_ratio - 1.0) / 1.2, 0.0, 1.0))
        polarity_confidence = float(np.clip(distance_margin / max(vector_norm * 0.35, 2.0), 0.0, 1.0))
        attachment_confidence = float(np.exp(-attachment_distance / 36.0))
        detection_confidence = float(scores[index]) if scores is not None and index < len(scores) else 1.0
        confidence = detection_confidence * attachment_confidence * polarity_confidence * (0.3 + 0.7 * elongation_confidence)
        failure_reason = ""
        if attachment_distance > max_attachment_distance:
            failure_reason = "far_from_skeleton"
        elif vector_norm < 4.0:
            failure_reason = "short_axis"
        elif confidence < min_confidence:
            failure_reason = "ambiguous_polarity"
        directed.append(DirectedBudOrientation(
            bud_index=index,
            base_global=(float(base[0]), float(base[1])),
            tip_global=(float(tip[0]), float(tip[1])),
            centroid_global=orientation.centroid_global,
            vector_xy=(float(unit_vector[0]), float(unit_vector[1])),
            confidence=float(confidence),
            aspect_ratio=float(orientation.aspect_ratio),
            attachment_distance=float(attachment_distance),
            endpoint_distance_margin=float(distance_margin),
            method=method,
            is_reliable=not failure_reason,
            is_latent_spur=False,
            failure_reason=failure_reason,
        ))
    return directed
