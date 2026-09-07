"""
条带感知路由 — 使用条带分割结果修复主干断裂, 替代过度连接的桥接算法.

问题: _bridge_vertical_gaps / _bridge_trunk_gaps_on_skeleton 在mask断裂处过度搜索连接,
      当枝条恰巧弯曲经过断裂处时会被误认为主干 (如 tree_009).

方案: 只在条带(标定带)覆盖处尝试连接 — 因为条带只会绑在主干上,
      条带位置 = 主干必经之处.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize


def _trace_endpoint_tangent(
    skeleton: np.ndarray,
    endpoint_rc: Tuple[int, int],
    max_steps: int = 24,
) -> np.ndarray:
    path = [endpoint_rc]
    previous = None
    current = endpoint_rc
    for _ in range(max_steps):
        row, col = current
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                candidate = (row + dr, col + dc)
                if not (0 <= candidate[0] < skeleton.shape[0] and 0 <= candidate[1] < skeleton.shape[1]):
                    continue
                if skeleton[candidate] and candidate != previous:
                    neighbors.append(candidate)
        if not neighbors:
            break
        next_point = max(
            neighbors,
            key=lambda point: (point[0] - path[0][0]) ** 2 + (point[1] - path[0][1]) ** 2,
        )
        path.append(next_point)
        previous, current = current, next_point
        if len(neighbors) > 1:
            break
    if len(path) < 2:
        return np.zeros((2,), dtype=np.float32)
    vector = np.asarray([path[-1][1] - path[0][1], path[-1][0] - path[0][0]], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.zeros((2,), dtype=np.float32)


def _collect_boundary_approaches(
    skeleton: np.ndarray,
    tape_dilated: np.ndarray,
    max_distance: float = 20.0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    distance_to_tape = distance_transform_edt(tape_dilated == 0)
    near_boundary = ((skeleton > 0) & (distance_to_tape <= float(max_distance))).astype(np.uint8)
    label_count, labels = cv2.connectedComponents(near_boundary, connectivity=8)
    points: List[Tuple[int, int]] = []
    tangents: List[np.ndarray] = []
    for label in range(1, label_count):
        coords = np.column_stack(np.where(labels == label))
        if len(coords) < 3:
            continue
        distances = distance_to_tape[coords[:, 0], coords[:, 1]]
        boundary_pool = coords[distances <= float(distances.min()) + 2.0]
        point_rc = tuple(map(int, np.rint(np.median(boundary_pool, axis=0))))
        farthest_rc = coords[int(np.argmax(np.sum((coords - np.asarray(point_rc)[None, :]) ** 2, axis=1)))]
        tangent = np.asarray(
            [farthest_rc[1] - point_rc[1], farthest_rc[0] - point_rc[0]],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-6:
            continue
        points.append(point_rc)
        tangents.append(tangent / norm)
    return np.asarray(points, dtype=np.int32), tangents


def _bridge_oriented_gaps(
    mask: np.ndarray,
    tape_binary: np.ndarray,
    tape_dilated: np.ndarray,
) -> np.ndarray:
    tape_coords_rc = np.column_stack(np.where(tape_binary > 0))
    if len(tape_coords_rc) < 20:
        return mask.copy()

    tape_points_xy = tape_coords_rc[:, ::-1].astype(np.float32)
    tape_center = np.mean(tape_points_xy, axis=0)
    covariance = np.cov(tape_points_xy - tape_center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    major_axis /= max(float(np.linalg.norm(major_axis)), 1e-6)
    normal_axis = np.asarray([-major_axis[1], major_axis[0]], dtype=np.float32)

    outside_mask = ((mask > 0) & (tape_dilated == 0)).astype(np.uint8)
    skeleton = skeletonize(outside_mask > 0).astype(np.uint8)
    degree = convolve(skeleton, np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    endpoint_coords_rc = np.column_stack(np.where((skeleton > 0) & (degree == 1)))
    distance_to_tape = distance_transform_edt(tape_dilated == 0)
    endpoint_coords_rc = np.asarray(
        [point for point in endpoint_coords_rc if distance_to_tape[tuple(point)] <= 16.0],
        dtype=np.int32,
    )
    if len(endpoint_coords_rc) < 2:
        endpoint_coords_rc, tangents = _collect_boundary_approaches(skeleton, tape_dilated)
    else:
        tangents = [_trace_endpoint_tangent(skeleton, tuple(map(int, point))) for point in endpoint_coords_rc]
    if len(endpoint_coords_rc) < 2:
        return mask.copy()

    endpoint_points_xy = endpoint_coords_rc[:, ::-1].astype(np.float32)
    signed_normal = (endpoint_points_xy - tape_center[None, :]) @ normal_axis
    side_a = np.where(signed_normal < 0)[0]
    side_b = np.where(signed_normal > 0)[0]
    if len(side_a) == 0 or len(side_b) == 0:
        return mask.copy()

    tape_normal_span = float(np.ptp((tape_points_xy - tape_center[None, :]) @ normal_axis))
    max_gap = max(80.0, tape_normal_span * 2.2)
    max_shift = max(80.0, tape_normal_span * 1.6)
    invalid_cost = 1e6
    costs = np.full((len(side_a), len(side_b)), invalid_cost, dtype=np.float32)

    for row_idx, endpoint_a_idx in enumerate(side_a):
        point_a = endpoint_points_xy[endpoint_a_idx]
        outward_a = -tangents[endpoint_a_idx]
        for col_idx, endpoint_b_idx in enumerate(side_b):
            point_b = endpoint_points_xy[endpoint_b_idx]
            delta = point_b - point_a
            gap = float(np.linalg.norm(delta))
            if gap < 1e-6 or gap > max_gap:
                continue
            direction = delta / gap
            outward_b = -tangents[endpoint_b_idx]
            alignment_a = float(np.dot(outward_a, direction))
            alignment_b = float(np.dot(outward_b, -direction))
            tangent_shift = abs(float(np.dot(delta, major_axis)))
            if alignment_a < 0.45 or alignment_b < 0.45 or tangent_shift > max_shift:
                continue
            samples = np.linspace(point_a, point_b, num=max(12, int(round(gap / 4.0))))
            sample_x = np.clip(np.rint(samples[:, 0]).astype(np.int32), 0, mask.shape[1] - 1)
            sample_y = np.clip(np.rint(samples[:, 1]).astype(np.int32), 0, mask.shape[0] - 1)
            tape_support = float(np.mean(tape_dilated[sample_y, sample_x] > 0))
            if tape_support < 0.65:
                continue
            costs[row_idx, col_idx] = gap + 1.5 * tangent_shift + 90.0 * (2.0 - alignment_a - alignment_b)

    row_indices, col_indices = linear_sum_assignment(costs)
    if not any(costs[row_idx, col_idx] < invalid_cost for row_idx, col_idx in zip(row_indices, col_indices)):
        endpoint_coords_rc, tangents = _collect_boundary_approaches(skeleton, tape_dilated)
        if len(endpoint_coords_rc) < 2:
            return mask.copy()
        endpoint_points_xy = endpoint_coords_rc[:, ::-1].astype(np.float32)
        signed_normal = (endpoint_points_xy - tape_center[None, :]) @ normal_axis
        side_a = np.where(signed_normal < 0)[0]
        side_b = np.where(signed_normal > 0)[0]
        if len(side_a) == 0 or len(side_b) == 0:
            return mask.copy()
        max_gap = max(160.0, tape_normal_span * 2.5)
        max_shift = max(180.0, tape_normal_span * 1.6)
        costs = np.full((len(side_a), len(side_b)), invalid_cost, dtype=np.float32)
        for row_idx, endpoint_a_idx in enumerate(side_a):
            point_a = endpoint_points_xy[endpoint_a_idx]
            outward_a = -tangents[endpoint_a_idx]
            for col_idx, endpoint_b_idx in enumerate(side_b):
                point_b = endpoint_points_xy[endpoint_b_idx]
                delta = point_b - point_a
                gap = float(np.linalg.norm(delta))
                if gap < 1e-6 or gap > max_gap:
                    continue
                direction = delta / gap
                outward_b = -tangents[endpoint_b_idx]
                alignment_a = float(np.dot(outward_a, direction))
                alignment_b = float(np.dot(outward_b, -direction))
                tangent_shift = abs(float(np.dot(delta, major_axis)))
                if alignment_a < 0.20 or alignment_b < 0.20 or tangent_shift > max_shift:
                    continue
                samples = np.linspace(point_a, point_b, num=max(12, int(round(gap / 4.0))))
                sample_x = np.clip(np.rint(samples[:, 0]).astype(np.int32), 0, mask.shape[1] - 1)
                sample_y = np.clip(np.rint(samples[:, 1]).astype(np.int32), 0, mask.shape[0] - 1)
                tape_support = float(np.mean(tape_dilated[sample_y, sample_x] > 0))
                if tape_support < 0.55:
                    continue
                costs[row_idx, col_idx] = gap + 1.8 * tangent_shift + 70.0 * (2.0 - alignment_a - alignment_b)
        row_indices, col_indices = linear_sum_assignment(costs)
    result = mask.copy()
    mask_radius = distance_transform_edt(mask > 0)
    for row_idx, col_idx in zip(row_indices, col_indices):
        if costs[row_idx, col_idx] >= invalid_cost:
            continue
        endpoint_a_idx = int(side_a[row_idx])
        endpoint_b_idx = int(side_b[col_idx])
        point_a = tuple(map(int, np.rint(endpoint_points_xy[endpoint_a_idx])))
        point_b = tuple(map(int, np.rint(endpoint_points_xy[endpoint_b_idx])))
        radius_a = float(mask_radius[tuple(endpoint_coords_rc[endpoint_a_idx])])
        radius_b = float(mask_radius[tuple(endpoint_coords_rc[endpoint_b_idx])])
        thickness = int(np.clip(round(max(2.0, min(radius_a, radius_b) * 1.5)), 2, 15))
        cv2.line(result, point_a, point_b, 1, thickness=thickness, lineType=cv2.LINE_8)
    return result


def _bridge_root_trunk_gap(
    mask: np.ndarray,
    tape_dilated: np.ndarray,
) -> np.ndarray:
    tape_coords_rc = np.column_stack(np.where(tape_dilated > 0))
    if len(tape_coords_rc) < 20:
        return mask.copy()
    tape_points_xy = tape_coords_rc[:, ::-1].astype(np.float32)
    tape_center = np.mean(tape_points_xy, axis=0)
    covariance = np.cov(tape_points_xy - tape_center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    major_axis /= max(float(np.linalg.norm(major_axis)), 1e-6)
    normal_axis = np.asarray([-major_axis[1], major_axis[0]], dtype=np.float32)

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    component_ids = list(range(1, label_count))
    candidates = [
        label
        for label in component_ids
        if int(stats[label, cv2.CC_STAT_AREA]) >= 80
        and int(stats[label, cv2.CC_STAT_HEIGHT]) >= 40
    ]
    if not candidates:
        return mask.copy()
    root_label = max(
        candidates,
        key=lambda label: (
            int(stats[label, cv2.CC_STAT_TOP]) + int(stats[label, cv2.CC_STAT_HEIGHT]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
            int(stats[label, cv2.CC_STAT_AREA]),
        ),
    )
    root_top = int(stats[root_label, cv2.CC_STAT_TOP])
    root_height = int(stats[root_label, cv2.CC_STAT_HEIGHT])
    root_bottom = root_top + root_height
    bottom_start = max(root_top, root_bottom - max(8, int(round(root_height * 0.08))))
    bottom_coords = np.column_stack(np.where(labels[bottom_start:root_bottom] == root_label))
    if len(bottom_coords) == 0:
        return mask.copy()
    root_x = float(np.median(bottom_coords[:, 1]))

    outside_mask = ((mask > 0) & (tape_dilated == 0)).astype(np.uint8)
    skeleton = skeletonize(outside_mask > 0).astype(np.uint8)
    degree = convolve(skeleton, np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    endpoint_coords, _ = _collect_boundary_approaches(
        skeleton,
        tape_dilated,
        max_distance=48.0,
    )
    if len(endpoint_coords) < 2:
        endpoint_coords = np.column_stack(np.where((skeleton > 0) & (degree == 1)))
        distance_to_tape = distance_transform_edt(tape_dilated == 0)
        endpoint_coords = np.asarray(
            [point for point in endpoint_coords if distance_to_tape[tuple(point)] <= 48.0],
            dtype=np.int32,
        )
    if len(endpoint_coords) < 2:
        return mask.copy()
    endpoint_points_xy = endpoint_coords[:, ::-1].astype(np.float32)
    signed_normal = (endpoint_points_xy - tape_center[None, :]) @ normal_axis
    side_a = np.where(signed_normal < 0)[0]
    side_b = np.where(signed_normal > 0)[0]
    if len(side_a) == 0 or len(side_b) == 0:
        return mask.copy()

    tape_normal_span = float(np.ptp((tape_points_xy - tape_center[None, :]) @ normal_axis))
    max_dx = max(96.0, tape_normal_span * 0.45)
    valid_pairs = [
        (int(index_a), int(index_b))
        for index_a in side_a
        for index_b in side_b
        if abs(float(endpoint_points_xy[index_a, 0] - endpoint_points_xy[index_b, 0])) <= max_dx
    ]
    if not valid_pairs:
        endpoint_coords, _ = _collect_boundary_approaches(skeleton, tape_dilated)
        if len(endpoint_coords) < 2:
            return mask.copy()
        endpoint_points_xy = endpoint_coords[:, ::-1].astype(np.float32)
        signed_normal = (endpoint_points_xy - tape_center[None, :]) @ normal_axis
        side_a = np.where(signed_normal < 0)[0]
        side_b = np.where(signed_normal > 0)[0]
        max_dx = max(160.0, tape_normal_span * 1.2)
        valid_pairs = [
            (int(index_a), int(index_b))
            for index_a in side_a
            for index_b in side_b
            if abs(float(endpoint_points_xy[index_a, 0] - endpoint_points_xy[index_b, 0])) <= max_dx
        ]
    if not valid_pairs:
        return mask.copy()
    mask_radius = distance_transform_edt(mask > 0)
    index_a, index_b = min(
        valid_pairs,
        key=lambda pair: (
            abs(0.5 * (endpoint_points_xy[pair[0], 0] + endpoint_points_xy[pair[1], 0]) - root_x) * 3.0
            + abs(endpoint_points_xy[pair[0], 0] - endpoint_points_xy[pair[1], 0]) * 2.0
            + np.linalg.norm(endpoint_points_xy[pair[0]] - endpoint_points_xy[pair[1]])
            - 12.0 * (
                float(mask_radius[tuple(endpoint_coords[pair[0]])])
                + float(mask_radius[tuple(endpoint_coords[pair[1]])])
            )
        ),
    )
    point_a = tuple(map(int, endpoint_coords[index_a]))
    point_b = tuple(map(int, endpoint_coords[index_b]))

    def recenter_on_mask_run(point: Tuple[int, int]) -> Tuple[Tuple[int, int], int]:
        row, col = point
        row_values = mask[row] > 0
        left = col
        while left > 0 and row_values[left - 1]:
            left -= 1
        right = col
        while right + 1 < mask.shape[1] and row_values[right + 1]:
            right += 1
        return (row, int(round(0.5 * (left + right)))), right - left + 1

    point_a, width_a = recenter_on_mask_run(point_a)
    point_b, width_b = recenter_on_mask_run(point_b)
    thickness = int(np.clip(round(min(width_a, width_b) * 0.9), 5, 121))
    bridge = np.zeros_like(mask, dtype=np.uint8)
    cv2.line(
        bridge,
        (int(point_a[1]), int(point_a[0])),
        (int(point_b[1]), int(point_b[0])),
        1,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )
    result = mask.copy()
    result[bridge > 0] = 1
    return result


def merge_tape_into_mask(
    mask: np.ndarray,
    tape_mask: np.ndarray,
    dilate_kernel_size: int = 31,
) -> np.ndarray:
    """将条带mask膨胀后合并入分割mask, 填充因条带遮挡导致的主干断裂.

    策略: 在条带两侧提取骨架端点, 根据局部切线、距离和条带区域支持度
    进行一对一匹配, 只添加通过条带的细连接.

    Args:
        mask: (H, W) uint8 原始分割mask
        tape_mask: (H, W) uint8 条带检测mask (0-255)
        dilate_kernel_size: 膨胀核大小 (椭圆), 需足够大以覆盖mask断裂的上下边界

    Returns:
        (H, W) uint8 增强后的mask
    """
    if tape_mask is None or not np.any(tape_mask):
        return mask

    mask_u8 = mask.astype(np.uint8)

    # 膨胀条带mask, 确保覆盖遮挡区域的上方和下方mask边界
    tape_binary = (tape_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    tape_dilated = cv2.dilate(tape_binary, kernel)

    result = _bridge_root_trunk_gap(mask_u8, tape_dilated)
    return _bridge_oriented_gaps(result, tape_binary, tape_dilated)


def compute_tape_guided_cost_adjustment(
    cost: np.ndarray,
    mask: np.ndarray,
    tape_mask: np.ndarray,
    dt_norm: np.ndarray,
    reduction: float = 0.15,
) -> np.ndarray:
    """在条带区域降低routing cost, 鼓励路径经过条带位置.

    条带区域的主干是确信的, 即使在原始mask中不可见.
    降低cost使得Dijkstra路由更倾向于穿过条带区域.

    Args:
        cost: (H, W) float32 原始cost map
        mask: (H, W) bool 分割mask
        tape_mask: (H, W) uint8 条带mask
        dt_norm: (H, W) float32 归一化距离变换
        reduction: 条带区域cost降低比例

    Returns:
        (H, W) float32 调整后的cost
    """
    if tape_mask is None or not np.any(tape_mask):
        return cost

    tape_binary = (tape_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tape_dilated = cv2.dilate(tape_binary, kernel)

    # 条带区域内降低cost
    adjustment = np.ones_like(cost, dtype=np.float32)
    adjustment[tape_dilated > 0] = 1.0 - reduction

    return (cost * adjustment).astype(np.float32)
