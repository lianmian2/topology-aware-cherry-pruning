"""ROI 边界闭环的轻量几何判定，不加载检测或分割模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class BudSupport:
    index: int
    state: str
    mask_overlap: float
    mask_distance: float
    roi_edge_distance: float
    component_label: int


@dataclass(frozen=True)
class RoiEdgeTouch:
    component_label: int
    directions: tuple[str, ...]
    pixels: int


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    return (mask > 0).astype(np.uint8)


def _global_bud_mask(
    shape: tuple[int, int],
    box: Sequence[float],
    mask_info: Optional[tuple[np.ndarray, int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    if mask_info is not None:
        local, offset_x, offset_y = mask_info
        local = _binary(local)
        y1 = max(0, int(offset_y))
        x1 = max(0, int(offset_x))
        y2 = min(shape[0], y1 + local.shape[0])
        x2 = min(shape[1], x1 + local.shape[1])
        if y2 > y1 and x2 > x1:
            out[y1:y2, x1:x2] = local[: y2 - y1, : x2 - x1]
        return out
    x1, y1, x2, y2 = np.round(box).astype(int)
    x1, x2 = sorted((max(0, x1), min(shape[1], x2)))
    y1, y2 = sorted((max(0, y1), min(shape[0], y2)))
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = 1
    return out


def classify_bud_supports(
    boxes: np.ndarray,
    masks_info: Sequence[tuple[np.ndarray, int, int]],
    combined_mask: np.ndarray,
    roi_mask: np.ndarray,
    *,
    near_mask_radius: float = 24.0,
    roi_edge_band: float = 48.0,
    min_mask_overlap: float = 0.05,
) -> list[BudSupport]:
    """Classify buds as supported, border_candidate, or background_reject.

    A bud is not required to have its centre inside a woody mask. Instance-mask
    overlap and the nearest mask distance preserve buds protruding from branches.
    """
    combined = _binary(combined_mask)
    roi = _binary(roi_mask)
    if combined.shape != roi.shape:
        raise ValueError("combined_mask and roi_mask must have the same shape")
    _, labeled = cv2.connectedComponents(combined, connectivity=8)
    distance_to_tree = cv2.distanceTransform((1 - combined).astype(np.uint8), cv2.DIST_L2, 5)
    distance_inside_roi = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    distance_outside_roi = cv2.distanceTransform(1 - roi, cv2.DIST_L2, 5)
    distance_to_roi_edge = np.where(roi > 0, distance_inside_roi, distance_outside_roi)
    supports: list[BudSupport] = []
    for index, box in enumerate(boxes):
        mask_info = masks_info[index] if index < len(masks_info) else None
        bud_mask = _global_bud_mask(combined.shape, box, mask_info)
        pixels = int(bud_mask.sum())
        overlap = float(np.logical_and(bud_mask > 0, combined > 0).sum() / pixels) if pixels else 0.0
        if pixels:
            mask_distance = float(distance_to_tree[bud_mask > 0].min())
            edge_distance = float(distance_to_roi_edge[bud_mask > 0].min())
            labels = labeled[bud_mask > 0]
            labels = labels[labels > 0]
            component = int(np.bincount(labels).argmax()) if labels.size else -1
        else:
            cx = int(np.clip(round((box[0] + box[2]) / 2), 0, combined.shape[1] - 1))
            cy = int(np.clip(round((box[1] + box[3]) / 2), 0, combined.shape[0] - 1))
            mask_distance = float(distance_to_tree[cy, cx])
            edge_distance = float(distance_to_roi_edge[cy, cx])
            component = int(labeled[cy, cx])
        if overlap >= min_mask_overlap or mask_distance <= near_mask_radius:
            state = "supported"
        elif edge_distance <= roi_edge_band:
            state = "border_candidate"
        else:
            state = "background_reject"
        supports.append(BudSupport(index, state, overlap, mask_distance, edge_distance, component))
    return supports


def roi_edge_touches(
    roi_mask: np.ndarray,
    branch_mask: np.ndarray,
    *,
    edge_band: int = 24,
    min_component_pixels: int = 128,
) -> list[RoiEdgeTouch]:
    """Find non-trivial branch components that contact an ROI boundary band."""
    roi = _binary(roi_mask)
    branch = _binary(branch_mask) & roi
    if roi.shape != branch.shape:
        raise ValueError("roi_mask and branch_mask must have the same shape")
    distance = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    labels_count, labels = cv2.connectedComponents(branch.astype(np.uint8), connectivity=8)
    ys, xs = np.where(roi > 0)
    if not len(xs):
        return []
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    touches: list[RoiEdgeTouch] = []
    for component in range(1, labels_count):
        component_mask = labels == component
        pixels = int(component_mask.sum())
        contact = component_mask & (distance <= edge_band)
        if pixels < min_component_pixels or not contact.any():
            continue
        cy, cx = np.where(contact)
        directions = []
        if np.any(cx <= x1 + edge_band):
            directions.append("left")
        if np.any(cx >= x2 - edge_band):
            directions.append("right")
        if np.any(cy <= y1 + edge_band):
            directions.append("top")
        if np.any(cy >= y2 - edge_band):
            directions.append("bottom")
        if directions:
            touches.append(RoiEdgeTouch(component, tuple(directions), pixels))
    return touches


def directional_roi_extension(
    roi_mask: np.ndarray,
    directions: Iterable[str],
    *,
    width: int = 96,
) -> np.ndarray:
    """Expand only the requested sides of an ROI bounding extent."""
    roi = _binary(roi_mask)
    ys, xs = np.where(roi > 0)
    if not len(xs):
        return roi
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    expanded = roi.copy()
    requested = set(directions)
    if "left" in requested:
        expanded[y1:y2, max(0, x1 - width):x1] = 1
    if "right" in requested:
        expanded[y1:y2, x2:min(roi.shape[1], x2 + width)] = 1
    if "top" in requested:
        expanded[max(0, y1 - width):y1, x1:x2] = 1
    if "bottom" in requested:
        expanded[y2:min(roi.shape[0], y2 + width), x1:x2] = 1
    return expanded


def extension_has_connected_branch(
    base_roi: np.ndarray,
    base_branch: np.ndarray,
    expanded_branch: np.ndarray,
    *,
    connection_radius: int = 12,
    min_new_pixels: int = 64,
) -> bool:
    """Accept new segmentation only when it is connected to original branch support."""
    base = _binary(base_branch) & _binary(base_roi)
    expanded = _binary(expanded_branch)
    new_pixels = expanded & (1 - _binary(base_roi))
    if int(new_pixels.sum()) < min_new_pixels or not base.any():
        return False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (connection_radius * 2 + 1,) * 2)
    return bool(np.logical_and(cv2.dilate(base, kernel), new_pixels).any())


def extension_has_bud_evidence(
    supports: Sequence[BudSupport],
    scores: Sequence[float],
    *,
    min_count: int = 2,
    min_score: float = 0.5,
) -> bool:
    return sum(
        support.state == "border_candidate" and index < len(scores) and scores[index] >= min_score
        for index, support in enumerate(supports)
    ) >= min_count


def apply_roi_mask(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    if image.shape[:2] != roi_mask.shape:
        raise ValueError("image and roi_mask shapes do not match")
    out = np.zeros_like(image)
    out[roi_mask > 0] = image[roi_mask > 0]
    return out
