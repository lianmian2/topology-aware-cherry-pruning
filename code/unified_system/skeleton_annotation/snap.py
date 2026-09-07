from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .state import AnnotationState, PointXY


def snap_to_existing_line(state: AnnotationState, click: PointXY,
                          target_group_id: Optional[str] = None,
                          max_dist: float = 15) -> Optional[Tuple[PointXY, str]]:
    cx, cy = click
    best: Optional[Tuple[PointXY, str, float]] = None
    for gid, group in state.groups.items():
        if target_group_id is not None and gid != target_group_id:
            continue
        if len(group.points) < 2:
            continue
        pts = group.points
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            snap_point, dist = _project_to_segment(cx, cy, x1, y1, x2, y2)
            if dist <= max_dist:
                if best is None or dist < best[2]:
                    best = (snap_point, gid, dist)
    if best is None:
        return None
    return (best[0], best[1])


def snap_to_existing_point(state: AnnotationState, click: PointXY,
                           max_dist: float = 12,
                           target_group_id: Optional[str] = None,
                           exclude_first_point: bool = False) -> Optional[Tuple[PointXY, str, int]]:
    cx, cy = click
    best: Optional[Tuple[PointXY, str, int, float]] = None
    for gid, group in state.groups.items():
        if target_group_id is not None and gid != target_group_id:
            continue
        for idx, (px, py) in enumerate(group.points):
            if exclude_first_point and idx == 0:
                continue
            dist = float(np.sqrt((cx - px) ** 2 + (cy - py) ** 2))
            if dist <= max_dist:
                if best is None or dist < best[3]:
                    best = ((px, py), gid, idx, dist)
    if best is None:
        return None
    return (best[0], best[1], best[2])


def _project_to_segment(px: int, py: int, x1: int, y1: int,
                        x2: int, y2: int) -> Tuple[PointXY, float]:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return ((x1, y1), float(np.sqrt((px - x1) ** 2 + (py - y1) ** 2)))
    t = max(0.0, min(1.0,
                     ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = int(round(x1 + t * dx))
    proj_y = int(round(y1 + t * dy))
    dist = float(np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2))
    return ((proj_x, proj_y), dist)
