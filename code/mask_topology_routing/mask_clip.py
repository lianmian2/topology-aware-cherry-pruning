"""Mask 边界裁剪 — 剔除越出 CSNet 分割 mask 的骨架边.

核心思路:
  - EDT 量化每条边"越出mask多远"
  - 区分两类越界:
    a) mask 间隙桥接: 边穿过短黑色区域后回到 mask → 保留（骨架正确补全了分割断裂）
    b) 真正越界: 边穿出 mask 后不再回来 → 移除
  - 擦边（只越出1-2px）→ 保留并 snap
  - 清除无边的游离点

用法:
    from mask_topology_routing.mask_clip import clip_annotation_groups
    clipped_groups, stats = clip_annotation_groups(groups, mask)
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt


def _bresenham_line(x0, y0, x1, y1):
    pts = []
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy
    return pts


def _find_exit_segments(edt_vals, threshold):
    """找出 EDT > threshold 的连续段, 返回 [(start, end), ...]."""
    segments = []
    start = None
    for i, v in enumerate(edt_vals):
        if v > threshold:
            if start is None: start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(edt_vals) - 1))
    return segments


def clip_annotation_groups(
    groups: List[Dict],
    mask: np.ndarray,
    max_exit_distance: float = 3.0,
    max_gap: float = 20.0,
    snap_radius: int = 5,
) -> Tuple[List[Dict], Dict]:
    """剔除真正越出 mask 的边, 保留 mask 间隙桥接。

    判定逻辑（逐边）:
      1. 沿边扫描 EDT, 找出所有 EDT > max_exit_distance 的连续段
      2. 对每个越界段:
         - 若段两端都被 mask 包围（不接触端点）且段长 ≤ max_gap
           → mask 间隙桥接 → 保留
         - 否则 → 真正越界 → 整条边移除
      3. 无越界段的边: 若有点擦边(0 < EDT ≤ max_exit_distance) → snap 后保留

    Args:
        groups: annotation groups (points + edges)
        mask: (H, W) uint8 二值 mask (非零=前景)
        max_exit_distance: EDT 距离阈值 (px)
        max_gap: mask 间隙最大桥接长度 (px)
        snap_radius: 擦边点 snap 搜索半径

    Returns:
        (clipped_groups, stats)
    """
    H, W = mask.shape[:2]
    bin_mask = (mask > 0).astype(np.uint8)
    edt = distance_transform_edt(bin_mask == 0)

    stats = {
        "total_edges_before": 0,
        "total_points_before": 0,
        "edges_removed": 0,
        "edges_grazing_snapped": 0,
        "edges_gap_bridged": 0,
        "points_orphaned": 0,
    }

    clipped_groups = []
    for group in groups:
        pts = [(int(round(p[0])), int(round(p[1]))) for p in group.get("points", [])]
        edges = group.get("edges", [])
        stats["total_edges_before"] += len(edges)
        stats["total_points_before"] += len(pts)

        if not pts or not edges:
            clipped_groups.append({**group})
            continue

        edge_verdicts = []
        new_pts_append = []

        for e in edges:
            if len(e) != 2: continue
            s, d = int(e[0]), int(e[1])
            if s >= len(pts) or d >= len(pts): continue
            x0, y0 = pts[s]; x1, y1 = pts[d]

            x0_c = max(0, min(x0, W - 1)); y0_c = max(0, min(y0, H - 1))
            x1_c = max(0, min(x1, W - 1)); y1_c = max(0, min(y1, H - 1))

            line = _bresenham_line(x0_c, y0_c, x1_c, y1_c)
            if not line: continue

            edt_vals = np.array([
                edt[min(y, H - 1), min(x, W - 1)] for x, y in line
            ], dtype=np.float32)

            n = len(edt_vals)
            exit_segs = _find_exit_segments(edt_vals, max_exit_distance)

            # --- 判定 ---
            has_true_exit = False
            for seg_start, seg_end in exit_segs:
                seg_len = seg_end - seg_start + 1
                touches_start = (seg_start == 0)
                touches_end = (seg_end == n - 1)

                if (not touches_start) and (not touches_end) and (seg_len <= max_gap):
                    # 被 mask 两端包围的短间隙 → 桥接, 不视为越界
                    stats["edges_gap_bridged"] += 1
                else:
                    # 接触端点 或 太长 → 真正越界
                    has_true_exit = True
                    break

            if has_true_exit:
                stats["edges_removed"] += 1
                continue

            # --- 保留边, 处理擦边 ---
            max_edt = float(edt_vals.max())
            if max_edt > 0:
                stats["edges_grazing_snapped"] += 1

            dist_src = edt[min(y0_c, H - 1), min(x0_c, W - 1)]
            dist_dst = edt[min(y1_c, H - 1), min(x1_c, W - 1)]

            if dist_src > 0:
                nsx, nsy = _snap_to_mask(x0_c, y0_c, bin_mask, snap_radius)
                new_pts_append.append((nsx, nsy))
                src_idx = len(pts) + len(new_pts_append) - 1
            else:
                src_idx = s

            if dist_dst > 0:
                ndx, ndy = _snap_to_mask(x1_c, y1_c, bin_mask, snap_radius)
                new_pts_append.append((ndx, ndy))
                dst_idx = len(pts) + len(new_pts_append) - 1
            else:
                dst_idx = d

            edge_verdicts.append((src_idx, dst_idx))

        all_pts = list(pts) + list(new_pts_append)

        if not edge_verdicts:
            clipped_groups.append({**group, "points": [], "edges": []})
            continue

        used = set()
        for s, d in edge_verdicts:
            used.add(s); used.add(d)

        old_to_new = {}
        new_pts_list = []
        for old_idx in sorted(used):
            old_to_new[old_idx] = len(new_pts_list)
            new_pts_list.append(list(all_pts[old_idx]))

        new_edges = [[old_to_new[s], old_to_new[d]] for s, d in edge_verdicts]
        stats["points_orphaned"] += len(all_pts) - len(new_pts_list)

        clipped_groups.append({
            **{k: v for k, v in group.items() if k not in ("points", "edges")},
            "points": new_pts_list,
            "edges": new_edges,
        })

    return clipped_groups, stats


def _snap_to_mask(x, y, mask, radius):
    H, W = mask.shape[:2]
    x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
    y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
    roi = mask[y0:y1, x0:x1]
    fg = np.where(roi > 0)
    if len(fg[0]) == 0:
        return (max(0, min(x, W - 1)), max(0, min(y, H - 1)))
    dists = (fg[1] - (x - x0)) ** 2 + (fg[0] - (y - y0)) ** 2
    nearest = np.argmin(dists)
    return (int(x0 + fg[1][nearest]), int(y0 + fg[0][nearest]))
