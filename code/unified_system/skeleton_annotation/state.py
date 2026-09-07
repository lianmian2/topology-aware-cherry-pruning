from __future__ import annotations

from typing import Dict, List, Optional, Tuple

PointXY = Tuple[int, int]

GROUP_COLORS_BGR = [
    (50, 50, 255),
    (50, 255, 50),
    (255, 150, 50),
    (50, 255, 255),
    (255, 50, 255),
    (50, 150, 255),
    (255, 255, 50),
    (150, 50, 255),
    (100, 255, 150),
    (200, 150, 50),
]

GROUP_COLORS_HEX = [
    "#FF3232", "#32FF32", "#3296FF", "#32FFFF", "#FF32FF",
    "#FF9632", "#FFFF32", "#9632FF", "#64FF96", "#C89632",
]


class AnnotationGroup:
    def __init__(self, group_id: str, group_type: str,
                 color_bgr: Tuple[int, int, int], color_hex: str) -> None:
        self.group_id = group_id
        self.group_type = group_type
        self.color_bgr = color_bgr
        self.color_hex = color_hex
        self.points: List[PointXY] = []
        self.edges: List[Tuple[int, int]] = []
        self.fork_origin_group: Optional[str] = None
        self.current_anchor_idx: Optional[int] = None

    def add_point(self, point: PointXY) -> int:
        self.points.append(point)
        idx = len(self.points) - 1
        self.current_anchor_idx = idx
        return idx

    def add_edge(self, start_idx: int, end_idx: int) -> None:
        self.edges.append((start_idx, end_idx))

    def rebuild_linear_edges(self) -> None:
        self.edges = [(i, i + 1) for i in range(len(self.points) - 1)]
        if self.points:
            self.current_anchor_idx = len(self.points) - 1
        else:
            self.current_anchor_idx = None

    def has_geometry(self) -> bool:
        return bool(self.points) or bool(self.edges)


class AnnotationState:
    def __init__(self) -> None:
        self.groups: Dict[str, AnnotationGroup] = {}
        self.active_group_id: Optional[str] = None
        self.branch_count: int = 0
        self.color_index: int = 0

    def clear(self) -> None:
        self.groups.clear()
        self.active_group_id = None
        self.branch_count = 0
        self.color_index = 0

    def add_branch_group(self, color_index: Optional[int] = None) -> str:
        self.branch_count += 1
        gid = f"branch_{self.branch_count:02d}"
        idx = (color_index if color_index is not None else self.color_index) % len(GROUP_COLORS_BGR)
        if color_index is None:
            self.color_index += 1
        color_bgr = GROUP_COLORS_BGR[idx]
        color_hex = GROUP_COLORS_HEX[idx]
        self.groups[gid] = AnnotationGroup(gid, "branch", color_bgr, color_hex)
        return gid

    def ensure_trunk_group(self) -> str:
        if "trunk" not in self.groups:
            self.groups["trunk"] = AnnotationGroup("trunk", "trunk",
                                                    GROUP_COLORS_BGR[0],
                                                    GROUP_COLORS_HEX[0])
        return "trunk"

    def any_group_has_points(self, min_points: int = 2) -> bool:
        return any(len(g.points) >= min_points for g in self.groups.values())
