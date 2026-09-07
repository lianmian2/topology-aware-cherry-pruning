from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from .state import AnnotationState, PointXY


class CanvasRenderer:
    def __init__(self, canvas, image_rgb: Optional[np.ndarray],
                 width: int, height: int,
                 branch_mask: Optional[np.ndarray],
                 show_mask: bool) -> None:
        self.canvas = canvas
        self.image_rgb = image_rgb
        self.width = width
        self.height = height
        self.branch_mask = branch_mask
        self.show_mask = show_mask
        self.photos = {}
        self.view_scale = 1.0
        self.view_offset_x = 0
        self.view_offset_y = 0
        self._hovered_image_point: Optional[Tuple[PointXY, str]] = None
        self._highlight_trunk: bool = False
        self._new_branch_waiting: bool = False

    def reset_view(self) -> None:
        self.view_scale = 1.0
        self.view_offset_x = 0
        self.view_offset_y = 0

    def canvas_to_image_xy(self, event_x: int, event_y: int) -> Optional[PointXY]:
        if self.image_rgb is None:
            return None
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return None
        base_scale = min(cw / self.width, ch / self.height) * 0.95
        ratio = base_scale * self.view_scale
        nw = int(self.width * ratio)
        nh = int(self.height * ratio)
        center_x = cw // 2 + self.view_offset_x
        center_y = ch // 2 + self.view_offset_y
        x0 = center_x - nw // 2
        y0 = center_y - nh // 2
        if not (x0 <= event_x < x0 + nw and y0 <= event_y < y0 + nh):
            return None
        ix = int(round((event_x - x0) / ratio))
        iy = int(round((event_y - y0) / ratio))
        ix = int(np.clip(ix, 0, self.width - 1))
        iy = int(np.clip(iy, 0, self.height - 1))
        return (ix, iy)

    def redraw(self, state: AnnotationState) -> None:
        if self.image_rgb is None:
            self.canvas.delete("all")
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="请点击「打开图片文件夹」加载图片", fill="#888888",
                font=("Microsoft YaHei", 14),
            )
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            self.canvas.after(100, lambda: self.redraw(state))
            return

        display = self.image_rgb.copy()

        if self.show_mask and self.branch_mask is not None:
            overlay = display.copy()
            overlay[self.branch_mask > 0] = (180, 255, 180)
            display = cv2.addWeighted(overlay, 0.28, display, 0.72, 0.0)

        self._draw_hints(display, state)
        self._draw_groups(display, state)

        base_scale = min(cw / self.width, ch / self.height) * 0.95
        ratio = base_scale * self.view_scale
        nw = max(1, int(self.width * ratio))
        nh = max(1, int(self.height * ratio))
        resized = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized)
        self.photos["main"] = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        center_x = cw // 2 + self.view_offset_x
        center_y = ch // 2 + self.view_offset_y
        self.canvas.create_image(center_x, center_y, image=self.photos["main"])

        self._draw_mode_hint(state)

    def _draw_hints(self, display: np.ndarray, state: AnnotationState) -> None:
        branch_active = (
            state.active_group_id is not None
            and state.active_group_id in state.groups
            and state.groups[state.active_group_id].group_type == "branch"
        )

        if self._new_branch_waiting:
            trunk = state.groups.get("trunk")
            if trunk and len(trunk.points) >= 2:
                for i in range(len(trunk.points) - 1):
                    cv2.line(display, trunk.points[i], trunk.points[i + 1],
                             (100, 255, 255), thickness=5, lineType=cv2.LINE_AA)

        if self._hovered_image_point is not None and (branch_active or self._new_branch_waiting):
            (hx, hy), _hgid = self._hovered_image_point
            cv2.circle(display, (hx, hy), 14, (0, 255, 255), thickness=3,
                       lineType=cv2.LINE_AA)
            cv2.circle(display, (hx, hy), 16, (0, 200, 200), thickness=1,
                       lineType=cv2.LINE_AA)

    def _draw_groups(self, display: np.ndarray, state: AnnotationState) -> None:
        for gid, group in state.groups.items():
            if len(group.points) < 1:
                continue
            b, g, r = int(group.color_bgr[0]), int(group.color_bgr[1]), int(group.color_bgr[2])
            color_rgb = (r, g, b)
            is_active = (gid == state.active_group_id)
            line_thickness = 3 if is_active else 2

            if group.edges:
                for start_idx, end_idx in group.edges:
                    if not (0 <= start_idx < len(group.points) and 0 <= end_idx < len(group.points)):
                        continue
                    cv2.line(display, group.points[start_idx], group.points[end_idx], color_rgb,
                             thickness=line_thickness, lineType=cv2.LINE_AA)
            else:
                for i in range(len(group.points) - 1):
                    cv2.line(display, group.points[i], group.points[i + 1], color_rgb,
                             thickness=line_thickness, lineType=cv2.LINE_AA)

            for i, (px, py) in enumerate(group.points):
                radius = 6 if is_active else 4
                if i == 0 and group.group_type == "branch":
                    cv2.drawMarker(display, (px, py), color_rgb,
                                   markerType=cv2.MARKER_DIAMOND,
                                   markerSize=radius + 4, thickness=2, line_type=cv2.LINE_AA)
                else:
                    cv2.circle(display, (px, py), radius, color_rgb,
                               thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(display, (px, py), radius + 1, (40, 40, 40),
                           thickness=1, lineType=cv2.LINE_AA)
                if is_active and group.current_anchor_idx == i:
                    cv2.circle(display, (px, py), radius + 6, (255, 255, 255),
                               thickness=2, lineType=cv2.LINE_AA)

    def _draw_mode_hint(self, state: AnnotationState) -> None:
        cw = self.canvas.winfo_width()
        if self._new_branch_waiting:
            self.canvas.create_text(
                cw // 2, 20,
                text="◇ 新建枝条 — 请在主干线段上点击起点（青色高亮）",
                fill="#00ffff", font=("Microsoft YaHei", 12, "bold"),
            )
        elif self._hovered_image_point is not None and state.active_group_id is not None:
            g = state.groups.get(state.active_group_id)
            if g and g.group_type == "branch":
                (hx, hy), hgid = self._hovered_image_point
                self.canvas.create_text(
                    cw // 2, 20,
                    text=f"◇ hover 节点 ({hx}, {hy}) 来自 '{hgid}' — 点击从此处分叉",
                    fill="#00ffff", font=("Microsoft YaHei", 12, "bold"),
                )
