from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .state import AnnotationGroup, AnnotationState, GROUP_COLORS_HEX, GROUP_COLORS_BGR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_image_unicode(file_path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), flags)
    if data is None:
        raise RuntimeError(f"无法读取图像: {file_path}")
    return data


def collect_image_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            files.append(f)
    return files


def load_image(file_path: Path) -> np.ndarray:
    return _read_image_unicode(file_path, cv2.IMREAD_COLOR)


def load_mask(mask_path: Optional[str], target_h: int, target_w: int) -> Optional[np.ndarray]:
    if mask_path is None:
        return None
    try:
        mask = _read_image_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8) * 255
    except Exception:
        return None


def load_annotation(state: AnnotationState, image_folder: Path, stem: str) -> Tuple[bool, Dict]:
    json_path = image_folder / "skeleton_annotations" / f"{stem}_skeleton.json"
    if not json_path.exists():
        return False, {}
    try:
        with json_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        return False, {}

    state.clear()
    for g_data in data.get("groups", []):
        gid = g_data["group_id"]
        gtype = g_data["group_type"]
        color_hex = g_data.get("color_hex", GROUP_COLORS_HEX[0])
        color_bgr = _hex_to_bgr(color_hex)
        group = AnnotationGroup(gid, gtype, color_bgr, color_hex)
        group.points = [tuple(p) for p in g_data.get("points", [])]
        edges = [tuple(edge) for edge in g_data.get("edges", [])]
        if edges:
            group.edges = edges
        else:
            group.rebuild_linear_edges()
        group.fork_origin_group = g_data.get("fork_origin_group")
        current_anchor_idx = g_data.get("current_anchor_idx")
        if isinstance(current_anchor_idx, int) and 0 <= current_anchor_idx < len(group.points):
            group.current_anchor_idx = current_anchor_idx
        state.groups[gid] = group
        if gtype == "branch":
            try:
                num = int(gid.split("_")[-1])
                state.branch_count = max(state.branch_count, num)
            except (ValueError, IndexError):
                state.branch_count = max(state.branch_count, state.branch_count + 1)
    state.color_index = state.branch_count
    session = data.get("session", {})
    return True, session


def save_annotation(image_folder: Path, stem: str, image_bgr: np.ndarray,
                    height: int, width: int, image_path_str: str,
                    state: AnnotationState, session: Optional[Dict] = None) -> None:
    save_dir = image_folder / "skeleton_annotations"
    save_dir.mkdir(parents=True, exist_ok=True)

    annotation_data = {
        "image_path": image_path_str,
        "image_width": width,
        "image_height": height,
        "groups": [
            {
                "group_id": g.group_id,
                "group_type": g.group_type,
                "color_hex": g.color_hex,
                "points": [list(p) for p in g.points],
                "edges": [list(edge) for edge in g.edges],
                "fork_origin_group": g.fork_origin_group,
                "current_anchor_idx": g.current_anchor_idx,
            }
            for g in state.groups.values()
        ],
        "session": session or {},
    }
    json_path = save_dir / f"{stem}_skeleton.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(annotation_data, fp, indent=2, ensure_ascii=False)

    skeleton_mask = np.zeros((height, width), dtype=np.uint8)
    for group in state.groups.values():
        if len(group.points) < 1:
            continue
        edges = group.edges if group.edges else [(i, i + 1) for i in range(len(group.points) - 1)]
        for start_idx, end_idx in edges:
            if not (0 <= start_idx < len(group.points) and 0 <= end_idx < len(group.points)):
                continue
            cv2.line(skeleton_mask, group.points[start_idx], group.points[end_idx], 255,
                     thickness=2, lineType=cv2.LINE_AA)
        for i, (px, py) in enumerate(group.points):
            if i == 0 and group.group_type == "branch":
                cv2.drawMarker(skeleton_mask, (px, py), 255,
                               markerType=cv2.MARKER_DIAMOND,
                               markerSize=8, thickness=2, line_type=cv2.LINE_AA)
            else:
                cv2.circle(skeleton_mask, (px, py), 5, 255,
                           thickness=-1, lineType=cv2.LINE_AA)

    skeleton_path = save_dir / f"{stem}_skeleton_mask.png"
    success, encoded = cv2.imencode(".png", skeleton_mask)
    if success:
        encoded.tofile(str(skeleton_path))

    overlay_bgr = image_bgr.copy()
    for group in state.groups.values():
        if len(group.points) < 1:
            continue
        b, g, r = int(group.color_bgr[0]), int(group.color_bgr[1]), int(group.color_bgr[2])
        edges = group.edges if group.edges else [(i, i + 1) for i in range(len(group.points) - 1)]
        for start_idx, end_idx in edges:
            if not (0 <= start_idx < len(group.points) and 0 <= end_idx < len(group.points)):
                continue
            cv2.line(overlay_bgr, group.points[start_idx], group.points[end_idx], (b, g, r),
                     thickness=3, lineType=cv2.LINE_AA)
        for i, (px, py) in enumerate(group.points):
            if i == 0 and group.group_type == "branch":
                cv2.drawMarker(overlay_bgr, (px, py), (b, g, r),
                               markerType=cv2.MARKER_DIAMOND,
                               markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            else:
                cv2.circle(overlay_bgr, (px, py), 6, (b, g, r),
                           thickness=-1, lineType=cv2.LINE_AA)

    overlay_path = save_dir / f"{stem}_skeleton_overlay.png"
    success, encoded = cv2.imencode(".png", overlay_bgr)
    if success:
        encoded.tofile(str(overlay_path))


def _hex_to_bgr(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16)
    g_val = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g_val, r)
