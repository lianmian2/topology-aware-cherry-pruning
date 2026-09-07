"""
Mask 拓扑路由工具函数.

包含:
- 路径解析、分割接口适配
- 单 Combined Mask 的几何骨架恢复管线
- 可视化与结果导出
- 通用读写工具
"""

from __future__ import annotations

import json
import importlib.util
import os
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import networkx as nx
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array
from skimage.morphology import skeletonize

# ---- 路径常量 ----
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "01_data" / "01_raw" / "final_data"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "01_data" / "03_processed"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "03_models" / "mask_topology_routing"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "04_results" / "mask_topology_routing"
DEFAULT_JUNCTION_PRIOR_PATH = CURRENT_DIR / "gt_junction_priors.json"
GUI_ROOT = PROJECT_ROOT / "07_graphical_interface" / "unified_system"
CODE_ROOT = PROJECT_ROOT / "02_code"
MODEL_ROOT = PROJECT_ROOT / "03_models"


def _first_existing_path(*candidates: os.PathLike) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("未找到可用文件，候选路径如下:\n" + "\n".join(map(str, candidates)))


def _load_module_from_path(module_name: str, module_path: os.PathLike):
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从路径加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(
    annotation_dir: Optional[str] = None,
    raw_root: Optional[str] = None,
    processed_root: Optional[str] = None,
    model_root: Optional[str] = None,
    result_root: Optional[str] = None,
) -> Dict[str, Path]:
    return {
        "annotation_dir": Path(annotation_dir) if annotation_dir else (PROJECT_ROOT / "01_data" / "02_annotated" / "skeleton_annotation"),
        "raw_root": Path(raw_root) if raw_root else DEFAULT_RAW_ROOT,
        "processed_root": Path(processed_root) if processed_root else DEFAULT_PROCESSED_ROOT,
        "model_root": Path(model_root) if model_root else DEFAULT_MODEL_ROOT,
        "result_root": Path(result_root) if result_root else DEFAULT_RESULT_ROOT,
    }


def ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict, path: os.PathLike) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: os.PathLike) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_image_rgb(path: os.PathLike) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================================
# 核心后处理数据结构
# ============================================================================

@dataclass
class PredictionResult:
    points: List[Tuple[int, int]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    node_types: List[str] = field(default_factory=list)
    graph: nx.Graph = field(default_factory=nx.Graph)
    trunk_path: List[int] = field(default_factory=list)
    trunk_node_prob: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    branch_junction_prob: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    branch_endpoint_prob: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    trunk_edge_prob: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    branch_edge_prob: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    trunk_thin: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    branch_thin: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    routing_stats: dict = field(default_factory=dict)
    cost_matrix: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    root_point: Tuple[int, int] = (-1, -1)
    trunk_line: List[Tuple[int, int]] = field(default_factory=list)
    branch_lines: List[List[Tuple[int, int]]] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    dt_map: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    skeleton_map: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    trunk_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    endpoint_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    raw_endpoints: List[Tuple[int, int]] = field(default_factory=list)
    filtered_endpoints: List[Tuple[int, int]] = field(default_factory=list)
    annotation_groups: List[Dict] = field(default_factory=list)


# ============================================================================
# SegmentationAdapter
# ============================================================================

class SegmentationAdapter:
    def __init__(self, use_real_models: bool = True, allow_mock_fallback: bool = False, device: Optional[str] = None):
        self.use_real_models = use_real_models
        self.allow_mock_fallback = allow_mock_fallback
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._ready = False
        self._fallback_reason = ""
        self._roi_model = None
        self._branch_model = None
        self._apply_roi_filter = None
        self._run_branch_segmentation = None

    def _load_roi_model(self):
        if self._roi_model is not None:
            return self._roi_model
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules
        roi_config = CODE_ROOT / "02_models" / "roi_locator_V2" / "configs" / "mask_rcnn_r50_fpn_roi_v2.py"
        roi_checkpoint = _first_existing_path(
            MODEL_ROOT / "roi" / "epoch_12.pth", MODEL_ROOT / "roi" / "epoch_11.pth", MODEL_ROOT / "roi" / "epoch_10.pth",
        )
        register_all_modules()
        self._roi_model = init_detector(str(roi_config), str(roi_checkpoint), device=self.device)
        self._roi_model.eval()
        return self._roi_model

    def _load_branch_model(self):
        if self._branch_model is not None:
            return self._branch_model
        branch_model_module = _load_module_from_path("branch_seg_csnet_v2_model", CODE_ROOT / "02_models" / "branch_seg_csnet_v2" / "model.py")
        CSNet = branch_model_module.CSNet
        checkpoint_path = _first_existing_path(
            MODEL_ROOT / "branch_segmentation" / "best_dice_model.pth",
            MODEL_ROOT / "branch_segmentation" / "best_iou_model.pth",
            MODEL_ROOT / "branch_segmentation" / "best_loss_model.pth",
        )
        self._branch_model = CSNet(in_channels=3, n_classes=1)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self._branch_model.load_state_dict(state_dict)
        self._branch_model.eval().to(self.device)
        return self._branch_model

    def _lazy_init(self) -> None:
        if self._ready or not self.use_real_models:
            return
        try:
            logic_roi_module = _load_module_from_path("unified_logic_roi", GUI_ROOT / "logic_roi.py")
            logic_branch_module = _load_module_from_path("unified_logic_branch", GUI_ROOT / "logic_branch.py")
            self._apply_roi_filter = logic_roi_module.apply_roi_filter
            self._run_branch_segmentation = logic_branch_module.run_branch_segmentation
            self._load_roi_model()
            self._load_branch_model()
            self._ready = True
        except Exception as exc:
            self._ready = False
            self._fallback_reason = str(exc)
            if self.allow_mock_fallback:
                self.use_real_models = False
                warnings.warn(f"真实枝条分割接口初始化失败，已回退到全1掩码。失败原因: {exc}")
            else:
                raise RuntimeError(f"真实枝条分割接口初始化失败。失败原因: {exc}") from exc

    def describe_mode(self) -> str:
        if self.use_real_models and self._ready:
            return "real_two_stage"
        if not self.use_real_models:
            return "simulated_all_one" if self.allow_mock_fallback else "simulated_explicit"
        return "uninitialized"

    def predict_mask(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_real_models:
            mask = np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255
            return mask, image_rgb.copy()
        self._lazy_init()
        roi_model = self._load_roi_model()
        branch_model = self._load_branch_model()
        _, roi_filtered_rgb = self._apply_roi_filter(roi_model, image_rgb)
        branch_mask = self._run_branch_segmentation(branch_model, roi_filtered_rgb, device=self.device)
        return branch_mask.astype(np.uint8), roi_filtered_rgb


# ============================================================================
# 模型保存/加载
# ============================================================================

def save_checkpoint(
    checkpoint_path: os.PathLike, model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, scheduler: Optional,
    epoch: int, best_score: float, extra: Optional[Dict] = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        checkpoint["extra"] = extra
    torch.save(checkpoint, str(checkpoint_path))


def load_checkpoint(
    checkpoint_path: os.PathLike,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional = None,
    map_location: str = "cpu",
) -> Dict:
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    if model is not None and checkpoint.get("model_state_dict") is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


# ============================================================================
# 纯几何后处理: 主干折线 + Dijkstra 寻根路由
# ============================================================================

def save_image_rgb(image: np.ndarray, image_path: os.PathLike) -> None:
    image_path = Path(image_path)
    ensure_dir(image_path.parent)
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_mask(mask: np.ndarray, mask_path: os.PathLike) -> None:
    mask_path = Path(mask_path)
    ensure_dir(mask_path.parent)
    cv2.imwrite(str(mask_path), mask.astype(np.uint8))


def parse_annotation_stem(annotation_name: str) -> Tuple[str, str, str]:
    stem = Path(annotation_name).stem.replace("_skeleton", "")
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"无法从文件名解析树编号和视角: {annotation_name}")
    tree_id = "_".join(parts[:2])
    status = parts[2]
    view_id = "_".join(parts[3:5])
    return tree_id, status, view_id


def resolve_image_path(annotation_path: os.PathLike, raw_root: os.PathLike = DEFAULT_RAW_ROOT) -> Path:
    annotation = load_json(annotation_path)
    raw_root = Path(raw_root)
    old_path = annotation.get("image_path")
    if old_path:
        basename = Path(old_path).name
        if basename.startswith("tree_") and basename.endswith(".jpg"):
            tree_id, status, view_id = parse_annotation_stem(Path(annotation_path).name)
            candidate = raw_root / tree_id / status / f"{view_id}.jpg"
            if candidate.exists():
                return candidate
    tree_id, status, view_id = parse_annotation_stem(Path(annotation_path).name)
    candidate = raw_root / tree_id / status / f"{view_id}.jpg"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"未找到与标注对应的原始图像: {annotation_path}\n预期路径: {candidate}")


def dynamic_batch_size(default_batch_size: int = 8) -> int:
    if not torch.cuda.is_available():
        return 1
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gb >= 20:
        return max(default_batch_size, 12)
    elif total_gb >= 10:
        return default_batch_size
    else:
        return max(default_batch_size // 2, 2)


def thin_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """细化为 1px 宽骨架 (用于 visualizer 兼容)."""
    mask = (np.asarray(binary_mask) > 0)
    if not np.any(mask):
        return np.zeros_like(np.asarray(binary_mask), dtype=np.uint8)
    return skeletonize(mask).astype(np.uint8) * 255


def _neighbor_offsets() -> List[Tuple[int, int]]:
    return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _normalize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(binary_mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8)


def _compute_degree_map(binary_skeleton: np.ndarray) -> np.ndarray:
    skeleton = (np.asarray(binary_skeleton) > 0).astype(np.uint8)
    degree = np.zeros_like(skeleton, dtype=np.uint8)
    for dr, dc in _neighbor_offsets():
        shifted = np.zeros_like(skeleton, dtype=np.uint8)
        r0_src = max(-dr, 0)
        r1_src = skeleton.shape[0] - max(dr, 0)
        c0_src = max(-dc, 0)
        c1_src = skeleton.shape[1] - max(dc, 0)
        r0_dst = max(dr, 0)
        r1_dst = r0_dst + (r1_src - r0_src)
        c0_dst = max(dc, 0)
        c1_dst = c0_dst + (c1_src - c0_src)
        shifted[r0_dst:r1_dst, c0_dst:c1_dst] = skeleton[r0_src:r1_src, c0_src:c1_src]
        degree += shifted
    degree *= skeleton
    return degree


def _neighbors_of(point_rc: Tuple[int, int], binary_skeleton: np.ndarray) -> List[Tuple[int, int]]:
    r, c = point_rc
    neighbors: List[Tuple[int, int]] = []
    for dr, dc in _neighbor_offsets():
        nr, nc = r + dr, c + dc
        if 0 <= nr < binary_skeleton.shape[0] and 0 <= nc < binary_skeleton.shape[1] and binary_skeleton[nr, nc] > 0:
            neighbors.append((nr, nc))
    return neighbors


def _prune_short_branches(binary_skeleton: np.ndarray, min_length: int = 10) -> np.ndarray:
    skeleton = (np.asarray(binary_skeleton) > 0).astype(np.uint8)
    if min_length <= 1:
        return skeleton

    while True:
        degree = _compute_degree_map(skeleton)
        endpoints = list(zip(*np.where((skeleton > 0) & (degree == 1))))
        removed_any = False
        for start in endpoints:
            if skeleton[start] == 0:
                continue
            path = [start]
            prev = None
            current = start
            while True:
                neighbors = [point for point in _neighbors_of(current, skeleton) if point != prev]
                if len(neighbors) != 1:
                    break
                nxt = neighbors[0]
                path.append(nxt)
                prev, current = current, nxt
                if int(_compute_degree_map(skeleton)[current]) != 2:
                    break
            end_degree = int(_compute_degree_map(skeleton)[current])
            if len(path) < min_length and end_degree >= 3:
                for point in path[:-1]:
                    skeleton[point] = 0
                    removed_any = True
        if not removed_any:
            break
    return skeleton


def _thin_and_extract_nodes(binary_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    skeleton = (thin_binary_mask(binary_mask) > 0).astype(np.uint8)
    degree_map = _compute_degree_map(skeleton)
    endpoints_rc = list(zip(*np.where((skeleton > 0) & (degree_map == 1))))
    junctions_rc = list(zip(*np.where((skeleton > 0) & (degree_map >= 3))))
    endpoints_xy = [(int(c), int(r)) for r, c in endpoints_rc]
    junctions_xy = [(int(c), int(r)) for r, c in junctions_rc]
    return skeleton, degree_map, endpoints_xy, junctions_xy


def _weld_endpoints_to_trunk(binary_skeleton: np.ndarray, trunk_mask: np.ndarray, max_dist: float = 5.0) -> np.ndarray:
    skeleton = (np.asarray(binary_skeleton) > 0).astype(np.uint8)
    trunk = (np.asarray(trunk_mask) > 0)
    if not np.any(skeleton) or not np.any(trunk):
        return skeleton
    degree = _compute_degree_map(skeleton)
    endpoints = list(zip(*np.where((skeleton > 0) & (degree == 1))))
    dist_to_trunk, nearest = distance_transform_edt(~trunk, return_indices=True)
    for r, c in endpoints:
        if dist_to_trunk[r, c] <= max_dist:
            tr = int(nearest[0, r, c])
            tc = int(nearest[1, r, c])
            cv2.line(skeleton, (c, r), (tc, tr), color=1, thickness=1)
    return skeleton


def _dedupe_consecutive(points_xy: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    deduped: List[Tuple[int, int]] = []
    for point in points_xy:
        item = (int(point[0]), int(point[1]))
        if not deduped or item != deduped[-1]:
            deduped.append(item)
    return deduped


def _polyline_length(points_xy: Sequence[Tuple[int, int]]) -> float:
    if len(points_xy) < 2:
        return 0.0
    pts = np.asarray(points_xy, dtype=np.float32)
    return float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())


def simplify_polyline(points_xy: Sequence[Tuple[int, int]], epsilon: float) -> List[Tuple[int, int]]:
    if len(points_xy) <= 2:
        return _dedupe_consecutive(points_xy)
    curve = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(curve, epsilon=max(float(epsilon), 0.5), closed=False).reshape(-1, 2)
    return _dedupe_consecutive([(int(round(x)), int(round(y))) for x, y in approx])


def _simplify_with_forced_points(
    dense_line_xy: Sequence[Tuple[int, int]],
    forced_points_xy: Sequence[Tuple[int, int]],
    epsilon: float,
) -> List[Tuple[int, int]]:
    dense_line_xy = _dedupe_consecutive(dense_line_xy)
    if len(dense_line_xy) <= 2:
        return list(dense_line_xy)
    dense_arr = np.asarray(dense_line_xy, dtype=np.float32)
    forced_indices = {0, len(dense_line_xy) - 1}
    for point in forced_points_xy:
        point_arr = np.asarray(point, dtype=np.float32)
        idx = int(np.argmin(np.linalg.norm(dense_arr - point_arr[None, :], axis=1)))
        forced_indices.add(idx)
    ordered = sorted(forced_indices)
    merged: List[Tuple[int, int]] = []
    for start_idx, end_idx in zip(ordered[:-1], ordered[1:]):
        segment = dense_line_xy[start_idx:end_idx + 1]
        if len(segment) < 2:
            continue
        simplified = simplify_polyline(segment, epsilon=epsilon)
        if merged and simplified and simplified[0] == merged[-1]:
            simplified = simplified[1:]
        merged.extend(simplified)
    return _dedupe_consecutive(merged) if merged else list(dense_line_xy)


def _draw_polyline_mask(polylines: Sequence[Sequence[Tuple[int, int]]], shape: Tuple[int, int], thickness: int = 1) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for line in polylines:
        if len(line) == 1:
            cv2.circle(canvas, line[0], radius=max(thickness // 2, 1), color=1, thickness=-1)
            continue
        for p0, p1 in zip(line[:-1], line[1:]):
            cv2.line(canvas, p0, p1, color=1, thickness=thickness, lineType=cv2.LINE_8)
    return canvas


def render_topology_groups_to_mask(groups: Sequence[Dict], shape: Tuple[int, int], thickness: int = 1) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for group in groups:
        points = [tuple(map(int, point)) for point in group.get("points", [])]
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(points) and 0 <= dst < len(points):
                cv2.line(canvas, points[src], points[dst], color=1, thickness=thickness, lineType=cv2.LINE_8)
    return canvas


def _bridge_vertical_gaps(
    mask: np.ndarray,
    max_gap: int = 96,
    max_dx: int = 28,
    min_component_area: int = 80,
    min_component_height: int = 40,
    interface_band: int = 6,
    max_bridges: int = 6,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    mask_u8 = (_normalize_binary_mask(mask) > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return mask_u8, []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 2:
        return mask_u8, []

    components: List[Dict[str, int]] = []
    for label in range(1, num_labels):
        top = int(stats[label, cv2.CC_STAT_TOP])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area and height < min_component_height:
            continue
        right = left + width - 1
        bottom = top + height - 1
        components.append(
            {
                "label": label,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "width": width,
                "height": height,
                "area": area,
                "cx": int(round(left + width * 0.5)),
                "vertical_ratio": float(height / max(width, 1)),
            }
        )

    bridges: List[Dict[str, int]] = []
    if len(components) < 2:
        return mask_u8, bridges

    candidate_pairs: List[Tuple[float, Dict[str, int], Dict[str, int]]] = []
    for upper in components:
        for lower in components:
            if upper["label"] == lower["label"]:
                continue
            gap = lower["top"] - upper["bottom"] - 1
            if gap <= 0 or gap > max_gap:
                continue
            if max(float(upper["vertical_ratio"]), float(lower["vertical_ratio"])) < 1.8:
                continue
            dx = abs(upper["cx"] - lower["cx"])
            overlap = min(upper["right"], lower["right"]) - max(upper["left"], lower["left"]) + 1
            adaptive_dx = min(max_dx, int(round(0.65 * max(min(upper["width"], lower["width"]), 1))))
            adaptive_dx = max(adaptive_dx, 8)
            if dx > adaptive_dx and overlap < 0:
                continue
            score = float(
                gap
                + 0.75 * dx
                - 0.03 * min(upper["area"], lower["area"])
                - 4.0 * min(float(upper["vertical_ratio"]), float(lower["vertical_ratio"]))
            )
            candidate_pairs.append((score, upper, lower))

    candidate_pairs.sort(key=lambda item: item[0])
    used_labels = set()
    bridge_mask = mask_u8.copy()
    for _, upper, lower in candidate_pairs:
        if len(bridges) >= max_bridges:
            break
        if upper["label"] in used_labels or lower["label"] in used_labels:
            continue

        upper_pixels = np.column_stack(np.where((labels == upper["label"]) & (np.arange(labels.shape[0])[:, None] >= upper["bottom"] - interface_band)))
        lower_pixels = np.column_stack(np.where((labels == lower["label"]) & (np.arange(labels.shape[0])[:, None] <= lower["top"] + interface_band)))
        if upper_pixels.size == 0 or lower_pixels.size == 0:
            continue

        bridge_center_x = 0.5 * (upper["cx"] + lower["cx"])
        best_pair = None
        best_pair_score = float("inf")
        for uy, ux in upper_pixels:
            for ly, lx in lower_pixels:
                if ly <= uy:
                    continue
                local_dx = abs(int(lx) - int(ux))
                local_gap = int(ly - uy - 1)
                if local_dx > max_dx or local_gap > max_gap:
                    continue
                if local_gap <= 0:
                    continue
                if float(local_dx) / max(float(local_gap), 1.0) > 0.38:
                    continue
                pair_score = float(local_gap + 2.4 * local_dx + 0.35 * (abs(ux - bridge_center_x) + abs(lx - bridge_center_x)))
                if pair_score < best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (int(ux), int(uy), int(lx), int(ly))
        if best_pair is None:
            continue

        x0, y0, x1, y1 = best_pair
        thickness = int(np.clip(round(min(upper["width"], lower["width"]) * 0.22), 2, 7))
        cv2.line(bridge_mask, (x0, y0), (x1, y1), color=1, thickness=thickness, lineType=cv2.LINE_8)
        bridges.append(
            {
                "upper_label": int(upper["label"]),
                "lower_label": int(lower["label"]),
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(x1),
                "y1": int(y1),
                "gap": int(y1 - y0 - 1),
                "dx": int(abs(x1 - x0)),
                "thickness": int(thickness),
            }
        )
        used_labels.add(upper["label"])
        used_labels.add(lower["label"])

    return bridge_mask.astype(np.uint8), bridges


def _select_bottom_component(mask: np.ndarray) -> Tuple[np.ndarray, dict]:
    mask_u8 = (_normalize_binary_mask(mask) > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return mask_u8, {"selected_label": 0, "component_count": 0, "component_area": 0, "bottom_y": -1}
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 2:
        ys = np.where(mask_u8 > 0)[0]
        return mask_u8, {
            "selected_label": 1,
            "component_count": num_labels - 1,
            "component_area": int(mask_u8.sum()),
            "bottom_y": int(ys.max()) if ys.size else -1,
        }

    bottoms: List[int] = []
    components: List[Tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        top = int(stats[label, cv2.CC_STAT_TOP])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        bottom_y = top + height - 1
        bottoms.append(bottom_y)
        components.append((label, bottom_y, area, height))

    global_bottom = max(bottoms) if bottoms else -1
    bottom_tol = max(24, int(round(mask_u8.shape[0] * 0.03)))
    max_area = max((item[2] for item in components), default=0)
    max_height = max((item[3] for item in components), default=0)
    structural_components = [
        item for item in components
        if item[2] >= max(32, int(round(max_area * 0.08))) or item[3] >= max(24, int(round(max_height * 0.25)))
    ]
    if not structural_components:
        structural_components = components
    structural_bottom = max((item[1] for item in structural_components), default=global_bottom)
    candidates = [item for item in structural_components if item[1] >= structural_bottom - bottom_tol]
    if not candidates:
        candidates = structural_components

    best_label = 1
    best_key = (-1, -1, -1)
    for label, bottom, area, height in candidates:
        key = (area, height, bottom)
        if key > best_key:
            best_key = key
            best_label = label
    selected = (labels == best_label).astype(np.uint8)
    return selected, {
        "selected_label": int(best_label),
        "component_count": int(num_labels - 1),
        "component_area": int(stats[best_label, cv2.CC_STAT_AREA]),
        "bottom_y": int(global_bottom),
    }

def _find_root_base(mask: np.ndarray, dt_map: np.ndarray, band_height: int = 12) -> Tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, 0
    max_y = int(ys.max())
    min_y = max(max_y - int(band_height), 0)
    band = (mask > 0) & (np.arange(mask.shape[0])[:, None] >= min_y)
    band_ys, band_xs = np.where(band)
    if band_ys.size == 0:
        band_ys, band_xs = ys, xs
    center_x = float(np.mean(band_xs))
    score = dt_map[band_ys, band_xs] - 0.15 * np.abs(band_xs.astype(np.float32) - center_x)
    best_idx = int(np.argmax(score))
    return int(band_ys[best_idx]), int(band_xs[best_idx])


def _snap_to_nearest_true(point_rc: Tuple[int, int], binary_map: np.ndarray) -> Tuple[int, int]:
    r, c = point_rc
    mask = np.asarray(binary_map) > 0
    if not np.any(mask):
        return int(r), int(c)
    r = int(np.clip(r, 0, mask.shape[0] - 1))
    c = int(np.clip(c, 0, mask.shape[1] - 1))
    if mask[r, c]:
        return r, c
    _, nearest = distance_transform_edt(~mask, return_indices=True)
    return int(nearest[0, r, c]), int(nearest[1, r, c])


def _sparsify_points(points_rc: Sequence[Tuple[int, int]], min_dist: float = 8.0, max_points: int = 12) -> List[Tuple[int, int]]:
    selected: List[Tuple[int, int]] = []
    for point in points_rc:
        if all((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 >= min_dist ** 2 for other in selected):
            selected.append((int(point[0]), int(point[1])))
        if len(selected) >= max_points:
            break
    return selected


def _extract_top_candidates(
    skeleton: np.ndarray,
    endpoints_rc: Sequence[Tuple[int, int]],
    root_rc: Tuple[int, int],
    dt_map: np.ndarray,
    max_candidates: int = 12,
) -> List[Tuple[int, int]]:
    root_r, root_c = root_rc
    candidates = [point for point in endpoints_rc if point[0] < root_r - 8]
    candidates.sort(key=lambda p: (p[0], -float(dt_map[p]), abs(p[1] - root_c)))
    if not candidates:
        ys, xs = np.where(skeleton > 0)
        if ys.size == 0:
            return []
        top_cut = int(np.percentile(ys, 15)) if ys.size > 4 else int(ys.min())
        candidates = [(int(r), int(c)) for r, c in zip(ys, xs) if r <= top_cut]
        candidates.sort(key=lambda p: (p[0], -float(dt_map[p]), abs(p[1] - root_c)))
    return _sparsify_points(candidates, min_dist=10.0, max_points=max_candidates)


def _route_path(cost: np.ndarray, start_rc: Tuple[int, int], end_rc: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    try:
        path_rc, _ = route_through_array(cost, start_rc, end_rc, fully_connected=True, geometric=True)
    except Exception:
        return None
    return [(int(r), int(c)) for r, c in path_rc]


def _build_skeleton_cost(dt_norm: np.ndarray, skeleton: np.ndarray, outside_cost: float) -> np.ndarray:
    cost = 1.0 - np.asarray(dt_norm, dtype=np.float32)
    cost[skeleton == 0] = float(outside_cost)
    return np.clip(cost, 1e-3, float(outside_cost)).astype(np.float32)


def _score_trunk_path(path_rc: Sequence[Tuple[int, int]], dt_norm: np.ndarray, root_rc: Tuple[int, int], image_shape: Tuple[int, int]) -> float:
    if not path_rc:
        return -1e9
    rows = np.asarray([p[0] for p in path_rc], dtype=np.float32)
    cols = np.asarray([p[1] for p in path_rc], dtype=np.float32)
    dt_values = np.asarray([dt_norm[p] for p in path_rc], dtype=np.float32)
    h, w = image_shape
    vertical_span = max(float(root_rc[0]) - float(rows.min()), 0.0) / max(float(h - 1), 1.0)
    centrality = np.abs(cols[-1] - float(root_rc[1])) / max(float(w - 1), 1.0)
    lateral_drift = float(np.mean(np.abs(cols - float(root_rc[1])))) / max(float(w - 1), 1.0)
    steps = np.diff(np.column_stack([rows, cols]), axis=0)
    arc_length = float(np.linalg.norm(steps, axis=1).sum()) if len(steps) else 0.0
    vertical_pixels = max(float(root_rc[0]) - float(rows.min()), 1.0)
    tortuosity = max(arc_length / vertical_pixels - 1.0, 0.0)
    downward_backtrack = float(np.maximum(steps[:, 0], 0.0).sum()) / vertical_pixels if len(steps) else 0.0
    return float(
        0.60 * dt_values.mean()
        + 0.25 * np.percentile(dt_values, 75)
        + 0.20 * vertical_span
        - 0.30 * centrality
        - 0.16 * lateral_drift
        - 0.20 * tortuosity
        - 0.12 * downward_backtrack
    )


def _path_rc_to_xy(path_rc: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(int(c), int(r)) for r, c in path_rc]


def _rescale_points_xy(points_xy: Sequence[Tuple[int, int]], scale_x: float, scale_y: float) -> List[Tuple[int, int]]:
    return _dedupe_consecutive([(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in points_xy])


def _resize_binary_mask(mask: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float, float]:
    h, w = mask.shape[:2]
    if max(h, w) <= max_dim:
        return mask.astype(np.uint8), 1.0, 1.0
    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8), float(w) / float(new_w), float(h) / float(new_h)


def _branch_trace_length(start_rc: Tuple[int, int], skeleton: np.ndarray, degree_map: np.ndarray) -> int:
    path_len = 0
    prev = None
    current = start_rc
    while True:
        neighbors = [point for point in _neighbors_of(current, skeleton) if point != prev]
        if len(neighbors) != 1:
            break
        nxt = neighbors[0]
        path_len += 1
        prev, current = current, nxt
        if int(degree_map[current]) != 2:
            break
    return path_len


def _trace_endpoint_direction(
    endpoint_rc: Tuple[int, int],
    skeleton: np.ndarray,
    max_steps: int = 8,
) -> Tuple[np.ndarray, float]:
    path = [endpoint_rc]
    prev = None
    current = endpoint_rc
    for _ in range(max_steps):
        neighbors = [point for point in _neighbors_of(current, skeleton) if point != prev]
        if len(neighbors) != 1:
            break
        nxt = neighbors[0]
        path.append(nxt)
        prev, current = current, nxt
        if skeleton[current] == 0:
            break
    polyline_xy = _path_rc_to_xy(path)
    tangent = _estimate_tangent_from_polyline(polyline_xy)
    return tangent, _polyline_length(polyline_xy)


def _bridge_trunk_gaps_on_skeleton(
    skeleton: np.ndarray,
    dt_map: np.ndarray,
    root_rc: Tuple[int, int],
    max_gap: int = 72,
    max_dx: int = 18,
    min_radius: float = 2.2,
    max_bridges: int = 3,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    skeleton_u8 = (np.asarray(skeleton) > 0).astype(np.uint8)
    if not np.any(skeleton_u8):
        return skeleton_u8, []
    degree = _compute_degree_map(skeleton_u8)
    endpoints = list(zip(*np.where((skeleton_u8 > 0) & (degree == 1))))
    if len(endpoints) < 2:
        return skeleton_u8, []

    num_labels, labels = cv2.connectedComponents(skeleton_u8, connectivity=8)
    root_c = float(root_rc[1])
    candidates: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = []
    up_vec = np.asarray([0.0, -1.0], dtype=np.float32)
    for idx_a in range(len(endpoints)):
        for idx_b in range(idx_a + 1, len(endpoints)):
            a = (int(endpoints[idx_a][0]), int(endpoints[idx_a][1]))
            b = (int(endpoints[idx_b][0]), int(endpoints[idx_b][1]))
            if labels[a] == labels[b]:
                continue
            upper, lower = (a, b) if a[0] < b[0] else (b, a)
            gap = int(lower[0] - upper[0] - 1)
            if gap <= 0 or gap > max_gap:
                continue
            dx = int(abs(lower[1] - upper[1]))
            if dx > max_dx:
                continue
            upper_radius = float(dt_map[upper])
            lower_radius = float(dt_map[lower])
            if min(upper_radius, lower_radius) < min_radius:
                continue
            radius_ratio = max(upper_radius, lower_radius) / max(min(upper_radius, lower_radius), 1e-3)
            if radius_ratio > 1.6:
                continue
            vec = _normalize_vector(np.asarray([lower[1] - upper[1], lower[0] - upper[0]], dtype=np.float32))
            if abs(float(np.dot(vec, up_vec))) < 0.86:
                continue
            upper_tangent, upper_len = _trace_endpoint_direction(upper, skeleton_u8)
            lower_tangent, lower_len = _trace_endpoint_direction(lower, skeleton_u8)
            if upper_len < 4.0 or lower_len < 4.0:
                continue
            if abs(float(np.dot(upper_tangent, up_vec))) < 0.72 or abs(float(np.dot(lower_tangent, up_vec))) < 0.72:
                continue
            if abs(float(upper[1]) - root_c) > max_dx * 2.0 and abs(float(lower[1]) - root_c) > max_dx * 2.0:
                continue
            score = float(
                gap
                + 1.3 * dx
                + 6.0 * abs(radius_ratio - 1.0)
                + 0.6 * (abs(float(upper[1]) - root_c) + abs(float(lower[1]) - root_c))
                - 2.0 * (upper_radius + lower_radius)
            )
            candidates.append((score, upper, lower))

    candidates.sort(key=lambda item: item[0])
    bridges: List[Dict[str, int]] = []
    used_points = set()
    bridged = skeleton_u8.copy()
    for _, upper, lower in candidates:
        if len(bridges) >= max_bridges:
            break
        if upper in used_points or lower in used_points:
            continue
        x0, y0 = int(upper[1]), int(upper[0])
        x1, y1 = int(lower[1]), int(lower[0])
        cv2.line(bridged, (x0, y0), (x1, y1), color=1, thickness=1, lineType=cv2.LINE_8)
        bridges.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "gap": int(y1 - y0 - 1), "dx": int(abs(x1 - x0))})
        used_points.add(upper)
        used_points.add(lower)
    return bridged.astype(np.uint8), bridges


def _choose_trunk_path_on_skeleton(
    skeleton: np.ndarray,
    dt_map: np.ndarray,
    dt_norm: np.ndarray,
    root_rc: Tuple[int, int],
    outside_cost: float,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float, np.ndarray]:
    endpoints_rc = list(zip(*np.where((skeleton > 0) & (_compute_degree_map(skeleton) == 1))))
    top_candidates = _extract_top_candidates(skeleton, endpoints_rc, root_rc, dt_map)
    skeleton_cost = _build_skeleton_cost(dt_norm, skeleton, outside_cost=outside_cost)

    best_trunk_path: Optional[List[Tuple[int, int]]] = None
    best_score = -1e9
    for candidate_rc in top_candidates:
        path_rc = _route_path(skeleton_cost, root_rc, candidate_rc)
        if not path_rc or len(path_rc) < 2:
            continue
        score = _score_trunk_path(path_rc, dt_norm, root_rc, skeleton.shape)
        if score > best_score:
            best_score = score
            best_trunk_path = path_rc

    if not best_trunk_path:
        topmost_rc = min(endpoints_rc, key=lambda p: p[0]) if endpoints_rc else root_rc
        best_trunk_path = _route_path(skeleton_cost, root_rc, topmost_rc) or [root_rc, topmost_rc]
        best_score = _score_trunk_path(best_trunk_path, dt_norm, root_rc, skeleton.shape)
    return best_trunk_path, top_candidates, float(best_score), skeleton_cost


def _find_component_attachment(
    component_mask: np.ndarray,
    trunk_mask: np.ndarray,
    dt_map: np.ndarray,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    attachments = _find_component_attachments(component_mask, trunk_mask, dt_map)
    if attachments:
        return attachments[0]

    component_coords = list(zip(*np.where(component_mask > 0)))
    trunk_coords = list(zip(*np.where(trunk_mask > 0)))
    if not component_coords or not trunk_coords:
        return None, None
    center_r = float(np.mean([point[0] for point in component_coords]))
    center_c = float(np.mean([point[1] for point in component_coords]))
    branch_attach = min(component_coords, key=lambda p: (p[0] - center_r) ** 2 + (p[1] - center_c) ** 2)
    trunk_attach = _snap_to_nearest_true(branch_attach, trunk_mask)
    return (int(branch_attach[0]), int(branch_attach[1])), (int(trunk_attach[0]), int(trunk_attach[1]))


def _find_component_attachments(
    component_mask: np.ndarray,
    trunk_mask: np.ndarray,
    dt_map: np.ndarray,
    trunk_points: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    component_coords = list(zip(*np.where(component_mask > 0)))
    if not component_coords or not np.any(trunk_mask):
        return []

    contact_mask = np.zeros_like(component_mask, dtype=np.uint8)
    branch_to_trunk: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for branch_rc in component_coords:
        neighbors = [trunk_rc for trunk_rc in _neighbors_of(branch_rc, trunk_mask) if trunk_mask[trunk_rc] > 0]
        if not neighbors:
            continue
        contact_mask[branch_rc] = 1
        branch_to_trunk[(int(branch_rc[0]), int(branch_rc[1]))] = [(int(rc[0]), int(rc[1])) for rc in neighbors]

    attachments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if np.any(contact_mask):
        num_labels, labels = cv2.connectedComponents(contact_mask, connectivity=8)
        for label in range(1, num_labels):
            coords = [(int(r), int(c)) for r, c in zip(*np.where(labels == label))]
            if not coords:
                continue
            branch_attach = max(coords, key=lambda rc: float(dt_map[rc]))
            trunk_candidates: List[Tuple[int, int]] = []
            for rc in coords:
                trunk_candidates.extend(branch_to_trunk.get(rc, []))
            if trunk_candidates:
                trunk_attach = min(
                    trunk_candidates,
                    key=lambda rc: (rc[0] - branch_attach[0]) ** 2 + (rc[1] - branch_attach[1]) ** 2,
                )
            else:
                trunk_attach = _snap_to_nearest_true(branch_attach, trunk_mask)
            attachments.append(((int(branch_attach[0]), int(branch_attach[1])), (int(trunk_attach[0]), int(trunk_attach[1]))))

    if attachments:
        # ---- arc-length clustering (preferred) or 2D merge (fallback) ----
        trunk_pts_xy = (
            [(int(tp[0]), int(tp[1])) for tp in trunk_points]
            if trunk_points and len(trunk_points) >= 2
            else None
        )
        if False and trunk_pts_xy:
            # SHORT-CIRCUIT: 06-26 1D arc-length clustering bypassed → use 2D fallback
            # 1D clustering by trunk arc-length position
            trunk_arc = [0.0]
            for i in range(1, len(trunk_pts_xy)):
                dx = trunk_pts_xy[i][0] - trunk_pts_xy[i - 1][0]
                dy = trunk_pts_xy[i][1] - trunk_pts_xy[i - 1][1]
                trunk_arc.append(trunk_arc[-1] + np.sqrt(float(dx * dx + dy * dy)))

            def _project_to_trunk(rc: Tuple[int, int]) -> float:
                y, x = float(rc[0]), float(rc[1])
                best_t = 0.0
                best_arc = trunk_arc[0]
                best_dist_sq = float("inf")
                for seg in range(len(trunk_pts_xy) - 1):
                    ax, ay = float(trunk_pts_xy[seg][0]), float(trunk_pts_xy[seg][1])
                    bx, by = float(trunk_pts_xy[seg + 1][0]), float(trunk_pts_xy[seg + 1][1])
                    abx, aby = bx - ax, by - ay
                    seg_len_sq = abx * abx + aby * aby
                    if seg_len_sq < 1e-6:
                        d_sq = (x - ax) ** 2 + (y - ay) ** 2
                        if d_sq < best_dist_sq:
                            best_dist_sq = d_sq
                            best_arc = trunk_arc[seg]
                        continue
                    t = max(0.0, min(1.0, ((x - ax) * abx + (y - ay) * aby) / seg_len_sq))
                    px = ax + t * abx
                    py = ay + t * aby
                    d_sq = (x - px) ** 2 + (y - py) ** 2
                    if d_sq < best_dist_sq:
                        best_dist_sq = d_sq
                        best_t = t
                        best_arc = trunk_arc[seg] + t * (trunk_arc[seg + 1] - trunk_arc[seg])
                return best_arc

            arc_positions = [_project_to_trunk(tr) for _, tr in attachments]
            cluster_threshold = min(trunk_arc[-1] * 0.03, 15.0)
            cluster_labels = [0] * len(attachments)
            next_label = 1
            for i in range(len(attachments)):
                if cluster_labels[i] != 0:
                    continue
                cluster_labels[i] = next_label
                for j in range(i + 1, len(attachments)):
                    if cluster_labels[j] == 0 and abs(arc_positions[j] - arc_positions[i]) < cluster_threshold:
                        cluster_labels[j] = next_label
                next_label += 1

            merged = []
            for lbl in range(1, next_label):
                cluster = [attachments[i] for i in range(len(attachments)) if cluster_labels[i] == lbl]
                best = max(cluster, key=lambda item: float(dt_map[item[0]]))
                merged.append(best)
            attachments = merged
        else:
            # Fallback: 2D Euclidean merge (18px radius)
            merged: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
            merge_radius_sq = float(18 ** 2)
            for branch_attach_rc, trunk_attach_rc in attachments:
                merged_idx = None
                for idx, (exist_branch_rc, exist_trunk_rc) in enumerate(merged):
                    branch_dist_sq = float((branch_attach_rc[0] - exist_branch_rc[0]) ** 2 + (branch_attach_rc[1] - exist_branch_rc[1]) ** 2)
                    trunk_dist_sq = float((trunk_attach_rc[0] - exist_trunk_rc[0]) ** 2 + (trunk_attach_rc[1] - exist_trunk_rc[1]) ** 2)
                    if branch_dist_sq <= merge_radius_sq and trunk_dist_sq <= merge_radius_sq:
                        merged_idx = idx
                        break
                if merged_idx is None:
                    merged.append((branch_attach_rc, trunk_attach_rc))
                else:
                    prev_branch_rc, prev_trunk_rc = merged[merged_idx]
                    prev_score = float(dt_map[prev_branch_rc])
                    curr_score = float(dt_map[branch_attach_rc])
                    if curr_score > prev_score:
                        merged[merged_idx] = (branch_attach_rc, trunk_attach_rc)
            attachments = merged

        attachments.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))
        return attachments

    trunk_coords = list(zip(*np.where(trunk_mask > 0)))
    if not trunk_coords:
        return []
    center_r = float(np.mean([point[0] for point in component_coords]))
    center_c = float(np.mean([point[1] for point in component_coords]))
    branch_attach = min(component_coords, key=lambda p: (p[0] - center_r) ** 2 + (p[1] - center_c) ** 2)
    trunk_attach = _snap_to_nearest_true(branch_attach, trunk_mask)
    return [((int(branch_attach[0]), int(branch_attach[1])), (int(trunk_attach[0]), int(trunk_attach[1])))]


def _prepend_anchor_to_group(group: Optional[Dict], anchor_xy: Tuple[int, int], attach_xy: Tuple[int, int]) -> Optional[Dict]:
    if group is None:
        return None
    group = {**group}
    points = [list(map(int, point)) for point in group.get("points", [])]
    edges = [list(map(int, edge)) for edge in group.get("edges", [])]
    anchor_xy = (int(anchor_xy[0]), int(anchor_xy[1]))
    attach_xy = (int(attach_xy[0]), int(attach_xy[1]))
    if not points:
        return group
    anchor_distance = float(np.linalg.norm(
        np.asarray(anchor_xy, dtype=np.float32) - np.asarray(attach_xy, dtype=np.float32)
    ))
    if anchor_distance > 64.0:
        group["root_anchor_rejected_distance_px"] = float(anchor_distance)
        return group
    point_keys = [tuple(point) for point in points]
    if attach_xy in point_keys:
        attach_idx = point_keys.index(attach_xy)
    else:
        attach_arr = np.asarray(attach_xy, dtype=np.float32)
        best_edge = None
        best_distance = float("inf")
        for edge_idx, (src, dst) in enumerate(edges):
            if not (0 <= src < len(points) and 0 <= dst < len(points)):
                continue
            a = np.asarray(points[src], dtype=np.float32)
            b = np.asarray(points[dst], dtype=np.float32)
            ab = b - a
            denom = float(np.dot(ab, ab))
            t = 0.0 if denom < 1e-6 else float(np.clip(np.dot(attach_arr - a, ab) / denom, 0.0, 1.0))
            distance = float(np.linalg.norm(attach_arr - (a + t * ab)))
            if distance < best_distance:
                best_distance = distance
                best_edge = (edge_idx, int(src), int(dst))
        if best_edge is None:
            attach_idx = int(np.argmin(np.linalg.norm(
                np.asarray(points, dtype=np.float32) - attach_arr[None, :], axis=1,
            )))
        else:
            edge_idx, src, dst = best_edge
            attach_idx = len(points)
            points.append([int(attach_xy[0]), int(attach_xy[1])])
            edges.pop(edge_idx)
            edges.extend([[src, attach_idx], [attach_idx, dst]])
            point_keys.append(attach_xy)
    if anchor_xy in point_keys:
        anchor_idx = point_keys.index(anchor_xy)
    else:
        anchor_idx = len(points)
        points.append([int(anchor_xy[0]), int(anchor_xy[1])])
    edge_keys = {tuple(sorted((int(src), int(dst)))) for src, dst in edges if src != dst}
    connectivity = nx.Graph()
    connectivity.add_nodes_from(range(len(points)))
    connectivity.add_edges_from(edge_keys)
    already_connected = (
        anchor_idx in connectivity
        and attach_idx in connectivity
        and nx.has_path(connectivity, anchor_idx, attach_idx)
    )
    if anchor_idx != attach_idx and not already_connected:
        edge_keys.add(tuple(sorted((anchor_idx, attach_idx))))
    group["points"] = points
    group["edges"] = [list(edge) for edge in sorted(edge_keys)]
    group["root_anchor_xy"] = [int(anchor_xy[0]), int(anchor_xy[1])]
    group["root_attach_xy"] = [int(attach_xy[0]), int(attach_xy[1])]
    return group


def _group_total_edge_length(group: Dict) -> float:
    points = [tuple(map(int, point)) for point in group.get("points", [])]
    total = 0.0
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if 0 <= src < len(points) and 0 <= dst < len(points):
            total += float(np.linalg.norm(np.asarray(points[src], dtype=np.float32) - np.asarray(points[dst], dtype=np.float32)))
    return total


def _count_trunk_contacts(
    group: Dict,
    trunk_mask: np.ndarray,
    contact_radius: int = 2,
    trunk_points: Optional[List[Tuple[int, int]]] = None,
) -> int:
    """Count distinct trunk contact regions for a branch group.

    When trunk_points is provided, uses arc-length spread to distinguish
    genuine multi-contact (branches following the trunk) from pixel jitter
    around a single anchor point.
    """
    points = [tuple(map(int, p)) for p in group.get("points", [])]
    edges = [(int(e[0]), int(e[1])) for e in group.get("edges", [])]
    if not points or not edges:
        return 0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    pad = contact_radius + 4
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(trunk_mask.shape[1], x_max + pad + 1)
    y_max = min(trunk_mask.shape[0], y_max + pad + 1)
    h, w = y_max - y_min, x_max - x_min
    if h <= 0 or w <= 0:
        return 0
    group_canvas = np.zeros((h, w), dtype=np.uint8)
    local_points = [(int(p[0]) - x_min, int(p[1]) - y_min) for p in points]
    for src, dst in edges:
        if 0 <= src < len(local_points) and 0 <= dst < len(local_points):
            p1 = local_points[src]
            p2 = local_points[dst]
            cv2.line(group_canvas, p1, p2, 1, 1, lineType=cv2.LINE_4)
    if contact_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (contact_radius * 2 + 1, contact_radius * 2 + 1))
        group_canvas = cv2.dilate(group_canvas, kernel, iterations=1)
    local_trunk = trunk_mask[y_min:y_max, x_min:x_max]
    contact = (group_canvas > 0) & (local_trunk > 0)
    n_labels, labels = cv2.connectedComponents(contact.astype(np.uint8), connectivity=8)
    contact_count = n_labels - 1
    if contact_count <= 1:
        return contact_count

    # Use arc-length spread to filter out pixel jitter around a single point
    if trunk_points and len(trunk_points) >= 2:
        tp = trunk_points
        trunk_arc = [0.0]
        for i in range(1, len(tp)):
            dx = tp[i][0] - tp[i - 1][0]
            dy = tp[i][1] - tp[i - 1][1]
            trunk_arc.append(trunk_arc[-1] + np.sqrt(float(dx * dx + dy * dy)))
        trunk_arc_total = trunk_arc[-1]

        # Collect arc positions of all contact pixels
        arc_positions: List[float] = []
        for lbl in range(1, n_labels):
            coords = np.where(labels == lbl)
            if coords[0].size == 0:
                continue
            # Global trunk coords of first contact pixel in this CC
            gy = int(coords[0][0]) + y_min
            gx = int(coords[1][0]) + x_min
            # Project to trunk arc-length
            best_a = trunk_arc[0]
            best_d = float("inf")
            yf, xf = float(gy), float(gx)
            for s in range(len(tp) - 1):
                ax, ay = float(tp[s][0]), float(tp[s][1])
                bx, by = float(tp[s + 1][0]), float(tp[s + 1][1])
                abx, aby = bx - ax, by - ay
                seg_len_sq = abx * abx + aby * aby
                if seg_len_sq < 1e-6:
                    d = (xf - ax) ** 2 + (yf - ay) ** 2
                    if d < best_d:
                        best_d = d
                        best_a = trunk_arc[s]
                    continue
                t = max(0.0, min(1.0, ((xf - ax) * abx + (yf - ay) * aby) / seg_len_sq))
                px_proj = ax + t * abx
                py_proj = ay + t * aby
                d = (xf - px_proj) ** 2 + (yf - py_proj) ** 2
                if d < best_d:
                    best_d = d
                    best_a = trunk_arc[s] + t * (trunk_arc[s + 1] - trunk_arc[s])
            arc_positions.append(best_a)

        if arc_positions:
            arc_spread = max(arc_positions) - min(arc_positions)
            # Genuine multi-contact: arc spread > 8% of trunk length AND > 30px absolute
            min_spread = max(trunk_arc_total * 0.08, 30.0)
            if arc_spread < min_spread:
                return 1  # pixel jitter, treat as single contact

    return contact_count


def _build_group_from_subgraph(
    points: Sequence[Tuple[int, int]],
    graph: nx.Graph,
    node_ids: Sequence[int],
    anchor_node: int,
    group_id: str,
    color_hex: str,
    fork_origin_group: Optional[str],
) -> Optional[Dict]:
    component = graph.subgraph(node_ids).copy()
    if component.number_of_edges() == 0:
        return None
    ordered_nodes = list(nx.bfs_tree(component, source=anchor_node).nodes())
    if len(ordered_nodes) < component.number_of_nodes():
        for node_id in component.nodes():
            if node_id not in ordered_nodes:
                ordered_nodes.append(node_id)
    local_index = {node_id: idx for idx, node_id in enumerate(ordered_nodes)}
    local_points = [[int(points[node_id][0]), int(points[node_id][1])] for node_id in ordered_nodes]
    local_edges = [[local_index[src], local_index[dst]] for src, dst in component.edges()]
    return {
        "group_id": group_id,
        "group_type": "branch",
        "color_hex": color_hex,
        "points": local_points,
        "edges": local_edges,
        "fork_origin_group": fork_origin_group,
    }


def _assign_tree_nodes_to_anchors(
    component: nx.Graph,
    anchor_nodes: Sequence[int],
    force_split: bool = False,
) -> Dict[int, int]:
    if not force_split:
        return {}
    queue: deque[int] = deque()
    owner: Dict[int, int] = {}
    dist: Dict[int, int] = {}
    for anchor_node in sorted(set(int(node) for node in anchor_nodes)):
        if anchor_node not in component:
            continue
        owner[anchor_node] = int(anchor_node)
        dist[anchor_node] = 0
        queue.append(anchor_node)

    while queue:
        node = queue.popleft()
        for neighbor in component.neighbors(node):
            cand_owner = owner[node]
            cand_dist = dist[node] + 1
            if neighbor not in dist or cand_dist < dist[neighbor] or (cand_dist == dist[neighbor] and cand_owner < owner[neighbor]):
                dist[neighbor] = cand_dist
                owner[neighbor] = cand_owner
                queue.append(neighbor)
    return owner


def _merge_root_family_groups(groups: Sequence[Dict]) -> List[Dict]:
    families: Dict[str, List[Dict]] = {}
    order: List[str] = []
    for group in groups:
        group_id = str(group.get("group_id", "branch"))
        family_id = group_id.split("_root_", 1)[0] if "_root_" in group_id else group_id
        if family_id not in families:
            families[family_id] = []
            order.append(family_id)
        families[family_id].append(group)
    merged_groups: List[Dict] = []
    for family_id in order:
        members = families[family_id]
        if len(members) == 1:
            merged_groups.append(members[0])
            continue
        points: List[List[int]] = []
        point_lookup: Dict[Tuple[int, int], int] = {}
        edge_keys = set()
        for member in members:
            remap = {}
            for old_idx, point in enumerate(member.get("points", [])):
                key = tuple(map(int, point))
                if key not in point_lookup:
                    point_lookup[key] = len(points)
                    points.append([key[0], key[1]])
                remap[old_idx] = point_lookup[key]
            for edge in member.get("edges", []):
                if len(edge) != 2:
                    continue
                src, dst = remap.get(int(edge[0])), remap.get(int(edge[1]))
                if src is not None and dst is not None and src != dst:
                    edge_keys.add(tuple(sorted((src, dst))))
        merged_groups.append({
            **members[0],
            "group_id": family_id,
            "points": points,
            "edges": [list(edge) for edge in sorted(edge_keys)],
            "root_family_members": [member.get("group_id") for member in members],
            "root_family_attachments": [
                {
                    "anchor_xy": list(map(int, member.get("root_anchor_xy"))),
                    "attach_xy": list(map(int, member.get("root_attach_xy"))),
                }
                for member in members
                if isinstance(member.get("root_anchor_xy"), (list, tuple))
                and len(member.get("root_anchor_xy")) == 2
                and isinstance(member.get("root_attach_xy"), (list, tuple))
                and len(member.get("root_attach_xy")) == 2
            ],
        })
    return merged_groups


def _split_group_by_attachments_edge_conserving(
    group: Dict,
    attachments: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    group_prefix: str,
    palette: Sequence[str],
    min_branch_length: float,
    trunk_distance: Optional[np.ndarray] = None,
    source_mask: Optional[np.ndarray] = None,
) -> List[Dict]:
    working_group = {
        **group,
        "points": [list(map(int, point)) for point in group.get("points", [])],
        "edges": [list(map(int, edge)) for edge in group.get("edges", [])],
    }
    anchor_coordinates: List[Tuple[int, int]] = []
    for branch_attach_rc, trunk_attach_rc in attachments:
        attach_xy = (int(branch_attach_rc[1]), int(branch_attach_rc[0]))
        anchor_xy = (int(trunk_attach_rc[1]), int(trunk_attach_rc[0]))
        working_group = _prepend_anchor_to_group(working_group, anchor_xy, attach_xy)
        anchor_coordinates.append(anchor_xy)
    points, graph = _group_to_local_graph(working_group)
    if graph.number_of_edges() == 0 or not points:
        return []
    point_lookup = {tuple(map(int, point)): idx for idx, point in enumerate(points)}
    anchors = sorted({point_lookup[point] for point in anchor_coordinates if point in point_lookup})
    if len(anchors) <= 1:
        return [working_group]
    anchor_coordinate_by_node = {
        point_lookup[point]: point for point in anchor_coordinates if point in point_lookup
    }
    weighted_graph = graph.copy()
    for src, dst in weighted_graph.edges():
        weighted_graph.edges[src, dst]["length"] = float(np.linalg.norm(
            np.asarray(points[src], dtype=np.float32) - np.asarray(points[dst], dtype=np.float32)
        ))
    tree = nx.minimum_spanning_tree(weighted_graph, weight="length")
    family_edge_length = float(_group_total_edge_length(working_group))
    effective_min_length = max(float(min_branch_length), 0.05 * family_edge_length)
    steiner_edges = set()
    for anchor_a, anchor_b in combinations(anchors, 2):
        try:
            path = nx.shortest_path(tree, anchor_a, anchor_b)
        except nx.NetworkXNoPath:
            continue
        steiner_edges.update(tuple(sorted((int(src), int(dst)))) for src, dst in zip(path[:-1], path[1:]))

    def boundary_priority(edge: Tuple[int, int]) -> Tuple[float, float]:
        src, dst = edge
        degree_bonus = 0.0 if max(tree.degree[src], tree.degree[dst]) >= 3 else 20.0
        unsupported_bonus = 0.0
        if source_mask is not None:
            rr, cc = _line_pixels_xy(points[src], points[dst], source_mask.shape)
            unsupported_bonus = -4.0 * float(_max_false_run(source_mask[rr, cc] > 0))
        length = float(weighted_graph.edges[src, dst].get("length", 0.0))
        return degree_bonus + unsupported_bonus, length

    candidate_edges = sorted(steiner_edges, key=boundary_priority)
    if len(candidate_edges) > 24:
        candidate_edges = candidate_edges[:24]

    def partition_score(cut_edges: Sequence[Tuple[int, int]]) -> Optional[Tuple[float, Dict[int, int]]]:
        forest = tree.copy()
        forest.remove_edges_from(cut_edges)
        components = list(nx.connected_components(forest))
        if len(components) != len(anchors):
            return None
        owner_map: Dict[int, int] = {}
        score = 0.0
        for component in components:
            component_anchors = [anchor for anchor in anchors if anchor in component]
            if len(component_anchors) != 1:
                return None
            anchor = component_anchors[0]
            for node in component:
                owner_map[int(node)] = int(anchor)
            subgraph = forest.subgraph(component).copy()
            parent = {anchor: None}
            for node in list(nx.bfs_tree(subgraph, anchor))[1:]:
                parent_node = next(neighbor for neighbor in subgraph.neighbors(node) if neighbor in parent)
                parent[node] = parent_node
            component_length = sum(
                float(weighted_graph.edges[src, dst].get("length", 0.0))
                for src, dst in subgraph.edges()
            )
            if component_length < effective_min_length:
                score += 120.0 * (effective_min_length - component_length) / max(effective_min_length, 1e-6)
            longest_root_path = 0.0
            best_root_straightness = 0.0
            for leaf in [node for node in subgraph.nodes() if subgraph.degree[node] == 1 and node != anchor]:
                path = nx.shortest_path(subgraph, anchor, leaf)
                path_length = sum(
                    float(weighted_graph.edges[src, dst].get("length", 0.0))
                    for src, dst in zip(path[:-1], path[1:])
                )
                displacement = float(np.linalg.norm(
                    np.asarray(points[leaf], dtype=np.float32)
                    - np.asarray(points[anchor], dtype=np.float32)
                ))
                straightness = displacement / max(path_length, 1e-6)
                if path_length > longest_root_path:
                    longest_root_path = path_length
                    best_root_straightness = straightness
            score -= 0.30 * longest_root_path
            score += 80.0 * (1.0 - best_root_straightness)
        for edge in cut_edges:
            score += boundary_priority(edge)[0]
        return float(score), owner_map

    best_partition = None
    required_cuts = len(anchors) - 1
    if required_cuts <= 4 and len(candidate_edges) >= required_cuts:
        for cut_edges in combinations(candidate_edges, required_cuts):
            result = partition_score(cut_edges)
            if result is None:
                continue
            score, owner_map = result
            if best_partition is None or score < best_partition[0]:
                best_partition = (score, owner_map, set(cut_edges))
    if best_partition is None:
        owner = _assign_tree_nodes_to_anchors(tree, anchors, force_split=True)
        cut_edge_set = {
            tuple(sorted((int(src), int(dst))))
            for src, dst in tree.edges()
            if owner.get(int(src)) != owner.get(int(dst))
        }
    else:
        _, owner, cut_edge_set = best_partition
    edge_coordinates_by_anchor: Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {
        anchor: [] for anchor in anchors
    }
    for src, dst in tree.edges():
        src_owner = owner.get(int(src))
        dst_owner = owner.get(int(dst))
        if src_owner is None or dst_owner is None:
            continue
        src_xy = tuple(map(int, points[int(src)]))
        dst_xy = tuple(map(int, points[int(dst)]))
        edge_key = tuple(sorted((int(src), int(dst))))
        if src_owner == dst_owner and edge_key not in cut_edge_set:
            edge_coordinates_by_anchor[int(src_owner)].append((src_xy, dst_xy))
            continue
        midpoint = (
            int(round((src_xy[0] + dst_xy[0]) / 2.0)),
            int(round((src_xy[1] + dst_xy[1]) / 2.0)),
        )
        if midpoint != src_xy:
            edge_coordinates_by_anchor[int(src_owner)].append((src_xy, midpoint))
        if midpoint != dst_xy:
            edge_coordinates_by_anchor[int(dst_owner)].append((midpoint, dst_xy))

    split_groups: List[Dict] = []
    for output_idx, anchor in enumerate(anchors, start=1):
        assigned_edges = edge_coordinates_by_anchor.get(anchor, [])
        if not assigned_edges:
            continue
        local_points: List[List[int]] = []
        local_lookup: Dict[Tuple[int, int], int] = {}
        local_edges = []
        for src_xy, dst_xy in assigned_edges:
            for point in (src_xy, dst_xy):
                if point not in local_lookup:
                    local_lookup[point] = len(local_points)
                    local_points.append([int(point[0]), int(point[1])])
            src_local = local_lookup[src_xy]
            dst_local = local_lookup[dst_xy]
            if src_local != dst_local:
                local_edges.append([src_local, dst_local])
        anchor_xy = anchor_coordinate_by_node[anchor]
        split_group = {
            **group,
            "group_id": f"{group_prefix}_{output_idx:02d}",
            "group_type": "branch",
            "color_hex": palette[(output_idx - 1) % len(palette)] if palette else "#00FF00",
            "points": local_points,
            "edges": local_edges,
            "fork_origin_group": "trunk",
            "root_partition_method": "edge_conserving_multi_source_midpoint",
            "root_anchor_xy": [int(anchor_xy[0]), int(anchor_xy[1])],
        }
        if _group_total_edge_length(split_group) >= effective_min_length:
            split_groups.append(split_group)
    return split_groups


def _split_group_by_attachments(
    group: Dict,
    attachments: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    group_prefix: str,
    palette: Sequence[str],
    min_branch_length: float,
    force_split: bool = False,
    trunk_distance: Optional[np.ndarray] = None,
    source_mask: Optional[np.ndarray] = None,
) -> List[Dict]:
    if not force_split:
        return [group]
    if not attachments:
        return [group] if _group_total_edge_length(group) >= float(min_branch_length) else []

    if len(attachments) <= 1:
        branch_attach_rc, trunk_attach_rc = attachments[0]
        anchor_xy = (int(trunk_attach_rc[1]), int(trunk_attach_rc[0]))
        attach_xy = (int(branch_attach_rc[1]), int(branch_attach_rc[0]))
        single_group = _prepend_anchor_to_group(
            {
                **group,
                "group_id": f"{group_prefix}_01",
                "group_type": "branch",
                "color_hex": palette[0] if palette else "#00FF00",
                "fork_origin_group": "trunk",
            },
            anchor_xy=anchor_xy,
            attach_xy=attach_xy,
        )
        if single_group is None or _group_total_edge_length(single_group) < float(min_branch_length):
            return []
        return [single_group]

    return _split_group_by_attachments_edge_conserving(
        group,
        attachments=attachments,
        group_prefix=group_prefix,
        palette=palette,
        min_branch_length=min_branch_length,
        trunk_distance=trunk_distance,
        source_mask=source_mask,
    )

    resolved_groups = [group]

    final_groups: List[Dict] = []
    next_group_idx = 1
    for resolved_group in resolved_groups:
        points, graph = _group_to_local_graph(resolved_group)
        if graph.number_of_edges() == 0 or not points:
            continue
        point_arr = np.asarray(points, dtype=np.float32)
        attachment_infos: List[Dict] = []
        for attach_idx, (branch_attach_rc, trunk_attach_rc) in enumerate(attachments):
            branch_xy = np.asarray([branch_attach_rc[1], branch_attach_rc[0]], dtype=np.float32)
            dists = np.linalg.norm(point_arr - branch_xy[None, :], axis=1)
            node_idx = int(np.argmin(dists))
            distance = float(dists[node_idx])
            attachment_infos.append(
                {
                    "attach_idx": int(attach_idx),
                    "node_idx": node_idx,
                    "distance": distance,
                    "branch_xy": (int(branch_attach_rc[1]), int(branch_attach_rc[0])),
                    "trunk_xy": (int(trunk_attach_rc[1]), int(trunk_attach_rc[0])),
                }
            )

        for comp_idx, component_nodes in enumerate(nx.connected_components(graph), start=1):
            component = graph.subgraph(component_nodes).copy()
            comp_points = np.asarray([points[node_id] for node_id in component.nodes()], dtype=np.float32)
            comp_center = np.mean(comp_points, axis=0) if comp_points.size else np.zeros((2,), dtype=np.float32)
            comp_attach_infos = [info for info in attachment_infos if info["node_idx"] in component]
            if not comp_attach_infos:
                best_info = min(
                    attachment_infos,
                    key=lambda info: float(np.linalg.norm(np.asarray(info["branch_xy"], dtype=np.float32) - comp_center)),
                )
                comp_attach_infos = [best_info]

            unique_anchor_infos: Dict[int, Dict] = {}
            for info in comp_attach_infos:
                node_idx = int(info["node_idx"])
                if node_idx not in unique_anchor_infos or info["distance"] < unique_anchor_infos[node_idx]["distance"]:
                    unique_anchor_infos[node_idx] = info
            comp_attach_infos = list(unique_anchor_infos.values())
            if len(comp_attach_infos) > 1:
                # Deduplicate by trunk_xy: same trunk contact point → keep best distance
                # Different trunk_xy → always keep (each is a distinct botanical branch)
                trunk_dedup: Dict[Tuple[int, int], Dict] = {}
                for info in comp_attach_infos:
                    tk = info["trunk_xy"]
                    if tk not in trunk_dedup or info["distance"] < trunk_dedup[tk]["distance"]:
                        trunk_dedup[tk] = info
                comp_attach_infos = list(trunk_dedup.values())

            if len(comp_attach_infos) == 1:
                info = comp_attach_infos[0]
                sub_group = _build_group_from_subgraph(
                    points,
                    component,
                    list(component.nodes()),
                    anchor_node=int(info["node_idx"]),
                    group_id=f"{group_prefix}_{next_group_idx:02d}",
                    color_hex=palette[(next_group_idx - 1) % len(palette)],
                    fork_origin_group="trunk",
                )
                if sub_group is None:
                    continue
                sub_group = _prepend_anchor_to_group(sub_group, info["trunk_xy"], tuple(map(int, points[int(info["node_idx"])])))
                if sub_group is not None and _group_total_edge_length(sub_group) >= float(min_branch_length):
                    final_groups.append(sub_group)
                    next_group_idx += 1
                continue

            owner = _assign_tree_nodes_to_anchors(
                component,
                [int(info["node_idx"]) for info in comp_attach_infos],
                force_split=force_split,
            )
            produced_count = 0
            unserved_anchors: List[Dict] = []
            for info in comp_attach_infos:
                anchor_node = int(info["node_idx"])
                owned_nodes = [node_id for node_id, owner_id in owner.items() if owner_id == anchor_node and node_id in component]
                if len(owned_nodes) < 2:
                    unserved_anchors.append(info)
                    continue
                sub_group = _build_group_from_subgraph(
                    points,
                    component,
                    owned_nodes,
                    anchor_node=anchor_node,
                    group_id=f"{group_prefix}_{next_group_idx:02d}",
                    color_hex=palette[(next_group_idx - 1) % len(palette)],
                    fork_origin_group="trunk",
                )
                if sub_group is None:
                    unserved_anchors.append(info)
                    continue
                sub_group = _prepend_anchor_to_group(sub_group, info["trunk_xy"], tuple(map(int, points[anchor_node])))
                if sub_group is not None and _group_total_edge_length(sub_group) >= float(min_branch_length):
                    final_groups.append(sub_group)
                    next_group_idx += 1
                    produced_count += 1
                else:
                    unserved_anchors.append(info)

            # Fallback: for unserved anchors, force-split by cutting shortest-path edges
            if unserved_anchors and produced_count >= 1:
                for info in unserved_anchors:
                    anchor_node = int(info["node_idx"])
                    if anchor_node not in component:
                        continue
                    # Find nearest node owned by a different served anchor
                    best_path = None
                    best_target = None
                    for nbr in component.nodes():
                        if nbr == anchor_node:
                            continue
                        try:
                            path = nx.shortest_path(component, source=anchor_node, target=nbr)
                        except nx.NetworkXNoPath:
                            continue
                        if len(path) < 2:
                            continue
                        mid = len(path) // 2
                        if best_path is None or len(path) < len(best_path):
                            best_path = path
                            best_target = nbr
                    if best_path and len(best_path) >= 2:
                        # Remove midpoint edge to split the component
                        mid = len(best_path) // 2
                        u, v = best_path[mid - 1], best_path[mid]
                        if component.has_edge(u, v):
                            component.remove_edge(u, v)
                        elif component.has_edge(v, u):
                            component.remove_edge(v, u)
                        # Re-run BFS from this anchor on the cut graph
                        sub_owned = list(nx.bfs_tree(component, source=anchor_node).nodes())
                        if len(sub_owned) >= 2:
                            sub_group = _build_group_from_subgraph(
                                points, component, sub_owned,
                                anchor_node=anchor_node,
                                group_id=f"{group_prefix}_{next_group_idx:02d}",
                                color_hex=palette[(next_group_idx - 1) % len(palette)],
                                fork_origin_group="trunk",
                            )
                            if sub_group is not None:
                                sub_group = _prepend_anchor_to_group(
                                    sub_group, info["trunk_xy"], tuple(map(int, points[anchor_node])))
                                if sub_group is not None and _group_total_edge_length(sub_group) >= float(min_branch_length):
                                    final_groups.append(sub_group)
                                    next_group_idx += 1
    return final_groups


def _sample_dt_radius(dt_map: np.ndarray, point_xy: Tuple[int, int]) -> float:
    x = int(np.clip(point_xy[0], 0, dt_map.shape[1] - 1))
    y = int(np.clip(point_xy[1], 0, dt_map.shape[0] - 1))
    return float(dt_map[y, x])


def _normalize_vector(vec_xy: np.ndarray) -> np.ndarray:
    vec_xy = np.asarray(vec_xy, dtype=np.float32)
    norm = float(np.linalg.norm(vec_xy))
    if norm < 1e-6:
        return np.zeros((2,), dtype=np.float32)
    return vec_xy / norm


def _angle_deg_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    unit_a = _normalize_vector(vec_a)
    unit_b = _normalize_vector(vec_b)
    dot = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _load_junction_priors(prior_path: os.PathLike = DEFAULT_JUNCTION_PRIOR_PATH) -> Dict[str, float]:
    path = Path(prior_path)
    priors = {
        "crossing_angle_mean": 176.0,
        "crossing_angle_std": 8.0,
        "branching_angle_mean": 46.0,
        "branching_angle_std": 14.0,
        "crossing_radius_ratio_max": 1.7,
        "branching_radius_ratio_min": 1.15,
        "junction_cluster_radius": 12.0,
        "crossing_cost_threshold": 1.35,
    }
    if path.exists():
        try:
            loaded = load_json(path)
            priors.update({key: float(value) for key, value in loaded.items() if isinstance(value, (int, float))})
        except Exception:
            pass
    priors["crossing_angle_min"] = max(150.0, priors["crossing_angle_mean"] - 2.5 * priors["crossing_angle_std"])
    priors["branching_angle_low"] = max(15.0, priors["branching_angle_mean"] - 1.5 * priors["branching_angle_std"])
    priors["branching_angle_high"] = min(95.0, priors["branching_angle_mean"] + 1.5 * priors["branching_angle_std"])
    return priors


def _build_group_topology_graph(groups: Sequence[Dict]) -> Tuple[nx.Graph, Dict[Tuple[int, int], int]]:
    graph = nx.Graph()
    point_to_node: Dict[Tuple[int, int], int] = {}
    next_node_id = 0

    def ensure_node(point_xy: Sequence[int], group_type: str) -> int:
        nonlocal next_node_id
        point = (int(point_xy[0]), int(point_xy[1]))
        if point in point_to_node:
            node_id = point_to_node[point]
            graph.nodes[node_id]["is_trunk"] = bool(graph.nodes[node_id]["is_trunk"] or group_type == "trunk")
            graph.nodes[node_id]["group_types"].add(group_type)
            return node_id
        node_id = next_node_id
        next_node_id += 1
        point_to_node[point] = node_id
        graph.add_node(node_id, point=point, is_trunk=bool(group_type == "trunk"), group_types={group_type})
        return node_id

    for group in groups:
        group_type = group.get("group_type", "branch")
        local_to_global: Dict[int, int] = {}
        for idx, point_xy in enumerate(group.get("points", [])):
            local_to_global[idx] = ensure_node(point_xy, group_type)
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src = local_to_global.get(int(edge[0]))
            dst = local_to_global.get(int(edge[1]))
            if src is None or dst is None or src == dst:
                continue
            graph.add_edge(src, dst, group_type=group_type, group_id=group.get("group_id", group_type))
    return graph, point_to_node


def _cluster_junction_nodes(graph: nx.Graph, radius: float) -> List[List[int]]:
    junction_nodes = [node_id for node_id in graph.nodes() if graph.degree[node_id] >= 3]
    junction_set = set(junction_nodes)
    adjacency = nx.Graph()
    adjacency.add_nodes_from(junction_nodes)
    radius_sq = float(radius * radius)
    for index, node_a in enumerate(junction_nodes):
        point_a = np.asarray(graph.nodes[node_a]["point"], dtype=np.float32)
        for node_b in junction_nodes[index + 1:]:
            point_b = np.asarray(graph.nodes[node_b]["point"], dtype=np.float32)
            if float(np.sum((point_a - point_b) ** 2)) <= radius_sq:
                adjacency.add_edge(node_a, node_b, reason="spatial_cluster")

    # A raster X is often simplified into two nearby degree-3 T nodes joined by
    # a short overlap segment.  Euclidean radius clustering misses that motif,
    # so neither T can expose the four external arms required by the crossing
    # solver.  Merge only non-trunk degree-3 pairs whose connecting path has no
    # intervening junction; the downstream two-pair geometry and bud-flow
    # solver still decides whether it is truly an X or two botanical forks.
    weighted = graph.copy()
    for src, dst in weighted.edges():
        src_xy = np.asarray(weighted.nodes[src]["point"], dtype=np.float32)
        dst_xy = np.asarray(weighted.nodes[dst]["point"], dtype=np.float32)
        weighted.edges[src, dst]["junction_distance"] = float(np.linalg.norm(src_xy - dst_xy))
    topology_radius = max(float(radius) * 4.0, 48.0)
    for index, node_a in enumerate(junction_nodes):
        if graph.degree[node_a] != 3 or bool(graph.nodes[node_a].get("is_trunk")):
            continue
        for node_b in junction_nodes[index + 1:]:
            if graph.degree[node_b] != 3 or bool(graph.nodes[node_b].get("is_trunk")):
                continue
            try:
                path = nx.shortest_path(weighted, node_a, node_b, weight="junction_distance")
                path_length = nx.path_weight(weighted, path, weight="junction_distance")
            except nx.NetworkXNoPath:
                continue
            if path_length > topology_radius:
                continue
            if any(node in junction_set for node in path[1:-1]):
                continue
            adjacency.add_edge(node_a, node_b, reason="degree3_x_motif")
    return [sorted(component) for component in nx.connected_components(adjacency)]


def _trace_arm_polyline(
    graph: nx.Graph,
    inside_id: int,
    outside_id: int,
    cluster_nodes: Sequence[int],
    max_hops: int = 6,
    max_arc_length: float = 48.0,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    cluster_set = set(int(node_id) for node_id in cluster_nodes)
    node_ids = [int(inside_id), int(outside_id)]
    polyline_xy = [
        tuple(map(int, graph.nodes[inside_id]["point"])),
        tuple(map(int, graph.nodes[outside_id]["point"])),
    ]
    prev_id = int(inside_id)
    current_id = int(outside_id)
    arc_length = _polyline_length(polyline_xy)
    while len(node_ids) < max_hops + 2 and arc_length < float(max_arc_length):
        if current_id not in graph or graph.degree[current_id] != 2:
            break
        next_candidates = [nbr for nbr in graph.neighbors(current_id) if nbr != prev_id and nbr not in cluster_set]
        if len(next_candidates) != 1:
            break
        next_id = int(next_candidates[0])
        if next_id in node_ids:
            break
        node_ids.append(next_id)
        polyline_xy.append(tuple(map(int, graph.nodes[next_id]["point"])))
        prev_id, current_id = current_id, next_id
        arc_length = _polyline_length(polyline_xy)
    return node_ids, _dedupe_consecutive(polyline_xy)


def _estimate_tangent_from_polyline(polyline_xy: Sequence[Tuple[int, int]]) -> np.ndarray:
    if len(polyline_xy) < 2:
        return np.zeros((2,), dtype=np.float32)
    pts = np.asarray(polyline_xy, dtype=np.float32)
    deltas = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > 1e-6
    if not np.any(valid):
        return _normalize_vector(pts[-1] - pts[0])
    weighted = (deltas[valid] * lengths[valid][:, None]).sum(axis=0)
    return _normalize_vector(weighted)


def _polyline_curvature_features(polyline_xy: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    if len(polyline_xy) < 2:
        return {
            "arc_length": 0.0,
            "chord_length": 0.0,
            "straightness": 1.0,
            "mean_turn_deg": 0.0,
            "max_turn_deg": 0.0,
        }
    pts = np.asarray(polyline_xy, dtype=np.float32)
    deltas = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > 1e-6
    arc_length = float(lengths.sum())
    chord_length = float(np.linalg.norm(pts[-1] - pts[0]))
    if np.count_nonzero(valid) < 2:
        return {
            "arc_length": arc_length,
            "chord_length": chord_length,
            "straightness": chord_length / max(arc_length, 1e-6),
            "mean_turn_deg": 0.0,
            "max_turn_deg": 0.0,
        }
    unit = deltas[valid] / lengths[valid][:, None]
    turn_angles: List[float] = []
    for vec_a, vec_b in zip(unit[:-1], unit[1:]):
        turn_angles.append(_angle_deg_between(vec_a, vec_b))
    mean_turn_deg = float(np.mean(turn_angles)) if turn_angles else 0.0
    max_turn_deg = float(np.max(turn_angles)) if turn_angles else 0.0
    return {
        "arc_length": arc_length,
        "chord_length": chord_length,
        "straightness": chord_length / max(arc_length, 1e-6),
        "mean_turn_deg": mean_turn_deg,
        "max_turn_deg": max_turn_deg,
    }


def _segment_tangent(polyline_xy: Sequence[Tuple[int, int]], from_start: bool = True, segment_count: int = 3) -> np.ndarray:
    if len(polyline_xy) < 2:
        return np.zeros((2,), dtype=np.float32)
    pts = np.asarray(polyline_xy, dtype=np.float32)
    if not from_start:
        pts = pts[::-1]
    max_seg = min(max(len(pts) - 1, 1), max(int(segment_count), 1))
    deltas = pts[1:max_seg + 1] - pts[:max_seg]
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > 1e-6
    if not np.any(valid):
        return _normalize_vector(pts[-1] - pts[0])
    weighted = (deltas[valid] * lengths[valid][:, None]).sum(axis=0)
    return _normalize_vector(weighted)


def _combined_polyline_features(polyline_a: Sequence[Tuple[int, int]], polyline_b: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    merged = list(reversed([tuple(map(int, point)) for point in polyline_a]))
    tail = [tuple(map(int, point)) for point in polyline_b]
    if merged and tail and merged[-1] == tail[0]:
        merged.extend(tail[1:])
    else:
        merged.extend(tail)
    return _polyline_curvature_features(_dedupe_consecutive(merged))


def _arm_growth_priors(point_xy: Tuple[int, int], vector: np.ndarray, root_point_xy: Optional[Tuple[int, int]]) -> Dict[str, float]:
    up_vec = np.asarray([0.0, -1.0], dtype=np.float32)
    up_alignment = float(np.dot(_normalize_vector(vector), up_vec))
    if root_point_xy is None:
        return {"up_alignment": up_alignment, "away_from_root_alignment": up_alignment}
    root_vec = _normalize_vector(np.asarray(point_xy, dtype=np.float32) - np.asarray(root_point_xy, dtype=np.float32))
    away_alignment = float(np.dot(_normalize_vector(vector), root_vec))
    return {"up_alignment": up_alignment, "away_from_root_alignment": away_alignment}


def _trunk_axis_from_pair(arm_a: Dict, arm_b: Dict) -> np.ndarray:
    axis = _normalize_vector(arm_a["vector"] - arm_b["vector"])
    if float(np.linalg.norm(axis)) < 1e-6:
        axis = _normalize_vector(arm_a["axis_vector"] - arm_b["axis_vector"])
    if float(np.linalg.norm(axis)) < 1e-6:
        axis = np.asarray([0.0, -1.0], dtype=np.float32)
    return axis


def _adjust_cost_by_bud_centers(
    cost: np.ndarray,
    mask: np.ndarray,
    bud_centers: List[Tuple[float, float]],
    dt_norm: np.ndarray,
    influence_radius: float = 35.0,
    max_reduction: float = 0.25,
) -> np.ndarray:
    """Module 6: 芽点辅助分割消歧 — 降低芽点附近 cost.

    细枝区域 (低 dt) 额外加权, 因为细枝芽点信号更重要.
    """
    if not bud_centers:
        return cost

    H, W = cost.shape
    mask_bool = mask.astype(bool)
    bud_boost = np.zeros((H, W), dtype=np.float32)
    sigma = influence_radius / 3.0

    for cx, cy in bud_centers:
        sx, sy = int(round(cx)), int(round(cy))
        if not (0 <= sx < W and 0 <= sy < H):
            continue
        if not mask_bool[sy, sx]:
            continue

        r = int(np.ceil(influence_radius))
        y1, y2 = max(0, sy - r), min(H, sy + r + 1)
        x1, x2 = max(0, sx - r), min(W, sx + r + 1)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        bud_boost[y1:y2, x1:x2] = np.maximum(bud_boost[y1:y2, x1:x2], gaussian)

    thin_penalty = np.maximum(0.0, 0.5 - dt_norm) * 2.0
    adjustment = 1.0 - bud_boost * max_reduction * (1.0 + thin_penalty)
    adjustment = np.clip(adjustment, 1.0 - max_reduction * 2.0, 1.0)
    adjustment[~mask_bool] = 1.0

    return (cost * adjustment).astype(np.float32)


def _point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """每个 point 到折线段的最近距离 (向量化).

    Args:
        points: (N, 2) float32 array of query points.
        polyline: (M, 2) float32 array of polyline vertices.

    Returns:
        (N,) float32 array of minimum distances.
    """
    if len(polyline) < 2:
        return np.linalg.norm(points - polyline[0], axis=1)
    seg_vecs = polyline[1:] - polyline[:-1]
    seg_lens_sq = np.sum(seg_vecs ** 2, axis=1)
    seg_lens_sq = np.where(seg_lens_sq < 1e-9, 1e-9, seg_lens_sq)
    best = np.full(len(points), np.inf, dtype=np.float32)
    for i, (seg, seg_len_sq) in enumerate(zip(seg_vecs, seg_lens_sq)):
        diff = points - polyline[i]
        t = np.clip(np.sum(diff * seg, axis=1) / seg_len_sq, 0.0, 1.0)
        proj = polyline[i] + t[:, np.newaxis] * seg
        dists = np.linalg.norm(points - proj, axis=1)
        best = np.minimum(best, dists)
    return best


def _extract_cluster_arms(
    graph: nx.Graph,
    cluster_nodes: Sequence[int],
    dt_map: np.ndarray,
    root_point_xy: Optional[Tuple[int, int]] = None,
) -> List[Dict]:
    cluster_set = set(int(node_id) for node_id in cluster_nodes)
    cluster_points = np.asarray([graph.nodes[node_id]["point"] for node_id in cluster_nodes], dtype=np.float32)
    center_xy = np.mean(cluster_points, axis=0) if cluster_points.size else np.zeros((2,), dtype=np.float32)
    arms: List[Dict] = []
    seen_edges = set()
    for inside_id in cluster_nodes:
        for outside_id in graph.neighbors(inside_id):
            if outside_id in cluster_set:
                continue
            edge_key = tuple(sorted((int(inside_id), int(outside_id))))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            node_ids, polyline_xy = _trace_arm_polyline(graph, int(inside_id), int(outside_id), cluster_nodes)
            bud_support_xy = list(polyline_xy)
            support_ids = list(node_ids)
            support_length = float(_polyline_length(bud_support_xy))
            while len(support_ids) >= 2 and support_length < 320.0:
                previous_id, current_id = support_ids[-2], support_ids[-1]
                current_xy = np.asarray(graph.nodes[current_id]["point"], dtype=np.float32)
                previous_xy = np.asarray(graph.nodes[previous_id]["point"], dtype=np.float32)
                incoming = _normalize_vector(current_xy - previous_xy)
                candidates = [
                    neighbor for neighbor in graph.neighbors(current_id)
                    if neighbor != previous_id and neighbor not in cluster_set and neighbor not in support_ids
                ]
                if not candidates:
                    break
                scored = []
                for neighbor in candidates:
                    neighbor_xy = np.asarray(graph.nodes[neighbor]["point"], dtype=np.float32)
                    alignment = float(np.dot(incoming, _normalize_vector(neighbor_xy - current_xy)))
                    scored.append((alignment, int(neighbor)))
                alignment, next_id = max(scored)
                if alignment < 0.2:
                    break
                next_xy = tuple(map(int, graph.nodes[next_id]["point"]))
                support_length += float(np.linalg.norm(np.asarray(next_xy, dtype=np.float32) - current_xy))
                support_ids.append(next_id)
                bud_support_xy.append(next_xy)
            tangent = _segment_tangent(polyline_xy, from_start=True, segment_count=4)
            if float(np.linalg.norm(tangent)) < 1e-6 and len(polyline_xy) >= 2:
                tangent = _normalize_vector(np.asarray(polyline_xy[-1], dtype=np.float32) - np.asarray(polyline_xy[0], dtype=np.float32))
            tail_tangent = _segment_tangent(polyline_xy, from_start=False, segment_count=4)
            axis_vector = _normalize_vector(np.asarray(polyline_xy[-1], dtype=np.float32) - np.asarray(polyline_xy[0], dtype=np.float32))
            radii = [_sample_dt_radius(dt_map, point_xy) for point_xy in polyline_xy]
            curve_stats = _polyline_curvature_features(polyline_xy)
            edge_attrs = graph.edges[inside_id, outside_id]
            edge_group_type = edge_attrs.get("group_type", "branch")
            is_trunk_arm = bool(
                edge_group_type == "trunk"
                or graph.nodes[inside_id].get("is_trunk")
                or graph.nodes[outside_id].get("is_trunk")
            )
            growth_priors = _arm_growth_priors(tuple(map(int, polyline_xy[-1])), tangent, root_point_xy=root_point_xy)
            support_length = float(_polyline_length(polyline_xy))
            arms.append(
                {
                    "inside_id": int(inside_id),
                    "outside_id": int(outside_id),
                    "inside_xy": tuple(map(int, graph.nodes[inside_id]["point"])),
                    "outside_xy": tuple(map(int, graph.nodes[outside_id]["point"])),
                    "center_xy": tuple(map(float, center_xy)),
                    "vector": tangent,
                    "tail_vector": tail_tangent,
                    "axis_vector": axis_vector,
                    "radius": float(np.median(radii)) if radii else float(max(_sample_dt_radius(dt_map, graph.nodes[inside_id]["point"]), _sample_dt_radius(dt_map, graph.nodes[outside_id]["point"]))),
                    "group_type": edge_group_type,
                    "group_id": edge_attrs.get("group_id", edge_group_type),
                    "polyline_xy": [tuple(map(int, point)) for point in polyline_xy],
                    "bud_support_polyline_xy": [tuple(map(int, point)) for point in bud_support_xy],
                    "bud_support_length": float(support_length),
                    "support_length": support_length,
                    "length_confidence": float(np.clip(support_length / 18.0, 0.15, 1.0)),
                    "straightness": float(curve_stats["straightness"]),
                    "mean_turn_deg": float(curve_stats["mean_turn_deg"]),
                    "max_turn_deg": float(curve_stats["max_turn_deg"]),
                    "arc_length": float(curve_stats["arc_length"]),
                    "chord_length": float(curve_stats["chord_length"]),
                    "is_trunk_arm": bool(is_trunk_arm),
                    "up_alignment": float(growth_priors["up_alignment"]),
                    "away_from_root_alignment": float(growth_priors["away_from_root_alignment"]),
                    "node_ids": [int(node_id) for node_id in node_ids],
                }
            )
    return arms


def _enumerate_pairings(indices: Sequence[int]) -> List[List[Tuple[int, int]]]:
    indices = list(indices)
    if not indices:
        return [[]]
    if len(indices) % 2 == 1:
        return []
    first = indices[0]
    solutions: List[List[Tuple[int, int]]] = []
    for pos in range(1, len(indices)):
        second = indices[pos]
        remain = indices[1:pos] + indices[pos + 1:]
        for tail in _enumerate_pairings(remain):
            solutions.append([(first, second)] + tail)
    return solutions


def _compute_arm_bud_direction_alignment(
    arm: Dict,
    bud_centers: np.ndarray,
    bud_angles: np.ndarray,
    bud_elongated: np.ndarray,
    search_radius: float = 30.0,
) -> Dict[str, float]:
    """计算arm上elongated芽点方向与arm切线方向的对齐程度.

    bud axis是无符号的[0,π), arm tangent是有符号方向.
    使用min(|diff|, π-|diff|)做无符号比较, 输出[0, π/2]范围的角差.

    Returns:
        n_elongated_buds: arm附近elongated芽点数
        mean_angle_diff: 平均角差(度), 越小越对齐
        alignment_score: [0,1], 1=完美对齐, 0=垂直
        direction_std: 芽点方向标准差(度), 越小越一致
    """
    polyline = np.asarray(arm.get("bud_support_polyline_xy", arm.get("polyline_xy", [])), dtype=np.float32)
    if len(polyline) < 2:
        return {"n_elongated_buds": 0, "mean_angle_diff": 45.0,
                "alignment_score": 0.5, "direction_std": 45.0}

    tangent = np.array(arm.get("vector", [0.0, 0.0]), dtype=np.float32)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-6:
        return {"n_elongated_buds": 0, "mean_angle_diff": 45.0,
                "alignment_score": 0.5, "direction_std": 45.0}
    arm_angle = float(np.arctan2(tangent[1], tangent[0]))

    dists = _point_to_polyline_distances(bud_centers, polyline)
    nearby = (dists <= search_radius) & bud_elongated
    n_buds = int(np.sum(nearby))

    if n_buds == 0:
        return {"n_elongated_buds": 0, "mean_angle_diff": 45.0,
                "alignment_score": 0.5, "direction_std": 45.0}

    nearby_angles = bud_angles[nearby]
    # 无符号比较: axis在θ和θ+π等价
    angle_diffs = np.abs(nearby_angles - arm_angle)
    angle_diffs = np.minimum(angle_diffs, np.pi - angle_diffs)
    angle_diffs_deg = np.rad2deg(angle_diffs)

    mean_diff = float(np.mean(angle_diffs_deg))
    std_diff = float(np.std(angle_diffs_deg))
    alignment_score = max(0.0, 1.0 - mean_diff / 90.0)

    return {
        "n_elongated_buds": n_buds,
        "mean_angle_diff": mean_diff,
        "alignment_score": alignment_score,
        "direction_std": std_diff,
    }


def _compute_arm_bud_flow(
    arm: Dict,
    bud_centers: np.ndarray,
    bud_directions: Optional[Sequence],
    search_radius: float,
) -> Dict[str, float]:
    polyline = np.asarray(arm.get("bud_support_polyline_xy", arm.get("polyline_xy", [])), dtype=np.float32)
    support_length = max(float(arm.get("bud_support_length", arm.get("support_length", _polyline_length(polyline)))), 1.0)
    if len(polyline) < 2 or len(bud_centers) == 0:
        return {
            "bud_count": 0.0,
            "effective_buds": 0.0,
            "density_per_1000px": 0.0,
            "terminal_progress": 0.5,
            "flow_vote": 0.0,
            "flow_confidence": 0.0,
        }
    dists = _point_to_polyline_distances(bud_centers, polyline)
    nearby_indices = np.where(dists <= search_radius)[0]
    if len(nearby_indices) == 0:
        return {
            "bud_count": 0.0,
            "effective_buds": 0.0,
            "density_per_1000px": 0.0,
            "terminal_progress": 0.5,
            "flow_vote": 0.0,
            "flow_confidence": 0.0,
        }
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(polyline[1:] - polyline[:-1], axis=1))])
    progress_values = []
    for bud_idx in nearby_indices:
        nearest_vertex = int(np.argmin(np.linalg.norm(polyline - bud_centers[bud_idx], axis=1)))
        progress_values.append(float(cumulative[nearest_vertex] / max(cumulative[-1], 1e-6)))
    direction_by_index = {}
    if bud_directions is not None:
        for direction in bud_directions:
            get = (lambda key: getattr(direction, key)) if hasattr(direction, "vector_xy") else (lambda key: direction.get(key))
            direction_by_index[int(get("bud_index"))] = direction
    arm_vector = _normalize_vector(np.asarray(arm.get("vector", [0.0, 0.0]), dtype=np.float32))
    signed_sum = 0.0
    direction_weight = 0.0
    effective_buds = 0.0
    for bud_idx in nearby_indices:
        direction = direction_by_index.get(int(bud_idx))
        if direction is None:
            continue
        get = (lambda key: getattr(direction, key)) if hasattr(direction, "vector_xy") else (lambda key: direction.get(key))
        confidence = float(get("confidence") or 0.0)
        reliable = bool(get("is_reliable"))
        latent_spur = bool(get("is_latent_spur"))
        if not reliable or latent_spur or confidence <= 0.0:
            continue
        vector = _normalize_vector(np.asarray(get("vector_xy"), dtype=np.float32))
        projection = float(np.dot(vector, arm_vector))
        weight = confidence * max(abs(projection) - 0.15, 0.0)
        signed_sum += weight * float(np.sign(projection))
        direction_weight += weight
        effective_buds += confidence
    flow_vote = signed_sum / max(direction_weight, 1e-6) if direction_weight > 0.0 else 0.0
    return {
        "bud_count": float(len(nearby_indices)),
        "effective_buds": float(effective_buds),
        "density_per_1000px": float(len(nearby_indices) / support_length * 1000.0),
        "terminal_progress": float(np.mean(progress_values)) if progress_values else 0.5,
        "flow_vote": float(flow_vote),
        "flow_confidence": float(np.clip(direction_weight / 1.5, 0.0, 1.0)),
    }


def _pair_cost(arm_a: Dict, arm_b: Dict, priors: Dict[str, float],
              bud_counts: Optional[Dict[int, int]] = None,
              idx_a: int = -1, idx_b: int = -1,
              arm_bud_dir_scores: Optional[Dict[int, Dict[str, float]]] = None,
              arm_bud_flows: Optional[Dict[int, Dict[str, float]]] = None) -> Dict[str, float]:
    angle_deg = _angle_deg_between(arm_a["vector"], arm_b["vector"])
    min_radius = max(min(float(arm_a["radius"]), float(arm_b["radius"])), 1e-3)
    radius_ratio = max(float(arm_a["radius"]), float(arm_b["radius"])) / min_radius
    combined_curve = _combined_polyline_features(arm_a["polyline_xy"], arm_b["polyline_xy"])
    curvature_cost = combined_curve["mean_turn_deg"] / 18.0 + combined_curve["max_turn_deg"] / 75.0
    curvature_cost += max(0.0, 0.92 - combined_curve["straightness"]) * 6.0
    length_conf = float(np.sqrt(max(arm_a.get("length_confidence", 0.15), 1e-3) * max(arm_b.get("length_confidence", 0.15), 1e-3)))
    length_penalty = max(0.0, 0.85 - length_conf) * 3.5
    growth_penalty = 0.0
    if arm_a.get("is_trunk_arm") != arm_b.get("is_trunk_arm"):
        growth_penalty += 6.0
    branch_growth_scores = [
        arm.get("away_from_root_alignment", 0.0)
        for arm in (arm_a, arm_b)
        if not arm.get("is_trunk_arm")
    ]
    if branch_growth_scores:
        growth_penalty += max(0.0, 0.10 - min(branch_growth_scores)) * 4.0
        growth_penalty += max(0.0, -max(arm.get("up_alignment", 0.0) for arm in (arm_a, arm_b) if not arm.get("is_trunk_arm"))) * 2.0
    bud_penalty = 0.0
    if bud_counts is not None and idx_a >= 0 and idx_b >= 0:
        density_a = float(bud_counts.get(idx_a, 0.0))
        density_b = float(bud_counts.get(idx_b, 0.0))
        total = density_a + density_b
        if density_a > 0.0 and density_b > 0.0:
            bud_penalty = abs(density_a - density_b) / max(total, 1e-6) * 0.35
    bud_dir_penalty = 0.0
    if arm_bud_dir_scores is not None and idx_a >= 0 and idx_b >= 0:
        score_a = arm_bud_dir_scores.get(idx_a, {}).get("alignment_score", 0.5)
        score_b = arm_bud_dir_scores.get(idx_b, {}).get("alignment_score", 0.5)
        avg_score = (score_a + score_b) / 2.0
        # 高对齐 → 低惩罚; 低对齐 → 高惩罚 (max 4.0)
        bud_dir_penalty = (1.0 - avg_score) * 4.0
    bud_flow_penalty = 0.0
    if arm_bud_flows is not None and idx_a >= 0 and idx_b >= 0:
        flow_a = arm_bud_flows.get(idx_a, {})
        flow_b = arm_bud_flows.get(idx_b, {})
        confidence = min(float(flow_a.get("flow_confidence", 0.0)), float(flow_b.get("flow_confidence", 0.0)))
        if confidence > 0.0:
            same_direction = max(0.0, float(flow_a.get("flow_vote", 0.0)) * float(flow_b.get("flow_vote", 0.0)))
            bud_flow_penalty += same_direction * confidence * 1.25
        count_a = float(flow_a.get("bud_count", 0.0))
        count_b = float(flow_b.get("bud_count", 0.0))
        if count_a >= 2.0 and count_b >= 2.0:
            progress_sum = float(flow_a.get("terminal_progress", 0.5)) + float(flow_b.get("terminal_progress", 0.5))
            bud_flow_penalty += abs(progress_sum - 1.0) * 0.25
    geometry_crossing_cost = abs(angle_deg - priors["crossing_angle_mean"]) / max(priors["crossing_angle_std"], 1.0)
    geometry_crossing_cost += max(0.0, radius_ratio - priors["crossing_radius_ratio_max"])
    geometry_crossing_cost += curvature_cost + length_penalty + growth_penalty
    if arm_a.get("group_type") != arm_b.get("group_type"):
        geometry_crossing_cost += 0.35
    crossing_cost = geometry_crossing_cost + bud_penalty + bud_dir_penalty + bud_flow_penalty
    branch_cost = abs(angle_deg - priors["branching_angle_mean"]) / max(priors["branching_angle_std"], 1.0)
    branch_cost += max(0.0, priors["branching_radius_ratio_min"] - radius_ratio) * 2.0
    branch_cost += max(0.0, 0.75 - length_conf) * 1.2
    return {
        "angle_deg": float(angle_deg),
        "radius_ratio": float(radius_ratio),
        "crossing_cost": float(crossing_cost),
        "geometry_crossing_cost": float(geometry_crossing_cost),
        "branch_cost": float(branch_cost),
        "curvature_cost": float(curvature_cost),
        "length_confidence": float(length_conf),
        "growth_penalty": float(growth_penalty),
        "bud_penalty": float(bud_penalty),
        "bud_dir_penalty": float(bud_dir_penalty),
        "bud_flow_penalty": float(bud_flow_penalty),
        "combined_mean_turn_deg": float(combined_curve["mean_turn_deg"]),
        "combined_straightness": float(combined_curve["straightness"]),
        "is_crossing_like": bool(angle_deg >= priors["crossing_angle_min"] and radius_ratio <= priors["crossing_radius_ratio_max"] + 0.35),
        "is_branch_like": bool(priors["branching_angle_low"] <= angle_deg <= priors["branching_angle_high"] and radius_ratio >= priors["branching_radius_ratio_min"]),
    }


def _bud_flow_supports_independent_roots(
    group: Dict,
    attachments: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    bud_centers: Optional[Sequence[Tuple[float, float]]],
    bud_directions: Optional[Sequence],
    search_radius: float = 20.0,
) -> bool:
    if not bud_centers or not bud_directions or len(attachments) < 2:
        return False
    points = np.asarray(group.get("points", []), dtype=np.float32)
    if len(points) < 2:
        return False
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if 0 <= src < len(points) and 0 <= dst < len(points) and src != dst:
            graph.add_edge(src, dst, weight=max(float(np.linalg.norm(points[dst] - points[src])), 1e-6))
    if graph.number_of_edges() == 0:
        return False
    root_indices = []
    for endpoint_rc, _ in attachments:
        endpoint_xy = np.asarray([endpoint_rc[1], endpoint_rc[0]], dtype=np.float32)
        root_indices.append(int(np.argmin(np.linalg.norm(points - endpoint_xy, axis=1))))
    root_distances = [nx.single_source_dijkstra_path_length(graph, root, weight="weight") for root in root_indices]
    direction_by_index = {}
    for direction in bud_directions:
        get = (lambda key: getattr(direction, key)) if hasattr(direction, "vector_xy") else (lambda key: direction.get(key))
        direction_by_index[int(get("bud_index"))] = direction
    support = [{"count": 0, "signed": 0.0, "weight": 0.0} for _ in root_indices]
    centers = np.asarray(bud_centers, dtype=np.float32)
    for bud_index, center in enumerate(centers):
        point_distances = np.linalg.norm(points - center, axis=1)
        nearest_node = int(np.argmin(point_distances))
        if float(point_distances[nearest_node]) > search_radius:
            continue
        direction = direction_by_index.get(bud_index)
        if direction is None:
            continue
        get = (lambda key: getattr(direction, key)) if hasattr(direction, "vector_xy") else (lambda key: direction.get(key))
        if not bool(get("is_reliable")) or bool(get("is_latent_spur")):
            continue
        root_slot = int(np.argmin([distances.get(nearest_node, float("inf")) for distances in root_distances]))
        root = root_indices[root_slot]
        try:
            path = nx.shortest_path(graph, root, nearest_node, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if len(path) >= 2:
            src, dst = path[-2], path[-1]
        else:
            neighbors = list(graph.neighbors(root))
            if not neighbors:
                continue
            src, dst = root, max(neighbors, key=lambda node: root_distances[root_slot].get(node, 0.0))
        tangent = _normalize_vector(points[dst] - points[src])
        vector = _normalize_vector(np.asarray(get("vector_xy"), dtype=np.float32))
        projection = float(np.dot(vector, tangent))
        if abs(projection) < 0.2:
            continue
        confidence = float(get("confidence") or 0.0)
        weight = confidence * abs(projection)
        support[root_slot]["count"] += 1
        support[root_slot]["signed"] += weight * float(np.sign(projection))
        support[root_slot]["weight"] += weight
    return all(
        item["count"] >= 2
        and item["weight"] >= 0.4
        and item["signed"] / max(item["weight"], 1e-6) > 0.2
        for item in support
    )


def _solve_crossing_pairs(arms: Sequence[Dict], priors: Dict[str, float],
                         bud_counts: Optional[Dict[int, int]] = None,
                         arm_bud_dir_scores: Optional[Dict[int, Dict[str, float]]] = None,
                         arm_bud_flows: Optional[Dict[int, Dict[str, float]]] = None) -> Optional[Dict]:
    if len(arms) < 2:
        return None

    pair_candidates: List[Dict] = []
    relaxed_cost_limit = float(priors["crossing_cost_threshold"]) + 1.25
    relaxed_angle_min = float(priors["crossing_angle_min"]) - 10.0
    for idx_a in range(len(arms)):
        for idx_b in range(idx_a + 1, len(arms)):
            info = _pair_cost(arms[idx_a], arms[idx_b], priors,
                            bud_counts=bud_counts, idx_a=idx_a, idx_b=idx_b,
                            arm_bud_dir_scores=arm_bud_dir_scores,
                            arm_bud_flows=arm_bud_flows)
            info["indices"] = (int(idx_a), int(idx_b))
            if info["angle_deg"] < relaxed_angle_min:
                continue
            if info["geometry_crossing_cost"] > relaxed_cost_limit:
                continue
            pair_candidates.append(info)

    if not pair_candidates:
        return None

    best_geometry_cost = min(float(info["geometry_crossing_cost"]) for info in pair_candidates)
    for info in pair_candidates:
        geometry_cost = float(info["geometry_crossing_cost"])
        bud_adjustment = float(info["crossing_cost"] - geometry_cost)
        info["bud_rerank_active"] = bool(geometry_cost <= best_geometry_cost + 0.75)
        info["selection_cost"] = geometry_cost + bud_adjustment if info["bud_rerank_active"] else geometry_cost

    pair_candidates.sort(
        key=lambda info: (
            info["selection_cost"],
            0 if arms[info["indices"][0]].get("is_trunk_arm") == arms[info["indices"][1]].get("is_trunk_arm") else 1,
            -info["angle_deg"],
            abs(info["radius_ratio"] - 1.0),
            info["combined_mean_turn_deg"],
        )
    )
    used_indices = set()
    selected: List[Dict] = []
    for info in pair_candidates:
        idx_a, idx_b = info["indices"]
        if idx_a in used_indices or idx_b in used_indices:
            continue
        selected.append(info)
        used_indices.add(idx_a)
        used_indices.add(idx_b)

    if not selected:
        return None

    paired_arm_count = len(used_indices)
    leftover_indices = [idx for idx in range(len(arms)) if idx not in used_indices]
    if paired_arm_count < 2:
        return None
    if len(leftover_indices) == 1:
        return None

    total_cost = float(sum(info["selection_cost"] for info in selected))
    avg_cost = total_cost / max(len(selected), 1)
    geometry_avg_cost = float(sum(info["geometry_crossing_cost"] for info in selected)) / max(len(selected), 1)
    if geometry_avg_cost > relaxed_cost_limit:
        return None
    return {
        "pairing": [tuple(map(int, info["indices"])) for info in selected],
        "pair_infos": selected,
        "total_cost": total_cost,
        "avg_cost": float(avg_cost),
        "leftover_indices": [int(idx) for idx in leftover_indices],
    }


def _solve_degree3_x_motif(
    arms: Sequence[Dict],
    arm_bud_flows: Optional[Dict[int, Dict[str, float]]] = None,
) -> Optional[Dict]:
    """Pair the four external arms of two short-linked degree-3 junctions."""
    if len(arms) != 4:
        return None
    candidates = []
    for pairing in _enumerate_pairings(range(4)):
        pair_infos = []
        geometry_cost = 0.0
        veto_count = 0
        reliable_support_count = 0
        minimum_angle = 180.0
        for idx_a, idx_b in pairing:
            angle = _angle_deg_between(arms[idx_a]["vector"], arms[idx_b]["vector"])
            combined = _combined_polyline_features(
                arms[idx_a].get("polyline_xy", []), arms[idx_b].get("polyline_xy", []),
            )
            geometry_cost += abs(180.0 - angle) + 20.0 * (1.0 - float(combined["straightness"]))
            minimum_angle = min(minimum_angle, float(angle))
            flow_a = (arm_bud_flows or {}).get(idx_a, {})
            flow_b = (arm_bud_flows or {}).get(idx_b, {})
            confidence = min(float(flow_a.get("flow_confidence", 0.0)),
                             float(flow_b.get("flow_confidence", 0.0)))
            product = float(flow_a.get("flow_vote", 0.0)) * float(flow_b.get("flow_vote", 0.0))
            # Reliable flow on the two outward-pointing arms of one continuous
            # branch should normally have opposite signs.  Same-sign strong
            # evidence vetoes that slash pairing; weak evidence cannot force it.
            evidence_sufficient = bool(
                float(flow_a.get("effective_buds", 0.0)) >= 2.0
                and float(flow_b.get("effective_buds", 0.0)) >= 2.0
                and confidence >= 0.4
            )
            bud_veto = bool(evidence_sufficient and product > 0.2)
            reliable_support = bool(evidence_sufficient and product < -0.2)
            veto_count += int(bud_veto)
            reliable_support_count += int(reliable_support)
            pair_infos.append({
                "indices": (int(idx_a), int(idx_b)),
                "angle_deg": float(angle),
                "combined_straightness": float(combined["straightness"]),
                "bud_flow_product": float(product),
                "bud_flow_confidence": float(confidence),
                "bud_evidence_sufficient": bool(evidence_sufficient),
                "bud_continuation_support": bool(reliable_support),
                "bud_flow_veto": bool(bud_veto),
            })
        mean_angle = float(np.mean([info["angle_deg"] for info in pair_infos]))
        candidates.append((int(veto_count), -int(reliable_support_count), float(geometry_cost),
                           float(minimum_angle), mean_angle, pairing, pair_infos))
    # GT calibration: the 95th percentile of the best-paired minimum angle
    # among genuine within-group short T+T motifs is 149.6 degrees.  Requiring
    # 150 degrees keeps the node-cancellation rule outside 95% of true motifs.
    viable = [
        candidate for candidate in candidates
        if candidate[4] >= 160.0
        and (candidate[3] >= 150.0 or (candidate[3] >= 140.0 and candidate[1] <= -1))
    ]
    if not viable:
        return None
    veto_count, negative_support_count, geometry_cost, minimum_angle, mean_angle, pairing, pair_infos = min(
        viable, key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
    )
    if veto_count > 0:
        return None
    return {
        "mode": "degree3_x_motif",
        "pairing": [tuple(map(int, pair)) for pair in pairing],
        "pair_infos": pair_infos,
        "total_cost": float(geometry_cost),
        "avg_cost": float(geometry_cost / 2.0),
        "reliable_bud_supported_pairs": int(-negative_support_count),
        "leftover_indices": [],
    }


def _score_trunk_continuity(
    arm_a: Dict, arm_b: Dict, angle_deg: float,
    arm_bud_dir_scores: Optional[Dict[int, Dict[str, float]]] = None,
    arm_bud_flows: Optional[Dict[int, Dict[str, float]]] = None,
    idx_a: int = -1, idx_b: int = -1,
) -> float:
    """Score an arm pair as trunk continuation by path continuity (higher = better).

    Key priors:
    - Direction alignment: arms 180° apart → trunk goes straight through junction
    - Path smoothness: smooth arms are more likely trunk than zigzag branches
    - Bud direction: consistent bud alignment along both arms suggests same branch
    """
    direction_score = angle_deg / 180.0  # [0, 1], 180° = perfect straight-through
    smoothness_a = float(arm_a.get("straightness", 0.5))
    smoothness_b = float(arm_b.get("straightness", 0.5))
    smoothness_score = (smoothness_a + smoothness_b) / 2.0
    trunk_bonus = 0.2 if (arm_a.get("is_trunk_arm") and arm_b.get("is_trunk_arm")) else 0.0
    len_a = float(arm_a.get("length_confidence", 0.5))
    len_b = float(arm_b.get("length_confidence", 0.5))
    length_score = (len_a + len_b) / 2.0
    bud_score = 0.5
    if arm_bud_dir_scores is not None and idx_a >= 0 and idx_b >= 0:
        score_a = arm_bud_dir_scores.get(idx_a, {}).get("alignment_score", 0.5)
        score_b = arm_bud_dir_scores.get(idx_b, {}).get("alignment_score", 0.5)
        bud_score = (score_a + score_b) / 2.0
    flow_score = 0.5
    if arm_bud_flows is not None and idx_a >= 0 and idx_b >= 0:
        flow_a = arm_bud_flows.get(idx_a, {})
        flow_b = arm_bud_flows.get(idx_b, {})
        confidence = min(float(flow_a.get("flow_confidence", 0.0)), float(flow_b.get("flow_confidence", 0.0)))
        if confidence > 0.0:
            product = float(flow_a.get("flow_vote", 0.0)) * float(flow_b.get("flow_vote", 0.0))
            flow_score = (1.0 - product) * 0.5
            flow_score = 0.5 + (flow_score - 0.5) * confidence
    return direction_score * 3.5 + smoothness_score * 2.5 + length_score * 1.0 + trunk_bonus + bud_score * 1.5 + flow_score * 0.8


def _solve_trunk_involved_junction(
    arms: Sequence[Dict],
    priors: Dict[str, float],
    root_point_xy: Optional[Tuple[int, int]] = None,
    bud_counts: Optional[Dict[int, int]] = None,
    arm_bud_dir_scores: Optional[Dict[int, Dict[str, float]]] = None,
    arm_bud_flows: Optional[Dict[int, Dict[str, float]]] = None,
    use_path_continuity: bool = True,
) -> Optional[Dict]:
    if len(arms) < 3:
        return None

    up_vec = np.asarray([0.0, -1.0], dtype=np.float32)
    trunk_pair = None
    trunk_pair_cost = float("inf")
    trunk_candidates = []
    for idx_a in range(len(arms)):
        for idx_b in range(idx_a + 1, len(arms)):
            arm_a = arms[idx_a]
            arm_b = arms[idx_b]
            angle_deg = _angle_deg_between(arm_a["vector"], arm_b["vector"])

            if use_path_continuity:
                base_continuity = _score_trunk_continuity(
                    arm_a, arm_b, angle_deg, None, None, idx_a, idx_b)
                continuity = _score_trunk_continuity(
                    arm_a, arm_b, angle_deg, arm_bud_dir_scores, arm_bud_flows, idx_a, idx_b)
                geometry_candidate_cost = -base_continuity
                candidate_cost = -continuity  # negate: higher continuity → lower cost
                pair_info = {
                    "angle_deg": float(angle_deg),
                    "bud_dir_penalty": 0.0,
                    "curvature_cost": 0.0,
                }
            else:
                pair_info = _pair_cost(arm_a, arm_b, priors,
                                      bud_counts=bud_counts, idx_a=idx_a, idx_b=idx_b,
                                      arm_bud_dir_scores=arm_bud_dir_scores,
                                      arm_bud_flows=arm_bud_flows)
                angle_cost = abs(pair_info["angle_deg"] - priors["crossing_angle_mean"]) / max(priors["crossing_angle_std"], 1.0)
                vertical_cost = (2.0 - abs(float(np.dot(arm_a["vector"], up_vec))) - abs(float(np.dot(arm_b["vector"], up_vec)))) * 1.4
                radius_boost = -0.16 * (float(arm_a["radius"]) + float(arm_b["radius"]))
                trunk_bonus = -1.8 if arm_a.get("is_trunk_arm") and arm_b.get("is_trunk_arm") else 0.0
                continuity_cost = pair_info["curvature_cost"] * 0.7
                candidate_cost = angle_cost + vertical_cost + continuity_cost + radius_boost + trunk_bonus + pair_info["bud_penalty"] + pair_info["bud_dir_penalty"]
                geometry_candidate_cost = candidate_cost - pair_info["bud_penalty"] - pair_info["bud_dir_penalty"] - pair_info.get("bud_flow_penalty", 0.0)

            trunk_candidates.append((float(geometry_candidate_cost), float(candidate_cost), int(idx_a), int(idx_b), pair_info))

    if trunk_candidates:
        best_geometry_cost = min(candidate[0] for candidate in trunk_candidates)
        chosen = min(
            trunk_candidates,
            key=lambda candidate: candidate[1] if candidate[0] <= best_geometry_cost + 0.75 else candidate[0],
        )
        geometry_cost, candidate_cost, idx_a, idx_b, pair_info = chosen
        trunk_pair_cost = candidate_cost if geometry_cost <= best_geometry_cost + 0.75 else geometry_cost
        trunk_pair = (idx_a, idx_b, pair_info)

    if trunk_pair is None:
        return None

    trunk_idx_a, trunk_idx_b, trunk_pair_info = trunk_pair
    if trunk_pair_info["angle_deg"] < max(145.0, priors["crossing_angle_min"] - 12.0):
        return None
    trunk_axis = _trunk_axis_from_pair(arms[trunk_idx_a], arms[trunk_idx_b])
    trunk_radius = float(max(arms[trunk_idx_a]["radius"], arms[trunk_idx_b]["radius"]))

    branch_attach_indices: List[int] = []
    remaining_branch_indices: List[int] = []
    for idx, arm in enumerate(arms):
        if idx in {trunk_idx_a, trunk_idx_b}:
            continue
        acute_angle = min(_angle_deg_between(arm["vector"], trunk_axis), _angle_deg_between(arm["vector"], -trunk_axis))
        radius_ratio = trunk_radius / max(float(arm["radius"]), 1e-3)
        growth_ok = bool(arm.get("away_from_root_alignment", 0.0) >= -0.05 and arm.get("up_alignment", 0.0) >= -0.55)
        if 25.0 <= acute_angle <= 70.0 and radius_ratio >= priors["branching_radius_ratio_min"] and growth_ok:
            branch_attach_indices.append(int(idx))
        else:
            remaining_branch_indices.append(int(idx))

    crossing_solution = None
    if len(remaining_branch_indices) >= 2:
        sub_bud_counts = None
        if bud_counts is not None:
            sub_bud_counts = {i: bud_counts.get(orig_idx, 0) for i, orig_idx in enumerate(remaining_branch_indices)}
        crossing_solution = _solve_crossing_pairs(
            [arms[idx] for idx in remaining_branch_indices], priors,
            bud_counts=sub_bud_counts,
            arm_bud_flows={i: arm_bud_flows.get(orig_idx, {}) for i, orig_idx in enumerate(remaining_branch_indices)} if arm_bud_flows is not None else None,
        )

    mapped_pairs: List[Tuple[int, int]] = []
    mapped_pair_infos: List[Dict] = []
    used_remaining = set()
    if crossing_solution is not None:
        for pair_info in crossing_solution["pair_infos"]:
            local_a, local_b = pair_info["indices"]
            idx_a = int(remaining_branch_indices[local_a])
            idx_b = int(remaining_branch_indices[local_b])
            mapped_pairs.append((idx_a, idx_b))
            mapped_info = dict(pair_info)
            mapped_info["indices"] = (idx_a, idx_b)
            mapped_pair_infos.append(mapped_info)
            used_remaining.add(idx_a)
            used_remaining.add(idx_b)

    leftover_indices = [idx for idx in remaining_branch_indices if idx not in used_remaining]
    return {
        "mode": "trunk_involved",
        "trunk_pair": (int(trunk_idx_a), int(trunk_idx_b)),
        "trunk_axis": trunk_axis,
        "trunk_radius": float(trunk_radius),
        "trunk_pair_info": trunk_pair_info,
        "pairing": mapped_pairs,
        "pair_infos": mapped_pair_infos,
        "trunk_branch_indices": [int(idx) for idx in branch_attach_indices],
        "leftover_indices": [int(idx) for idx in leftover_indices],
        "root_point_xy": tuple(root_point_xy) if root_point_xy is not None else None,
    }


def _uncross_graph_cluster(graph: nx.Graph, cluster_nodes: Sequence[int], pairing_solution: Dict) -> Dict[str, int]:
    center_xy = np.mean(np.asarray([graph.nodes[node_id]["point"] for node_id in cluster_nodes], dtype=np.float32), axis=0)
    arms = pairing_solution["arms"]
    created_nodes = 0
    new_node_id = max(graph.nodes(), default=-1) + 1
    removed_nodes = list(cluster_nodes)
    graph.remove_nodes_from(removed_nodes)
    trunk_node_id = None
    trunk_pair = pairing_solution.get("trunk_pair")
    trunk_branch_indices = [int(idx) for idx in pairing_solution.get("trunk_branch_indices", [])]
    if trunk_pair is not None or trunk_branch_indices:
        trunk_node_id = new_node_id
        new_node_id += 1
        created_nodes += 1
        graph.add_node(trunk_node_id, point=tuple(map(int, np.round(center_xy))), is_trunk=True, group_types={"trunk", "branch"})
        trunk_indices = set(trunk_branch_indices)
        if trunk_pair is not None:
            trunk_indices.update([int(trunk_pair[0]), int(trunk_pair[1])])
        for idx in sorted(trunk_indices):
            arm = arms[idx]
            if arm["outside_id"] not in graph:
                continue
            graph.add_edge(
                trunk_node_id,
                arm["outside_id"],
                group_type="trunk" if idx in set(trunk_pair or []) else arm.get("group_type", "branch"),
                group_id="trunk" if idx in set(trunk_pair or []) else arm.get("group_id", "branch"),
            )

    for pair_idx, (idx_a, idx_b) in enumerate(pairing_solution["pairing"], start=1):
        arm_a = arms[idx_a]
        arm_b = arms[idx_b]
        vec = _normalize_vector(arm_a["vector"] - arm_b["vector"])
        if float(np.linalg.norm(vec)) < 1e-6:
            vec = np.asarray([1.0, 0.0], dtype=np.float32)
        perp = np.asarray([-vec[1], vec[0]], dtype=np.float32)
        offset = perp * float(1.6 * (pair_idx - (len(pairing_solution["pairing"]) + 1) * 0.5))
        new_point = tuple(map(int, np.round(center_xy + offset)))
        current_node = new_node_id
        new_node_id += 1
        created_nodes += 1
        pair_types = {arm_a.get("group_type", "branch"), arm_b.get("group_type", "branch")}
        graph.add_node(current_node, point=new_point, is_trunk=bool("trunk" in pair_types), group_types=pair_types)
        for arm in (arm_a, arm_b):
            if arm["outside_id"] not in graph:
                continue
            graph.add_edge(
                current_node,
                arm["outside_id"],
                group_type=arm.get("group_type", "branch"),
                group_id=arm.get("group_id", f"crossing_pair_{pair_idx:02d}"),
            )

    leftover_indices = [int(idx) for idx in pairing_solution.get("leftover_indices", [])]
    if len(leftover_indices) >= 2:
        residual_node = new_node_id
        new_node_id += 1
        created_nodes += 1
        residual_types = {arms[idx].get("group_type", "branch") for idx in leftover_indices}
        graph.add_node(residual_node, point=tuple(map(int, np.round(center_xy))), is_trunk=bool("trunk" in residual_types), group_types=residual_types)
        for idx in leftover_indices:
            arm = arms[idx]
            if arm["outside_id"] not in graph:
                continue
            graph.add_edge(
                residual_node,
                arm["outside_id"],
                group_type=arm.get("group_type", "branch"),
                group_id=arm.get("group_id", "junction_residual"),
            )
    return {
        "removed_nodes": len(removed_nodes),
        "created_nodes": int(created_nodes),
    }


def _score_residual_attachment(
    residual_nodes: Sequence[int],
    graph: nx.Graph,
    owner_nodes: Sequence[int],
    seed_node: int,
) -> float:
    residual_pts = np.asarray([graph.nodes[node_id]["point"] for node_id in residual_nodes], dtype=np.float32)
    owner_pts = np.asarray([graph.nodes[node_id]["point"] for node_id in owner_nodes], dtype=np.float32)
    if residual_pts.size == 0 or owner_pts.size == 0:
        return -1e9
    diff = residual_pts[:, None, :] - owner_pts[None, :, :]
    dist = float(np.linalg.norm(diff, axis=2).min())
    component_axis = _normalize_vector(residual_pts[-1] - residual_pts[0]) if len(residual_pts) >= 2 else np.zeros((2,), dtype=np.float32)
    seed_to_component = _normalize_vector(np.mean(residual_pts, axis=0) - np.asarray(graph.nodes[seed_node]["point"], dtype=np.float32))
    smoothness = float(abs(np.dot(component_axis, seed_to_component))) if float(np.linalg.norm(component_axis)) > 1e-6 else 0.0
    return float(1.8 * smoothness - 0.06 * dist)


def _nearest_trunk_segment_side(point_xy: Tuple[int, int], trunk_points: Sequence[Tuple[int, int]], margin: float = 5.0) -> int:
    if len(trunk_points) < 2:
        return 0
    point = np.asarray(point_xy, dtype=np.float32)
    best_dist = float("inf")
    best_cross = 0.0
    for p0_xy, p1_xy in zip(trunk_points[:-1], trunk_points[1:]):
        p0 = np.asarray(p0_xy, dtype=np.float32)
        p1 = np.asarray(p1_xy, dtype=np.float32)
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-6:
            continue
        t = float(np.clip(np.dot(point - p0, seg) / seg_len_sq, 0.0, 1.0))
        proj = p0 + t * seg
        diff = point - proj
        dist = float(np.linalg.norm(diff))
        if dist < best_dist:
            best_dist = dist
            best_cross = float(seg[0] * diff[1] - seg[1] * diff[0])
    if best_dist <= margin or abs(best_cross) <= 1e-6:
        return 0
    return 1 if best_cross > 0.0 else -1


def _split_owner_graph_by_trunk_side(
    owner_graph: nx.Graph,
    graph: nx.Graph,
    trunk_points: Sequence[Tuple[int, int]],
) -> List[nx.Graph]:
    if owner_graph.number_of_nodes() == 0:
        return []
    side_map = {node_id: _nearest_trunk_segment_side(tuple(map(int, graph.nodes[node_id]["point"])), trunk_points) for node_id in owner_graph.nodes()}
    pruned = owner_graph.copy()
    for src, dst in list(pruned.edges()):
        src_side = side_map.get(src, 0)
        dst_side = side_map.get(dst, 0)
        if src_side != 0 and dst_side != 0 and src_side != dst_side:
            pruned.remove_edge(src, dst)

    components: List[nx.Graph] = []
    for component_nodes in nx.connected_components(pruned):
        component = pruned.subgraph(component_nodes).copy()
        nonzero_sides = {side_map[node_id] for node_id in component.nodes() if side_map.get(node_id, 0) != 0}
        if len(nonzero_sides) <= 1:
            components.append(component)
            continue
        for side in sorted(nonzero_sides):
            side_nodes = [node_id for node_id in component.nodes() if side_map.get(node_id, 0) in {0, side}]
            side_graph = component.subgraph(side_nodes).copy()
            for sub_nodes in nx.connected_components(side_graph):
                components.append(side_graph.subgraph(sub_nodes).copy())
    return [component for component in components if component.number_of_edges() > 0]


def _split_component_mask_by_trunk_side(
    component_mask: np.ndarray,
    trunk_points: Sequence[Tuple[int, int]],
    min_pixels: int = 12,
) -> List[np.ndarray]:
    mask = (np.asarray(component_mask) > 0).astype(np.uint8)
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0 or len(trunk_points) < 2:
        return [mask]

    side_values = np.zeros((len(coords),), dtype=np.int8)
    for idx, (r, c) in enumerate(coords):
        side_values[idx] = int(_nearest_trunk_segment_side((int(c), int(r)), trunk_points, margin=4.0))

    present_sides = {int(side) for side in np.unique(side_values) if int(side) != 0}
    if len(present_sides) < 2:
        return [mask]

    side_masks: Dict[int, np.ndarray] = {}
    for side in (-1, 1):
        side_mask = np.zeros_like(mask, dtype=np.uint8)
        side_coords = coords[side_values == side]
        if len(side_coords) >= min_pixels:
            side_mask[side_coords[:, 0], side_coords[:, 1]] = 1
            side_masks[side] = side_mask

    if len(side_masks) < 2:
        return [mask]

    neutral_coords = coords[side_values == 0]
    if len(neutral_coords) > 0:
        left_dist = distance_transform_edt(side_masks[-1] == 0)
        right_dist = distance_transform_edt(side_masks[1] == 0)
        for r, c in neutral_coords:
            if float(left_dist[r, c]) <= float(right_dist[r, c]):
                side_masks[-1][r, c] = 1
            else:
                side_masks[1][r, c] = 1

    split_masks: List[np.ndarray] = []
    for side in (-1, 1):
        side_mask = side_masks[side]
        num_labels, labels = cv2.connectedComponents(side_mask, connectivity=8)
        for label in range(1, num_labels):
            sub_mask = (labels == label).astype(np.uint8)
            if int(sub_mask.sum()) >= min_pixels:
                split_masks.append(sub_mask)
    return split_masks if split_masks else [mask]


def _prune_cross_trunk_edges(
    graph: nx.Graph,
    trunk_points: Sequence[Tuple[int, int]],
    margin: float = 5.0,
) -> Dict[str, int]:
    if graph.number_of_edges() == 0 or len(trunk_points) < 2:
        return {"removed_edges": 0}
    side_map = {
        node_id: _nearest_trunk_segment_side(tuple(map(int, graph.nodes[node_id]["point"])), trunk_points, margin=margin)
        for node_id in graph.nodes()
    }
    removed_edges = 0
    for src, dst, edge_attrs in list(graph.edges(data=True)):
        if edge_attrs.get("group_type") == "trunk":
            continue
        if bool(graph.nodes[src].get("is_trunk")) or bool(graph.nodes[dst].get("is_trunk")):
            continue
        src_side = side_map.get(src, 0)
        dst_side = side_map.get(dst, 0)
        if src_side != 0 and dst_side != 0 and src_side != dst_side:
            graph.remove_edge(src, dst)
            removed_edges += 1
    return {"removed_edges": int(removed_edges)}


def _build_branch_groups_from_refined_graph(
    graph: nx.Graph,
    trunk_group: Dict,
    min_branch_length: float,
    root_point_xy: Optional[Tuple[int, int]] = None,
) -> List[Dict]:
    trunk_points = [tuple(map(int, point)) for point in trunk_group.get("points", [])]
    trunk_anchor_infos: List[Tuple[int, float, Tuple[int, int]]] = []

    def project_to_trunk(point_xy: Tuple[int, int]) -> Tuple[int, float, Tuple[int, int]]:
        point = np.asarray(point_xy, dtype=np.float32)
        best: Optional[Tuple[float, int, float, Tuple[int, int]]] = None
        for seg_idx, (start_xy, end_xy) in enumerate(zip(trunk_points[:-1], trunk_points[1:])):
            start = np.asarray(start_xy, dtype=np.float32)
            segment = np.asarray(end_xy, dtype=np.float32) - start
            segment_sq = float(np.dot(segment, segment))
            ratio = 0.0 if segment_sq < 1e-6 else float(np.clip(np.dot(point - start, segment) / segment_sq, 0.0, 1.0))
            projected = start + ratio * segment
            distance = float(np.linalg.norm(point - projected))
            projected_xy = (int(round(float(projected[0]))), int(round(float(projected[1]))))
            candidate = (distance, int(seg_idx), ratio, projected_xy)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None:
            return 0, 0.0, tuple(map(int, trunk_points[0]))
        return best[1], best[2], best[3]

    trunk_set = {tuple(point) for point in trunk_points}
    trunk_nodes = {node_id for node_id in graph.nodes() if tuple(graph.nodes[node_id]["point"]) in trunk_set or bool(graph.nodes[node_id].get("is_trunk"))}
    branch_graph = graph.copy()
    branch_graph.remove_nodes_from(trunk_nodes)
    if branch_graph.number_of_nodes() == 0:
        return []
    palette = ["#00FF00", "#00D0FF", "#FFCC00", "#AA66FF", "#FF8800", "#66FFAA", "#FF66AA", "#66A0FF"]
    seed_infos: Dict[int, Dict] = {}
    for node_id in branch_graph.nodes():
        trunk_neighbors = [nbr for nbr in graph.neighbors(node_id) if nbr in trunk_nodes]
        if not trunk_neighbors:
            continue
        trunk_neighbor = min(
            trunk_neighbors,
            key=lambda nbr: float(np.linalg.norm(np.asarray(graph.nodes[nbr]["point"], dtype=np.float32) - np.asarray(graph.nodes[node_id]["point"], dtype=np.float32))),
        )
        seed_infos[int(node_id)] = {
            "seed_node": int(node_id),
            "trunk_node": int(trunk_neighbor),
            "group_id": "",
        }

    if not seed_infos:
        for component_nodes in nx.connected_components(branch_graph):
            component_list = list(component_nodes)
            if not component_list or not trunk_nodes:
                continue
            seed_node = min(
                component_list,
                key=lambda node_id: min(
                    float(np.linalg.norm(np.asarray(graph.nodes[node_id]["point"], dtype=np.float32) - np.asarray(graph.nodes[trunk_id]["point"], dtype=np.float32)))
                    for trunk_id in trunk_nodes
                ),
            )
            trunk_neighbor = min(
                trunk_nodes,
                key=lambda trunk_id: float(np.linalg.norm(np.asarray(graph.nodes[seed_node]["point"], dtype=np.float32) - np.asarray(graph.nodes[trunk_id]["point"], dtype=np.float32))),
            )
            seed_infos[int(seed_node)] = {"seed_node": int(seed_node), "trunk_node": int(trunk_neighbor), "group_id": ""}

    owners: Dict[int, int] = {}
    parents: Dict[int, Optional[int]] = {}
    distances: Dict[int, int] = {}
    queue: deque[int] = deque()
    for seed_node in sorted(seed_infos):
        owners[seed_node] = int(seed_node)
        parents[seed_node] = None
        distances[seed_node] = 0
        queue.append(seed_node)

    while queue:
        node_id = queue.popleft()
        owner_id = owners[node_id]
        for nbr in branch_graph.neighbors(node_id):
            cand_dist = distances[node_id] + 1
            if nbr not in distances or cand_dist < distances[nbr] or (cand_dist == distances[nbr] and owner_id < owners[nbr]):
                distances[nbr] = cand_dist
                owners[nbr] = owner_id
                parents[nbr] = int(node_id)
                queue.append(int(nbr))

    owner_nodes: Dict[int, set] = {seed_node: {seed_node} for seed_node in seed_infos}
    owner_edges: Dict[int, set] = {seed_node: set() for seed_node in seed_infos}
    for node_id, owner_id in owners.items():
        owner_nodes.setdefault(owner_id, set()).add(int(node_id))
        parent_id = parents.get(node_id)
        if parent_id is not None and owners.get(parent_id) == owner_id:
            owner_edges.setdefault(owner_id, set()).add(tuple(sorted((int(node_id), int(parent_id)))))

    orphan_nodes = [node_id for node_id in branch_graph.nodes() if node_id not in owners]
    orphan_graph = branch_graph.subgraph(orphan_nodes).copy()
    for component_nodes in nx.connected_components(orphan_graph):
        component_list = list(component_nodes)
        if not component_list or not owner_nodes:
            continue
        best_owner = max(
            owner_nodes.keys(),
            key=lambda owner_id: _score_residual_attachment(component_list, graph, sorted(owner_nodes[owner_id]), seed_node=owner_id),
        )
        comp_root = min(
            component_list,
            key=lambda node_id: min(
                float(np.linalg.norm(np.asarray(graph.nodes[node_id]["point"], dtype=np.float32) - np.asarray(graph.nodes[owner_node]["point"], dtype=np.float32)))
                for owner_node in owner_nodes[best_owner]
            ),
        )
        comp_tree = nx.bfs_tree(orphan_graph.subgraph(component_list), source=comp_root)
        owner_nodes.setdefault(best_owner, set()).update(int(node_id) for node_id in component_list)
        for src, dst in comp_tree.edges():
            owner_edges.setdefault(best_owner, set()).add(tuple(sorted((int(src), int(dst)))))
        bridge_target = min(
            owner_nodes[best_owner] - set(component_list),
            key=lambda owner_node: float(np.linalg.norm(np.asarray(graph.nodes[comp_root]["point"], dtype=np.float32) - np.asarray(graph.nodes[owner_node]["point"], dtype=np.float32))),
            default=None,
        )
        if bridge_target is not None:
            owner_edges.setdefault(best_owner, set()).add(tuple(sorted((int(comp_root), int(bridge_target)))))

    groups: List[Dict] = []
    branch_idx = 1
    for seed_node, info in sorted(seed_infos.items()):
        nodes = sorted(owner_nodes.get(seed_node, set()))
        edges = sorted(owner_edges.get(seed_node, set()))
        if len(nodes) < 2 and not edges:
            continue
        local_graph = nx.Graph()
        local_graph.add_nodes_from(nodes)
        local_graph.add_edges_from(edges)
        if local_graph.number_of_edges() == 0:
            continue
        side_components = _split_owner_graph_by_trunk_side(local_graph, graph, trunk_points=trunk_points)
        if not side_components:
            side_components = [local_graph]
        for side_component in side_components:
            if seed_node not in side_component.nodes():
                side_seed = min(
                    side_component.nodes(),
                    key=lambda node_id: float(np.linalg.norm(
                        np.asarray(graph.nodes[node_id]["point"], dtype=np.float32)
                        - np.asarray(graph.nodes[seed_node]["point"], dtype=np.float32)
                    )),
                )
            else:
                side_seed = seed_node
            bfs_nodes = list(nx.bfs_tree(side_component, source=side_seed).nodes())
            for node_id in side_component.nodes():
                if node_id not in bfs_nodes:
                    bfs_nodes.append(node_id)
            local_map = {node_id: idx for idx, node_id in enumerate(bfs_nodes)}
            group = {
                "group_id": f"branch_{branch_idx:02d}",
                "group_type": "branch",
                "color_hex": palette[(branch_idx - 1) % len(palette)],
                "points": [[int(graph.nodes[node_id]["point"][0]), int(graph.nodes[node_id]["point"][1])] for node_id in bfs_nodes],
                "edges": [[local_map[src], local_map[dst]] for src, dst in side_component.edges() if src in local_map and dst in local_map],
                "fork_origin_group": "trunk",
            }
            attach_xy = tuple(map(int, graph.nodes[side_seed]["point"]))
            anchor_seg_idx, anchor_ratio, trunk_anchor_xy = project_to_trunk(attach_xy)
            group = _prepend_anchor_to_group(group, trunk_anchor_xy, attach_xy)
            if group is not None and _group_total_edge_length(group) >= float(min_branch_length):
                trunk_anchor_infos.append((anchor_seg_idx, anchor_ratio, trunk_anchor_xy))
                groups.append(group)
                branch_idx += 1
    if trunk_anchor_infos:
        anchors_by_segment: Dict[int, List[Tuple[float, Tuple[int, int]]]] = {}
        for seg_idx, ratio, anchor_xy in trunk_anchor_infos:
            anchors_by_segment.setdefault(int(seg_idx), []).append((float(ratio), anchor_xy))
        refined_trunk_points: List[Tuple[int, int]] = []
        for seg_idx, start_xy in enumerate(trunk_points[:-1]):
            refined_trunk_points.append(tuple(map(int, start_xy)))
            for _, anchor_xy in sorted(anchors_by_segment.get(seg_idx, []), key=lambda item: item[0]):
                if anchor_xy != refined_trunk_points[-1] and anchor_xy != trunk_points[seg_idx + 1]:
                    refined_trunk_points.append(anchor_xy)
        refined_trunk_points.append(tuple(map(int, trunk_points[-1])))
        refined_trunk_points = _dedupe_consecutive(refined_trunk_points)
        trunk_group["points"] = [list(point) for point in refined_trunk_points]
        trunk_group["edges"] = [[idx, idx + 1] for idx in range(len(refined_trunk_points) - 1)]
    return groups


def reconstruct_branch_groups_with_junction_pairing(
    groups: Sequence[Dict],
    dt_map: np.ndarray,
    min_branch_length: float,
    prior_path: os.PathLike = DEFAULT_JUNCTION_PRIOR_PATH,
    cluster_radius: Optional[float] = None,
    root_point_xy: Optional[Tuple[int, int]] = None,
    bud_centers: Optional[Sequence[Tuple[float, float]]] = None,
    bud_orientations: Optional[Sequence] = None,
    bud_directions: Optional[Sequence] = None,
    protected_tape_mask: Optional[np.ndarray] = None,
    enable_bud_density_prior: bool = True,
    enable_bud_direction_flow: bool = True,
    enable_bud_root_split: bool = True,
    structural_tolerance: float = 8.0,
) -> Tuple[List[Dict], Dict[str, float]]:
    if not groups:
        return [], {"junction_clusters": 0, "crossing_clusters": 0, "crossing_pairs": 0}
    trunk_group = next((group for group in groups if group.get("group_type") == "trunk"), None)
    if trunk_group is None:
        return list(groups), {"junction_clusters": 0, "crossing_clusters": 0, "crossing_pairs": 0}
    if root_point_xy is None and trunk_group.get("points"):
        trunk_points = [tuple(map(int, point)) for point in trunk_group.get("points", [])]
        root_point_xy = max(trunk_points, key=lambda point: point[1]) if trunk_points else None

    priors = _load_junction_priors(prior_path)
    graph, _ = _build_group_topology_graph(groups)
    trunk_points = [tuple(map(int, point)) for point in trunk_group.get("points", [])]
    trunk_side_stats = _prune_cross_trunk_edges(graph, trunk_points=trunk_points, margin=5.0)
    clusters = _cluster_junction_nodes(graph, radius=float(cluster_radius or priors["junction_cluster_radius"]))
    crossing_clusters = 0
    crossing_pairs = 0
    trunk_involved_clusters = 0
    bud_flow_evidence_clusters = 0
    pairing_debug: List[Dict] = []
    for cluster_nodes in clusters:
        arms = _extract_cluster_arms(graph, cluster_nodes, dt_map, root_point_xy=root_point_xy)
        if len(arms) < 2:
            continue

        arm_bud_counts: Optional[Dict[int, int]] = None
        arm_bud_dir_scores: Optional[Dict[int, Dict[str, float]]] = None
        arm_bud_flows: Optional[Dict[int, Dict[str, float]]] = None
        if bud_centers:
            buds_arr = np.asarray(bud_centers, dtype=np.float32)
            if enable_bud_density_prior:
                arm_bud_counts = {}
                for arm_idx, arm in enumerate(arms):
                    polyline = np.asarray(arm.get("bud_support_polyline_xy", arm["polyline_xy"]), dtype=np.float32)
                    if len(polyline) == 0:
                        arm_bud_counts[arm_idx] = 0
                        continue
                    arm_radius = float(arm.get("radius", 8.0))
                    search_radius = max(arm_radius * 2.5, 12.0)
                    dists = _point_to_polyline_distances(buds_arr, polyline)
                    arm_bud_counts[arm_idx] = float(np.sum(dists <= search_radius)) / max(float(arm.get("bud_support_length", arm.get("support_length", 1.0))), 1.0) * 1000.0

            if enable_bud_direction_flow:
                arm_bud_flows = {}
                for arm_idx, arm in enumerate(arms):
                    arm_radius = float(arm.get("radius", 8.0))
                    search_radius = max(arm_radius * 2.5, 12.0)
                    arm_bud_flows[arm_idx] = _compute_arm_bud_flow(
                        arm,
                        buds_arr,
                        bud_directions,
                        search_radius=search_radius,
                    )
                if sum(flow.get("flow_confidence", 0.0) > 0.0 for flow in arm_bud_flows.values()) >= 2:
                    bud_flow_evidence_clusters += 1

        # 芽点方向对齐评分: 只使用 elongated buds
        if bud_orientations and bud_centers:
            buds_arr = np.asarray(bud_centers, dtype=np.float32)
            bud_angles = np.array([o.axis_angle if hasattr(o, 'axis_angle') else o.get('axis_angle', 0)
                                   for o in bud_orientations], dtype=np.float32)
            bud_elongated = np.array([o.is_elongated if hasattr(o, 'is_elongated') else o.get('is_elongated', False)
                                      for o in bud_orientations])
            arm_bud_dir_scores = {}
            for arm_idx, arm in enumerate(arms):
                arm_radius = float(arm.get("radius", 8.0))
                search_radius = max(arm_radius * 3.0, 20.0)
                arm_bud_dir_scores[arm_idx] = _compute_arm_bud_direction_alignment(
                    arm, buds_arr, bud_angles, bud_elongated,
                    search_radius=search_radius,
                )

        if any(arm.get("is_trunk_arm") for arm in arms):
            pairing_solution = _solve_trunk_involved_junction(
                arms, priors, root_point_xy=root_point_xy,
                bud_counts=arm_bud_counts,
                arm_bud_dir_scores=arm_bud_dir_scores,
                arm_bud_flows=arm_bud_flows,
            )
            if pairing_solution is not None:
                trunk_involved_clusters += 1
        else:
            pairing_solution = None
            if len(cluster_nodes) >= 2:
                pairing_solution = _solve_degree3_x_motif(
                    arms, arm_bud_flows=arm_bud_flows,
                )
            if pairing_solution is None:
                pairing_solution = _solve_crossing_pairs(arms, priors,
                    bud_counts=arm_bud_counts,
                    arm_bud_dir_scores=arm_bud_dir_scores,
                    arm_bud_flows=arm_bud_flows)
        if pairing_solution is None:
            continue
        pairing_solution["arms"] = arms
        pairing_debug.append({
            "center_xy": list(map(float, arms[0].get("center_xy", (0.0, 0.0)))),
            "mode": pairing_solution.get("mode", "crossing"),
            "pairing": [list(map(int, pair)) for pair in pairing_solution.get("pairing", [])],
            "trunk_pair": list(map(int, pairing_solution["trunk_pair"])) if pairing_solution.get("trunk_pair") is not None else None,
            "pair_infos": [
                {
                    key: (list(value) if isinstance(value, tuple) else value)
                    for key, value in info.items()
                    if isinstance(value, (int, float, bool, tuple))
                }
                for info in pairing_solution.get("pair_infos", [])
            ],
            "arm_bud_flows": arm_bud_flows or {},
            "arms": [
                {
                    "polyline_xy": [list(map(int, point)) for point in arm.get("polyline_xy", [])],
                    "is_trunk_arm": bool(arm.get("is_trunk_arm", False)),
                    "group_type": arm.get("group_type", "branch"),
                }
                for arm in arms
            ],
        })
        _uncross_graph_cluster(graph, cluster_nodes, pairing_solution)
        crossing_clusters += 1
        crossing_pairs += len(pairing_solution["pairing"])

    # 环检测: 汇接配对错误 → 经过主干的环
    from .cycle_detector import (
        detect_trunk_cycles, collect_cycle_stats, attempt_bud_consistency_cycle_repair,
    )
    trunk_cycles = detect_trunk_cycles(graph, trunk_points)
    cycle_stats = collect_cycle_stats(trunk_cycles, clusters)

    refined_groups = [trunk_group]
    refined_groups.extend(
        _build_branch_groups_from_refined_graph(
            graph,
            trunk_group,
            min_branch_length=min_branch_length,
            root_point_xy=root_point_xy,
        )
    )

    multi_contact_groups_detected = 0
    multi_contact_crossing_candidates = 0
    multi_contact_groups_split = 0
    bud_flow_root_splits = 0
    tape_root_contacts_pruned = 0
    secondary_root_contacts_pruned = 0
    trunk_mask_refined = np.zeros(dt_map.shape, dtype=np.uint8)
    refined_trunk_points = [tuple(map(int, point)) for point in trunk_group.get("points", [])]
    refined_trunk_arc = [0.0]
    for start_xy, end_xy in zip(refined_trunk_points[:-1], refined_trunk_points[1:]):
        refined_trunk_arc.append(
            refined_trunk_arc[-1]
            + float(np.linalg.norm(np.asarray(end_xy, dtype=np.float32) - np.asarray(start_xy, dtype=np.float32)))
        )

    def contact_arc_position(contact_rc: Tuple[int, int]) -> float:
        contact_xy = np.asarray([contact_rc[1], contact_rc[0]], dtype=np.float32)
        best_distance = float("inf")
        best_arc = 0.0
        for seg_idx, (start_xy, end_xy) in enumerate(zip(refined_trunk_points[:-1], refined_trunk_points[1:])):
            start = np.asarray(start_xy, dtype=np.float32)
            segment = np.asarray(end_xy, dtype=np.float32) - start
            segment_sq = float(np.dot(segment, segment))
            ratio = 0.0 if segment_sq < 1e-6 else float(np.clip(np.dot(contact_xy - start, segment) / segment_sq, 0.0, 1.0))
            projection = start + ratio * segment
            distance = float(np.linalg.norm(contact_xy - projection))
            if distance < best_distance:
                best_distance = distance
                best_arc = refined_trunk_arc[seg_idx] + ratio * (refined_trunk_arc[seg_idx + 1] - refined_trunk_arc[seg_idx])
        return best_arc

    for start_xy, end_xy in zip(refined_trunk_points[:-1], refined_trunk_points[1:]):
        cv2.line(trunk_mask_refined, start_xy, end_xy, 1, 3, lineType=cv2.LINE_4)
    trunk_distance_refined = distance_transform_edt(trunk_mask_refined == 0)

    def prune_group_to_single_attachment(
        source_group: Dict,
        source_attachments: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> Optional[Dict]:
        local_points, local_graph = _group_to_local_graph(source_group)
        if not local_points or local_graph.number_of_edges() == 0:
            return None
        point_arr = np.asarray(local_points, dtype=np.float32)
        attachment_infos = []
        for branch_attach_rc, trunk_attach_rc in source_attachments:
            branch_xy = np.asarray([branch_attach_rc[1], branch_attach_rc[0]], dtype=np.float32)
            node_idx = int(np.argmin(np.linalg.norm(point_arr - branch_xy[None, :], axis=1)))
            attachment_infos.append(
                {
                    "node_idx": node_idx,
                    "trunk_xy": (int(trunk_attach_rc[1]), int(trunk_attach_rc[0])),
                }
            )
        anchor_nodes = [int(info["node_idx"]) for info in attachment_infos]
        owners = _assign_tree_nodes_to_anchors(local_graph, anchor_nodes, force_split=True)

        def upward_growth(info: Dict) -> Tuple[float, float, int]:
            anchor_node = int(info["node_idx"])
            owned_nodes = [node for node, owner in owners.items() if owner == anchor_node]
            root_y = float(local_points[anchor_node][1])
            min_y = min((float(local_points[node][1]) for node in owned_nodes), default=root_y)
            max_distance = max(
                (
                    float(np.linalg.norm(np.asarray(local_points[node]) - np.asarray(local_points[anchor_node])))
                    for node in owned_nodes
                ),
                default=0.0,
            )
            return (root_y - min_y, max_distance, root_y)

        retained_info = max(attachment_infos, key=upward_growth)
        retained_anchor = int(retained_info["node_idx"])
        trimmed_graph = local_graph.copy()
        removed_nodes = set()
        for info in attachment_infos:
            current = int(info["node_idx"])
            if current == retained_anchor:
                continue
            previous = None
            while current in local_graph and local_graph.degree[current] <= 2:
                x, y = map(int, local_points[current])
                if not (0 <= y < trunk_distance_refined.shape[0] and 0 <= x < trunk_distance_refined.shape[1]):
                    break
                if float(trunk_distance_refined[y, x]) > 8.0:
                    break
                removed_nodes.add(current)
                next_nodes = [node for node in local_graph.neighbors(current) if node != previous]
                if len(next_nodes) != 1:
                    break
                previous, current = current, int(next_nodes[0])
        trimmed_graph.remove_nodes_from(removed_nodes)
        if retained_anchor not in trimmed_graph:
            return None
        retained_component = nx.node_connected_component(trimmed_graph, retained_anchor)
        pruned_group = _build_group_from_subgraph(
            local_points,
            trimmed_graph,
            list(retained_component),
            anchor_node=retained_anchor,
            group_id=str(source_group.get("group_id", "branch")),
            color_hex=str(source_group.get("color_hex", "#00FF00")),
            fork_origin_group="trunk",
        )
        if pruned_group is not None:
            pruned_group = _prepend_anchor_to_group(
                pruned_group,
                retained_info["trunk_xy"],
                tuple(map(int, local_points[retained_anchor])),
            )
        if pruned_group is None or _group_total_edge_length(pruned_group) < float(min_branch_length):
            return None
        return pruned_group

    repaired_groups = [trunk_group]
    split_palette = ["#00FF00", "#00D0FF", "#FFCC00", "#AA66FF", "#FF8800", "#66FFAA", "#FF66AA", "#66A0FF"]
    for group in refined_groups[1:]:
        group_mask = np.zeros(dt_map.shape, dtype=np.uint8)
        group_points = [tuple(map(int, point)) for point in group.get("points", [])]
        group_degree = [0] * len(group_points)
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(group_points) and 0 <= dst < len(group_points):
                cv2.line(group_mask, group_points[src], group_points[dst], 1, 1, lineType=cv2.LINE_4)
                group_degree[src] += 1
                group_degree[dst] += 1
        contact_mask = ((group_mask > 0) & (trunk_distance_refined <= float(structural_tolerance))).astype(np.uint8)
        contact_count, contact_labels = cv2.connectedComponents(contact_mask, connectivity=8)
        attachments = []
        used_endpoint_indices = set()
        endpoint_indices = [idx for idx, degree in enumerate(group_degree) if degree == 1]
        for contact_label in range(1, contact_count):
            contact_coords = np.column_stack(np.where(contact_labels == contact_label))
            if contact_coords.size == 0 or not endpoint_indices:
                continue
            endpoint_coords = np.asarray(
                [[group_points[idx][1], group_points[idx][0]] for idx in endpoint_indices],
                dtype=np.float32,
            )
            distances = np.linalg.norm(endpoint_coords[:, None, :] - contact_coords[None, :, :], axis=2)
            endpoint_pos, contact_idx = np.unravel_index(int(np.argmin(distances)), distances.shape)
            endpoint_idx = endpoint_indices[int(endpoint_pos)]
            if float(distances[endpoint_pos, contact_idx]) > max(6.0, float(structural_tolerance) + 2.0) or endpoint_idx in used_endpoint_indices:
                continue
            used_endpoint_indices.add(endpoint_idx)
            contact_rc = tuple(map(int, contact_coords[contact_idx]))
            endpoint_xy = group_points[endpoint_idx]
            endpoint_rc = (int(endpoint_xy[1]), int(endpoint_xy[0]))
            attachments.append((endpoint_rc, contact_rc))
        if len(attachments) <= 1:
            repaired_groups.append(group)
            continue
        multi_contact_groups_detected += 1
        contact_arcs = [contact_arc_position(trunk_rc) for _, trunk_rc in attachments]
        max_local_root_span = max(60.0, refined_trunk_arc[-1] * 0.15)
        # Every spatially distinct trunk-contact cluster remains a candidate root.
        # Do not compress three or more botanical roots into a fixed two-way split.
        protected_endpoint_indices = {
            group_points.index((int(endpoint_rc[1]), int(endpoint_rc[0])))
            for endpoint_rc, contact_rc in attachments
            if protected_tape_mask is not None
            and 0 <= contact_rc[0] < protected_tape_mask.shape[0]
            and 0 <= contact_rc[1] < protected_tape_mask.shape[1]
            and protected_tape_mask[contact_rc] > 0
            and (int(endpoint_rc[1]), int(endpoint_rc[0])) in group_points
        }
        far_root_span = max(contact_arcs) - min(contact_arcs) > max_local_root_span
        bud_supported_split = bool(
            enable_bud_root_split
            and enable_bud_direction_flow
            and _bud_flow_supports_independent_roots(group, attachments, bud_centers, bud_directions)
        )
        if not protected_endpoint_indices and not bud_supported_split and far_root_span:
            multi_contact_crossing_candidates += 1
        extreme_root_span = (
            max(contact_arcs) - min(contact_arcs)
            > max(120.0, refined_trunk_arc[-1] * 0.40)
        )
        if extreme_root_span and not protected_endpoint_indices and not bud_supported_split:
            # A single long branch can cross the trunk again far from its botanical
            # root.  Splitting it at the geodesic midpoint creates two false roots.
            # Keep the attachment whose owned portion grows furthest upward, and
            # peel the short leaf chain at every secondary trunk contact.
            pruned_group = prune_group_to_single_attachment(group, attachments)
            if pruned_group is not None:
                repaired_groups.append(pruned_group)
                secondary_root_contacts_pruned += max(0, len(attachments) - 1)
                continue
        split_groups = _split_group_by_attachments(
            group,
            attachments=attachments,
            group_prefix=f"{group.get('group_id', 'branch')}_root",
            palette=split_palette,
            min_branch_length=min_branch_length,
            force_split=True,
        )
        split_is_supported = len(split_groups) >= 2 and all(
            len(split_group.get("points", [])) >= (2 if protected_endpoint_indices else 4)
            and _group_total_edge_length(split_group) >= float(min_branch_length) * (1.0 if protected_endpoint_indices else 2.0)
            for split_group in split_groups
        )
        if split_is_supported:
            repaired_groups.extend(split_groups)
            multi_contact_groups_split += 1
            bud_flow_root_splits += int(bud_supported_split and far_root_span)
        else:
            repaired_groups.append(group)
    refined_groups = repaired_groups

    validated_groups = [trunk_group]
    for group in refined_groups[1:]:
        group_mask = np.zeros(dt_map.shape, dtype=np.uint8)
        group_points = [tuple(map(int, point)) for point in group.get("points", [])]
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(group_points) and 0 <= dst < len(group_points):
                cv2.line(group_mask, group_points[src], group_points[dst], 1, 1, lineType=cv2.LINE_4)
        remaining_attachments = _find_component_attachments(
            group_mask,
            trunk_mask_refined,
            dt_map,
            trunk_points=refined_trunk_points,
        )
        if len(remaining_attachments) <= 1:
            validated_groups.append(group)
            continue
        if "_root_" in str(group.get("group_id", "")):
            pruned_group = prune_group_to_single_attachment(group, remaining_attachments)
            if pruned_group is not None:
                validated_groups.append(pruned_group)
                secondary_root_contacts_pruned += max(0, len(remaining_attachments) - 1)
                continue
        split_groups = _split_group_by_attachments(
            group,
            attachments=remaining_attachments,
            group_prefix=f"{group.get('group_id', 'branch')}_validated",
            palette=split_palette,
            min_branch_length=min_branch_length,
            force_split=True,
        )
        if len(split_groups) >= 2:
            validated_groups.extend(split_groups)
            multi_contact_groups_split += 1
        else:
            validated_groups.append(group)
    refined_groups = validated_groups

    # 芽点方向一致性破圈: 对含环的分支, 移除芽点方向最不一致的分支
    cycles_repaired = 0
    repair_debug = []
    if trunk_cycles and bud_orientations and bud_centers:
        buds_arr = np.asarray(bud_centers, dtype=np.float32)
        bud_angles = np.array([o['axis_angle'] if isinstance(o, dict) else o.axis_angle
                               for o in bud_orientations], dtype=np.float32)
        bud_elongated = np.array([o['is_elongated'] if isinstance(o, dict) else o.is_elongated
                                  for o in bud_orientations])
        refined_groups, cycles_repaired, repair_debug = attempt_bud_consistency_cycle_repair(
            refined_groups, graph, trunk_cycles,
            buds_arr, bud_angles, bud_elongated,
        )
        if cycles_repaired > 0:
            # 重建图并重新检测环
            new_graph, _ = _build_group_topology_graph(refined_groups)
            trunk_cycles = detect_trunk_cycles(new_graph, trunk_points)
            cycle_stats = collect_cycle_stats(trunk_cycles, clusters)

    stats = {
        "junction_clusters": float(len(clusters)),
        "crossing_clusters": float(crossing_clusters),
        "crossing_pairs": float(crossing_pairs),
        "trunk_involved_clusters": float(trunk_involved_clusters),
        "bud_flow_evidence_clusters": float(bud_flow_evidence_clusters),
        "trunk_side_removed_edges": float(trunk_side_stats.get("removed_edges", 0)),
        "junction_cluster_radius": float(cluster_radius or priors["junction_cluster_radius"]),
        "crossing_angle_mean": float(priors["crossing_angle_mean"]),
        "branching_angle_mean": float(priors["branching_angle_mean"]),
        "trunk_cycles_detected": cycle_stats["trunk_cycles_total"],
        "trunk_cycles_junction_related": cycle_stats["trunk_cycles_junction_related"],
        "trunk_cycles_unexplained": cycle_stats["trunk_cycles_unexplained"],
        "bud_consistency_cycles_repaired": float(cycles_repaired),
        "multi_contact_groups_detected": float(multi_contact_groups_detected),
        "multi_contact_crossing_candidates": float(multi_contact_crossing_candidates),
        "multi_contact_groups_split": float(multi_contact_groups_split),
        "bud_flow_root_splits": float(bud_flow_root_splits),
        "tape_root_contacts_pruned": float(tape_root_contacts_pruned),
        "secondary_root_contacts_pruned": float(secondary_root_contacts_pruned),
        "bud_consistency_repair_debug": repair_debug,
        "junction_pairing_debug": pairing_debug,
    }
    # Guard: warn if any branch group has >1 trunk contact after junction pairing
    if trunk_group and len(refined_groups) > 1:
        trunk_pts_xy = [tuple(map(int, p)) for p in trunk_group.get("points", [])]
        if trunk_pts_xy:
            tx, ty = max(p[0] for p in trunk_pts_xy) + 4, max(p[1] for p in trunk_pts_xy) + 4
            trunk_mask_ref = np.zeros((ty, tx), dtype=np.uint8)
            for i in range(len(trunk_pts_xy) - 1):
                cv2.line(trunk_mask_ref, trunk_pts_xy[i], trunk_pts_xy[i + 1], 1, 3, lineType=cv2.LINE_4)
            multi_contact_groups = []
            for g in refined_groups[1:]:
                if g.get("group_type") != "branch":
                    continue
                nc = _count_trunk_contacts(g, trunk_mask_ref, contact_radius=2)
                if nc > 1:
                    multi_contact_groups.append((g["group_id"], nc))
            stats["junction_multi_contact_groups"] = multi_contact_groups
    return refined_groups, stats


def _build_graph_from_polylines(
    trunk_line: Sequence[Tuple[int, int]],
    branch_lines: Sequence[Sequence[Tuple[int, int]]],
) -> Tuple[List[Tuple[int, int]], List[str], nx.Graph, List[int]]:
    graph = nx.Graph()
    points: List[Tuple[int, int]] = []
    node_types: List[str] = []
    point_to_id: Dict[Tuple[int, int], int] = {}

    def ensure_node(point_xy: Tuple[int, int], node_type: str) -> int:
        point_xy = (int(point_xy[0]), int(point_xy[1]))
        if point_xy in point_to_id:
            idx = point_to_id[point_xy]
            if node_types[idx] == "trunk_node" and node_type == "branch_junction":
                node_types[idx] = "branch_junction"
            return idx
        idx = len(points)
        point_to_id[point_xy] = idx
        points.append(point_xy)
        node_types.append(node_type)
        graph.add_node(idx, point=point_xy)
        return idx

    trunk_path: List[int] = []
    for idx, point in enumerate(trunk_line):
        node_id = ensure_node(point, "trunk_node")
        trunk_path.append(node_id)
        if idx > 0:
            graph.add_edge(trunk_path[idx - 1], node_id)

    for branch_line in branch_lines:
        if not branch_line:
            continue
        branch_ids: List[int] = []
        for idx, point in enumerate(branch_line):
            if idx == 0:
                node_type = "branch_junction"
            elif idx == len(branch_line) - 1:
                node_type = "branch_endpoint"
            else:
                node_type = "branch_path"
            branch_ids.append(ensure_node(point, node_type))
        for src, dst in zip(branch_ids[:-1], branch_ids[1:]):
            if src != dst:
                graph.add_edge(src, dst)
    return points, node_types, graph, trunk_path


def _rescale_annotation_groups(groups: Sequence[Dict], scale_x: float, scale_y: float) -> List[Dict]:
    scaled_groups: List[Dict] = []
    for group in groups:
        scaled_group = {
            **group,
            "points": [
                [int(round(point[0] * scale_x)), int(round(point[1] * scale_y))]
                for point in group.get("points", [])
            ],
        }
        for key in ("root_anchor_xy", "root_attach_xy"):
            value = group.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                scaled_group[key] = [
                    int(round(float(value[0]) * scale_x)),
                    int(round(float(value[1]) * scale_y)),
                ]
        scaled_groups.append(scaled_group)
    return scaled_groups


def _line_pixels_xy(a: Tuple[int, int], b: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.line(canvas, tuple(map(int, a)), tuple(map(int, b)), 1, 1, lineType=cv2.LINE_4)
    return np.where(canvas > 0)


def _group_contact_count(group: Dict, trunk_distance: np.ndarray, tolerance: float) -> int:
    mask = render_topology_groups_to_mask([group], trunk_distance.shape).astype(np.uint8)
    contacts = ((mask > 0) & (trunk_distance <= float(tolerance))).astype(np.uint8)
    count, _ = cv2.connectedComponents(contacts, connectivity=8)
    return max(0, int(count) - 1)


def _nearest_component_id(point_xy: Tuple[int, int], labels: np.ndarray, radius: int = 3) -> int:
    x, y = map(int, point_xy)
    y0, y1 = max(0, y - radius), min(labels.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(labels.shape[1], x + radius + 1)
    values = labels[y0:y1, x0:x1]
    values = values[values > 0]
    if values.size == 0:
        return 0
    ids, counts = np.unique(values, return_counts=True)
    return int(ids[int(np.argmax(counts))])


def _subgroup_from_nodes(group: Dict, points: List[Tuple[int, int]], graph: nx.Graph,
                         nodes: Sequence[int], suffix: str) -> Optional[Dict]:
    node_set = set(nodes)
    component = graph.subgraph(node_set).copy()
    if component.number_of_edges() == 0:
        return None
    ordered = sorted(component.nodes())
    remap = {node: idx for idx, node in enumerate(ordered)}
    return {
        **group,
        "group_id": f"{group.get('group_id', 'branch')}_{suffix}",
        "points": [list(map(int, points[node])) for node in ordered],
        "edges": [[remap[src], remap[dst]] for src, dst in component.edges()],
    }


def _reliable_bud_flow_vote(group: Dict, bud_directions: Optional[Sequence], root_xy: Tuple[int, int]) -> Optional[float]:
    if not bud_directions:
        return None
    points = np.asarray(group.get("points", []), dtype=np.float32)
    if len(points) < 2:
        return None
    root = np.asarray(root_xy, dtype=np.float32)
    outward = points[np.argmax(np.linalg.norm(points - root[None, :], axis=1))] - root
    outward = _normalize_vector(outward)
    signed_weight = 0.0
    total_weight = 0.0
    count = 0
    edges = [edge for edge in group.get("edges", []) if len(edge) == 2]

    def distance_to_group(center: np.ndarray) -> float:
        best = float(np.min(np.linalg.norm(points - center[None, :], axis=1)))
        for edge in edges:
            src, dst = int(edge[0]), int(edge[1])
            if not (0 <= src < len(points) and 0 <= dst < len(points)):
                continue
            a, b = points[src], points[dst]
            ab = b - a
            denom = float(np.dot(ab, ab))
            t = 0.0 if denom < 1e-6 else float(np.clip(np.dot(center - a, ab) / denom, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(center - (a + t * ab))))
        return best

    for direction in bud_directions:
        if not bool(direction.get("is_reliable", False)):
            continue
        center = np.asarray(direction.get("centroid_global", (0.0, 0.0)), dtype=np.float32)
        if distance_to_group(center) > 32.0:
            continue
        weight = max(0.0, float(direction.get("confidence", 0.0)))
        vector = _normalize_vector(np.asarray(direction.get("vector_xy", (0.0, 0.0)), dtype=np.float32))
        signed_weight += weight * float(np.dot(vector, outward))
        total_weight += weight
        count += 1
    if count < 2 or total_weight < 0.4:
        return None
    vote = signed_weight / max(total_weight, 1e-6)
    return float(vote) if abs(vote) > 0.2 else None


def _group_geometry_diagnostics(group: Dict, trunk_distance: np.ndarray,
                                trunk_points: Sequence[Tuple[int, int]], tolerance: float,
                                bud_directions: Optional[Sequence]) -> Dict:
    points, graph = _group_to_local_graph(group)
    if not points or graph.number_of_edges() == 0:
        return {"fork_depth": 0, "max_return_to_trunk_px": 0.0,
                "max_child_angle_deg": 0.0, "bud_flow_vote": None}
    # mask_clip can sever an edge while leaving the surviving pieces in the same
    # annotation group.  Diagnostics must not silently discard either piece, nor
    # assume that a minimum-spanning forest has a path from one global root to
    # every leaf.  Measure each retained connected component independently and
    # aggregate the hard-constraint extrema across the complete group.
    if not nx.is_connected(graph):
        component_diagnostics = []
        for index, nodes in enumerate(nx.connected_components(graph), start=1):
            component = graph.subgraph(nodes).copy()
            if component.number_of_edges() == 0:
                continue
            component_group = _subgroup_from_nodes(
                group, points, graph, list(component.nodes()), f"diagnostic_component_{index:02d}",
            )
            if component_group is None:
                continue
            component_diagnostics.append(
                _group_geometry_diagnostics(
                    component_group, trunk_distance, trunk_points, tolerance, bud_directions,
                )
            )
        if not component_diagnostics:
            return {"fork_depth": 0, "max_return_to_trunk_px": 0.0,
                    "max_child_angle_deg": 0.0, "bud_flow_vote": None}
        votes = [item.get("bud_flow_vote") for item in component_diagnostics
                 if item.get("bud_flow_vote") is not None]
        vote = max(votes, key=abs) if votes else None
        return {
            "fork_depth": int(max(item["fork_depth"] for item in component_diagnostics)),
            "max_return_to_trunk_px": float(max(item["max_return_to_trunk_px"] for item in component_diagnostics)),
            "max_child_angle_deg": float(max(item["max_child_angle_deg"] for item in component_diagnostics)),
            "bud_flow_vote": vote,
        }
    root = min(graph.nodes(), key=lambda node: float(trunk_distance[points[node][1], points[node][0]]))
    weighted = graph.copy()
    for src, dst in weighted.edges():
        weighted.edges[src, dst]["length"] = float(
            np.linalg.norm(np.asarray(points[src], dtype=np.float32) - np.asarray(points[dst], dtype=np.float32))
        )
    tree = nx.minimum_spanning_tree(weighted, weight="length")
    parent = {root: None}
    fork_depth = {root: int(tree.degree[root] >= 3)}
    directed_tree = nx.bfs_tree(tree, root)
    order = list(directed_tree)
    for node in order[1:]:
        p = next(n for n in tree.neighbors(node) if n in parent)
        parent[node] = p
        fork_depth[node] = fork_depth[p] + int(tree.degree[node] >= 3)

    max_angle = 0.0
    for node in tree.nodes():
        # A degree-2 root can be an inserted attachment lying on a continuous
        # path; it is not an internal child-child fork.  Counting its two
        # directions as a 180-degree botanical split generated false alarms.
        if tree.degree[node] < 3:
            continue
        children = [n for n in tree.neighbors(node) if parent.get(n) == node]
        center = np.asarray(points[node], dtype=np.float32)
        for i in range(len(children)):
            va = _normalize_vector(np.asarray(points[children[i]], dtype=np.float32) - center)
            for j in range(i + 1, len(children)):
                vb = _normalize_vector(np.asarray(points[children[j]], dtype=np.float32) - center)
                angle = np.degrees(np.arccos(np.clip(float(np.dot(va, vb)), -1.0, 1.0)))
                max_angle = max(max_angle, float(angle))

    max_return = 0.0
    for leaf in [n for n in tree.nodes() if tree.degree[n] == 1 and n != root]:
        path = nx.shortest_path(tree, root, leaf)
        distances = [float(trunk_distance[points[n][1], points[n][0]]) for n in path]
        peak = distances[0]
        for distance in distances[1:]:
            peak = max(peak, distance)
            max_return = max(max_return, peak - distance)
    root_xy = points[root]
    return {
        "fork_depth": int(max(fork_depth.values(), default=0)),
        "max_return_to_trunk_px": float(max_return),
        "max_child_angle_deg": float(max_angle),
        "bud_flow_vote": _reliable_bud_flow_vote(group, bud_directions, root_xy),
    }


def _prune_short_terminal_spurs(group: Dict, trunk_distance: np.ndarray,
                                max_length: float = 8.0) -> Tuple[Dict, int]:
    points, graph = _group_to_local_graph(group)
    if not points or graph.number_of_edges() == 0:
        return group, 0
    removed_edges = 0
    while True:
        candidate = None
        for leaf in [node for node in graph.nodes() if graph.degree[node] == 1]:
            x, y = points[leaf]
            if float(trunk_distance[y, x]) <= 3.0:
                continue
            path = [leaf]
            previous = None
            current = leaf
            path_length = 0.0
            while graph.degree[current] <= 2:
                neighbors = [node for node in graph.neighbors(current) if node != previous]
                if not neighbors:
                    break
                neighbor = neighbors[0]
                path_length += float(np.linalg.norm(
                    np.asarray(points[current], dtype=np.float32)
                    - np.asarray(points[neighbor], dtype=np.float32)
                ))
                previous, current = current, neighbor
                path.append(current)
                if graph.degree[current] != 2:
                    break
            if graph.degree[current] >= 3 and path_length <= float(max_length):
                candidate = path
                break
        if candidate is None:
            break
        for src, dst in zip(candidate[:-1], candidate[1:]):
            if graph.has_edge(src, dst):
                graph.remove_edge(src, dst)
                removed_edges += 1
        graph.remove_nodes_from([node for node in candidate[:-1] if graph.degree[node] == 0])
    if removed_edges == 0:
        return group, 0
    active_nodes = sorted(node for node in graph.nodes() if graph.degree[node] > 0)
    remap = {node: idx for idx, node in enumerate(active_nodes)}
    cleaned = {
        **group,
        "points": [list(map(int, points[node])) for node in active_nodes],
        "edges": [[remap[int(src)], remap[int(dst)]] for src, dst in graph.edges()
                  if src in remap and dst in remap],
        "short_terminal_spurs_removed": int(removed_edges),
    }
    return cleaned, int(removed_edges)


def _max_false_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values.tolist():
        if bool(value):
            current = 0
        else:
            current += 1
            best = max(best, current)
    return int(best)


def _reroute_edge_on_supported_mask(points: List[Tuple[int, int]], graph: nx.Graph,
                                    src: int, dst: int, support_mask: np.ndarray) -> bool:
    start_xy = tuple(map(int, points[src]))
    end_xy = tuple(map(int, points[dst]))
    start_rc = _snap_to_nearest_true((start_xy[1], start_xy[0]), support_mask)
    end_rc = _snap_to_nearest_true((end_xy[1], end_xy[0]), support_mask)
    distance = float(np.linalg.norm(np.asarray(start_xy, dtype=np.float32) - np.asarray(end_xy, dtype=np.float32)))
    margin = int(max(24.0, distance * 0.5))
    y0 = max(0, min(start_rc[0], end_rc[0]) - margin)
    y1 = min(support_mask.shape[0], max(start_rc[0], end_rc[0]) + margin + 1)
    x0 = max(0, min(start_rc[1], end_rc[1]) - margin)
    x1 = min(support_mask.shape[1], max(start_rc[1], end_rc[1]) + margin + 1)
    local_support = support_mask[y0:y1, x0:x1] > 0
    if not local_support.any():
        return False
    cost = np.where(local_support, 1.0, 1e5).astype(np.float32)
    try:
        path_rc, _ = route_through_array(
            cost,
            (int(start_rc[0] - y0), int(start_rc[1] - x0)),
            (int(end_rc[0] - y0), int(end_rc[1] - x0)),
            fully_connected=True,
        )
    except (ValueError, nx.NetworkXNoPath):
        return False
    if any(not local_support[int(row), int(col)] for row, col in path_rc):
        return False
    dense_xy = [(int(col + x0), int(row + y0)) for row, col in path_rc]
    sampled_xy = [start_xy]
    sampled_xy.extend(dense_xy[index] for index in range(0, len(dense_xy), 4))
    sampled_xy.extend([dense_xy[-1], end_xy])
    deduplicated = []
    for point in sampled_xy:
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if len(deduplicated) < 2:
        return False
    if graph.has_edge(src, dst):
        graph.remove_edge(src, dst)
    previous = src
    for point in deduplicated[1:-1]:
        node = len(points)
        points.append((int(point[0]), int(point[1])))
        graph.add_node(node)
        graph.add_edge(previous, node)
        previous = node
    graph.add_edge(previous, dst)
    return True


def _partition_hard_violations(group: Dict, trunk_distance: np.ndarray,
                               tolerance: float,
                               bud_directions: Optional[Sequence] = None) -> Tuple[List[Dict], Dict[str, int]]:
    points, graph = _group_to_local_graph(group)
    counts = {
        "hierarchy_splits": 0,
        "angle_splits": 0,
        "return_splits": 0,
        "bud_vetoes": 0,
        "secondary_groups_created": 0,
        "hierarchy_violations": 0,
        "angle_violations": 0,
        "return_violations": 0,
        "bud_veto_candidates": 0,
    }
    if graph.number_of_edges() == 0:
        return [], counts
    weighted = graph.copy()
    for src, dst in weighted.edges():
        weighted.edges[src, dst]["length"] = float(
            np.linalg.norm(np.asarray(points[src], dtype=np.float32) - np.asarray(points[dst], dtype=np.float32))
        )
    tree = nx.minimum_spanning_tree(weighted, weight="length")
    root = min(tree.nodes(), key=lambda node: float(trunk_distance[points[node][1], points[node][0]]))
    parent = {root: None}
    fork_depth = {root: int(tree.degree[root] >= 3)}
    directed_tree = nx.bfs_tree(tree, root)
    order = list(directed_tree)
    cut_edges: Dict[Tuple[int, int], str] = {}
    for node in order[1:]:
        p = next(n for n in tree.neighbors(node) if n in parent)
        parent[node] = p
        candidate_depth = fork_depth[p] + int(tree.degree[node] >= 3)
        if candidate_depth > 3:
            cut_edges[tuple(sorted((p, node)))] = "fork_depth_gt_3"
            counts["hierarchy_violations"] += 1
            fork_depth[node] = int(tree.degree[node] >= 3)
        else:
            fork_depth[node] = candidate_depth

    for node in tree.nodes():
        children = [n for n in tree.neighbors(node) if parent.get(n) == node]
        if len(children) < 2:
            continue
        center = np.asarray(points[node], dtype=np.float32)
        for idx_a in range(len(children)):
            for idx_b in range(idx_a + 1, len(children)):
                a, b = children[idx_a], children[idx_b]
                va = _normalize_vector(np.asarray(points[a], dtype=np.float32) - center)
                vb = _normalize_vector(np.asarray(points[b], dtype=np.float32) - center)
                angle = float(np.degrees(np.arccos(np.clip(float(np.dot(va, vb)), -1.0, 1.0))))
                if angle <= 150.0:
                    continue
                candidates = []
                for child in (a, b):
                    descendants = nx.descendants(directed_tree, child) | {child}
                    distances = [float(trunk_distance[points[n][1], points[n][0]]) for n in descendants]
                    child_return = max(distances, default=0.0) - min(distances, default=0.0)
                    candidates.append((child_return, child))
                child_return, child = max(candidates)
                if child_return > float(tolerance):
                    edge = tuple(sorted((node, child)))
                    if edge not in cut_edges:
                        cut_edges[edge] = "angle_gt_150_with_return"
                        counts["angle_violations"] += 1
                        counts["return_violations"] += 1

    bud_vote = _reliable_bud_flow_vote(group, bud_directions, points[root])
    if bud_vote is not None and bud_vote < -0.2:
        best_return = 0.0
        best_edge = None
        for leaf in [node for node in tree.nodes() if tree.degree[node] == 1 and node != root]:
            path = nx.shortest_path(tree, root, leaf)
            distances = [float(trunk_distance[points[node][1], points[node][0]]) for node in path]
            peak_value = distances[0]
            peak_idx = 0
            for idx, value in enumerate(distances[1:], start=1):
                if value > peak_value:
                    peak_value = value
                    peak_idx = idx
                return_distance = peak_value - value
                if return_distance > best_return and peak_idx < idx:
                    best_return = return_distance
                    best_edge = tuple(sorted((path[peak_idx], path[peak_idx + 1])))
        if best_edge is not None and best_return > float(tolerance) and best_edge not in cut_edges:
            cut_edges[best_edge] = "reliable_bud_flow_veto"
            counts["return_violations"] += 1
            counts["bud_veto_candidates"] += 1

    # A geometric violation is not sufficient evidence for creating a new
    # botanical branch.  Splitting here produced rootless groups and visual
    # rings across groups.  Keep every supported edge in its rooted group and
    # expose the candidate boundaries only as diagnostics.  A real split is
    # performed earlier only when another independent trunk attachment exists.
    annotated_group = {
        **group,
        "partition_parent_group": group.get("group_id"),
        "partition_reasons": sorted(set(cut_edges.values())),
        "edge_disposition": "retained_in_rooted_group",
        "edge_diagnostics": [
            {
                "edge": [int(src), int(dst)],
                "disposition": "retained_in_rooted_group",
                "candidate_boundary_reason": cut_edges.get(tuple(sorted((int(src), int(dst))))),
            }
            for src, dst in graph.edges()
        ],
    }
    return [annotated_group], counts

    partition_tree = tree.copy()
    partition_tree.remove_edges_from(cut_edges.keys())
    components = list(nx.connected_components(partition_tree))
    owner: Dict[int, int] = {}
    for component_idx, nodes in enumerate(components):
        for node in nodes:
            owner[int(node)] = int(component_idx)
    root_owner = owner[int(root)]
    node_depth = dict(nx.shortest_path_length(tree, source=root))
    edges_by_owner: Dict[int, List[Tuple[int, int]]] = {idx: [] for idx in range(len(components))}
    reasons_by_owner: Dict[int, set] = {idx: set() for idx in range(len(components))}
    for src, dst in graph.edges():
        src_owner = owner[int(src)]
        dst_owner = owner[int(dst)]
        edge_key = tuple(sorted((int(src), int(dst))))
        if src_owner == dst_owner:
            edge_owner = src_owner
        else:
            deeper = int(src) if node_depth.get(int(src), 0) >= node_depth.get(int(dst), 0) else int(dst)
            edge_owner = owner[deeper]
            reasons_by_owner[edge_owner].add(cut_edges.get(edge_key, "cross_partition_edge"))
        edges_by_owner[edge_owner].append((int(src), int(dst)))

    partitioned: List[Dict] = []
    root_group_id = f"{group.get('group_id', 'branch')}_constrained_01"
    ordered_owners = [root_owner] + [idx for idx in range(len(components)) if idx != root_owner]
    for output_idx, component_owner in enumerate(ordered_owners, start=1):
        assigned_edges = edges_by_owner.get(component_owner, [])
        if not assigned_edges:
            continue
        node_ids = sorted({node for edge in assigned_edges for node in edge})
        remap = {node: idx for idx, node in enumerate(node_ids)}
        is_root_group = component_owner == root_owner
        partition_group = {
            **group,
            "group_id": f"{group.get('group_id', 'branch')}_constrained_{output_idx:02d}",
            "group_type": "branch" if is_root_group else "secondary_branch",
            "points": [list(map(int, points[node])) for node in node_ids],
            "edges": [[remap[src], remap[dst]] for src, dst in assigned_edges],
            "edge_diagnostics": [
                {
                    "edge": [remap[src], remap[dst]],
                    "source_edge": [int(src), int(dst)],
                    "disposition": "retained" if is_root_group else "reassigned",
                }
                for src, dst in assigned_edges
            ],
            "fork_origin_group": group.get("fork_origin_group", "trunk") if is_root_group else root_group_id,
            "partition_parent_group": group.get("group_id"),
            "partition_reasons": sorted(reasons_by_owner.get(component_owner, set())),
            "edge_disposition": "retained" if is_root_group else "reassigned",
        }
        partitioned.append(partition_group)
        if not is_root_group:
            counts["secondary_groups_created"] += 1
    return partitioned, counts


def _tolerance_contact_attachments(group: Dict, trunk_mask: np.ndarray,
                                   trunk_distance: np.ndarray, tolerance: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    group_mask = render_topology_groups_to_mask([group], trunk_mask.shape).astype(np.uint8)
    contact_mask = ((group_mask > 0) & (trunk_distance <= float(tolerance))).astype(np.uint8)
    count, labels = cv2.connectedComponents(contact_mask, connectivity=8)
    points = np.asarray(group.get("points", []), dtype=np.float32)
    attachments = []
    for component_id in range(1, count):
        coords = np.column_stack(np.where(labels == component_id))
        if coords.size == 0:
            continue
        component_distances = trunk_distance[coords[:, 0], coords[:, 1]]
        coord_idx = int(np.argmin(component_distances))
        branch_rc = (int(coords[coord_idx, 0]), int(coords[coord_idx, 1]))
        trunk_rc = _snap_to_nearest_true(branch_rc, trunk_mask)
        attachments.append((branch_rc, (int(trunk_rc[0]), int(trunk_rc[1]))))
    return attachments


def _augment_residual_skeleton_coverage(groups: Sequence[Dict], trunk_mask: np.ndarray,
                                        source_mask: np.ndarray) -> Tuple[List[Dict], Dict[str, float]]:
    augmented = [
        {**group,
         "points": [list(map(int, point)) for point in group.get("points", [])],
         "edges": [list(map(int, edge)) for edge in group.get("edges", [])]}
        for group in groups
    ]
    branch_indices = [idx for idx, group in enumerate(augmented) if group.get("group_type") == "branch"]
    if not branch_indices:
        return augmented, {"residual_components_added": 0, "residual_skeleton_pixels_added": 0.0}
    source_skeleton = skeletonize(source_mask > 0).astype(np.uint8)
    existing = render_topology_groups_to_mask(
        [augmented[idx] for idx in branch_indices], source_mask.shape,
    ).astype(np.uint8)
    existing = cv2.dilate(existing, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    trunk_distance = distance_transform_edt(trunk_mask == 0)
    residual = ((source_skeleton > 0) & (existing == 0) & (trunk_distance > 3.0)).astype(np.uint8)
    count, labels = cv2.connectedComponents(residual, connectivity=8)
    added_components = 0
    added_pixels = 0
    for component_id in range(1, count):
        component_mask = (labels == component_id).astype(np.uint8)
        pixel_count = int(component_mask.sum())
        if pixel_count < 8:
            continue
        coords = np.column_stack(np.where(component_mask > 0))
        best = None
        for target_idx in branch_indices:
            target_mask = render_topology_groups_to_mask([augmented[target_idx]], source_mask.shape).astype(np.uint8)
            target_distance = distance_transform_edt(target_mask == 0)
            values = target_distance[coords[:, 0], coords[:, 1]]
            position = int(np.argmin(values))
            distance = float(values[position])
            if best is None or distance < best[0]:
                best = (distance, target_idx, position, target_mask, values)
        if best is None or best[0] > 12.0:
            continue
        distance, target_idx, position, target_mask, target_distances = best
        # A long residual trace that stays beside an existing route is usually
        # the raw-mask medial axis running in parallel with an off-centre route.
        # Adding it creates the near-duplicate double skeleton observed in the
        # review overlays.  Genuine omitted channels touch an existing route at
        # an entrance but most of their length travels away from it.
        near_existing_fraction = float(np.mean(target_distances <= 12.0))
        if near_existing_fraction >= 0.65:
            continue
        residual_group = _component_to_group(
            component_mask,
            group_id=f"residual_{component_id:03d}",
            group_type="branch",
            color_hex=augmented[target_idx].get("color_hex", "#00FF00"),
            fork_origin_group=augmented[target_idx].get("group_id"),
            epsilon=2.0,
        )
        if residual_group is None:
            continue
        residual_points, residual_graph = _group_to_local_graph(residual_group)
        if residual_graph.number_of_edges() == 0:
            continue
        residual_tree = nx.minimum_spanning_tree(residual_graph)
        residual_group = _subgroup_from_nodes(
            residual_group, residual_points, residual_tree, list(residual_tree.nodes()), "tree",
        )
        if residual_group is None:
            continue
        target = augmented[target_idx]
        candidate_target = {
            **target,
            "points": [list(map(int, point)) for point in target.get("points", [])],
            "edges": [list(map(int, edge)) for edge in target.get("edges", [])],
            "residual_components": list(target.get("residual_components", [])),
        }
        target_points = [tuple(map(int, point)) for point in target.get("points", [])]
        residual_points_xy = [tuple(map(int, point)) for point in residual_group.get("points", [])]
        if not target_points or not residual_points_xy:
            continue
        target_array = np.asarray(target_points, dtype=np.float32)
        residual_array = np.asarray(residual_points_xy, dtype=np.float32)
        # Join through one terminal only.  Connecting an interior point turns a
        # continuous missing channel into an artificial fork; joining both
        # terminals can create a ring.  The residual graph is already an MST.
        _, normalized_residual_graph = _group_to_local_graph(residual_group)
        residual_degrees = dict(normalized_residual_graph.degree())
        terminal_nodes = [int(node) for node, degree in residual_degrees.items() if int(degree) == 1]
        if terminal_nodes:
            terminal_array = residual_array[terminal_nodes]
            terminal_distances = np.linalg.norm(
                target_array[:, None, :] - terminal_array[None, :, :], axis=2,
            )
            target_node, terminal_position = np.unravel_index(
                int(np.argmin(terminal_distances)), terminal_distances.shape,
            )
            residual_node = int(terminal_nodes[int(terminal_position)])
        else:
            distances = np.linalg.norm(target_array[:, None, :] - residual_array[None, :, :], axis=2)
            target_node, residual_node = np.unravel_index(int(np.argmin(distances)), distances.shape)
        connector_start = target_points[target_node]
        connector_end = residual_points_xy[residual_node]
        rr, cc = _line_pixels_xy(connector_start, connector_end, source_mask.shape)
        if _max_false_run(source_mask[rr, cc] > 0) > 3:
            continue
        point_lookup = {point: idx for idx, point in enumerate(target_points)}
        remap = {}
        for old_idx, point in enumerate(residual_points_xy):
            if point not in point_lookup:
                point_lookup[point] = len(candidate_target["points"])
                candidate_target["points"].append([int(point[0]), int(point[1])])
            remap[old_idx] = point_lookup[point]
        edge_keys = {tuple(sorted(map(int, edge))) for edge in candidate_target.get("edges", []) if len(edge) == 2}
        for src, dst in residual_group.get("edges", []):
            mapped = tuple(sorted((remap[int(src)], remap[int(dst)])))
            if mapped[0] != mapped[1]:
                edge_keys.add(mapped)
        connector_edge = tuple(sorted((int(target_node), int(remap[int(residual_node)]))))
        if connector_edge[0] != connector_edge[1]:
            edge_keys.add(connector_edge)
        candidate_target["edges"] = [list(edge) for edge in sorted(edge_keys)]
        candidate_target.setdefault("residual_components", []).append(f"residual_{component_id:03d}")
        _, candidate_graph = _group_to_local_graph(candidate_target)
        if not nx.is_forest(candidate_graph):
            continue
        before_diagnostics = _group_geometry_diagnostics(
            target, trunk_distance, [], 24.0, None,
        )
        after_diagnostics = _group_geometry_diagnostics(
            candidate_target, trunk_distance, [], 24.0, None,
        )
        # Coverage repair may add a real channel, but it must not manufacture a
        # deeper hierarchy, a flatter reverse fork, or extra return-to-trunk
        # behaviour beyond what the rooted candidate already contained.
        if int(after_diagnostics["fork_depth"]) > max(3, int(before_diagnostics["fork_depth"])):
            continue
        if float(after_diagnostics["max_child_angle_deg"]) > max(
            150.0, float(before_diagnostics["max_child_angle_deg"]),
        ) + 1e-6:
            continue
        if float(after_diagnostics["max_return_to_trunk_px"]) > max(
            24.0, float(before_diagnostics["max_return_to_trunk_px"]),
        ) + 1e-6:
            continue
        augmented[target_idx] = candidate_target
        added_components += 1
        added_pixels += pixel_count
    return augmented, {
        "residual_components_added": int(added_components),
        "residual_skeleton_pixels_added": float(added_pixels),
    }


def _uncross_group_degree3_motifs(
    group: Dict,
    dt_map: np.ndarray,
    bud_directions: Optional[Sequence],
    maximum_path_length: float = 130.0,
) -> Tuple[List[Dict], List[Dict]]:
    """Uncross short-linked T+T motifs after root-family graphs are merged."""
    points, graph = _group_to_local_graph(group)
    if not points or graph.number_of_edges() == 0:
        return [group], []
    for node in graph.nodes():
        graph.nodes[node]["is_trunk"] = False
    for src, dst in graph.edges():
        graph.edges[src, dst]["group_type"] = "branch"
        graph.edges[src, dst]["group_id"] = group.get("group_id", "branch")
    centers = np.asarray(
        [direction.get("centroid_global", (0.0, 0.0)) for direction in (bud_directions or [])],
        dtype=np.float32,
    )
    debug = []
    used_nodes = set()
    junctions = [node for node in graph.nodes() if graph.degree[node] == 3]
    candidates = []
    for index, node_a in enumerate(junctions):
        for node_b in junctions[index + 1:]:
            try:
                path = nx.shortest_path(graph, node_a, node_b)
            except nx.NetworkXNoPath:
                continue
            if any(graph.degree[node] >= 3 for node in path[1:-1]):
                continue
            path_length = sum(float(np.linalg.norm(
                np.asarray(graph.nodes[src]["point"], dtype=np.float32)
                - np.asarray(graph.nodes[dst]["point"], dtype=np.float32)
            )) for src, dst in zip(path[:-1], path[1:]))
            if path_length <= float(maximum_path_length):
                candidates.append((path_length, int(node_a), int(node_b), path))
    for path_length, node_a, node_b, path in sorted(candidates):
        if node_a in used_nodes or node_b in used_nodes or node_a not in graph or node_b not in graph:
            continue
        cluster_nodes = [int(node) for node in path]
        arms = _extract_cluster_arms(graph, cluster_nodes, dt_map)
        if len(arms) != 4:
            continue
        arm_flows = {}
        if len(centers) and bud_directions:
            for arm_idx, arm in enumerate(arms):
                arm_flows[arm_idx] = _compute_arm_bud_flow(
                    arm, centers, bud_directions,
                    search_radius=max(float(arm.get("radius", 8.0)) * 2.5, 12.0),
                )
        solution = _solve_degree3_x_motif(arms, arm_bud_flows=arm_flows)
        if solution is None:
            continue
        solution["arms"] = arms
        debug.append({
            "source_group_id": group.get("group_id", "branch"),
            "center_xy": list(map(float, arms[0].get("center_xy", (0.0, 0.0)))),
            "mode": solution.get("mode"),
            "junction_path_length": float(path_length),
            "corridor_points_processing": [
                list(map(int, graph.nodes[node]["point"])) for node in cluster_nodes
            ],
            "cancelled_node_points_processing": [
                list(map(int, graph.nodes[node]["point"]))
                for node in cluster_nodes if graph.degree[node] >= 3
            ],
            "arm_ports_processing": [
                list(map(int, arm.get("outside_xy", (0, 0)))) for arm in arms
            ],
            "pairing": [list(map(int, pair)) for pair in solution.get("pairing", [])],
            "pair_infos": solution.get("pair_infos", []),
            "arm_bud_flows": arm_flows,
        })
        _uncross_graph_cluster(graph, cluster_nodes, solution)
        used_nodes.update((node_a, node_b))
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted", label_attribute="old_node_id")
    normalized_points = [tuple(map(int, graph.nodes[node]["point"])) for node in graph.nodes()]
    pieces = []
    for piece_idx, nodes in enumerate(nx.connected_components(graph), start=1):
        piece = graph.subgraph(nodes).copy()
        if piece.number_of_edges() == 0:
            continue
        piece_group = _subgroup_from_nodes(
            group, normalized_points,
            graph, list(piece.nodes()), f"xpart_{piece_idx:02d}",
        )
        if piece_group is not None:
            pieces.append(piece_group)
    return pieces or [group], debug


def _apply_partition_constraints(groups: Sequence[Dict], trunk_mask: np.ndarray,
                                 source_mask: np.ndarray, tape_mask: Optional[np.ndarray],
                                 tolerance: float, bud_directions: Optional[Sequence]) -> Tuple[List[Dict], Dict[str, int]]:
    """Enforce raw-component provenance and attach auditable botanical diagnostics."""
    if not groups:
        return [], {}
    groups, residual_stats = _augment_residual_skeleton_coverage(groups, trunk_mask, source_mask)
    trunk = groups[0]
    trunk_distance = distance_transform_edt(trunk_mask == 0)
    _, source_labels = cv2.connectedComponents((source_mask > 0).astype(np.uint8), connectivity=8)
    stats = {
        "illegal_cross_component_edges": 0,
        "illegal_cross_component_edges_removed": 0,
        "supported_mask_reroutes": 0,
        "unattached_groups": 0,
        "hierarchy_splits": 0,
        "angle_splits": 0,
        "return_splits": 0,
        "bud_vetoes": 0,
        "secondary_groups_created": 0,
        "hierarchy_violations": 0,
        "angle_violations": 0,
        "return_violations": 0,
        "bud_veto_candidates": 0,
        "candidate_edge_length": 0.0,
        "pre_root_partition_edge_length": float(sum(_group_total_edge_length(group) for group in groups[1:])),
        "root_partition_edge_retention_ratio": 1.0,
        "supported_edge_length": 0.0,
        "output_edge_length": 0.0,
        "edge_retention_ratio": 1.0,
        "partition_x_clusters": 0,
        "partition_x_pairs": 0,
        "partition_x_debug": [],
        **residual_stats,
    }
    palette = ["#00FF00", "#00D0FF", "#FFCC00", "#AA66FF", "#FF8800", "#66FFAA", "#FF66AA", "#66A0FF"]
    expanded_groups: List[Dict] = []
    root_family_groups = _merge_root_family_groups(groups[1:])
    uncrossed_families = []
    branch_dt = distance_transform_edt(source_mask > 0)
    for family_group in root_family_groups:
        family_root_candidates = _tolerance_contact_attachments(
            family_group, trunk_mask, trunk_distance, float(tolerance),
        )
        declared_root_attachments = family_group.get("root_family_attachments", [])
        independent_roots = max(len(family_root_candidates), len(declared_root_attachments))
        if independent_roots >= 2:
            family_pieces, family_debug = _uncross_group_degree3_motifs(
                family_group, branch_dt, bud_directions,
            )
        else:
            family_pieces, family_debug = [family_group], []
        for item in family_debug:
            item["independent_root_count"] = int(independent_roots)
        uncrossed_families.extend(family_pieces)
        stats["partition_x_clusters"] += int(len(family_debug))
        stats["partition_x_pairs"] += int(sum(len(item.get("pairing", [])) for item in family_debug))
        stats["partition_x_debug"].extend(family_debug)
    root_family_groups = uncrossed_families
    for group in root_family_groups:
        explicit_attachments = []
        for item in group.get("root_family_attachments", []):
            anchor_xy = item.get("anchor_xy")
            attach_xy = item.get("attach_xy")
            if not (isinstance(anchor_xy, (list, tuple)) and len(anchor_xy) == 2
                    and isinstance(attach_xy, (list, tuple)) and len(attach_xy) == 2):
                continue
            explicit_attachments.append(
                ((int(attach_xy[1]), int(attach_xy[0])),
                 (int(anchor_xy[1]), int(anchor_xy[0])))
            )
        attachments = _tolerance_contact_attachments(
            group, trunk_mask, trunk_distance, float(tolerance),
        )
        if len(attachments) > 1:
            split_groups = _split_group_by_attachments(
                group,
                attachments=attachments,
                group_prefix=f"{group.get('group_id', 'branch')}_tolroot",
                palette=palette,
                min_branch_length=12.0,
                force_split=True,
                trunk_distance=trunk_distance,
                source_mask=source_mask,
            )
            expanded_groups.extend(split_groups or [group])
        else:
            expanded_groups.append(group)

    constrained: List[Dict] = [trunk]
    for group in expanded_groups:
        points, graph = _group_to_local_graph(group)
        if graph.number_of_edges() == 0:
            continue
        stats["candidate_edge_length"] += float(_group_total_edge_length(group))
        provenance = set()
        for src, dst in list(graph.edges()):
            rr, cc = _line_pixels_xy(points[src], points[dst], source_labels.shape)
            raw_support = source_mask[rr, cc] > 0
            tape_support = np.zeros_like(raw_support, dtype=bool) if tape_mask is None else tape_mask[rr, cc] > 0
            combined_support = raw_support | tape_support
            tape_supported = bool(np.any((~raw_support) & tape_support))
            if _max_false_run(combined_support) <= 3:
                if tape_supported:
                    provenance.add("tape_bridge")
                else:
                    provenance.add("raw_mask")
                continue
            src_component = _nearest_component_id(points[src], source_labels, radius=5)
            dst_component = _nearest_component_id(points[dst], source_labels, radius=5)
            if src_component > 0 and src_component == dst_component:
                routing_support = source_labels == int(src_component)
                edge_provenance = "raw_mask_reroute"
            elif tape_mask is not None and src_component > 0 and dst_component > 0:
                routing_support = (source_mask > 0) | (tape_mask > 0)
                edge_provenance = "tape_bridge_reroute"
            else:
                graph.remove_edge(src, dst)
                stats["illegal_cross_component_edges_removed"] += 1
                continue
            if _reroute_edge_on_supported_mask(points, graph, int(src), int(dst), routing_support):
                provenance.add(edge_provenance)
                stats["supported_mask_reroutes"] += 1
            else:
                if graph.has_edge(src, dst):
                    graph.remove_edge(src, dst)
                stats["illegal_cross_component_edges_removed"] += 1
                continue
            if tape_supported:
                provenance.add("tape_bridge")

        pieces = [graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph) if graph.subgraph(nodes).number_of_edges()]
        for piece_idx, piece in enumerate(pieces, start=1):
            piece_group = _subgroup_from_nodes(group, points, graph, list(piece.nodes()), f"part_{piece_idx:02d}")
            if piece_group is None:
                continue
            stats["supported_edge_length"] += float(_group_total_edge_length(piece_group))
            partitioned_groups, split_counts = _partition_hard_violations(
                piece_group, trunk_distance=trunk_distance, tolerance=tolerance,
                bud_directions=bud_directions,
            )
            for key, value in split_counts.items():
                stats[key] += int(value)
            for partition_group in partitioned_groups:
                component_ids = [_nearest_component_id(tuple(map(int, p)), source_labels) for p in partition_group["points"]]
                component_ids = [value for value in component_ids if value > 0]
                source_component_id = int(np.bincount(component_ids).argmax()) if component_ids else 0
                root_contacts = _group_contact_count(partition_group, trunk_distance, min(float(tolerance), 3.0))
                root_contacts_24px = _group_contact_count(partition_group, trunk_distance, float(tolerance))
                diagnostics = _group_geometry_diagnostics(
                    partition_group, trunk_distance, groups[0].get("points", []), tolerance, bud_directions,
                )
                if root_contacts == 0 and partition_group.get("group_type") == "branch":
                    partition_group["group_type"] = "secondary_branch"
                    partition_group["unattached_reason"] = "partition_has_no_independent_trunk_root"
                    stats["unattached_groups"] += 1
                elif root_contacts == 1:
                    partition_group["group_type"] = "branch"
                    partition_group["fork_origin_group"] = "trunk"
                partition_group.update({
                    "color_hex": palette[(len(constrained) - 1) % len(palette)],
                    "root_contact_count": int(root_contacts),
                    "root_contact_count_24px": int(root_contacts_24px),
                    "source_component_id": int(source_component_id),
                    "bridge_provenance": sorted(provenance) or ["raw_mask"],
                    **diagnostics,
                })
                stats["output_edge_length"] += float(_group_total_edge_length(partition_group))
                constrained.append(partition_group)
    stats["edge_retention_ratio"] = float(
        stats["output_edge_length"] / max(stats["supported_edge_length"], 1e-6)
    )
    stats["root_partition_edge_retention_ratio"] = float(
        stats["candidate_edge_length"] / max(stats["pre_root_partition_edge_length"], 1e-6)
    )
    return constrained, stats


def _build_graph_from_annotation_groups(groups: Sequence[Dict]) -> Tuple[List[Tuple[int, int]], List[str], nx.Graph, List[int]]:
    points: List[Tuple[int, int]] = []
    node_types: List[str] = []
    point_to_id: Dict[Tuple[int, int], int] = {}
    graph = nx.Graph()
    trunk_path: List[int] = []

    def ensure_node(point_xy: Sequence[int], node_type: str) -> int:
        point = (int(point_xy[0]), int(point_xy[1]))
        if point in point_to_id:
            node_id = point_to_id[point]
            if node_types[node_id] == "trunk_node" and node_type != "trunk_node":
                node_types[node_id] = node_type
            return node_id
        node_id = len(points)
        point_to_id[point] = node_id
        points.append(point)
        node_types.append(node_type)
        graph.add_node(node_id, point=point)
        return node_id

    for group in groups:
        group_type = group.get("group_type", "branch")
        group_points = group.get("points", [])
        group_edges = group.get("edges", [])
        local_degrees = [0] * len(group_points)
        for edge in group_edges:
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(group_points) and 0 <= dst < len(group_points) and src != dst:
                local_degrees[src] += 1
                local_degrees[dst] += 1

        local_to_global: Dict[int, int] = {}
        for local_idx, point in enumerate(group_points):
            if group_type == "trunk":
                node_type = "trunk_node"
            else:
                degree = local_degrees[local_idx]
                if degree <= 1:
                    node_type = "branch_endpoint"
                elif degree >= 3:
                    node_type = "branch_junction"
                else:
                    node_type = "branch_path"
            local_to_global[local_idx] = ensure_node(point, node_type)

        for edge in group_edges:
            if len(edge) != 2:
                continue
            src = local_to_global.get(int(edge[0]))
            dst = local_to_global.get(int(edge[1]))
            if src is not None and dst is not None and src != dst:
                graph.add_edge(src, dst)

        if group_type == "trunk":
            ordered = [local_to_global[idx] for idx in range(len(group_points)) if idx in local_to_global]
            trunk_path.extend(ordered)

    trunk_path = _dedupe_consecutive([points[idx] for idx in trunk_path])
    point_to_id_after = {point: idx for idx, point in enumerate(points)}
    trunk_ids = [point_to_id_after[point] for point in trunk_path if point in point_to_id_after]
    return points, node_types, graph, trunk_ids


def _component_to_group(
    component_mask: np.ndarray,
    group_id: str,
    group_type: str,
    color_hex: str,
    fork_origin_group: Optional[str],
    forced_points_xy: Optional[Sequence[Tuple[int, int]]] = None,
    epsilon: float = 2.0,
) -> Optional[Dict]:
    skeleton = (np.asarray(component_mask) > 0).astype(np.uint8)
    if not np.any(skeleton):
        return None

    degree_map = _compute_degree_map(skeleton)
    keypoints_rc = {
        (int(r), int(c))
        for r, c in zip(*np.where((skeleton > 0) & ((degree_map != 2) | (degree_map == 0))))
    }

    forced_points_xy = forced_points_xy or []
    for point_xy in forced_points_xy:
        snapped_r, snapped_c = _snap_to_nearest_true((int(point_xy[1]), int(point_xy[0])), skeleton)
        if skeleton[snapped_r, snapped_c] > 0:
            keypoints_rc.add((snapped_r, snapped_c))

    if not keypoints_rc:
        coords = list(zip(*np.where(skeleton > 0)))
        if not coords:
            return None
        coords = [(int(r), int(c)) for r, c in coords]
        keypoints_rc.add(min(coords))
        keypoints_rc.add(max(coords))

    visited_edges = set()
    point_to_local: Dict[Tuple[int, int], int] = {}
    local_points: List[List[int]] = []
    local_edges: List[List[int]] = []

    def ensure_local(point_xy: Tuple[int, int]) -> int:
        point_xy = (int(point_xy[0]), int(point_xy[1]))
        if point_xy in point_to_local:
            return point_to_local[point_xy]
        idx = len(local_points)
        point_to_local[point_xy] = idx
        local_points.append([point_xy[0], point_xy[1]])
        return idx

    for start_rc in sorted(keypoints_rc):
        for neighbor_rc in _neighbors_of(start_rc, skeleton):
            edge_key = tuple(sorted((start_rc, neighbor_rc)))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            path_rc = [start_rc, neighbor_rc]
            prev_rc = start_rc
            current_rc = neighbor_rc
            while current_rc not in keypoints_rc:
                next_candidates = [pt for pt in _neighbors_of(current_rc, skeleton) if pt != prev_rc]
                if not next_candidates:
                    break
                next_rc = next_candidates[0]
                next_edge = tuple(sorted((current_rc, next_rc)))
                if next_edge in visited_edges:
                    break
                visited_edges.add(next_edge)
                path_rc.append(next_rc)
                prev_rc, current_rc = current_rc, next_rc

            polyline_xy = _path_rc_to_xy(path_rc)
            simplified_xy = simplify_polyline(polyline_xy, epsilon=epsilon)
            if len(simplified_xy) < 2:
                simplified_xy = [polyline_xy[0], polyline_xy[-1]]
            local_ids = [ensure_local(point_xy) for point_xy in simplified_xy]
            for src, dst in zip(local_ids[:-1], local_ids[1:]):
                if src != dst:
                    edge = [src, dst]
                    if edge not in local_edges and edge[::-1] not in local_edges:
                        local_edges.append(edge)

    if len(local_points) < 2 or not local_edges:
        return None
    local_graph = nx.Graph()
    local_graph.add_nodes_from(range(len(local_points)))
    local_graph.add_edges_from((int(edge[0]), int(edge[1])) for edge in local_edges)
    if not nx.is_forest(local_graph):
        if forced_points_xy:
            local_arr = np.asarray(local_points, dtype=np.float32)
            root_id = 0
            best_dist = float("inf")
            for forced_point in forced_points_xy:
                forced_arr = np.asarray(forced_point, dtype=np.float32)
                candidate = int(np.argmin(np.linalg.norm(local_arr - forced_arr[None, :], axis=1)))
                dist = float(np.linalg.norm(local_arr[candidate] - forced_arr))
                if dist < best_dist:
                    best_dist = dist
                    root_id = candidate
        else:
            root_id = 0
        tree_edges = list(nx.bfs_edges(local_graph, source=root_id))
        if tree_edges:
            local_edges = [[int(src), int(dst)] for src, dst in tree_edges]
    return {
        "group_id": group_id,
        "group_type": group_type,
        "color_hex": color_hex,
        "points": local_points,
        "edges": local_edges,
        "fork_origin_group": fork_origin_group,
    }


def _build_annotation_groups(
    trunk_line_xy: Sequence[Tuple[int, int]],
    branch_dense_lines_xy: Sequence[Sequence[Tuple[int, int]]],
    branch_attach_points_xy: Sequence[Tuple[int, int]],
    branch_rdp_epsilon: float,
    image_shape: Tuple[int, int],
) -> List[Dict]:
    groups: List[Dict] = []
    trunk_points = [list(map(int, point)) for point in _dedupe_consecutive(trunk_line_xy)]
    if trunk_points:
        groups.append(
            {
                "group_id": "trunk",
                "group_type": "trunk",
                "color_hex": "#FF3232",
                "points": trunk_points,
                "edges": [[i, i + 1] for i in range(max(len(trunk_points) - 1, 0))],
                "fork_origin_group": None,
            }
        )

    if not branch_dense_lines_xy:
        return groups

    branch_union = _draw_polyline_mask(branch_dense_lines_xy, image_shape, thickness=1)
    branch_union = skeletonize(branch_union > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(branch_union, connectivity=8)
    palette = ["#00FF00", "#00D0FF", "#FFCC00", "#AA66FF", "#FF8800", "#66FFAA"]
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        forced_points = [point for point in branch_attach_points_xy if component_mask[min(max(point[1], 0), component_mask.shape[0] - 1), min(max(point[0], 0), component_mask.shape[1] - 1)] > 0]
        group = _component_to_group(
            component_mask=component_mask,
            group_id=f"branch_{label:02d}",
            group_type="branch",
            color_hex=palette[(label - 1) % len(palette)],
            fork_origin_group="trunk",
            forced_points_xy=forced_points,
            epsilon=branch_rdp_epsilon,
        )
        if group is not None:
            groups.append(group)
    return groups


class DijkstraSkeletonRouter:
    def __init__(
        self,
        outside_cost: float = 100.0,
        root_band_height: int = 16,
        trunk_exclusion_radius: float = 10.0,
        min_branch_length: float = 12.0,
        prune_spur_length: int = 8,
        trunk_rdp_epsilon_ratio: float = 0.01,
        branch_rdp_epsilon: float = 2.0,
        min_endpoint_branch_length: int = 16,
        max_endpoints_to_route: int = 24,
        max_processing_dim: int = 1280,
        enable_vertical_bridge: bool = False,
        bridge_max_gap: int = 96,
        bridge_max_dx: int = 28,
        bridge_min_component_area: int = 80,
        bridge_min_component_height: int = 40,
        enable_skeleton_bridge: bool = False,
        enable_junction_pairing: bool = True,
        enable_bud_cost_adjustment: bool = False,
        enable_bud_density_prior: bool = True,
        enable_bud_direction_flow: bool = True,
        enable_bud_root_split: bool = True,
        enable_partition_constraints: bool = True,
        structural_tolerance_px: float = 24.0,
        junction_cluster_radius: float = 12.0,
        junction_prior_path: os.PathLike = DEFAULT_JUNCTION_PRIOR_PATH,
    ):
        self.outside_cost = float(outside_cost)
        self.root_band_height = int(root_band_height)
        self.trunk_exclusion_radius = float(trunk_exclusion_radius)
        self.min_branch_length = float(min_branch_length)
        self.prune_spur_length = int(prune_spur_length)
        self.trunk_rdp_epsilon_ratio = float(trunk_rdp_epsilon_ratio)
        self.branch_rdp_epsilon = float(branch_rdp_epsilon)
        self.min_endpoint_branch_length = int(min_endpoint_branch_length)
        self.max_endpoints_to_route = int(max_endpoints_to_route)
        self.max_processing_dim = int(max_processing_dim)
        self.enable_vertical_bridge = bool(enable_vertical_bridge)
        self.bridge_max_gap = int(bridge_max_gap)
        self.bridge_max_dx = int(bridge_max_dx)
        self.bridge_min_component_area = int(bridge_min_component_area)
        self.bridge_min_component_height = int(bridge_min_component_height)
        self.enable_skeleton_bridge = bool(enable_skeleton_bridge)
        self.enable_junction_pairing = bool(enable_junction_pairing)
        self.enable_bud_cost_adjustment = bool(enable_bud_cost_adjustment)
        self.enable_bud_density_prior = bool(enable_bud_density_prior)
        self.enable_bud_direction_flow = bool(enable_bud_direction_flow)
        self.enable_bud_root_split = bool(enable_bud_root_split)
        self.enable_partition_constraints = bool(enable_partition_constraints)
        self.structural_tolerance_px = float(structural_tolerance_px)
        self.junction_cluster_radius = float(junction_cluster_radius)
        self.junction_prior_path = Path(junction_prior_path)

    def __call__(self, combined_mask: np.ndarray,
                bud_boxes: Optional[Sequence[Sequence[float]]] = None,
                bud_orientations: Optional[Sequence] = None,
                bud_masks_info: Optional[Sequence] = None,
                bud_scores: Optional[Sequence[float]] = None,
                bud_directions: Optional[Sequence] = None,
                protected_tape_mask: Optional[np.ndarray] = None,
                source_mask: Optional[np.ndarray] = None) -> PredictionResult:
        original_mask = _normalize_binary_mask(combined_mask)
        source_mask_original = _normalize_binary_mask(source_mask) if source_mask is not None else original_mask.copy()
        mask, scale_x, scale_y = _resize_binary_mask(original_mask, max_dim=self.max_processing_dim)
        source_mask_scaled = cv2.resize(
            source_mask_original.astype(np.uint8),
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        tape_mask_scaled = None
        if protected_tape_mask is not None and np.any(protected_tape_mask):
            tape_mask_scaled = cv2.resize(
                (protected_tape_mask > 0).astype(np.uint8),
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Module 6: 预计算芽点中心 (processing 坐标系)
        bud_centers_scaled: Optional[List[Tuple[float, float]]] = None
        bud_orientations_scaled: Optional[List] = None
        bud_directions_scaled: Optional[List] = None
        if bud_boxes is not None and len(bud_boxes) > 0:
            bud_centers_scaled = [
                (float((x1 + x2) * 0.5 / scale_x), float((y1 + y2) * 0.5 / scale_y))
                for x1, y1, x2, y2 in bud_boxes
            ]
        if bud_orientations is not None and len(bud_orientations) > 0:
            # 缩放centroid到processing坐标系, 保留axis_angle和is_elongated不变
            bud_orientations_scaled = []
            for o in bud_orientations:
                get = (lambda k: getattr(o, k)) if hasattr(o, 'axis_angle') else (lambda k: o[k])
                cx, cy = get('centroid_global')
                bud_orientations_scaled.append({
                    'axis_angle': get('axis_angle'),
                    'is_elongated': get('is_elongated'),
                    'centroid_global': (float(cx / scale_x), float(cy / scale_y)),
                })

        def scale_bud_directions(directions: Optional[Sequence]) -> Optional[List[Dict]]:
            if directions is None:
                return None
            scaled = []
            for direction in directions:
                get = (lambda key: getattr(direction, key)) if hasattr(direction, "vector_xy") else (lambda key: direction.get(key))
                base = get("base_global")
                tip = get("tip_global")
                centroid = get("centroid_global")
                base_scaled = np.asarray([float(base[0]) / scale_x, float(base[1]) / scale_y], dtype=np.float32)
                tip_scaled = np.asarray([float(tip[0]) / scale_x, float(tip[1]) / scale_y], dtype=np.float32)
                vector = _normalize_vector(tip_scaled - base_scaled)
                scaled.append({
                    "bud_index": int(get("bud_index")),
                    "base_global": tuple(map(float, base_scaled)),
                    "tip_global": tuple(map(float, tip_scaled)),
                    "centroid_global": (float(centroid[0]) / scale_x, float(centroid[1]) / scale_y),
                    "vector_xy": tuple(map(float, vector)),
                    "confidence": float(get("confidence") or 0.0),
                    "is_reliable": bool(get("is_reliable")),
                    "is_latent_spur": bool(get("is_latent_spur")),
                })
            return scaled

        bud_directions_scaled = scale_bud_directions(bud_directions)

        if np.any(mask):
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
        bridge_segments: List[Dict[str, int]] = []
        if self.enable_vertical_bridge and np.any(mask):
            mask, bridge_segments = _bridge_vertical_gaps(
                mask,
                max_gap=self.bridge_max_gap,
                max_dx=self.bridge_max_dx,
                min_component_area=self.bridge_min_component_area,
                min_component_height=self.bridge_min_component_height,
            )
        rooted_mask, component_stats = _select_bottom_component(mask)
        if not np.any(rooted_mask):
            return PredictionResult(mask=original_mask, routing_stats={"status": "empty_mask", **component_stats})

        dt_map = distance_transform_edt(rooted_mask > 0).astype(np.float32)
        max_dt = float(dt_map.max()) if np.any(dt_map) else 1.0
        dt_norm = dt_map / max(max_dt, 1e-6)

        skeleton = skeletonize(rooted_mask > 0).astype(np.uint8)
        if self.prune_spur_length > 1:
            skeleton = _prune_short_branches(skeleton, min_length=self.prune_spur_length)
        root_seed_rc = _find_root_base(rooted_mask, dt_map, band_height=self.root_band_height)
        root_rc = _snap_to_nearest_true(root_seed_rc, skeleton if np.any(skeleton) else rooted_mask)
        skeleton_trunk_bridges: List[Dict[str, int]] = []
        if self.enable_skeleton_bridge and np.any(skeleton):
            skeleton, skeleton_trunk_bridges = _bridge_trunk_gaps_on_skeleton(skeleton, dt_map=dt_map, root_rc=root_rc)
            root_rc = _snap_to_nearest_true(root_seed_rc, skeleton if np.any(skeleton) else rooted_mask)
        degree_map = _compute_degree_map(skeleton)
        endpoints_rc = list(zip(*np.where((skeleton > 0) & (degree_map == 1))))

        if bud_directions_scaled is None and bud_masks_info is not None and len(bud_masks_info) > 0:
            try:
                from bud_skeleton_fusion.bud_orientation import extract_directed_bud_orientations
                skeleton_reference = cv2.resize(
                    skeleton.astype(np.uint8),
                    (original_mask.shape[1], original_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                extracted_directions = extract_directed_bud_orientations(
                    list(bud_masks_info),
                    skeleton_reference,
                    scores=bud_scores,
                )
                bud_directions_scaled = scale_bud_directions(extracted_directions)
            except Exception as exc:
                raise RuntimeError("Bud-direction extraction failed; constrained routing was not run") from exc

        cost = 1.0 - dt_norm
        cost[rooted_mask == 0] = self.outside_cost
        cost = np.clip(cost, 1e-3, self.outside_cost).astype(np.float32)

        # Module 6: 芽点辅助分割消歧 — 降低芽点附近 cost, 引导路由经过有芽区域
        if self.enable_bud_cost_adjustment and bud_centers_scaled:
            cost = _adjust_cost_by_bud_centers(
                cost, rooted_mask, bud_centers_scaled, dt_norm,
            )

        best_trunk_path, top_candidates, best_score, skeleton_cost = _choose_trunk_path_on_skeleton(
            skeleton=skeleton,
            dt_map=dt_map,
            dt_norm=dt_norm,
            root_rc=root_rc,
            outside_cost=self.outside_cost,
        )

        trunk_dense_xy = _path_rc_to_xy(best_trunk_path)
        trunk_mask = _draw_polyline_mask([trunk_dense_xy], rooted_mask.shape, thickness=1)
        trunk_dist = distance_transform_edt(trunk_mask == 0)

        raw_endpoints_rc = list(endpoints_rc)
        filtered_endpoints_rc: List[Tuple[int, int]] = []
        endpoint_infos: List[Tuple[int, float, Tuple[int, int]]] = []
        for endpoint_rc in endpoints_rc:
            if endpoint_rc == root_rc:
                continue
            if trunk_dist[endpoint_rc] <= self.trunk_exclusion_radius:
                continue
            branch_trace_len = _branch_trace_length(endpoint_rc, skeleton, degree_map)
            if branch_trace_len < self.min_endpoint_branch_length:
                continue
            endpoint_infos.append((branch_trace_len, float(trunk_dist[endpoint_rc]), endpoint_rc))

        endpoint_infos.sort(key=lambda item: (-item[0], -item[1], item[2][0], item[2][1]))
        if self.max_endpoints_to_route > 0:
            endpoint_infos = endpoint_infos[:self.max_endpoints_to_route]
        filtered_endpoints_rc = [item[2] for item in endpoint_infos]
        filtered_endpoints_rc.sort(key=lambda p: (p[0], p[1]))
        branch_dense_lines_xy: List[List[Tuple[int, int]]] = []
        branch_attach_points_xy: List[Tuple[int, int]] = []
        routed_count = 0
        skipped_short = 0
        skipped_no_path = 0
        branch_component_count = 0
        trunk_path_mask = np.zeros_like(skeleton, dtype=np.uint8)
        for r, c in best_trunk_path:
            trunk_path_mask[int(r), int(c)] = 1
        residual_skeleton = skeleton.copy()
        residual_skeleton[trunk_path_mask > 0] = 0
        num_labels, branch_labels = cv2.connectedComponents(residual_skeleton, connectivity=8)
        annotation_groups: List[Dict] = []
        trunk_arc = max(_polyline_length(trunk_dense_xy), 1.0)
        trunk_epsilon = max(1.0, trunk_arc * self.trunk_rdp_epsilon_ratio)
        trunk_line_xy = simplify_polyline(trunk_dense_xy, epsilon=trunk_epsilon)
        annotation_groups.append(
            {
                "group_id": "trunk",
                "group_type": "trunk",
                "color_hex": "#FF3232",
                "points": [list(map(int, point)) for point in _dedupe_consecutive(trunk_line_xy)],
                "edges": [[i, i + 1] for i in range(max(len(_dedupe_consecutive(trunk_line_xy)) - 1, 0))],
                "fork_origin_group": None,
            }
        )

        for label in range(1, num_labels):
            component_mask = (branch_labels == label).astype(np.uint8)
            component_size = int(component_mask.sum())
            if component_size <= 1:
                continue
            side_masks = _split_component_mask_by_trunk_side(
                component_mask,
                trunk_points=trunk_line_xy,
                min_pixels=max(10, self.min_endpoint_branch_length // 2),
            )
            valid_group_count = 0
            for side_idx, side_mask in enumerate(side_masks, start=1):
                attachments = _find_component_attachments(side_mask, trunk_path_mask, dt_map, trunk_points=trunk_dense_xy)
                if not attachments:
                    skipped_no_path += 1
                    continue
                component_group = _component_to_group(
                    component_mask=side_mask,
                    group_id=f"branch_{label:02d}_{side_idx:02d}",
                    group_type="branch",
                    color_hex="#00FF00",
                    fork_origin_group="trunk",
                    forced_points_xy=[(int(branch_attach_rc[1]), int(branch_attach_rc[0])) for branch_attach_rc, _ in attachments],
                    epsilon=self.branch_rdp_epsilon,
                )
                if component_group is None:
                    skipped_no_path += 1
                    continue
                split_groups = _split_group_by_attachments(
                    component_group,
                    attachments=attachments,
                    group_prefix=f"branch_{label:02d}_{side_idx:02d}",
                    palette=["#00FF00", "#00D0FF", "#FFCC00", "#AA66FF", "#FF8800", "#66FFAA", "#FF66AA", "#66A0FF"],
                    min_branch_length=self.min_branch_length,
                )
                if not split_groups:
                    skipped_no_path += 1
                    continue

                # SHORT-CIRCUIT: 06-26 5-round validation loop bypassed to restore 06-22 behavior
                if False:
                    # Post-hoc validation: re-split any group with >1 trunk contact
                    _max_validation_iter = 5
                    for _v_iter in range(_max_validation_iter):
                        _re_split_needed = False
                        _validated_groups: List[Dict] = []
                        for _sg in split_groups:
                            _contacts = _count_trunk_contacts(_sg, trunk_path_mask, contact_radius=2)
                            if _contacts <= 1:
                                _validated_groups.append(_sg)
                                continue
                            _re_split_needed = True
                            # Render group mask for re-detection
                            _sg_pts = [tuple(map(int, p)) for p in _sg.get("points", [])]
                            _sg_edges_raw = _sg.get("edges", [])
                            if not _sg_pts or not _sg_edges_raw:
                                _validated_groups.append(_sg)
                                continue
                            _sg_edges = [(int(e[0]), int(e[1])) for e in _sg_edges_raw]
                            _x_vals = [p[0] for p in _sg_pts]
                            _y_vals = [p[1] for p in _sg_pts]
                            _x0, _x1 = max(0, min(_x_vals) - 4), min(trunk_path_mask.shape[1], max(_x_vals) + 5)
                            _y0, _y1 = max(0, min(_y_vals) - 4), min(trunk_path_mask.shape[0], max(_y_vals) + 5)
                            _gh, _gw = _y1 - _y0, _x1 - _x0
                            if _gh <= 0 or _gw <= 0:
                                _validated_groups.append(_sg)
                                continue
                            _gm = np.zeros((_gh, _gw), dtype=np.uint8)
                            _lpts = [(int(p[0]) - _x0, int(p[1]) - _y0) for p in _sg_pts]
                            for _s, _d in _sg_edges:
                                if 0 <= _s < len(_lpts) and 0 <= _d < len(_lpts):
                                    cv2.line(_gm, _lpts[_s], _lpts[_d], 1, 1, lineType=cv2.LINE_4)
                            _gm = cv2.dilate(_gm, np.ones((3, 3), dtype=np.uint8), iterations=1)
                            _local_trunk = trunk_path_mask[_y0:_y1, _x0:_x1]
                            _re_attachments = _find_component_attachments(
                                _gm, _local_trunk, dt_map[_y0:_y1, _x0:_x1],
                                trunk_points=None,  # local coord; use 2D fallback
                            )
                            # Shift attachments back to global coords
                            _re_attachments_global = [
                                ((br + _y0, bc + _x0), (tr + _y0, tc + _x0))
                                for (br, bc), (tr, tc) in _re_attachments
                            ]
                            if len(_re_attachments_global) <= 1:
                                _validated_groups.append(_sg)
                                continue
                            # Force re-split
                            _re_groups = _split_group_by_attachments(
                                _sg,
                                attachments=_re_attachments_global,
                                group_prefix=f"{_sg.get('group_id', 'branch')}_v{_v_iter}",
                                palette=["#00FF00", "#00D0FF", "#FFCC00", "#AA66FF", "#FF8800", "#66FFAA"],
                                min_branch_length=self.min_branch_length,
                            )
                            _validated_groups.extend(_re_groups if _re_groups else [_sg])
                        split_groups = _validated_groups
                        if not _re_split_needed:
                            break

                for split_group in split_groups:
                    component_points_xy = [tuple(map(int, point)) for point in split_group.get("points", [])]
                    if len(component_points_xy) < 2:
                        continue
                    group_edge_length = _group_total_edge_length(split_group)
                    if group_edge_length < self.min_branch_length:
                        skipped_short += 1
                        continue
                    component_edges = split_group.get("edges", [])
                    for edge in component_edges:
                        if len(edge) != 2:
                            continue
                        src, dst = int(edge[0]), int(edge[1])
                        if 0 <= src < len(component_points_xy) and 0 <= dst < len(component_points_xy):
                            branch_dense_lines_xy.append([component_points_xy[src], component_points_xy[dst]])
                    branch_attach_points_xy.append(tuple(map(int, split_group["points"][0])))
                    annotation_groups.append(split_group)
                    routed_count += 1
                    valid_group_count += 1
            if valid_group_count > 0:
                branch_component_count += 1
            else:
                skipped_short += 1

        trunk_line_xy = _simplify_with_forced_points(trunk_dense_xy, branch_attach_points_xy, epsilon=trunk_epsilon)
        annotation_groups[0]["points"] = [list(map(int, point)) for point in _dedupe_consecutive(trunk_line_xy)]
        annotation_groups[0]["edges"] = [[i, i + 1] for i in range(max(len(annotation_groups[0]["points"]) - 1, 0))]
        junction_stats: Dict[str, float] = {}
        if self.enable_junction_pairing and len(annotation_groups) > 1:
            annotation_groups, junction_stats = reconstruct_branch_groups_with_junction_pairing(
                annotation_groups,
                dt_map=dt_map,
                min_branch_length=self.min_branch_length,
                prior_path=self.junction_prior_path,
                cluster_radius=self.junction_cluster_radius,
                root_point_xy=(int(root_rc[1]), int(root_rc[0])),
                bud_centers=bud_centers_scaled,
                bud_orientations=bud_orientations_scaled,
                bud_directions=bud_directions_scaled,
                protected_tape_mask=tape_mask_scaled,
                enable_bud_density_prior=self.enable_bud_density_prior,
                enable_bud_direction_flow=self.enable_bud_direction_flow,
                enable_bud_root_split=self.enable_bud_root_split,
                structural_tolerance=max(2.0, self.structural_tolerance_px / max(scale_x, scale_y)),
            )

        structural_tolerance_scaled = max(2.0, self.structural_tolerance_px / max(scale_x, scale_y))
        partition_stats: Dict[str, object] = {"partition_constraints_enabled": self.enable_partition_constraints}
        if self.enable_partition_constraints:
            annotation_groups, applied_partition_stats = _apply_partition_constraints(
                annotation_groups,
                trunk_mask=_draw_polyline_mask(
                    [[tuple(map(int, point)) for point in annotation_groups[0].get("points", [])]],
                    rooted_mask.shape,
                    thickness=1,
                ),
                source_mask=source_mask_scaled,
                tape_mask=tape_mask_scaled,
                tolerance=structural_tolerance_scaled,
                bud_directions=bud_directions_scaled,
            )
            partition_stats.update(applied_partition_stats)
        for constrained_group in annotation_groups[1:]:
            constrained_group["max_return_to_trunk_px"] = float(
                constrained_group.get("max_return_to_trunk_px", 0.0) * max(scale_x, scale_y)
            )

        branch_lines_xy = [line_xy for line_xy in branch_dense_lines_xy if _polyline_length(line_xy) >= 1.0]
        branch_lines_xy = []
        for group in annotation_groups[1:]:
            points_xy = [tuple(map(int, point)) for point in group.get("points", [])]
            for edge in group.get("edges", []):
                if len(edge) != 2:
                    continue
                src, dst = int(edge[0]), int(edge[1])
                if 0 <= src < len(points_xy) and 0 <= dst < len(points_xy):
                    branch_lines_xy.append([points_xy[src], points_xy[dst]])

        endpoint_mask = np.zeros_like(rooted_mask, dtype=np.uint8)
        for r, c in filtered_endpoints_rc:
            endpoint_mask[r, c] = 1

        trunk_prob = _draw_polyline_mask([trunk_line_xy], rooted_mask.shape, thickness=1).astype(np.float32)
        branch_prob = render_topology_groups_to_mask(annotation_groups[1:], rooted_mask.shape).astype(np.float32)

        trunk_line_xy_full = _rescale_points_xy(trunk_line_xy, scale_x, scale_y)
        branch_lines_xy_full = [_rescale_points_xy(line_xy, scale_x, scale_y) for line_xy in branch_lines_xy]
        annotation_groups_full = _rescale_annotation_groups(annotation_groups, scale_x, scale_y)
        points_full, node_types_full, graph_full, trunk_path_full = _build_graph_from_annotation_groups(annotation_groups_full)
        endpoint_mask_full = np.zeros_like(original_mask, dtype=np.uint8)
        for r, c in filtered_endpoints_rc:
            rr = int(np.clip(round(r * scale_y), 0, original_mask.shape[0] - 1))
            cc = int(np.clip(round(c * scale_x), 0, original_mask.shape[1] - 1))
            endpoint_mask_full[rr, cc] = 1

        trunk_prob_full = _draw_polyline_mask([trunk_line_xy_full], original_mask.shape, thickness=1).astype(np.float32)
        branch_prob_full = _draw_polyline_mask(branch_lines_xy_full, original_mask.shape, thickness=1).astype(np.float32)
        skeleton_full = cv2.resize(skeleton.astype(np.uint8), (original_mask.shape[1], original_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        trunk_mask_full = _draw_polyline_mask([trunk_line_xy_full], original_mask.shape, thickness=1)
        cost_full = cv2.resize(skeleton_cost.astype(np.float32), (original_mask.shape[1], original_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Post-junction-pairing guardrail removed (Phase II revert)

        routing_stats = {
            "status": "ok",
            **component_stats,
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "processing_shape": [int(rooted_mask.shape[1]), int(rooted_mask.shape[0])],
            "root_seed": [int(root_seed_rc[1]), int(root_seed_rc[0])],
            "root_point": [int(root_rc[1]), int(root_rc[0])],
            "top_candidate_count": len(top_candidates),
            "bridge_segments": len(bridge_segments),
            "skeleton_trunk_bridges": len(skeleton_trunk_bridges),
            "endpoints_total": len(raw_endpoints_rc),
            "endpoints_filtered": len(filtered_endpoints_rc),
            "routed": routed_count,
            "branch_components": branch_component_count,
            "skipped_short": skipped_short,
            "skipped_no_path": skipped_no_path,
            "trunk_score": float(best_score),
            "directed_buds_total": float(len(bud_directions_scaled or [])),
            "directed_buds_reliable": float(sum(bool(direction.get("is_reliable")) for direction in (bud_directions_scaled or []))),
            "max_dt": float(max_dt),
            **junction_stats,
            **partition_stats,
        }

        return PredictionResult(
            points=points_full,
            confidences=[1.0] * len(points_full),
            node_types=node_types_full,
            graph=graph_full,
            trunk_path=trunk_path_full,
            trunk_node_prob=trunk_prob_full,
            branch_junction_prob=np.zeros_like(trunk_prob_full, dtype=np.float32),
            branch_endpoint_prob=endpoint_mask_full.astype(np.float32),
            trunk_edge_prob=trunk_prob_full,
            branch_edge_prob=branch_prob_full,
            trunk_thin=(trunk_prob_full > 0).astype(np.uint8) * 255,
            branch_thin=(branch_prob_full > 0).astype(np.uint8) * 255,
            routing_stats=routing_stats,
            cost_matrix=cost_full,
            root_point=(int(round(root_rc[1] * scale_x)), int(round(root_rc[0] * scale_y))),
            trunk_line=trunk_line_xy_full,
            branch_lines=branch_lines_xy_full,
            mask=original_mask.astype(np.uint8),
            dt_map=cv2.resize(dt_map.astype(np.float32), (original_mask.shape[1], original_mask.shape[0]), interpolation=cv2.INTER_LINEAR),
            skeleton_map=skeleton_full.astype(np.uint8),
            trunk_mask=(trunk_mask_full > 0).astype(np.uint8),
            endpoint_mask=endpoint_mask_full.astype(np.uint8),
            raw_endpoints=[(int(round(c * scale_x)), int(round(r * scale_y))) for r, c in raw_endpoints_rc],
            filtered_endpoints=[(int(round(c * scale_x)), int(round(r * scale_y))) for r, c in filtered_endpoints_rc],
            annotation_groups=annotation_groups_full,
        )


# ============================================================================
# Tape 模型 (懒加载, 不依赖 smp; torchvision/nn/F 在首次调用时才 import)
# ============================================================================

_TAPE_CKPT = PROJECT_ROOT / "03_models" / "Tape_Segmentation_V2_outputs" / "best_model.pth"
_tape_model = None


def _get_tape_model(device="cuda:0"):
    """懒加载 tape UNet, 首次调用时才 import torchvision/nn/re 并构建模型."""
    global _tape_model
    if _tape_model is not None:
        return _tape_model

    if device != "cuda:0":
        raise RuntimeError(f"条带分割仅允许使用 cuda:0，当前请求设备为 {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("条带分割需要本地 CUDA GPU，但当前 PyTorch 未检测到可用 CUDA")
    if not _TAPE_CKPT.is_file():
        raise FileNotFoundError(f"条带分割权重不存在: {_TAPE_CKPT}")

    import re
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import mobilenet_v2

    class _DecoderBlock(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x, skip=None):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    class _TapeUNet(nn.Module):
        """smp.Unet('mobilenet_v2', in_channels=3, classes=1) 的 torchvision 重构."""
        def __init__(self):
            super().__init__()
            enc = mobilenet_v2(weights=None, num_classes=1)
            self.encoder = enc.features
            self.decoder = nn.ModuleList([
                _DecoderBlock(1376, 256, 256),
                _DecoderBlock(288, 128, 128),
                _DecoderBlock(152, 64, 64),
                _DecoderBlock(80, 32, 32),
                _DecoderBlock(32, 16, 16),
            ])
            self.seg_head = nn.Conv2d(16, 1, 3, padding=1)

        def forward(self, x):
            skips = []
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                if i in (1, 3, 6, 13):
                    skips.append(x)
            x = self.decoder[0](x, skips[3])
            x = self.decoder[1](x, skips[2])
            x = self.decoder[2](x, skips[1])
            x = self.decoder[3](x, skips[0])
            x = self.decoder[4](x, None)
            return self.seg_head(x)

    model = _TapeUNet()
    sd = torch.load(str(_TAPE_CKPT), map_location=device, weights_only=True)
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        new_k = re.sub(r'^encoder\.features\.', 'encoder.', new_k)
        new_k = re.sub(r'^decoder\.blocks\.', 'decoder.', new_k)
        new_k = new_k.replace('segmentation_head.0', 'seg_head')
        new_sd[new_k] = v
    model.load_state_dict(new_sd)
    model.to(device)
    model.eval()
    _tape_model = model
    return _tape_model


def validate_tape_segmentation_runtime() -> str:
    """验证条带分割权重和 CUDA 推理环境；失败时阻止骨架生成。"""
    _get_tape_model("cuda:0")
    return torch.cuda.get_device_name(0)


def _filter_y_axis(pred_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    if num_labels <= 1:
        return pred_mask
    max_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i
    y_start = stats[max_label, cv2.CC_STAT_TOP]
    height = stats[max_label, cv2.CC_STAT_HEIGHT]
    y_end = y_start + height
    tolerance = 5
    y1 = max(0, y_start - tolerance)
    y2 = min(pred_mask.shape[0], y_end + tolerance)
    filtered = np.zeros_like(pred_mask)
    filtered[y1:y2, :] = pred_mask[y1:y2, :]
    return filtered


def _run_tape_segmentation(tape_model, img_rgb, device="cuda:0"):
    """输入 RGB 原图 (H,W,3), 返回 tape mask (H,W) uint8."""
    H, W = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_array = resized.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = tape_model(img_tensor)
        probs = torch.sigmoid(logits)
        pred_512 = (probs > 0.3).float().cpu().numpy()[0, 0]
        pred_512 = (pred_512 * 255).astype(np.uint8)
    pred_orig = cv2.resize(pred_512, (W, H), interpolation=cv2.INTER_NEAREST)
    return _filter_y_axis(pred_orig)


def _componentwise_morphological_close(mask_bin: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Close each raw component independently without allowing components to merge."""
    source = (mask_bin > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(source, connectivity=8)
    if count <= 2:
        return cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel)

    owner = np.zeros_like(labels, dtype=np.int32)
    claimed = np.zeros_like(source, dtype=np.uint8)
    for component_id in range(1, count):
        component = (labels == component_id).astype(np.uint8)
        closed = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)
        free = (closed > 0) & (claimed == 0)
        owner[free] = component_id
        claimed[free] = 1

    # If independently closed regions touch, remove only newly created boundary
    # pixels. Original segmentation pixels are never deleted or reassigned.
    conflict = np.zeros_like(source, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(owner, shift=(dy, dx), axis=(0, 1))
            conflict |= (owner > 0) & (shifted > 0) & (owner != shifted)
    claimed[conflict & (source == 0)] = 0
    claimed[source > 0] = 1
    return claimed.astype(np.uint8)


def prepare_processed_router_mask(
    combined_mask: np.ndarray,
    image_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mask_bin = (combined_mask > 127).astype(np.uint8) if combined_mask.max() > 1 else combined_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 35))
    mask_processed = _componentwise_morphological_close(mask_bin, kernel)
    if image_rgb is None:
        raise ValueError("标准骨架生成必须提供 image_rgb，以执行强制条带分割")
    from .tape_aware_routing import merge_tape_into_mask
    tape_model = _get_tape_model("cuda:0")
    tape_mask = _run_tape_segmentation(tape_model, image_rgb, device="cuda:0")
    mask_processed = merge_tape_into_mask(mask_processed, tape_mask)
    return mask_processed, tape_mask


def build_prediction_result(
    node_probs: Optional[np.ndarray] = None,
    edge_probs: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    combined_mask: Optional[np.ndarray] = None,
    image_rgb: Optional[np.ndarray] = None,
    edge_threshold: float = 0.3,
    bud_boxes: Optional[Sequence[Sequence[float]]] = None,
    bud_orientations: Optional[Sequence] = None,
    bud_masks_info: Optional[Sequence] = None,
    bud_scores: Optional[Sequence[float]] = None,
    bud_directions: Optional[Sequence] = None,
    processed_mask: Optional[np.ndarray] = None,
    protected_tape_mask: Optional[np.ndarray] = None,
    **kwargs,
) -> PredictionResult:
    if combined_mask is None:
        if reference_mask is not None:
            combined_mask = reference_mask
        elif edge_probs is not None:
            combined_mask = (np.max(edge_probs, axis=0) >= edge_threshold).astype(np.uint8)
        else:
            raise ValueError("build_prediction_result 需要 combined_mask 或 reference_mask 或 edge_probs")

    # Phase B 标准预处理: binarize → MORPH_CLOSE → tape-merge
    if processed_mask is None:
        mask_processed, tape_mask = prepare_processed_router_mask(combined_mask, image_rgb)
    else:
        mask_processed = (np.asarray(processed_mask) > 0).astype(np.uint8)
        tape_mask = protected_tape_mask

    router = DijkstraSkeletonRouter(**kwargs)
    return router(
        mask_processed,
        bud_boxes=bud_boxes,
        bud_orientations=bud_orientations,
        bud_masks_info=bud_masks_info,
        bud_scores=bud_scores,
        bud_directions=bud_directions,
        protected_tape_mask=tape_mask,
        source_mask=(np.asarray(combined_mask) > 0).astype(np.uint8),
    )


# ============================================================================
# 实验: 有向树结构 / 高阶交叉点拆分
# ============================================================================

def _group_to_local_graph(group: Dict) -> Tuple[List[Tuple[int, int]], nx.Graph]:
    points = [tuple(map(int, point)) for point in group.get("points", [])]
    graph = nx.Graph()
    for idx, point in enumerate(points):
        graph.add_node(idx, point=point)
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if 0 <= src < len(points) and 0 <= dst < len(points) and src != dst:
            graph.add_edge(src, dst)
    return points, graph


def _pair_neighbors_by_opposition(center_xy: Tuple[int, int], neighbor_ids: Sequence[int], positions: Dict[int, np.ndarray]) -> List[List[int]]:
    remaining = list(neighbor_ids)
    groups: List[List[int]] = []
    center = np.asarray(center_xy, dtype=np.float32)

    def unit_vec(node_id: int) -> np.ndarray:
        vec = positions[node_id] - center
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-6)

    while len(remaining) >= 2:
        best_pair = None
        best_score = float("inf")
        for idx_a in range(len(remaining)):
            for idx_b in range(idx_a + 1, len(remaining)):
                a = remaining[idx_a]
                b = remaining[idx_b]
                score = float(np.dot(unit_vec(a), unit_vec(b)))
                if score < best_score:
                    best_score = score
                    best_pair = (a, b)
        if best_pair is None:
            break
        a, b = best_pair
        groups.append([a, b])
        remaining.remove(a)
        remaining.remove(b)

    if remaining:
        leftover = remaining[0]
        if not groups:
            groups.append([leftover])
        else:
            left_vec = unit_vec(leftover)
            attach_idx = max(
                range(len(groups)),
                key=lambda idx: max(float(np.dot(left_vec, unit_vec(member))) for member in groups[idx]),
            )
            groups[attach_idx].append(leftover)
    return groups


def _resolve_high_degree_group(group: Dict, degree_threshold: int = 3, offset_radius: float = 2.5) -> Tuple[List[Dict], Dict[str, int]]:
    points, graph = _group_to_local_graph(group)
    if graph.number_of_nodes() == 0:
        return [], {"high_degree_before": 0, "high_degree_after": 0, "split_nodes": 0}

    positions: Dict[int, np.ndarray] = {idx: np.asarray(point, dtype=np.float32) for idx, point in enumerate(points)}
    next_id = len(points)
    split_nodes = 0
    high_before = sum(1 for _, degree in graph.degree() if degree >= degree_threshold + 1)

    while True:
        targets = [node for node, degree in graph.degree() if degree >= degree_threshold + 1]
        if not targets:
            break
        target = max(targets, key=lambda node: graph.degree[node])
        neighbors = list(graph.neighbors(target))
        center_xy = tuple(map(int, positions[target]))
        pair_groups = _pair_neighbors_by_opposition(center_xy, neighbors, positions)
        graph.remove_node(target)
        num_groups = max(len(pair_groups), 1)
        for group_idx, members in enumerate(pair_groups):
            angle = (2.0 * np.pi * group_idx) / float(num_groups)
            offset = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32) * float(offset_radius)
            new_id = next_id
            next_id += 1
            positions[new_id] = positions[target] + offset
            graph.add_node(new_id, point=tuple(map(int, np.round(positions[new_id]))))
            for member in members:
                graph.add_edge(new_id, member)
        split_nodes += 1

    resolved_groups: List[Dict] = []
    high_after = sum(1 for _, degree in graph.degree() if degree >= degree_threshold + 1)
    for comp_idx, component_nodes in enumerate(nx.connected_components(graph), start=1):
        component = graph.subgraph(component_nodes).copy()
        if component.number_of_edges() == 0:
            continue
        node_list = list(component.nodes())
        local_index = {node_id: idx for idx, node_id in enumerate(node_list)}
        local_points = [
            [int(round(float(positions[node_id][0]))), int(round(float(positions[node_id][1])))]
            for node_id in node_list
        ]
        local_edges = [[local_index[src], local_index[dst]] for src, dst in component.edges()]
        resolved_groups.append(
            {
                "group_id": f"{group.get('group_id', 'branch')}_resolved_{comp_idx:02d}",
                "group_type": group.get("group_type", "branch"),
                "color_hex": group.get("color_hex", "#00FF00"),
                "points": local_points,
                "edges": local_edges,
                "fork_origin_group": group.get("fork_origin_group"),
            }
        )
    return resolved_groups, {"high_degree_before": int(high_before), "high_degree_after": int(high_after), "split_nodes": int(split_nodes)}


def _choose_group_anchor(points: Sequence[Tuple[int, int]], parent_points: Sequence[Tuple[int, int]]) -> int:
    if not points:
        return 0
    if not parent_points:
        return int(np.argmax([point[1] for point in points]))
    parent_arr = np.asarray(parent_points, dtype=np.float32)
    point_arr = np.asarray(points, dtype=np.float32)
    dists = np.linalg.norm(point_arr[:, None, :] - parent_arr[None, :, :], axis=2)
    return int(np.argmin(dists.min(axis=1)))


def _build_directed_topology(groups: Sequence[Dict]) -> Dict:
    directed_nodes: List[Dict] = []
    directed_edges: List[Dict] = []
    node_id_counter = 0
    group_stats: List[Dict] = []
    trunk_points: List[Tuple[int, int]] = []
    trunk_group = next((group for group in groups if group.get("group_type") == "trunk"), None)
    if trunk_group is not None:
        trunk_points = [tuple(map(int, point)) for point in trunk_group.get("points", [])]

    total_high_before = 0
    total_high_after = 0
    total_split_nodes = 0

    for group in groups:
        group_type = group.get("group_type", "branch")
        group_id = group.get("group_id", "group")
        points, graph = _group_to_local_graph(group)
        if graph.number_of_nodes() == 0:
            continue

        if group_type == "trunk":
            resolved_groups = [group]
            split_stats = {"high_degree_before": 0, "high_degree_after": 0, "split_nodes": 0}
        else:
            resolved_groups, split_stats = _resolve_high_degree_group(group, degree_threshold=3)
            if not resolved_groups:
                resolved_groups = [group]

        total_high_before += int(split_stats["high_degree_before"])
        total_high_after += int(split_stats["high_degree_after"])
        total_split_nodes += int(split_stats["split_nodes"])
        group_stats.append({"group_id": group_id, **split_stats, "resolved_components": len(resolved_groups)})

        for resolved_idx, resolved_group in enumerate(resolved_groups, start=1):
            r_points, r_graph = _group_to_local_graph(resolved_group)
            if r_graph.number_of_nodes() == 0:
                continue
            if resolved_group.get("group_type") == "trunk":
                anchor_local = int(np.argmax([point[1] for point in r_points]))
            else:
                anchor_local = _choose_group_anchor(r_points, trunk_points)
            bfs_tree = nx.bfs_tree(r_graph, source=anchor_local)
            local_to_global: Dict[int, int] = {}
            for local_idx, point in enumerate(r_points):
                node_id = node_id_counter
                node_id_counter += 1
                local_to_global[local_idx] = node_id
                directed_nodes.append(
                    {
                        "id": int(node_id),
                        "x": int(point[0]),
                        "y": int(point[1]),
                        "group_id": resolved_group.get("group_id", group_id),
                        "group_type": resolved_group.get("group_type", group_type),
                        "is_anchor": bool(local_idx == anchor_local),
                    }
                )
            for src, dst in bfs_tree.edges():
                directed_edges.append(
                    {
                        "src": int(local_to_global[src]),
                        "dst": int(local_to_global[dst]),
                        "group_id": resolved_group.get("group_id", group_id),
                        "group_type": resolved_group.get("group_type", group_type),
                    }
                )

    digraph = nx.DiGraph()
    digraph.add_nodes_from(node["id"] for node in directed_nodes)
    digraph.add_edges_from((edge["src"], edge["dst"]) for edge in directed_edges)
    return {
        "nodes": directed_nodes,
        "edges": directed_edges,
        "stats": {
            "num_nodes": len(directed_nodes),
            "num_edges": len(directed_edges),
            "is_dag": bool(nx.is_directed_acyclic_graph(digraph)),
            "high_degree_before": int(total_high_before),
            "high_degree_after": int(total_high_after),
            "split_nodes": int(total_split_nodes),
        },
        "group_stats": group_stats,
    }


# ============================================================================
# 结果导出 (decode_prediction_to_annotation)
# ============================================================================

def decode_prediction_to_annotation(
    image_path: os.PathLike, image_shape: Tuple[int, int], prediction: PredictionResult,
) -> Dict:
    img_h, img_w = image_shape
    groups: List[Dict] = []

    if prediction.annotation_groups:
        groups = prediction.annotation_groups

    elif prediction.trunk_line:
        trunk_points = [list(map(int, point)) for point in prediction.trunk_line]
        trunk_edges = [[i, i + 1] for i in range(max(len(trunk_points) - 1, 0))]
        groups.append({"group_id": "trunk", "group_type": "trunk", "color_hex": "#FF3232", "points": trunk_points, "edges": trunk_edges, "fork_origin_group": None})
        for branch_idx, branch_line in enumerate(prediction.branch_lines, start=1):
            branch_points = [list(map(int, point)) for point in branch_line]
            branch_edges = [[i, i + 1] for i in range(max(len(branch_points) - 1, 0))]
            groups.append({"group_id": f"branch_{branch_idx:02d}", "group_type": "branch", "color_hex": "#00FF00", "points": branch_points, "edges": branch_edges, "fork_origin_group": "trunk"})
    else:
        trunk_set = set(prediction.trunk_path)
        trunk_points = [list(map(int, prediction.points[idx])) for idx in prediction.trunk_path]
        trunk_edges = [[i, i + 1] for i in range(max(len(trunk_points) - 1, 0))]
        groups.append({"group_id": "trunk", "group_type": "trunk", "color_hex": "#FF3232", "points": trunk_points, "edges": trunk_edges, "fork_origin_group": None})
        non_trunk_graph = prediction.graph.copy()
        non_trunk_graph.remove_nodes_from(trunk_set)
        branch_idx = 1
        for component_nodes in nx.connected_components(non_trunk_graph):
            component = prediction.graph.subgraph(component_nodes).copy()
            ordered_nodes = list(nx.dfs_preorder_nodes(component, source=next(iter(component_nodes))))
            branch_points = [list(map(int, prediction.points[idx])) for idx in ordered_nodes]
            branch_edges = [[i, i + 1] for i in range(max(len(branch_points) - 1, 0))]
            groups.append({"group_id": f"branch_{branch_idx:02d}", "group_type": "branch", "color_hex": "#00FF00", "points": branch_points, "edges": branch_edges, "fork_origin_group": "trunk"})
            branch_idx += 1

    directed_graph = _build_directed_topology(groups)
    return {"image_path": str(image_path), "image_width": img_w, "image_height": img_h, "groups": groups, "directed_graph": directed_graph}


def finalize_clipped_annotation_groups(
    groups: Sequence[Dict],
    source_mask: np.ndarray,
    structural_tolerance_px: float = 24.0,
    enable_postclip_crossing_reassignment: bool = True,
    _crossing_refine_pass: int = 0,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Recompute final root validity and merge clip-created rootless fragments."""
    if not groups:
        return [], {"postclip_unattached_merged": 0, "postclip_unattached_retained": 0}
    final_groups = [{**group, "points": [list(map(int, point)) for point in group.get("points", [])],
                     "edges": [list(map(int, edge)) for edge in group.get("edges", [])]}
                    for group in groups]
    trunk_idx = next((idx for idx, group in enumerate(final_groups) if group.get("group_type") == "trunk"), None)
    if trunk_idx is None:
        return final_groups, {"postclip_unattached_merged": 0, "postclip_unattached_retained": 0}
    trunk_group = final_groups[trunk_idx]
    trunk_mask = render_topology_groups_to_mask([trunk_group], source_mask.shape).astype(np.uint8)
    trunk_distance = distance_transform_edt(trunk_mask == 0)

    def reconnect_internal_components(group: Dict) -> Tuple[Dict, int]:
        """Reconnect clip-created pieces only through the same raw branch mask."""
        points, graph = _group_to_local_graph(group)
        if not points or graph.number_of_edges() == 0 or nx.is_connected(graph):
            return group, 0
        branch_support = ((source_mask > 0) & (trunk_distance > 3.0)).astype(np.uint8)
        _, support_labels = cv2.connectedComponents(branch_support, connectivity=8)
        root_node = min(
            graph.nodes(), key=lambda node: float(trunk_distance[points[node][1], points[node][0]]),
        )
        main_nodes = set(nx.node_connected_component(graph, root_node))
        reconnect_count = 0
        remaining = [set(nodes) for nodes in nx.connected_components(graph) if root_node not in nodes]
        remaining.sort(key=len, reverse=True)
        for component_nodes in remaining:
            candidates = []
            for main_node in main_nodes:
                main_xy = np.asarray(points[main_node], dtype=np.float32)
                for fragment_node in component_nodes:
                    distance = float(np.linalg.norm(
                        main_xy - np.asarray(points[fragment_node], dtype=np.float32),
                    ))
                    candidates.append((distance, int(main_node), int(fragment_node)))
            connected = False
            for _, main_node, fragment_node in sorted(candidates)[:32]:
                main_component = _nearest_component_id(points[main_node], support_labels, radius=3)
                fragment_component = _nearest_component_id(points[fragment_node], support_labels, radius=3)
                if main_component <= 0 or main_component != fragment_component:
                    continue
                support = (support_labels == int(main_component)).astype(np.uint8)
                if _reroute_edge_on_supported_mask(
                    points, graph, main_node, fragment_node, support,
                ):
                    main_nodes.update(nx.node_connected_component(graph, root_node))
                    reconnect_count += 1
                    connected = True
                    break
            if not connected:
                continue
        if reconnect_count == 0:
            return group, 0
        active_nodes = sorted(node for node in graph.nodes() if graph.degree[node] > 0)
        remap = {node: idx for idx, node in enumerate(active_nodes)}
        return {
            **group,
            "points": [list(map(int, points[node])) for node in active_nodes],
            "edges": [
                [remap[int(src)], remap[int(dst)]] for src, dst in graph.edges()
                if src in remap and dst in remap
            ],
            "postclip_internal_reconnections": int(reconnect_count),
        }, int(reconnect_count)

    def root_count(group: Dict) -> int:
        return _group_contact_count(group, trunk_distance, min(float(structural_tolerance_px), 3.0))

    def merge_into(target: Dict, fragment: Dict, connector_path: Sequence[Tuple[int, int]]) -> None:
        point_to_idx = {tuple(map(int, point)): idx for idx, point in enumerate(target.get("points", []))}
        remap = {}
        for old_idx, point in enumerate(fragment.get("points", [])):
            key = tuple(map(int, point))
            if key not in point_to_idx:
                point_to_idx[key] = len(target["points"])
                target["points"].append([key[0], key[1]])
            remap[old_idx] = point_to_idx[key]
        edge_set = {tuple(sorted(map(int, edge))) for edge in target.get("edges", []) if len(edge) == 2}
        for edge in fragment.get("edges", []):
            if len(edge) != 2:
                continue
            mapped = tuple(sorted((remap[int(edge[0])], remap[int(edge[1])])))
            if mapped[0] != mapped[1]:
                edge_set.add(mapped)
        connector_ids = []
        for point in connector_path:
            key = tuple(map(int, point))
            if key not in point_to_idx:
                point_to_idx[key] = len(target["points"])
                target["points"].append([key[0], key[1]])
            connector_ids.append(point_to_idx[key])
        for src, dst in zip(connector_ids[:-1], connector_ids[1:]):
            if src != dst:
                edge_set.add(tuple(sorted((src, dst))))
        target["edges"] = [list(edge) for edge in sorted(edge_set)]
        target.setdefault("merged_group_ids", []).append(fragment.get("group_id"))
        target["bridge_provenance"] = sorted(set(target.get("bridge_provenance", [])) | set(fragment.get("bridge_provenance", [])))

    for idx, group in enumerate(final_groups):
        if group.get("group_type") not in {"branch", "secondary_branch", "unattached_component"}:
            continue
        candidate_attachments = _tolerance_contact_attachments(
            group, trunk_mask, trunk_distance, float(structural_tolerance_px),
        )
        if len(candidate_attachments) != 1:
            continue
        branch_attach_rc, trunk_attach_rc = candidate_attachments[0]
        restored = _prepend_anchor_to_group(
            group,
            anchor_xy=(int(trunk_attach_rc[1]), int(trunk_attach_rc[0])),
            attach_xy=(int(branch_attach_rc[1]), int(branch_attach_rc[0])),
        )
        if restored is not None:
            final_groups[idx] = restored

    postclip_crossing_debug = []
    postclip_crossing_reassignments = 0
    branch_dt = distance_transform_edt(source_mask > 0)

    def family_members(group: Dict) -> set:
        members = group.get("root_family_members", [])
        return {str(value) for value in members if value is not None}

    def terminal_nodes(group: Dict) -> List[int]:
        _, graph = _group_to_local_graph(group)
        return [int(node) for node in graph.nodes() if graph.degree[node] == 1]

    for source_idx, source_group in enumerate(list(final_groups)) if enable_postclip_crossing_reassignment else []:
        if source_group.get("group_type") not in {"branch", "secondary_branch", "unattached_component"}:
            continue
        source_family = family_members(source_group)
        if len(source_family) < 2 or root_count(source_group) != 1:
            continue
        pieces, crossing_debug = _uncross_group_degree3_motifs(
            source_group, branch_dt, bud_directions=None, maximum_path_length=130.0,
        )
        if len(pieces) != 2 or len(crossing_debug) != 1:
            continue
        piece_root_counts = [root_count(piece) for piece in pieces]
        if sorted(piece_root_counts) != [0, 1]:
            continue
        rooted_piece = pieces[piece_root_counts.index(1)]
        rootless_piece = pieces[piece_root_counts.index(0)]
        debug_item = crossing_debug[0]
        corridor = [tuple(map(int, point)) for point in debug_item.get("corridor_points_processing", [])]
        if len(corridor) < 2:
            continue
        center = np.mean(np.asarray(corridor, dtype=np.float32), axis=0)
        rootless_points = [tuple(map(int, point)) for point in rootless_piece.get("points", [])]
        external_terminals = [
            node for node in terminal_nodes(rootless_piece)
            if float(np.linalg.norm(np.asarray(rootless_points[node], dtype=np.float32) - center)) > 40.0
        ]
        if not external_terminals:
            continue

        target_candidates = []
        for target_idx, target_group in enumerate(final_groups):
            if target_idx == source_idx or target_group.get("group_type") not in {"branch", "secondary_branch", "unattached_component"}:
                continue
            target_family = family_members(target_group)
            if len(source_family & target_family) < 2 or root_count(target_group) != 1:
                continue
            target_points = [tuple(map(int, point)) for point in target_group.get("points", [])]
            if not target_points:
                continue
            target_array = np.asarray(target_points, dtype=np.float32)
            for terminal_node in external_terminals:
                terminal_xy = np.asarray(rootless_points[terminal_node], dtype=np.float32)
                distances = np.linalg.norm(target_array - terminal_xy[None, :], axis=1)
                target_node = int(np.argmin(distances))
                target_candidates.append((
                    float(distances[target_node]), int(target_idx), int(terminal_node), int(target_node),
                ))
        if not target_candidates:
            continue
        distance, target_idx, terminal_node, target_node = min(target_candidates)
        if distance > float(structural_tolerance_px):
            continue
        rootless_xy = rootless_points[terminal_node]
        target_xy = tuple(map(int, final_groups[target_idx]["points"][target_node]))
        rr, cc = _line_pixels_xy(rootless_xy, target_xy, source_mask.shape)
        if _max_false_run(source_mask[rr, cc] > 0) > 3:
            continue

        _, rooted_graph = _group_to_local_graph(rooted_piece)
        _, rootless_graph = _group_to_local_graph(rootless_piece)
        _, target_graph = _group_to_local_graph(final_groups[target_idx])
        if not nx.is_forest(rooted_graph) or not nx.is_forest(rootless_graph) or not nx.is_forest(target_graph):
            continue

        source_group_id = str(source_group.get("group_id", f"branch_{source_idx:02d}"))
        target_group_id = str(final_groups[target_idx].get("group_id", f"branch_{target_idx:02d}"))
        rooted_piece = {
            **rooted_piece,
            "group_id": source_group_id,
            "crossing_reassigned_piece": "rooted_route",
        }
        rootless_piece = {
            **rootless_piece,
            "crossing_reassigned_piece": "transferred_route",
            "crossing_source_group_id": source_group_id,
        }
        merge_into(
            final_groups[target_idx], rootless_piece,
            connector_path=[target_xy, rootless_xy],
        )
        final_groups[source_idx] = rooted_piece

        pairing = [tuple(map(int, pair)) for pair in debug_item.get("pairing", [])]
        ports = [tuple(map(int, point)) for point in debug_item.get("arm_ports_processing", [])]
        resolved_pairs = []
        for pair in pairing:
            pair_points = [np.asarray(ports[index], dtype=np.float32) for index in pair]
            rooted_array = np.asarray(rooted_piece.get("points", []), dtype=np.float32)
            rootless_array = np.asarray(rootless_piece.get("points", []), dtype=np.float32)
            rooted_distance = sum(float(np.min(np.linalg.norm(rooted_array - point[None, :], axis=1))) for point in pair_points)
            rootless_distance = sum(float(np.min(np.linalg.norm(rootless_array - point[None, :], axis=1))) for point in pair_points)
            resolved_pairs.append({
                "group_id": source_group_id if rooted_distance <= rootless_distance else target_group_id,
                "port_indices": [int(pair[0]), int(pair[1])],
            })
        debug_item = {
            **debug_item,
            "coordinate_space": "full_image",
            "source_group_id": source_group_id,
            "target_group_id": target_group_id,
            "resolved_route_pairs": resolved_pairs,
            "transfer_connector": [list(map(int, target_xy)), list(map(int, rootless_xy))],
            "transfer_distance_px": float(distance),
            "independent_root_count": 2,
        }
        postclip_crossing_debug.append(debug_item)
        postclip_crossing_reassignments += 1

    structural_indices = [
        idx for idx, group in enumerate(final_groups)
        if group.get("group_type") in {"branch", "secondary_branch", "unattached_component"}
    ]
    rooted = [idx for idx in structural_indices if root_count(final_groups[idx]) == 1]
    rootless = [idx for idx in structural_indices if root_count(final_groups[idx]) == 0]
    merged = set()
    merge_count = 0
    for fragment_idx in rootless:
        fragment = final_groups[fragment_idx]
        fragment_points = np.asarray(fragment.get("points", []), dtype=np.float32)
        if fragment_points.size == 0:
            continue
        candidates = []
        for target_idx in rooted:
            if target_idx == fragment_idx:
                continue
            target = final_groups[target_idx]
            if int(fragment.get("source_component_id", 0)) != int(target.get("source_component_id", 0)):
                continue
            target_points = np.asarray(target.get("points", []), dtype=np.float32)
            if target_points.size == 0:
                continue
            distances = np.linalg.norm(fragment_points[:, None, :] - target_points[None, :, :], axis=2)
            frag_pos, target_pos = np.unravel_index(int(np.argmin(distances)), distances.shape)
            candidates.append((float(distances[frag_pos, target_pos]), target_idx, frag_pos, target_pos))
        if not candidates:
            continue
        distance, target_idx, frag_pos, target_pos = min(candidates)
        if distance > float(structural_tolerance_px) * 10.0:
            continue
        a = tuple(map(int, fragment_points[frag_pos]))
        b = tuple(map(int, np.asarray(final_groups[target_idx]["points"])[target_pos]))
        rr, cc = _line_pixels_xy(a, b, source_mask.shape)
        connector_path = [b, a]
        if _max_false_run(source_mask[rr, cc] > 0) > 3:
            start_rc = _snap_to_nearest_true((int(b[1]), int(b[0])), source_mask > 0)
            end_rc = _snap_to_nearest_true((int(a[1]), int(a[0])), source_mask > 0)
            margin = int(max(48.0, distance * 1.5))
            y0 = max(0, min(start_rc[0], end_rc[0]) - margin)
            y1 = min(source_mask.shape[0], max(start_rc[0], end_rc[0]) + margin + 1)
            x0 = max(0, min(start_rc[1], end_rc[1]) - margin)
            x1 = min(source_mask.shape[1], max(start_rc[1], end_rc[1]) + margin + 1)
            local_source = source_mask[y0:y1, x0:x1] > 0
            cost = np.where(local_source, 1.0, 1e4).astype(np.float32)
            try:
                path_rc, _ = route_through_array(
                    cost,
                    (int(start_rc[0] - y0), int(start_rc[1] - x0)),
                    (int(end_rc[0] - y0), int(end_rc[1] - x0)),
                    fully_connected=True,
                )
            except (ValueError, nx.NetworkXNoPath):
                continue
            if any(not local_source[int(r), int(c)] for r, c in path_rc):
                continue
            dense_xy = [(int(c + x0), int(r + y0)) for r, c in path_rc]
            connector_path = [b] + simplify_polyline(dense_xy, epsilon=2.0) + [a]
        merge_into(final_groups[target_idx], fragment, connector_path=connector_path)
        merged.add(fragment_idx)
        merge_count += 1

    output = []
    unattached_retained = 0
    short_spurs_removed = 0
    internal_reconnections = 0
    for idx, group in enumerate(final_groups):
        if idx in merged:
            continue
        if group.get("group_type") in {"branch", "secondary_branch", "unattached_component"}:
            group, reconnect_count = reconnect_internal_components(group)
            internal_reconnections += int(reconnect_count)
            group, removed_spurs = _prune_short_terminal_spurs(
                # This is terminal-spur cleanup, not subtree partitioning: only
                # a leaf-to-junction arm can be removed.  32 px remains below
                # the GT 5% quantile (34.1 px) and cannot erase a long channel.
                group, trunk_distance, max_length=32.0,
            )
            short_spurs_removed += int(removed_spurs)
            count = root_count(group)
            group["root_contact_count"] = int(count)
            group["root_contact_count_24px"] = int(
                _group_contact_count(group, trunk_distance, float(structural_tolerance_px))
            )
            diagnostics = _group_geometry_diagnostics(
                group, trunk_distance, trunk_group.get("points", []), float(structural_tolerance_px), None,
            )
            group.update({key: value for key, value in diagnostics.items() if key != "bud_flow_vote"})
            if count == 1:
                group["group_type"] = "branch"
                group["fork_origin_group"] = "trunk"
            elif group.get("group_type") == "branch":
                group["group_type"] = "secondary_branch"
                group["unattached_reason"] = "no_unique_24px_trunk_root_after_mask_clip"
                unattached_retained += 1
            group["edge_diagnostics"] = [
                {
                    "edge": [int(edge[0]), int(edge[1])],
                    "disposition": group.get("edge_disposition", "retained"),
                    "postclip_status": "retained_after_mask_clip",
                }
                for edge in group.get("edges", [])
                if len(edge) == 2
            ]
        output.append(group)
    if (enable_postclip_crossing_reassignment
            and _crossing_refine_pass == 0 and postclip_crossing_reassignments == 0):
        refined_output, refined_stats = finalize_clipped_annotation_groups(
            output,
            source_mask=source_mask,
            structural_tolerance_px=structural_tolerance_px,
            enable_postclip_crossing_reassignment=enable_postclip_crossing_reassignment,
            _crossing_refine_pass=1,
        )
        additive_keys = {
            "postclip_unattached_merged",
            "postclip_unattached_retained",
            "postclip_short_terminal_spurs_removed",
            "postclip_internal_reconnections",
        }
        for key in additive_keys:
            refined_stats[key] = int(refined_stats.get(key, 0)) + int({
                "postclip_unattached_merged": merge_count,
                "postclip_unattached_retained": unattached_retained,
                "postclip_short_terminal_spurs_removed": short_spurs_removed,
                "postclip_internal_reconnections": internal_reconnections,
            }[key])
        refined_stats["postclip_crossing_refinement_passes"] = 2
        return refined_output, refined_stats
    branch_number = 0
    for group in output:
        if group.get("group_type") != "branch":
            continue
        branch_number += 1
        display_id = f"B{branch_number:02d}"
        group["display_id"] = display_id
        points, graph = _group_to_local_graph(group)
        fork_points = []
        if points and graph.number_of_edges() > 0:
            root = min(
                graph.nodes(),
                key=lambda node: float(trunk_distance[points[node][1], points[node][0]]),
            )
            weighted = graph.copy()
            for src, dst in weighted.edges():
                weighted.edges[src, dst]["length"] = float(
                    np.linalg.norm(
                        np.asarray(points[src], dtype=np.float32)
                        - np.asarray(points[dst], dtype=np.float32)
                    )
                )
            distances = nx.single_source_dijkstra_path_length(weighted, root, weight="length")
            fork_nodes = sorted(
                (node for node in graph.nodes() if graph.degree[node] >= 3 and node in distances),
                key=lambda node: (float(distances[node]), int(node)),
            )
            for fork_number, node in enumerate(fork_nodes, start=1):
                fork_points.append({
                    "fork_id": f"{display_id}-F{fork_number:02d}",
                    "point": [int(points[node][0]), int(points[node][1])],
                    "degree": int(graph.degree[node]),
                    "distance_from_root_px": float(distances[node]),
                })
        group["fork_points"] = fork_points
    return output, {
        "postclip_unattached_merged": int(merge_count),
        "postclip_unattached_retained": int(unattached_retained),
        "postclip_short_terminal_spurs_removed": int(short_spurs_removed),
        "postclip_internal_reconnections": int(internal_reconnections),
        "postclip_crossing_reassignments": int(postclip_crossing_reassignments),
        "postclip_crossing_debug": postclip_crossing_debug,
        "postclip_crossing_refinement_passes": int(_crossing_refine_pass + 1),
    }


# ============================================================================
# 可视化
# ============================================================================

def _distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    hues = [18 + int(144 * i / max(n, 1)) for i in range(n)]
    colors = []
    for h in hues:
        rgb = cv2.cvtColor(np.uint8([[[h, 210, 180]]]), cv2.COLOR_HSV2RGB)[0, 0]
        colors.append((int(rgb[2]), int(rgb[1]), int(rgb[0])))
    return colors


def _draw_groups_overlay(canvas: np.ndarray, groups: Sequence[Dict], line_thickness: int = 6) -> np.ndarray:
    branch_colors = _distinct_colors(max(sum(1 for group in groups if group.get("group_type") != "trunk"), 1))
    branch_color_idx = 0
    for group in groups:
        group_type = group.get("group_type", "branch")
        points = [tuple(map(int, point)) for point in group.get("points", [])]
        edges = group.get("edges", [])
        if group_type == "trunk":
            color = (255, 64, 64)
            thickness = line_thickness + 1
        else:
            color = branch_colors[branch_color_idx % len(branch_colors)]
            branch_color_idx += 1
            thickness = line_thickness
        for edge in edges:
            if len(edge) != 2:
                continue
            src, dst = int(edge[0]), int(edge[1])
            if 0 <= src < len(points) and 0 <= dst < len(points):
                cv2.line(canvas, points[src], points[dst], color=color, thickness=thickness, lineType=cv2.LINE_AA)
        if group_type != "trunk":
            local_degrees = [0] * len(points)
            for edge in edges:
                if len(edge) != 2:
                    continue
                src, dst = int(edge[0]), int(edge[1])
                if 0 <= src < len(points) and 0 <= dst < len(points):
                    local_degrees[src] += 1
                    local_degrees[dst] += 1
            for idx, point in enumerate(points):
                if local_degrees[idx] <= 1:
                    cv2.circle(canvas, point, 6, (64, 255, 64), thickness=-1)
                elif local_degrees[idx] >= 3:
                    cv2.circle(canvas, point, 5, (255, 255, 255), thickness=2)
    return canvas


def draw_prediction_overlay(image_rgb: np.ndarray, prediction: PredictionResult, line_thickness: int = 6) -> np.ndarray:
    canvas = image_rgb.copy()
    if prediction.annotation_groups:
        canvas = _draw_groups_overlay(canvas, prediction.annotation_groups, line_thickness=line_thickness)
        if prediction.root_point != (-1, -1):
            cv2.circle(canvas, prediction.root_point, 7, (255, 255, 0), thickness=-1)
            cv2.circle(canvas, prediction.root_point, 7, (0, 0, 0), thickness=2)
        return canvas
    if prediction.trunk_line:
        for p0, p1 in zip(prediction.trunk_line[:-1], prediction.trunk_line[1:]):
            cv2.line(canvas, p0, p1, color=(255, 64, 64), thickness=line_thickness + 1, lineType=cv2.LINE_AA)
        for color, branch_line in zip(_distinct_colors(len(prediction.branch_lines)), prediction.branch_lines):
            for p0, p1 in zip(branch_line[:-1], branch_line[1:]):
                cv2.line(canvas, p0, p1, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
            if branch_line:
                cv2.circle(canvas, branch_line[0], 6, (255, 255, 255), thickness=2)
                cv2.circle(canvas, branch_line[-1], 6, (64, 255, 64), thickness=-1)
        if prediction.root_point != (-1, -1):
            cv2.circle(canvas, prediction.root_point, 7, (255, 255, 0), thickness=-1)
            cv2.circle(canvas, prediction.root_point, 7, (0, 0, 0), thickness=2)
        return canvas

    trunk_set = set(prediction.trunk_path)
    non_trunk_graph = prediction.graph.copy()
    non_trunk_graph.remove_nodes_from(trunk_set)
    comp_colors: Dict[int, tuple] = {}
    for color, comp_nodes in zip(_distinct_colors(max(non_trunk_graph.number_of_nodes(), 1)), nx.connected_components(non_trunk_graph)):
        for node_id in comp_nodes:
            comp_colors[node_id] = color
    for u, v in prediction.graph.edges():
        p1 = prediction.points[u]
        p2 = prediction.points[v]
        color = (255, 64, 64) if (u in trunk_set and v in trunk_set) else comp_colors.get(u, comp_colors.get(v, (64, 255, 64)))
        cv2.line(canvas, tuple(map(int, p1)), tuple(map(int, p2)), color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
    return canvas


# ============================================================================
# 兼容旧接口的函数
# ============================================================================

def gaussian_heatmap(height: int, width: int, center_x: float, center_y: float, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def pretty_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
