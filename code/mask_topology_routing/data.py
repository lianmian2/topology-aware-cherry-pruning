"""
Mask 拓扑路由数据辅助模块.

当前纯几何路线只复用其中的通用清单读取与路径解析能力；
训练相关的数据生成逻辑保留为历史兼容代码，不作为当前主线入口。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    SegmentationAdapter,
    ensure_dir,
    gaussian_heatmap,
    load_image_rgb,
    load_json,
    parse_annotation_stem,
    resolve_image_path,
    save_image_rgb,
    save_json,
    save_mask,
)

try:
    import albumentations as A
except Exception:
    A = None


DEFAULT_ANNOTATION_DIR = Path(__file__).resolve().parents[3] / "01_data" / "02_annotated" / "skeleton_annotation"
EXCLUDED_SAMPLE_NAMES = {
    "tree_137_before_view_05",
    "tree_163_before_view_05",
}


def load_manifest(manifest_path: Path) -> List[Dict]:
    """读取预处理后保存的样本索引。"""
    manifest = load_json(manifest_path)
    samples = manifest["samples"]
    return [sample for sample in samples if sample.get("sample_name") not in EXCLUDED_SAMPLE_NAMES]


def build_augmentor() -> Optional["A.Compose"]:
    """构建适合树木拓扑任务的安全增强 (仅翻转/旋转/亮度)。"""
    if A is None:
        return None
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def split_by_groups(annotation: Dict) -> Tuple[List[Tuple[float, float]], List[int]]:
    """把各 group 的点拍平，记录每个点属于哪个 group。"""
    points: List[Tuple[float, float]] = []
    group_ids: List[int] = []
    for group_idx, group in enumerate(annotation.get("groups", [])):
        for point in group.get("points", []):
            points.append((float(point[0]), float(point[1])))
            group_ids.append(group_idx)
    return points, group_ids


def rebuild_annotation_with_points(annotation: Dict, transformed_points: Sequence[Tuple[float, float]]) -> Dict:
    """把增强后的点坐标重新写回 annotation 结构。"""
    new_annotation = copy.deepcopy(annotation)
    cursor = 0
    for group in new_annotation.get("groups", []):
        num_points = len(group.get("points", []))
        group["points"] = [
            [float(transformed_points[cursor + i][0]), float(transformed_points[cursor + i][1])]
            for i in range(num_points)
        ]
        cursor += num_points
    return new_annotation


def clip_point(point: Sequence[float], width: int, height: int) -> Tuple[int, int]:
    """把点坐标裁剪到图像内部。"""
    x = int(np.clip(round(float(point[0])), 0, width - 1))
    y = int(np.clip(round(float(point[1])), 0, height - 1))
    return x, y


def build_dilated_skeleton_prior(
    annotation: Dict,
    image_shape: Tuple[int, int],
    dilate_radius: int = 40,
    line_thickness: int = 1,
) -> np.ndarray:
    """根据骨架标注生成膨胀后的连续拓扑先验图。"""
    from data import render_annotation_skeleton_mask  # local ref
    base_mask = render_annotation_skeleton_mask(
        annotation=annotation, image_shape=image_shape,
        line_thickness=line_thickness, draw_nodes=True, node_radius=max(1, line_thickness),
    )
    kernel_size = dilate_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(base_mask, kernel, iterations=1).astype(np.uint8)


def render_annotation_skeleton_mask(
    annotation: Dict,
    image_shape: Tuple[int, int],
    line_thickness: int = 1,
    draw_nodes: bool = True,
    node_radius: int = 1,
) -> np.ndarray:
    """把骨架 JSON 标注渲染成二值骨架掩码。"""
    image_h, image_w = image_shape
    canvas = np.zeros((image_h, image_w), dtype=np.uint8)
    for group in annotation.get("groups", []):
        points = group.get("points", [])
        for edge in group.get("edges", []):
            if len(edge) != 2:
                continue
            p1 = clip_point(points[int(edge[0])], image_w, image_h)
            p2 = clip_point(points[int(edge[1])], image_w, image_h)
            cv2.line(canvas, p1, p2, color=255, thickness=line_thickness, lineType=cv2.LINE_AA)
        if draw_nodes:
            for point in points:
                x, y = clip_point(point, image_w, image_h)
                cv2.circle(canvas, (x, y), radius=node_radius, color=255, thickness=-1)
    return canvas


def generate_targets(
    annotation: Dict,
    image_shape: Tuple[int, int],
    output_size: int,
    sigma: float = 2.0,
) -> Dict:
    """
    根据标注 JSON 的 group_type 生成历史兼容 GT 标签 (仅 endpoint 单通道).

    输出:
    - node_heatmaps   : [1, output_size, output_size] — branch_endpoint
    - trunk_edge      : [output_size, output_size] float — trunk group 的边
    - branch_edge     : [output_size, output_size] float — branch group 的边
    - junction        : [output_size, output_size] float — 全零 (兼容)
    - endpoint        : [output_size, output_size] float — branch_endpoint 热力图副本
    - trunk           : [output_size, output_size] float — trunk_edge 副本
    """
    img_h, img_w = image_shape
    trunk_edge_bin = np.zeros((output_size, output_size), dtype=np.uint8)
    branch_edge_bin = np.zeros((output_size, output_size), dtype=np.uint8)
    branch_endpoint_hm = np.zeros((output_size, output_size), dtype=np.float32)

    def scale_point(x: float, y: float) -> Tuple[int, int]:
        tx = int(round(float(x) / max(img_w - 1, 1) * max(output_size - 1, 1)))
        ty = int(round(float(y) / max(img_h - 1, 1) * max(output_size - 1, 1)))
        return int(np.clip(tx, 0, output_size - 1)), int(np.clip(ty, 0, output_size - 1))

    def place_heatmap(hm: np.ndarray, x: float, y: float) -> None:
        target_x = x / max(img_w - 1, 1) * max(output_size - 1, 1)
        target_y = y / max(img_h - 1, 1) * max(output_size - 1, 1)
        gaussian = gaussian_heatmap(output_size, output_size, target_x, target_y, sigma=sigma)
        np.maximum(hm, gaussian, out=hm)

    groups = annotation.get("groups", [])
    if len(groups) == 0:
        return {
            "node_heatmaps": np.zeros((1, output_size, output_size), dtype=np.float32),
            "trunk_edge": trunk_edge_bin.astype(np.float32),
            "branch_edge": branch_edge_bin.astype(np.float32),
            "junction": np.zeros((output_size, output_size), dtype=np.float32),
            "endpoint": np.zeros((output_size, output_size), dtype=np.float32),
            "trunk": trunk_edge_bin.astype(np.float32),
        }

    for group in groups:
        group_type = group.get("group_type", "branch")
        points = group.get("points", [])
        edges = group.get("edges", [])
        if len(points) == 0:
            continue

        if group_type == "trunk":
            for edge in edges:
                if len(edge) != 2:
                    continue
                src_idx, dst_idx = int(edge[0]), int(edge[1])
                if src_idx < 0 or src_idx >= len(points) or dst_idx < 0 or dst_idx >= len(points):
                    continue
                p1 = scale_point(points[src_idx][0], points[src_idx][1])
                p2 = scale_point(points[dst_idx][0], points[dst_idx][1])
                cv2.line(trunk_edge_bin, p1, p2, color=1, thickness=3, lineType=cv2.LINE_AA)
            # 纯 trunk group 不标注任何节点
        else:
            for edge in edges:
                if len(edge) != 2:
                    continue
                src_idx, dst_idx = int(edge[0]), int(edge[1])
                if src_idx < 0 or src_idx >= len(points) or dst_idx < 0 or dst_idx >= len(points):
                    continue
                p1 = scale_point(points[src_idx][0], points[src_idx][1])
                p2 = scale_point(points[dst_idx][0], points[dst_idx][1])
                cv2.line(branch_edge_bin, p1, p2, color=1, thickness=3, lineType=cv2.LINE_AA)
            # 仅标注 branch 的末点作为 endpoint
            last_point = points[-1]
            place_heatmap(branch_endpoint_hm, last_point[0], last_point[1])

    dilate_kernel = np.ones((3, 3), np.uint8)
    trunk_edge_bin = cv2.dilate(trunk_edge_bin, dilate_kernel, iterations=1)
    branch_edge_bin = cv2.dilate(branch_edge_bin, dilate_kernel, iterations=1)

    return {
        "node_heatmaps": branch_endpoint_hm[None, ...].astype(np.float32),  # [1, H, W]
        "trunk_edge": trunk_edge_bin.astype(np.float32),
        "branch_edge": branch_edge_bin.astype(np.float32),
        "junction": np.zeros((output_size, output_size), dtype=np.float32),
        "endpoint": branch_endpoint_hm.astype(np.float32),
        "trunk": trunk_edge_bin.astype(np.float32),
    }


def resize_image_and_mask(image_rgb: np.ndarray, mask: np.ndarray, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """把原图和掩码缩放到统一尺寸。"""
    resized_image = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


class SkeletonV11Dataset(Dataset):
    """历史兼容数据集类，当前纯几何主线不直接使用。"""

    def __init__(
        self,
        samples: Sequence[Dict],
        image_size: int = 256,
        output_size: int = 256,
        sigma: float = 2.0,
        training: bool = False,
        use_augmentation: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.output_size = output_size
        self.sigma = sigma
        self.training = training
        self.augmentor = build_augmentor() if use_augmentation and training else None
        self.prior_dilate_radius = 40

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        input_image_path = sample.get("roi_extended_by_skeleton_path", sample.get("roi_filtered_path", sample["image_path"]))
        image_rgb = load_image_rgb(input_image_path)
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码: {sample['mask_path']}")

        annotation = load_json(sample["annotation_path"])

        if self.augmentor is not None:
            points, _ = split_by_groups(annotation)
            transformed = self.augmentor(image=image_rgb, mask=mask, keypoints=points)
            image_rgb = transformed["image"]
            mask = transformed["mask"]
            annotation = rebuild_annotation_with_points(annotation, transformed["keypoints"])

        original_h, original_w = image_rgb.shape[:2]

        targets = generate_targets(annotation, image_shape=(original_h, original_w), output_size=self.output_size, sigma=self.sigma)

        prior_mask = build_dilated_skeleton_prior(
            annotation=annotation, image_shape=(original_h, original_w),
            dilate_radius=self.prior_dilate_radius, line_thickness=1,
        )

        image_rgb, mask = resize_image_and_mask(image_rgb, mask, image_size=self.image_size)

        input_tensor = np.concatenate(
            [image_rgb.astype(np.float32) / 255.0, (mask.astype(np.float32) / 255.0)[..., None]], axis=-1,
        )
        input_tensor = np.transpose(input_tensor, (2, 0, 1))

        loss_mask = cv2.resize(prior_mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        loss_mask = (loss_mask > 0).astype(np.float32)

        return {
            "input": torch.from_numpy(input_tensor).float(),
            "node_heatmaps": torch.from_numpy(targets["node_heatmaps"]).float(),
            "trunk_edge": torch.from_numpy(targets["trunk_edge"][None, ...]).float(),
            "branch_edge": torch.from_numpy(targets["branch_edge"][None, ...]).float(),
            "junction": torch.from_numpy(targets["junction"][None, ...]).float(),
            "endpoint": torch.from_numpy(targets["endpoint"][None, ...]).float(),
            "trunk": torch.from_numpy(targets["trunk"][None, ...]).float(),
            "loss_mask": torch.from_numpy(loss_mask[None, ...]).float(),
            "meta": {
                "sample_name": sample["sample_name"],
                "image_path": sample["image_path"],
                "input_image_path": input_image_path,
                "input_mode": "roi_extended_rgb_plus_mask",
                "mask_path": sample["mask_path"],
                "annotation_path": sample["annotation_path"],
                "original_size": (original_h, original_w),
            },
        }


def prepare_processed_dataset(
    annotation_dir: Path = DEFAULT_ANNOTATION_DIR,
    raw_root: Path = DEFAULT_RAW_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    project_name: str = "skeleton_prediction",
    test_ratio: float = 0.2,
    overwrite: bool = False,
    use_real_segmentation: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, Path]:
    """整理骨架预测所需数据到 `03_processed`。"""
    annotation_dir = Path(annotation_dir)
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)

    train_root = ensure_dir(processed_root / "train" / project_name)
    test_root = ensure_dir(processed_root / "test" / project_name)
    ann_root = ensure_dir(processed_root / "annotations" / project_name)

    manifest_train_path = ann_root / "manifest_train.json"
    manifest_test_path = ann_root / "manifest_test.json"

    if manifest_train_path.exists() and manifest_test_path.exists() and not overwrite:
        return {"train_manifest": manifest_train_path, "test_manifest": manifest_test_path, "annotation_root": ann_root}

    for split_root in [train_root, test_root]:
        ensure_dir(split_root / "images")
        ensure_dir(split_root / "branch_masks")
        ensure_dir(split_root / "roi_filtered")
        ensure_dir(split_root / "overlays")
    ensure_dir(ann_root / "raw_json")

    json_files = sorted(annotation_dir.glob("*_skeleton.json"))
    if max_samples is not None:
        json_files = json_files[:max_samples]

    segmenter = SegmentationAdapter(use_real_models=use_real_segmentation, allow_mock_fallback=not use_real_segmentation)
    samples_train: List[Dict] = []
    samples_test: List[Dict] = []

    for idx, annotation_path in enumerate(json_files):
        image_path = resolve_image_path(annotation_path, raw_root=raw_root)
        image_rgb = load_image_rgb(image_path)
        branch_mask, roi_filtered = segmenter.predict_mask(image_rgb)

        tree_id, status, view_id = parse_annotation_stem(annotation_path.name)
        sample_name = f"{tree_id}_{status}_{view_id}"
        if sample_name in EXCLUDED_SAMPLE_NAMES:
            continue
        is_test = (idx % max(int(round(1.0 / max(test_ratio, 1e-6))), 2) == 0)
        split_root = test_root if is_test else train_root

        image_out = split_root / "images" / f"{sample_name}.jpg"
        mask_out = split_root / "branch_masks" / f"{sample_name}.png"
        roi_out = split_root / "roi_filtered" / f"{sample_name}.jpg"
        overlay_out = split_root / "overlays" / f"{sample_name}.png"
        ann_out = ann_root / "raw_json" / annotation_path.name

        save_image_rgb(image_rgb, image_out)
        save_mask(branch_mask, mask_out)
        save_image_rgb(roi_filtered, roi_out)

        overlay = image_rgb.copy()
        overlay[branch_mask > 0] = (0.6 * overlay[branch_mask > 0] + 0.4 * np.array([64, 255, 64], dtype=np.float32)).astype(np.uint8)
        save_image_rgb(overlay, overlay_out)

        annotation = load_json(annotation_path)
        annotation["resolved_image_path"] = str(image_path)
        save_json(annotation, ann_out)

        sample_entry = {
            "sample_name": sample_name,
            "image_path": str(image_out),
            "mask_path": str(mask_out),
            "roi_filtered_path": str(roi_out),
            "overlay_path": str(overlay_out),
            "annotation_path": str(ann_out),
            "source_annotation_path": str(annotation_path),
            "source_image_path": str(image_path),
            "mask_mode": segmenter.describe_mode(),
            "segmentation_fallback_reason": segmenter._fallback_reason,
        }
        if is_test:
            samples_test.append(sample_entry)
        else:
            samples_train.append(sample_entry)

    save_json({"project_name": project_name, "num_samples": len(samples_train), "samples": samples_train}, manifest_train_path)
    save_json({"project_name": project_name, "num_samples": len(samples_test), "samples": samples_test}, manifest_test_path)
    print(f"数据准备完成: 训练集 {len(samples_train)} 张, 测试集 {len(samples_test)} 张")
    return {"train_manifest": manifest_train_path, "test_manifest": manifest_test_path, "annotation_root": ann_root}
