from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tree_topology_builder_2d import TreeTopologyBuilder2D


PointXY = Tuple[int, int]


def _skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    try:
        thinning = cv2.ximgproc.thinning  # type: ignore[attr-defined]
        return thinning(mask_u8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from skimage.morphology import skeletonize

        return skeletonize(mask_u8 > 0).astype(np.uint8) * 255
    except Exception:
        img = mask_u8.copy()
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            done = cv2.countNonZero(img) == 0
        return skel


def _extract_endpoints_and_junctions(skeleton_mask: np.ndarray) -> Tuple[List[PointXY], List[PointXY], np.ndarray]:
    skeleton = (skeleton_mask > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    degree = cv2.filter2D(skeleton, -1, kernel)
    endpoints_y, endpoints_x = np.where((skeleton > 0) & (degree == 1))
    junctions_y, junctions_x = np.where((skeleton > 0) & (degree > 2))
    endpoints = [(int(x), int(y)) for x, y in zip(endpoints_x, endpoints_y)]
    junctions = [(int(x), int(y)) for x, y in zip(junctions_x, junctions_y)]
    return endpoints, junctions, degree


def _neighbors8(point: PointXY) -> List[PointXY]:
    x, y = point
    return [
        (x - 1, y),
        (x + 1, y),
        (x, y - 1),
        (x, y + 1),
        (x - 1, y - 1),
        (x - 1, y + 1),
        (x + 1, y - 1),
        (x + 1, y + 1),
    ]


def _is_skeleton_pixel(point: PointXY, skeleton_bool: np.ndarray) -> bool:
    x, y = point
    h, w = skeleton_bool.shape[:2]
    return 0 <= x < w and 0 <= y < h and bool(skeleton_bool[y, x])


def _prune_short_spurs(skeleton_mask: np.ndarray, max_length: int) -> np.ndarray:
    if max_length <= 0:
        return (skeleton_mask > 0).astype(np.uint8) * 255

    skeleton = (skeleton_mask > 0).astype(np.uint8) * 255
    changed = True
    while changed:
        changed = False
        endpoints, _, _ = _extract_endpoints_and_junctions(skeleton)
        if not endpoints:
            break

        remove_points = set()
        skeleton_bool = skeleton > 0
        for endpoint in endpoints:
            path = [endpoint]
            previous = None
            current = endpoint
            hit_major = False
            while len(path) <= max_length:
                neighbors = [
                    nb for nb in _neighbors8(current)
                    if _is_skeleton_pixel(nb, skeleton_bool) and nb != previous
                ]
                if len(neighbors) == 0:
                    break
                if len(neighbors) > 1:
                    hit_major = True
                    break
                nxt = neighbors[0]
                path.append(nxt)
                previous, current = current, nxt
            if hit_major and len(path) <= max_length:
                remove_points.update(path[:-1])

        if remove_points:
            for x, y in remove_points:
                skeleton[y, x] = 0
            changed = True
    return skeleton


def _extract_bud_centroids(
    bud_boxes: Optional[Sequence[Sequence[float]]],
    bud_masks_info: Optional[Sequence[Any]] = None,
    img_shape: Optional[Tuple[int, ...]] = None,
) -> List[PointXY]:
    if bud_boxes is None and bud_masks_info is None:
        return []
    centroids: List[PointXY] = []
    total = 0
    if bud_masks_info is not None:
        total = max(total, len(bud_masks_info))
    if bud_boxes is not None:
        total = max(total, len(bud_boxes))

    for idx in range(total):
        mask_data = bud_masks_info[idx] if bud_masks_info is not None and idx < len(bud_masks_info) else None
        if mask_data is not None and len(mask_data) >= 3:
            local_mask, offset_x, offset_y = mask_data[:3]
            ys, xs = np.where(np.asarray(local_mask) > 0)
            if len(xs) > 0:
                cx = float(np.mean(xs)) + float(offset_x)
                cy = float(np.mean(ys)) + float(offset_y)
                centroids.append((int(round(cx)), int(round(cy))))
                continue

        if bud_boxes is not None and idx < len(bud_boxes):
            box = bud_boxes[idx]
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
            centroids.append((int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))))

    if img_shape is not None and len(img_shape) >= 2:
        h, w = int(img_shape[0]), int(img_shape[1])
        centroids = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in centroids]
    return centroids


def run_skeleton_topology(
    trunk_mask: np.ndarray,
    branch_mask: np.ndarray,
    bud_boxes: Optional[Sequence[Sequence[float]]] = None,
    bud_masks_info: Optional[Sequence[Any]] = None,
    img_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """
    当前 2D 骨架检查入口。

    说明：
    - 当前主关注点是 branch_mask 的直接细化结果
    - 保留 TreeTopologyBuilder2D 作为可视化与实验输出封装
    - 芽点中心默认由全图芽点检测结果恢复
    """

    bud_centroids = _extract_bud_centroids(bud_boxes, bud_masks_info, img_shape)
    builder = TreeTopologyBuilder2D(
        trunk_mask=trunk_mask,
        branch_mask=branch_mask,
        bud_centroids=bud_centroids,
    )
    graph, debug_data = builder.build_with_debug()
    return {
        "builder": builder,
        "graph": graph,
        "debug_data": debug_data,
        "bud_centroids": bud_centroids,
    }


def run_direct_skeletonization(
    branch_mask: np.ndarray,
    prune_length: int = 6,
) -> Dict[str, Any]:
    """
    旧版骨架流程入口：
    原图 -> ROI -> 枝条分割 -> 直接细化 -> 短毛刺裁剪。
    """

    raw_skeleton = _skeletonize_mask(branch_mask)
    pruned_skeleton = _prune_short_spurs(raw_skeleton, prune_length)
    endpoints, junctions, degree_map = _extract_endpoints_and_junctions(pruned_skeleton)
    return {
        "raw_skeleton": raw_skeleton,
        "skeleton_mask": pruned_skeleton,
        "endpoints": endpoints,
        "junctions": junctions,
        "degree_map": degree_map,
        "prune_length": prune_length,
    }


def draw_direct_skeleton_visualization(
    overlay: np.ndarray,
    skeleton_result: Dict[str, Any],
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    将旧版直接细化骨架结果叠加到现有 overlay 上。
    """

    if overlay is None:
        raise ValueError("overlay 不能为空。")
    if skeleton_result is None:
        return overlay

    result = overlay.copy()
    skeleton_mask = np.asarray(skeleton_result.get("skeleton_mask"))
    if skeleton_mask.size == 0:
        return result
    result[skeleton_mask > 0] = color
    return result


def draw_topology_visualization(
    overlay: np.ndarray,
    topology_result: Dict[str, Any],
    show_node_labels: bool = False,
) -> np.ndarray:
    """
    将新版 TreeTopologyBuilder2D 的拓扑结果叠加到现有 overlay 上。
    """

    if overlay is None:
        raise ValueError("overlay 不能为空。")
    if topology_result is None:
        return overlay

    builder: TreeTopologyBuilder2D = topology_result["builder"]
    graph = topology_result["graph"]
    return builder.visualize_on_image(
        image=overlay,
        graph=graph,
        show_trunk_mask=True,
        show_branch_mask=False,
        show_node_labels=show_node_labels,
        show_legend=True,
    )


def save_topology_visualization_bundle(
    topology_result: Dict[str, Any],
    output_dir: str,
    image: Optional[np.ndarray] = None,
    prefix: str = "topology",
) -> Dict[str, str]:
    """可选的导出封装，方便 GUI 或脚本侧直接保存整套结果。"""

    if topology_result is None:
        raise ValueError("topology_result 不能为空。")
    builder: TreeTopologyBuilder2D = topology_result["builder"]
    graph = topology_result["graph"]
    return builder.save_visualization_bundle(
        output_dir=output_dir,
        image=image,
        graph=graph,
        prefix=prefix,
        save_node_labels=False,
    )
