from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def save_bud_detection_cache(path: Path, result: Dict, image_shape: Tuple[int, int]) -> None:
    path = Path(path)
    masks_info = result.get("masks_info", [])
    packed_masks: List[np.ndarray] = []
    mask_shapes = []
    mask_offsets = []
    for mask, offset_x, offset_y in masks_info:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        packed_masks.append(np.packbits(binary.reshape(-1)))
        mask_shapes.append(binary.shape[:2])
        mask_offsets.append((int(offset_x), int(offset_y)))
    max_length = max((len(mask) for mask in packed_masks), default=0)
    packed_array = np.zeros((len(packed_masks), max_length), dtype=np.uint8)
    packed_lengths = np.zeros((len(packed_masks),), dtype=np.int32)
    for index, packed in enumerate(packed_masks):
        packed_array[index, :len(packed)] = packed
        packed_lengths[index] = len(packed)
    np.savez_compressed(
        path,
        boxes=np.asarray(result.get("boxes", []), dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(result.get("scores", []), dtype=np.float32),
        labels=np.asarray(result.get("labels", []), dtype=np.int16),
        packed_masks=packed_array,
        packed_lengths=packed_lengths,
        mask_shapes=np.asarray(mask_shapes, dtype=np.int16).reshape(-1, 2),
        mask_offsets=np.asarray(mask_offsets, dtype=np.int32).reshape(-1, 2),
        image_shape=np.asarray(image_shape, dtype=np.int32),
    )


def load_bud_detection_cache(path: Path) -> Dict:
    with np.load(Path(path), allow_pickle=False) as cache:
        boxes = cache["boxes"].astype(np.float32)
        scores = cache["scores"].astype(np.float32)
        labels = cache["labels"].astype(np.int16)
        packed_masks = cache["packed_masks"]
        packed_lengths = cache["packed_lengths"]
        mask_shapes = cache["mask_shapes"]
        mask_offsets = cache["mask_offsets"]
        image_shape = tuple(map(int, cache["image_shape"]))
        masks_info = []
        for index, ((height, width), (offset_x, offset_y)) in enumerate(zip(mask_shapes, mask_offsets)):
            bit_count = int(height) * int(width)
            packed = packed_masks[index, :int(packed_lengths[index])]
            mask = np.unpackbits(packed, count=bit_count).reshape(int(height), int(width)).astype(np.uint8)
            masks_info.append((mask, int(offset_x), int(offset_y)))
    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "masks_info": masks_info,
        "image_shape": image_shape,
        "total_count": int(len(boxes)),
    }
