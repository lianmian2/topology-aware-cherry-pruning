# bud_skeleton_fusion — 芽点-骨架联合分析

将全图芽点检测 (`logic_bud.py`) 与 `DijkstraSkeletonRouter` 拓扑路由骨架结合，利用芽点的位置、形态方向信息增强骨架拓扑质量。

## 模块总览

| # | 模块 | 文件 | 功能 |
|---|------|------|------|
| 1 | 芽点尖轴提取 | `bud_orientation.py` | 从芽点 mask 提取椭圆主轴角度、长宽比 |
| 5 | 芽点→骨架吸附 | `bud_skeleton_attachment.py` | 分割感知吸附：在连通分量内搜索最近骨架像素 |
| 4 | 分叉/交叉芽点辅助 | `junction_bud_refinement.py` | 芽点分布 + 方向增强汇接配对 + 环检测 |
| 3 | 无芽枝条裁剪 | `branch_bud_pruning.py` | 移除 annotation_groups 中无芽附着的枝条 |
| 2 | 芽点-骨架夹角验证 | `skeleton_angle_validation.py` | 芽轴 vs 骨架切线 / vs 分割边界法线夹角 |
| 6 | 芽点辅助分割消歧 | `bud_auxiliary_segmentation.py` | 生成芽点增强的区域置信度 heatmap |

## 快速使用

```python
from bud_skeleton_fusion import (
    extract_bud_orientations,
    attach_buds_to_skeleton,
    score_junction_cluster,
    prune_budless_branches,
    validate_bud_angles,
    compute_bud_confidence_heatmap,
)
```

### 典型 Pipeline

```python
# 1. 芽点检测 (复用 logic_bud.py)
from logic_bud import run_bud_detection_pipeline
bud_result = run_bud_detection_pipeline(image_bgr, use_roi_filter=True)

# 2. 骨架提取 (复用 mask_topology_routing)
from mask_topology_routing.utils import build_prediction_result
pred = build_prediction_result(combined_mask, ...)

# 3. 芽点尖轴提取
orientations = extract_bud_orientations(bud_result['masks_info'])

# 4. 分割感知芽点吸附
attachments = attach_buds_to_skeleton(
    bud_result['boxes'], bud_result['masks_info'],
    pred.skeleton_map, pred.mask, pred.annotation_groups,
)

# 5. 无芽枝条裁剪
from bud_skeleton_fusion import prune_budless_branches
prune_result = prune_budless_branches(pred.annotation_groups, attachments)
print(pruning_summary(prune_result))

# 6. 汇接点芽点增强评分 (需要 DijkstraSkeletonRouter 的内部 arm 数据)
# junction_score = score_junction_cluster(arms, attachments, orientations)
```

## 集成位置

所有模块面向 `DijkstraSkeletonRouter.__call__` 流程：

```
Stage 1-5: mask → skeleton → trunk → branch routing → groups
    │
    ├── [模块 5] 芽点吸附到 skeleton_map + 归属到 groups
    │
Stage 6: junction_pairing
    │
    ├── [模块 4] 芽点分布增强配对评分 + 破圈
    │
    ├── [模块 3] 移除无芽 branch groups
    │
PredictionResult (无环 + 芽点吸附)
```

## 依赖

- `logic_bud.py` (芽点检测)
- `mask_topology_routing/utils.py` (DijkstraSkeletonRouter)
- `opencv-python`, `numpy`, `scipy`, `networkx`, `scikit-image`
