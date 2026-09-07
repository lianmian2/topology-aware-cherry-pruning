"""
bud_skeleton_fusion — 芽点-骨架联合分析模块.

将全图芽点检测 (logic_bud.py) 与拓扑路由骨架 (DijkstraSkeletonRouter) 结合,
提供芽点方向提取、分割感知吸附、汇接点破圈、无芽剪枝等功能.

模块:
    1. bud_orientation              — 芽点尖轴提取
    2. bud_skeleton_attachment       — 芽点→骨架分割感知吸附
    3. branch_bud_pruning            — 无芽枝条裁剪
    4. junction_bud_refinement       — 分叉/交叉点芽点辅助判断
    5. skeleton_angle_validation     — 芽点-骨架夹角验证
    6. bud_auxiliary_segmentation    — 芽点辅助分割消歧
"""

from .bud_orientation import (
    BudOrientation,
    DirectedBudOrientation,
    extract_directed_bud_orientations,
    extract_bud_orientations,
    orientation_stats,
)
from .bud_skeleton_attachment import (
    BudAttachment,
    attach_buds_to_skeleton,
    attachment_stats,
    build_group_bud_map,
)
from .branch_bud_pruning import (
    PruningResult,
    prune_budless_branches,
    pruning_summary,
)
from .junction_bud_refinement import (
    ArmBudInfo,
    JunctionBudScore,
    score_junction_cluster,
    verify_junction_acyclic,
    rerank_crossing_pairs_with_buds,
    junction_refinement_report,
)
from .skeleton_angle_validation import (
    BudAngleValidation,
    validate_bud_angles,
    angle_validation_summary,
)
from .bud_auxiliary_segmentation import (
    compute_bud_confidence_heatmap,
    identify_ambiguous_regions,
    compute_cost_adjustment,
)

__all__ = [
    # Module 1
    'BudOrientation',
    'DirectedBudOrientation',
    'extract_directed_bud_orientations',
    'extract_bud_orientations',
    'orientation_stats',
    # Module 5
    'BudAttachment',
    'attach_buds_to_skeleton',
    'attachment_stats',
    'build_group_bud_map',
    # Module 3
    'PruningResult',
    'prune_budless_branches',
    'pruning_summary',
    # Module 4
    'ArmBudInfo',
    'JunctionBudScore',
    'score_junction_cluster',
    'verify_junction_acyclic',
    'rerank_crossing_pairs_with_buds',
    'junction_refinement_report',
    # Module 2
    'BudAngleValidation',
    'validate_bud_angles',
    'angle_validation_summary',
    # Module 6
    'compute_bud_confidence_heatmap',
    'identify_ambiguous_regions',
    'compute_cost_adjustment',
]
