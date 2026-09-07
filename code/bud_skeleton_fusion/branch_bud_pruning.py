"""
无芽枝条裁剪 — 从 DijkstraSkeletonRouter 的 annotation_groups 中移除无芽附着的枝条.

原理: 对剪枝决策而言, 没有芽点的枝条没有保留意义.
保留无芽枝条会增加 GNN 噪声节点/边, 不利于训练.

操作对象: PredictionResult.annotation_groups (group-based JSON 格式).
不修改原始 DijkstraSkeletonRouter 的图结构.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from .bud_skeleton_attachment import BudAttachment


@dataclass
class PruningResult:
    """剪枝结果."""

    pruned_groups: List[Dict]          # 裁剪后的 annotation_groups
    removed_group_ids: List[str]        # 被移除的 group ID
    original_count: int                 # 原始 branch group 数量
    pruned_count: int                   # 裁剪后 branch group 数量
    total_original_length: float        # 原始枝条总折线长度
    total_pruned_length: float          # 裁剪后枝条总折线长度


def _group_polyline_length(group: Dict) -> float:
    """计算 group 的折线总长度."""
    points = group.get("points", [])
    total = 0.0
    for i in range(len(points) - 1):
        p0 = np.array(points[i], dtype=np.float32)
        p1 = np.array(points[i + 1], dtype=np.float32)
        total += float(np.linalg.norm(p1 - p0))
    return total


def _build_group_tree(annotation_groups: List[Dict]) -> Dict[str, List[str]]:
    """
    构建 group 的父子关系树.

    group 间的 fork_origin_group 定义了分叉来源.
    返回: group_id → [child_group_id, ...]
    """
    children: Dict[str, List[str]] = {}
    for group in annotation_groups:
        gid = group.get("group_id", "")
        if gid not in children:
            children[gid] = []
        parent = group.get("fork_origin_group")
        if parent:
            if parent not in children:
                children[parent] = []
            children[parent].append(gid)
    return children


def _collect_group_buds(
    group_id: str,
    attachments: List[BudAttachment],
    bud_match_distance: float = 24.0,
) -> List[BudAttachment]:
    """收集归属于指定 group 的芽点."""
    return [a for a in attachments
            if a.group_id == group_id and a.skeleton_point is not None
            and a.distance <= bud_match_distance]


def _has_buds_recursive(
    group_id: str,
    children: Dict[str, List[str]],
    group_bud_counts: Dict[str, int],
    visited: Optional[Set[str]] = None,
) -> bool:
    """
    递归检查 group 及其子孙是否有芽点.

    用于剪枝决策: 如果 group 本身无芽且所有子孙也无芽 → 可安全删除.
    """
    if visited is None:
        visited = set()
    if group_id in visited:
        return False
    visited.add(group_id)

    if group_bud_counts.get(group_id, 0) > 0:
        return True

    for child_id in children.get(group_id, []):
        if _has_buds_recursive(child_id, children, group_bud_counts, visited):
            return True

    return False


def prune_budless_branches(
    annotation_groups: List[Dict],
    attachments: List[BudAttachment],
    bud_match_distance: float = 24.0,
    min_branch_length: float = 10.0,
) -> PruningResult:
    """
    移除无芽附着的枝条 groups.

    保留规则:
    - group_type == "trunk" 始终保留
    - group 自身有芽 → 保留
    - group 的任意子孙有芽 → 保留 (因为需要拓扑连通性)
    - 其他 → 删除

    Args:
        annotation_groups: DijkstraSkeletonRouter 输出的 groups
        attachments: 芽点吸附结果 (模块 5)
        bud_match_distance: 芽点吸附距离阈值
        min_branch_length: 即使有芽也裁剪的最小枝条长度 (可过滤碎片)

    Returns:
        PruningResult
    """
    # 1. 分组: trunk vs branch
    trunk_group = None
    branch_groups: List[Dict] = []
    for group in annotation_groups:
        if group.get("group_type") == "trunk":
            trunk_group = group
        else:
            branch_groups.append(group)

    if not branch_groups:
        return PruningResult(
            pruned_groups=annotation_groups,
            removed_group_ids=[],
            original_count=0,
            pruned_count=0,
            total_original_length=0.0,
            total_pruned_length=0.0,
        )

    # 2. 统计每个 group 的芽点数
    group_bud_counts: Dict[str, int] = {}
    for group in branch_groups:
        gid = group.get("group_id", "")
        buds = _collect_group_buds(gid, attachments, bud_match_distance)
        group_bud_counts[gid] = len(buds)

    # 3. 构建父子关系树
    children = _build_group_tree(branch_groups)

    # 4. 递归判断每个 branch group 是否可删除
    removed_ids: Set[str] = set()
    for group in branch_groups:
        gid = group.get("group_id", "")
        if gid in removed_ids:
            continue

        has_buds = _has_buds_recursive(gid, children, group_bud_counts)

        if not has_buds:
            removed_ids.add(gid)

    # 5. 构建裁剪后的 groups
    pruned = []
    if trunk_group is not None:
        pruned.append(trunk_group)
    for group in branch_groups:
        gid = group.get("group_id", "")
        if gid not in removed_ids:
            pruned.append(group)

    # 6. 统计
    original_total_len = sum(_group_polyline_length(g) for g in branch_groups)
    pruned_branch_groups = [g for g in pruned if g.get("group_type") != "trunk"]
    pruned_total_len = sum(_group_polyline_length(g) for g in pruned_branch_groups)

    return PruningResult(
        pruned_groups=pruned,
        removed_group_ids=sorted(removed_ids),
        original_count=len(branch_groups),
        pruned_count=len(pruned_branch_groups),
        total_original_length=original_total_len,
        total_pruned_length=pruned_total_len,
    )


def pruning_summary(result: PruningResult) -> str:
    """生成人类可读的剪枝摘要."""
    lines = [
        f"Branch groups: {result.original_count} → {result.pruned_count} "
        f"({result.original_count - result.pruned_count} removed)",
        f"Total branch length: {result.total_original_length:.0f} → "
        f"{result.total_pruned_length:.0f} px "
        f"({(1 - result.total_pruned_length / max(result.total_original_length, 1)) * 100:.1f}% reduced)",
    ]
    if result.removed_group_ids:
        lines.append(f"Removed: {', '.join(result.removed_group_ids)}")
    return "\n".join(lines)
