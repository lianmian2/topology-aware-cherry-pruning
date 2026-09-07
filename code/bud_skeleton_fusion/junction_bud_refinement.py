"""
分叉/交叉点芽点辅助判断 — 利用芽点分布 + 方向信息增强汇接配对.

当前 DijkstraSkeletonRouter 的汇接配对 (`_pair_cost`, `_solve_crossing_pairs`,
`_solve_trunk_involved_junction`) 纯基于几何先验 (角度、半径比、曲率),
在复杂汇接处可能配对错误 → 图中产生环 (botanically impossible).

本模块提供:
1. 臂的芽点密度/分布评分 — 增强 `_pair_cost`
2. 环检测 — 验证配对结果是否产生环
3. 重排序 — 基于芽点一致性重新排序候选配对

集成方式: 在 `reconstruct_branch_groups_with_junction_pairing` 之后作为验证/修正步骤,
或作为 `_pair_cost` 的额外评分项传入.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .bud_skeleton_attachment import BudAttachment
from .bud_orientation import BudOrientation


@dataclass
class ArmBudInfo:
    """单条臂上的芽点信息."""

    arm_index: int
    bud_count: int              # 附着芽点数
    bud_indices: List[int]       # 附着的芽点索引
    bud_density: float           # 芽点数 / 臂长度 (px⁻¹)
    mean_bud_angle: float        # 芽点方向均值 (弧度, [0, π))
    angle_consistency: float     # 芽点方向与臂方向的余弦相似度 [0, 1]


@dataclass
class JunctionBudScore:
    """芽点增强的汇接点评分."""

    cluster_nodes: List[int]
    arm_bud_infos: List[ArmBudInfo]
    pair_bud_scores: Dict[Tuple[int, int], float]  # (arm_a, arm_b) → bud_consistency_score
    has_cycle_risk: bool
    cycle_nodes: List[int]       # 如果产生环, 涉及的节点


def _collect_arm_buds(
    arm_polyline_xy: List[Tuple[int, int]],
    attachments: List[BudAttachment],
    orientations: Optional[List[BudOrientation]],
    max_distance: float = 24.0,
) -> ArmBudInfo:
    """
    收集一条臂上的芽点信息.

    Args:
        arm_polyline_xy: 臂的折线路径 [(x, y), ...]
        attachments: 全部芽点吸附结果
        orientations: 芽点方向信息 (可选)
        max_distance: 芽点吸附点到折线的最大距离

    Returns:
        ArmBudInfo
    """
    polyline = np.array(arm_polyline_xy, dtype=np.float32)
    if len(polyline) < 2:
        return ArmBudInfo(arm_index=-1, bud_count=0, bud_indices=[],
                          bud_density=0.0, mean_bud_angle=0.0, angle_consistency=0.0)

    # 臂的方向向量 (从起点到终点)
    arm_vec = polyline[-1] - polyline[0]
    arm_length = float(np.linalg.norm(arm_vec))
    if arm_length < 1e-6:
        return ArmBudInfo(arm_index=-1, bud_count=0, bud_indices=[],
                          bud_density=0.0, mean_bud_angle=0.0, angle_consistency=0.0)
    arm_dir = arm_vec / arm_length   # 归一化

    bud_indices: List[int] = []
    bud_angles: List[float] = []

    for att in attachments:
        if att.skeleton_point is None:
            continue
        sx, sy = att.skeleton_point

        # 计算吸附点到折线的最短距离
        min_dist = float('inf')
        for i in range(len(polyline) - 1):
            p0 = polyline[i]
            p1 = polyline[i + 1]
            seg = p1 - p0
            seg_len2 = float(np.dot(seg, seg))
            if seg_len2 < 1e-10:
                d = float(np.linalg.norm(np.array([sx, sy]) - p0))
            else:
                t = max(0.0, min(1.0, float(np.dot(np.array([sx, sy]) - p0, seg)) / seg_len2))
                proj = p0 + t * seg
                d = float(np.linalg.norm(np.array([sx, sy]) - proj))
            if d < min_dist:
                min_dist = d

        if min_dist <= max_distance:
            bud_indices.append(att.bud_index)
            if orientations is not None and att.bud_index < len(orientations):
                bud_angles.append(orientations[att.bud_index].axis_angle)

    bud_count = len(bud_indices)
    bud_density = bud_count / arm_length if arm_length > 0 else 0.0

    # 芽点方向一致性: 芽轴与臂方向应接近平行 (夹角应接近 0 或 π)
    mean_angle = 0.0
    angle_consistency = 0.0
    if bud_angles and arm_length > 0:
        mean_angle = float(np.mean(bud_angles))
        # 芽轴方向与臂方向的夹角余弦 (取绝对值, 因为芽轴无符号)
        cosines = [abs(np.cos(angle - float(np.arctan2(arm_dir[1], arm_dir[0])))) for angle in bud_angles]
        angle_consistency = float(np.mean(cosines))

    return ArmBudInfo(
        arm_index=-1,
        bud_count=bud_count,
        bud_indices=bud_indices,
        bud_density=bud_density,
        mean_bud_angle=mean_angle,
        angle_consistency=angle_consistency,
    )


def score_junction_cluster(
    arms: List[Dict],
    attachments: List[BudAttachment],
    orientations: Optional[List[BudOrientation]] = None,
) -> JunctionBudScore:
    """
    对单个汇接点簇计算芽点增强评分.

    Args:
        arms: _extract_cluster_arms 的输出, 每个 arm dict 含 polyline_xy, vector, is_trunk_arm 等
        attachments: 全部芽点吸附结果
        orientations: 芽点方向信息

    Returns:
        JunctionBudScore
    """
    arm_bud_infos: List[ArmBudInfo] = []
    pair_bud_scores: Dict[Tuple[int, int], float] = {}

    for i, arm in enumerate(arms):
        polyline = arm.get("polyline_xy", [])
        info = _collect_arm_buds(polyline, attachments, orientations)
        info.arm_index = i
        arm_bud_infos.append(info)

    # 计算每对臂的芽点一致性评分
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            bi = arm_bud_infos[i]
            bj = arm_bud_infos[j]
            score = _pair_bud_consistency(bi, bj, arms[i], arms[j])
            pair_bud_scores[(i, j)] = score

    return JunctionBudScore(
        cluster_nodes=[],
        arm_bud_infos=arm_bud_infos,
        pair_bud_scores=pair_bud_scores,
        has_cycle_risk=False,
        cycle_nodes=[],
    )


def _pair_bud_consistency(
    bi: ArmBudInfo, bj: ArmBudInfo,
    arm_i: Dict, arm_j: Dict,
) -> float:
    """
    计算一对臂的芽点一致性评分.

    评分逻辑:
    - 交叉对 (crossing): 两臂属于不同枝条 → 每臂应各自有芽, 且芽方向与臂方向一致 → 高分
    - 分叉对 (branch): 两臂共享上游 → 芽点分布应互补 → 适中的芽点密度差异 → 高分
    - 无芽臂 → 该臂可能不属于这棵树 → 该配对应为低分

    Returns:
        score [0, 1], 越高表示芽点证据越支持该配对
    """
    has_buds_i = bi.bud_count > 0
    has_buds_j = bj.bud_count > 0

    # 两条臂都有芽: 高基准分
    if has_buds_i and has_buds_j:
        base = 0.7
    elif has_buds_i or has_buds_j:
        base = 0.35     # 只有一条臂有芽 → 降低置信度
    else:
        base = 0.15     # 两条臂都无芽 → 低置信度

    # 芽点方向一致性加分 (芽轴与臂方向对齐)
    angle_bonus = (bi.angle_consistency + bj.angle_consistency) * 0.15

    # 芽点密度相似度 (密度差异大 → 可能不是同类型枝条)
    max_density = max(bi.bud_density, bj.bud_density, 1e-6)
    min_density = min(bi.bud_density, bj.bud_density)
    density_similarity = min_density / max_density if max_density > 0 else 0.0

    score = min(1.0, base + angle_bonus + density_similarity * 0.1)
    return score


def verify_junction_acyclic(
    graph: nx.Graph,
    junction_clusters: List[List[int]],
) -> Tuple[bool, List[int]]:
    """
    验证汇接修正后的图是否为无环图.

    树在植物学上应是 DAG (当方向从主干向外时), 所以无向图中不应存在无法定向的环.

    Args:
        graph: 汇接修正后的图
        junction_clusters: 所有汇接点簇

    Returns:
        (is_acyclic, cycle_nodes): 是否无环 + 涉及的节点
    """
    try:
        cycles = list(nx.cycle_basis(graph))
        if not cycles:
            return True, []

        cycle_nodes: Set[int] = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)

        # 检查环是否涉及汇接点簇 (如果环不涉及汇接点, 可能不是配对问题)
        junction_related = False
        cluster_nodes_set: Set[int] = set()
        for cluster in junction_clusters:
            cluster_nodes_set.update(cluster)

        for node in cycle_nodes:
            if node in cluster_nodes_set:
                junction_related = True
                break

        if not junction_related:
            return True, []   # 环与汇接点无关, 暂不报告

        return False, list(cycle_nodes)
    except Exception:
        return False, []


def rerank_crossing_pairs_with_buds(
    pair_candidates: List[Dict],
    pair_bud_scores: Dict[Tuple[int, int], float],
    bud_weight: float = 0.3,
) -> List[Dict]:
    """
    基于芽点评分重新排序候选交叉配对.

    原始排序 (crossing_cost, group_type_match, -angle, radius_diff, curvature)
    加入芽点评分作为额外排序键: bud_score 高的优先.

    Args:
        pair_candidates: _solve_crossing_pairs 中的候选配对列表
        pair_bud_scores: (arm_a, arm_b) → bud_consistency_score
        bud_weight: 芽点评分权重 (0 = 不使用芽点, 1 = 完全用芽点)

    Returns:
        重新排序后的候选列表
    """
    if bud_weight <= 0.0 or not pair_bud_scores:
        return pair_candidates

    def sort_key(info: Dict) -> float:
        idx_a, idx_b = info["indices"]
        bud_score = pair_bud_scores.get((int(idx_a), int(idx_b)), 0.5)
        bud_score += pair_bud_scores.get((int(idx_b), int(idx_a)), 0.5) / 2.0

        # 组合原始代价和芽点评分
        original_cost = info.get("crossing_cost", 1.0)
        # bud_score 高 → 降低代价; bud_weight 控制影响力
        adjusted_cost = original_cost * (1.0 - bud_weight * max(0.0, bud_score - 0.3))
        return float(adjusted_cost)

    return sorted(pair_candidates, key=sort_key)


def junction_refinement_report(
    clusters: List[JunctionBudScore],
) -> dict:
    """汇总所有汇接点的芽点增强评分."""
    total_clusters = len(clusters)
    high_confidence = sum(1 for c in clusters
                          if any(s > 0.6 for s in c.pair_bud_scores.values()))
    low_confidence = sum(1 for c in clusters
                         if all(s < 0.3 for s in c.pair_bud_scores.values()))
    cycle_risks = sum(1 for c in clusters if c.has_cycle_risk)

    total_buds_near_junctions = sum(
        sum(info.bud_count for info in c.arm_bud_infos) for c in clusters
    )

    return {
        'total_junction_clusters': total_clusters,
        'high_confidence_clusters': high_confidence,
        'low_confidence_clusters': low_confidence,
        'cycle_risk_clusters': cycle_risks,
        'total_buds_near_junctions': total_buds_near_junctions,
    }
