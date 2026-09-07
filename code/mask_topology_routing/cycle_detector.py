"""
汇接配对错误 → 树干环检测.

核心洞察:
- 交叉误判为分叉时，本该属于同一枝条的相对两臂被拆成两根独立分支
- 两"伪分支"都连接到主干 → 路径 + 主干段 = 经过主干的环
- 树在植物学上绝对无环 → 树干环 = 汇接错误的可检测签名

用法:
    from .cycle_detector import detect_trunk_cycles, collect_cycle_stats
    cycles = detect_trunk_cycles(graph, trunk_points)
    stats = collect_cycle_stats(cycles, junction_clusters)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


def _point_to_polyline_dist(point: np.ndarray, polyline: np.ndarray) -> float:
    """点到折线的最短距离."""
    if len(polyline) < 2:
        return float(np.linalg.norm(point - polyline[0]))
    seg_dx = polyline[1:, 0] - polyline[:-1, 0]
    seg_dy = polyline[1:, 1] - polyline[:-1, 1]
    seg_len_sq = seg_dx ** 2 + seg_dy ** 2
    px = point[0] - polyline[:-1, 0]
    py = point[1] - polyline[:-1, 1]
    t = np.clip((px * seg_dx + py * seg_dy) / np.maximum(seg_len_sq, 1e-12), 0.0, 1.0)
    proj_x = polyline[:-1, 0] + t * seg_dx
    proj_y = polyline[:-1, 1] + t * seg_dy
    return float(np.min(np.sqrt((point[0] - proj_x) ** 2 + (point[1] - proj_y) ** 2)))


def detect_trunk_cycles(
    graph: nx.Graph,
    trunk_points: Sequence[Tuple[int, int]],
    trunk_proximity_threshold: float = 12.0,
) -> List[List[int]]:
    """检测图中所有经过主干段的环.

    通过检查环的边端点是否靠近主干折线来识别树干环.
    (节点属性 is_trunk 在汇接处理后不可靠, 故用几何邻近判断.)

    Args:
        graph: 汇接修正后的无向图
        trunk_points: 主干折线点 [(x,y), ...]
        trunk_proximity_threshold: 节点到主干线距离阈值 (px)

    Returns:
        List[cycle_nodes]: 每个环的节点 ID 列表
    """
    try:
        all_cycles = nx.cycle_basis(graph)
    except Exception:
        return []

    if not all_cycles or len(trunk_points) < 2:
        return []

    trunk_poly = np.asarray(trunk_points, dtype=np.float32)
    trunk_cycles: List[List[int]] = []

    for cycle in all_cycles:
        trunk_adjacent_nodes = 0
        for node_id in cycle:
            if node_id not in graph.nodes:
                continue
            pt = np.asarray(graph.nodes[node_id]["point"], dtype=np.float32)
            dist = _point_to_polyline_dist(pt, trunk_poly)
            if dist <= trunk_proximity_threshold:
                trunk_adjacent_nodes += 1

        # 环至少要有 2 个靠近主干的节点才是"树干环"
        # (单个节点靠近主干可能是偶然的, 2个说明环路径经过主干)
        if trunk_adjacent_nodes >= 2:
            trunk_cycles.append(cycle)

    return trunk_cycles


def find_cycle_edges(graph: nx.Graph, cycle: List[int]) -> List[Tuple[int, int]]:
    """提取环中的所有边 (有序)."""
    edges: List[Tuple[int, int]] = []
    for i in range(len(cycle)):
        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
        if graph.has_edge(u, v):
            edges.append((u, v))
    return edges


def classify_cycle_segments(
    graph: nx.Graph,
    cycle: List[int],
    trunk_points: Sequence[Tuple[int, int]],
    trunk_proximity_threshold: float = 12.0,
) -> Dict[str, List[Tuple[int, int]]]:
    """将环的边分类为 trunk_segments 和 branch_segments.

    树干段: 两个端点都靠近主干折线的边
    分支段: 至少一个端点不靠近主干折线的边
    """
    if len(trunk_points) < 2:
        return {"trunk_segments": [], "branch_segments": []}
    trunk_poly = np.asarray(trunk_points, dtype=np.float32)

    trunk_edges: List[Tuple[int, int]] = []
    branch_edges: List[Tuple[int, int]] = []

    for i in range(len(cycle)):
        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
        if not graph.has_edge(u, v):
            continue
        pu = np.asarray(graph.nodes[u]["point"], dtype=np.float32)
        pv = np.asarray(graph.nodes[v]["point"], dtype=np.float32)
        u_trunk = _point_to_polyline_dist(pu, trunk_poly) <= trunk_proximity_threshold
        v_trunk = _point_to_polyline_dist(pv, trunk_poly) <= trunk_proximity_threshold
        if u_trunk and v_trunk:
            trunk_edges.append((u, v))
        else:
            branch_edges.append((u, v))

    return {"trunk_segments": trunk_edges, "branch_segments": branch_edges}


def map_cycles_to_junctions(
    cycles: List[List[int]],
    junction_clusters: List[List[int]],
) -> Dict[int, List[int]]:
    """将环映射回产生它们的汇接簇.

    Args:
        cycles: detect_trunk_cycles 的输出
        junction_clusters: 所有汇接点簇 (每个簇是 node_id 列表)

    Returns:
        {cycle_idx: [cluster_idx, ...]} — 每个环关联的汇接簇
    """
    cluster_sets = [set(c) for c in junction_clusters]
    mapping: Dict[int, List[int]] = {}

    for cycle_idx, cycle in enumerate(cycles):
        cycle_set = set(cycle)
        related = []
        for cl_idx, cl_set in enumerate(cluster_sets):
            overlap = cycle_set & cl_set
            if len(overlap) >= 2:
                related.append(cl_idx)
        mapping[cycle_idx] = related

    return mapping


def collect_cycle_stats(
    cycles: List[List[int]],
    junction_clusters: List[List[int]],
) -> Dict[str, float]:
    """收集环统计信息."""
    mapping = map_cycles_to_junctions(cycles, junction_clusters)

    junction_related = sum(1 for related in mapping.values() if related)
    return {
        "trunk_cycles_total": float(len(cycles)),
        "trunk_cycles_junction_related": float(junction_related),
        "trunk_cycles_unexplained": float(len(cycles) - junction_related),
    }


def describe_cycle(
    graph: nx.Graph,
    cycle: List[int],
    trunk_points: Optional[Sequence[Tuple[int, int]]] = None,
    trunk_proximity_threshold: float = 12.0,
) -> Dict:
    """诊断性地描述一个环."""
    if trunk_points and len(trunk_points) >= 2:
        segments = classify_cycle_segments(graph, cycle, trunk_points, trunk_proximity_threshold)
    else:
        segments = {"trunk_segments": [], "branch_segments": []}

    trunk_edge_count = len(segments["trunk_segments"])
    branch_edge_count = len(segments["branch_segments"])
    cycle_len = len(cycle)

    trunk_length = 0.0
    for u, v in segments["trunk_segments"]:
        pu = np.asarray(graph.nodes[u]["point"])
        pv = np.asarray(graph.nodes[v]["point"])
        trunk_length += float(np.linalg.norm(pv - pu))

    return {
        "cycle_length_nodes": cycle_len,
        "trunk_segments": trunk_edge_count,
        "branch_segments": branch_edge_count,
        "trunk_segment_length_px": round(trunk_length, 1),
        "nodes": cycle,
    }


def _point_to_polyline_distances(
    points: np.ndarray, polyline: np.ndarray,
) -> np.ndarray:
    """向量化点到折线距离. points: (N,2), polyline: (M,2) → (N,)"""
    if len(polyline) < 2:
        return np.linalg.norm(points - polyline[0], axis=1)
    seg_dx = polyline[1:, 0] - polyline[:-1, 0]
    seg_dy = polyline[1:, 1] - polyline[:-1, 1]
    seg_len_sq = seg_dx ** 2 + seg_dy ** 2
    seg_len_sq = np.maximum(seg_len_sq, 1e-12)
    px = points[:, None, 0] - polyline[None, :-1, 0]
    py = points[:, None, 1] - polyline[None, :-1, 1]
    t = np.clip((px * seg_dx[None, :] + py * seg_dy[None, :]) / seg_len_sq[None, :], 0.0, 1.0)
    proj_x = polyline[None, :-1, 0] + t * seg_dx[None, :]
    proj_y = polyline[None, :-1, 1] + t * seg_dy[None, :]
    return np.min(np.sqrt((points[:, None, 0] - proj_x) ** 2 + (points[:, None, 1] - proj_y) ** 2), axis=1)


def compute_branch_bud_consistency(
    branch_polyline: np.ndarray,
    bud_centers: np.ndarray,
    bud_angles: np.ndarray,
    bud_elongated: np.ndarray,
    search_radius: float = 40.0,
) -> Dict:
    """计算一条分支上elongated芽点方向的相互一致性.

    使用圆形统计的 mean resultant length R:
    - R → 1: 所有芽点方向一致 (正确的枝条)
    - R → 0: 芽点方向随机 (可能是错误连接)

    Args:
        branch_polyline: (M, 2) 分支折线
        bud_centers: (N, 2) 所有芽点中心
        bud_angles: (N,) 芽点主轴角 [0, π) 弧度
        bud_elongated: (N,) bool
        search_radius: 芽点到折线的搜索半径

    Returns:
        {n_buds, consistency_R, is_consistent, mean_direction_deg}
    """
    if len(branch_polyline) < 2:
        return {"n_buds": 0, "consistency_R": 0.5, "is_consistent": False, "mean_direction_deg": 0.0}

    dists = _point_to_polyline_distances(bud_centers, branch_polyline)
    nearby = (dists <= search_radius) & bud_elongated
    n_buds = int(np.sum(nearby))

    if n_buds < 2:
        return {"n_buds": n_buds, "consistency_R": 0.5, "is_consistent": False, "mean_direction_deg": 0.0}

    angles = bud_angles[nearby]
    # 圆形统计: 芽点轴是无符号的 → 角度加倍处理 ([0,π) → [0,2π))
    doubled = 2.0 * angles
    x = np.cos(doubled)
    y = np.sin(doubled)
    R = float(np.sqrt(np.mean(x) ** 2 + np.mean(y) ** 2))
    mean_dir = float(0.5 * np.arctan2(np.mean(y), np.mean(x)))  # 主方向, 回到 [0,π)
    if mean_dir < 0:
        mean_dir += np.pi

    return {
        "n_buds": n_buds,
        "consistency_R": R,
        "is_consistent": bool(R >= 0.6),
        "mean_direction_deg": float(np.rad2deg(mean_dir)),
    }


def find_cycle_branches(
    graph: nx.Graph,
    cycle: List[int],
    groups: List[Dict],
) -> List[int]:
    """找参与环的 branch group 索引.

    通过检查 cycle 中的节点是否在 group 的 points 附近来判断.
    """
    branch_indices = []
    for group_idx, group in enumerate(groups):
        if group.get("group_type") != "branch":
            continue
        group_points = np.asarray(group.get("points", []), dtype=np.float32)
        if len(group_points) < 2:
            continue
        # 检查 cycle 中的节点是否靠近这个 group 的折线
        for node_id in cycle:
            if node_id not in graph.nodes:
                continue
            pt = np.asarray(graph.nodes[node_id]["point"], dtype=np.float32)
            dist = _point_to_polyline_dist(pt, group_points)
            if dist < 15.0:
                branch_indices.append(group_idx)
                break
    return branch_indices


def attempt_bud_consistency_cycle_repair(
    groups: List[Dict],
    graph: nx.Graph,
    trunk_cycles: List[List[int]],
    bud_centers: np.ndarray,
    bud_angles: np.ndarray,
    bud_elongated: np.ndarray,
    min_consistency: float = 0.30,
    high_consistency: float = 0.55,
    search_radius: float = 40.0,
) -> Tuple[List[Dict], int, List[Dict]]:
    """用芽点方向一致性修复树干环.

    安全策略:
    1. 只移除 R < min_consistency 的 branch (非常不一致)
    2. 且同环中至少有一个 branch 的 R > high_consistency (存在明显对比)
    3. 两个 branch 都至少有 2 个 elongated buds

    Args:
        groups: 当前 annotation_groups
        graph: 当前拓扑图
        trunk_cycles: 检测到的树干环
        bud_centers: (N,2) 芽点中心
        bud_angles: (N,) 芽点角度
        bud_elongated: (N,) bool
        min_consistency: R < 此值才可能被移除
        high_consistency: 同环中需有 R > 此值的 branch 作为参照
        search_radius: 芽点搜索半径

    Returns:
        (modified_groups, n_repaired, debug_info)
    """
    if not trunk_cycles or len(bud_centers) == 0:
        return list(groups), 0, []

    branch_groups = [(i, g) for i, g in enumerate(groups) if g.get("group_type") == "branch"]
    if not branch_groups:
        return list(groups), 0, []

    # 对所有 branch group 计算芽点一致性
    branch_consistencies = {}
    for idx, group in enumerate(groups):
        if group.get("group_type") != "branch":
            continue
        polyline = np.asarray(group.get("points", []), dtype=np.float32)
        if len(polyline) < 2:
            branch_consistencies[idx] = {"n_buds": 0, "consistency_R": 0.5}
            continue
        branch_consistencies[idx] = compute_branch_bud_consistency(
            polyline, bud_centers, bud_angles, bud_elongated, search_radius,
        )

    debug_info = []
    removed_indices: Set[int] = set()

    for cycle_idx, cycle in enumerate(trunk_cycles):
        branch_idxs = find_cycle_branches(graph, cycle, groups)
        available = [bi for bi in branch_idxs if bi not in removed_indices]

        if len(available) < 2:
            continue

        # 收集每个 branch 的 consistency
        cycle_branch_R = {bi: branch_consistencies.get(bi, {}).get("consistency_R", 0.5) for bi in available}
        cycle_branch_n = {bi: branch_consistencies.get(bi, {}).get("n_buds", 0) for bi in available}

        worst_idx = min(available, key=lambda bi: cycle_branch_R[bi])
        best_idx = max(available, key=lambda bi: cycle_branch_R[bi])
        worst_R = cycle_branch_R[worst_idx]
        best_R = cycle_branch_R[best_idx]
        worst_n = cycle_branch_n[worst_idx]
        best_n = cycle_branch_n[best_idx]

        debug_info.append({
            "cycle_idx": cycle_idx,
            "cycle_nodes": len(cycle),
            "candidate_branches": [
                {"group_idx": bi, "R": cycle_branch_R[bi], "n_buds": cycle_branch_n[bi]}
                for bi in available
            ],
            "worst_idx": worst_idx, "worst_R": worst_R, "worst_n": worst_n,
            "best_idx": best_idx, "best_R": best_R, "best_n": best_n,
            "would_remove": False,
        })

        # 安全条件: 最差 branch 一致性很低, 且最好 branch 一致性很高, 且两者都有足够芽点
        if worst_R < min_consistency and best_R > high_consistency and worst_n >= 2 and best_n >= 2:
            debug_info[-1]["would_remove"] = True
            removed_indices.add(worst_idx)

    if not removed_indices:
        return list(groups), 0, debug_info

    remaining = [g for i, g in enumerate(groups) if i not in removed_indices]
    return remaining, len(removed_indices), debug_info
