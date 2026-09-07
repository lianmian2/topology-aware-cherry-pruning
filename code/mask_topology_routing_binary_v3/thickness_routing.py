from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mask_topology_routing import utils as base


LEGACY_TRUNK_SELECTOR = base._choose_trunk_path_on_skeleton


def _distribution(rulebook: dict[str, Any], name: str) -> dict[str, float]:
    return rulebook["summary"][name]


def _lower_tail_penalty(value: float, stats: dict[str, float]) -> float:
    scale = max(float(stats.get("iqr", 0.0)), 0.05)
    return max(float(stats["p05"]) - float(value), 0.0) / scale


def _robust_distance(value: float, stats: dict[str, float]) -> float:
    scale = max(float(stats.get("iqr", 0.0)), 0.05)
    return abs(float(value) - float(stats["median"])) / scale


def load_thickness_rulebook(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _junction_centers(skeleton: np.ndarray) -> list[tuple[int, int]]:
    degree = base._compute_degree_map(skeleton)
    junction = ((skeleton > 0) & (degree >= 3)).astype(np.uint8)
    count, labels = cv2.connectedComponents(junction, connectivity=8)
    centers: list[tuple[int, int]] = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if not len(xs):
            continue
        centers.append((int(round(float(np.mean(ys)))), int(round(float(np.mean(xs))))))
    return [base._snap_to_nearest_true(point, skeleton) for point in centers]


def _candidate_features(
    path_rc: list[tuple[int, int]],
    dt_map: np.ndarray,
    root_rc: tuple[int, int],
    component_top: int,
) -> dict[str, float]:
    points = np.asarray(path_rc, dtype=np.int32)
    radii = dt_map[points[:, 0], points[:, 1]].astype(np.float64)
    count = len(radii)
    third = max(count // 3, 1)
    terminal_window = max(count // 10, 3)
    lower = float(np.median(radii[:third]))
    upper = float(np.median(radii[-third:]))
    median = max(float(np.median(radii)), 1e-6)
    terminal = float(np.median(radii[-terminal_window:]))
    rows = points[:, 0].astype(np.float64)
    cols = points[:, 1].astype(np.float64)
    steps = np.diff(points.astype(np.float64), axis=0)
    arc = float(np.linalg.norm(steps, axis=1).sum()) if len(steps) else 0.0
    vertical = max(float(root_rc[0]) - float(rows.min()), 1.0)
    return {
        "median_radius_px": median,
        "p25_radius_px": float(np.percentile(radii, 25)),
        "sustained_to_median_ratio": float(np.percentile(radii, 25) / median),
        "upper_to_lower_radius_ratio": float(upper / max(lower, 1e-6)),
        "terminal_to_median_radius_ratio": float(terminal / median),
        "terminal_below_component_top_norm_height": float((path_rc[-1][0] - component_top) / max(dt_map.shape[0], 1)),
        "tortuosity": float(arc / vertical),
        "lateral_drift_norm_width": float(np.ptp(cols) / max(dt_map.shape[1], 1)),
    }


def _contact_cluster_count(
    source_points: set[tuple[int, int]],
    target_points: set[tuple[int, int]],
    shape: tuple[int, int],
) -> int:
    """Count physically distinct 8-neighbour contacts between two path parts."""
    if not source_points or not target_points:
        return 0
    contact = np.zeros(shape, dtype=np.uint8)
    for row, col in source_points:
        if any(
            (row + dr, col + dc) in target_points
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        ):
            contact[row, col] = 1
    count, _ = cv2.connectedComponents(contact, connectivity=8)
    return max(int(count) - 1, 0)


def _replacement_safety(
    legacy_path: list[tuple[int, int]],
    candidate_path: list[tuple[int, int]],
    candidate_type: str,
    skeleton_shape: tuple[int, int],
) -> dict[str, Any]:
    """Ensure a trunk reroute reclassifies the old tail instead of deleting it.

    The selector is allowed to stop at an internal junction (as in tree 149),
    but the removed legacy tail must remain a single, one-contact residual branch.
    Divergent reroutes additionally require both exclusive tails to meet only at
    the shared prefix.  This is deliberately conservative: ambiguous cycles or
    multi-contact replacements retain the legacy trunk.
    """
    shared_prefix = 0
    for old_point, new_point in zip(legacy_path, candidate_path):
        if old_point != new_point:
            break
        shared_prefix += 1

    old_set = set(legacy_path)
    new_set = set(candidate_path)
    old_exclusive = old_set - new_set
    new_exclusive = new_set - old_set
    old_contacts = _contact_cluster_count(old_exclusive, new_set, skeleton_shape)
    new_contacts = _contact_cluster_count(new_exclusive, old_set, skeleton_shape)
    is_prefix_stop = bool(old_exclusive) and not new_exclusive

    reasons: list[str] = []
    if shared_prefix < min(8, max(min(len(legacy_path), len(candidate_path)) // 4, 2)):
        reasons.append("insufficient_shared_root_prefix")
    if not old_exclusive:
        reasons.append("does_not_replace_legacy_tail")
    elif old_contacts != 1:
        reasons.append("legacy_tail_not_single_contact_residual")
    if is_prefix_stop:
        if candidate_type != "junction":
            reasons.append("early_stop_is_not_junction")
    elif new_exclusive and new_contacts != 1:
        reasons.append("new_route_has_multiple_legacy_contacts")

    return {
        "safe": not reasons,
        "reasons": reasons,
        "shared_prefix_px": int(shared_prefix),
        "legacy_residual_px": int(len(old_exclusive)),
        "new_exclusive_px": int(len(new_exclusive)),
        "legacy_residual_contact_clusters": int(old_contacts),
        "new_route_contact_clusters": int(new_contacts),
        "replacement_mode": "junction_stop" if is_prefix_stop else "junction_reroute",
    }


def choose_trunk_path_thickness_aware(
    skeleton: np.ndarray,
    dt_map: np.ndarray,
    dt_norm: np.ndarray,
    root_rc: tuple[int, int],
    outside_cost: float,
    thickness_rulebook: dict[str, Any],
    topology_rulebook: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], float, np.ndarray, list[dict[str, Any]]]:
    legacy_path, legacy_candidates, legacy_score, legacy_cost = LEGACY_TRUNK_SELECTOR(
        skeleton, dt_map, dt_norm, root_rc, outside_cost
    )
    degree = base._compute_degree_map(skeleton)
    endpoints = [(int(r), int(c)) for r, c in zip(*np.where((skeleton > 0) & (degree == 1)))]
    junctions = _junction_centers(skeleton)
    eligible = [point for point in endpoints + junctions if point[0] < root_rc[0] - 8]
    # Keep both high and thick landmarks. The old router kept only the highest endpoints.
    high = sorted(eligible, key=lambda point: (point[0], -float(dt_map[point])))[:24]
    thick = sorted(eligible, key=lambda point: (-float(dt_map[point]), point[0]))[:24]
    candidates = base._sparsify_points(high + thick, min_dist=10.0, max_points=36)
    skeleton_cost = base._build_skeleton_cost(dt_norm, skeleton, outside_cost=outside_cost)
    ys = np.where(skeleton > 0)[0]
    component_top = int(ys.min()) if len(ys) else 0
    taper_stats = _distribution(thickness_rulebook, "trunk_upper_to_lower_radius_ratio")
    terminal_stats = _distribution(thickness_rulebook, "trunk_terminal_to_median_radius_ratio")
    sustained_stats = thickness_rulebook["summary"].get("trunk_sustained_to_median_ratio", {"p05": 0.65, "median": 0.85, "iqr": 0.15})
    terminal_height_stats = None
    legacy_features = _candidate_features(legacy_path, dt_map, root_rc, component_top)
    gate_reasons: list[str] = []
    if legacy_features["upper_to_lower_radius_ratio"] < float(taper_stats["p05"]):
        gate_reasons.append("legacy_upper_thinner_than_truth_p05")
    if legacy_features["terminal_to_median_radius_ratio"] < float(terminal_stats["p05"]):
        gate_reasons.append("legacy_terminal_thinner_than_truth_p05")
    if legacy_features["sustained_to_median_ratio"] < float(sustained_stats["p05"]):
        gate_reasons.append("legacy_sustained_width_below_truth_p05")
    if topology_rulebook is not None:
        terminal_height_stats = topology_rulebook["summary"]["trunk_distributions"]["terminal_below_tree_top_norm_height"]
        topology_stats = topology_rulebook["summary"]["trunk_distributions"]
        tortuosity_stats = topology_stats["tortuosity"]
        # Pixel skeletons are rougher than simplified GT polylines. Trigger only
        # beyond the GT maximum plus four IQRs, not at the raw GT maximum.
        tortuosity_trigger = float(tortuosity_stats["max"] + 4.0 * tortuosity_stats["iqr"])
        if legacy_features["tortuosity"] > tortuosity_trigger:
            gate_reasons.append("legacy_tortuosity_gross_outlier")
        drift_stats = topology_stats["lateral_drift_norm_width"]
        if legacy_features["lateral_drift_norm_width"] > float(drift_stats["max"]):
            gate_reasons.append("legacy_lateral_drift_above_truth_max")

    ranked: list[dict[str, Any]] = []
    best_path: list[tuple[int, int]] | None = None
    best_score = -1e9
    junction_set = set(junctions)
    for candidate in candidates:
        path = base._route_path(skeleton_cost, root_rc, candidate)
        if not path or len(path) < 2:
            continue
        features = _candidate_features(path, dt_map, root_rc, component_top)
        base_score = base._score_trunk_path(path, dt_norm, root_rc, skeleton.shape)
        score = float(base_score)
        score += 0.22 * min(features["sustained_to_median_ratio"], 1.0)
        score -= 0.34 * _lower_tail_penalty(features["upper_to_lower_radius_ratio"], taper_stats)
        score -= 0.42 * _lower_tail_penalty(features["terminal_to_median_radius_ratio"], terminal_stats)
        score -= 0.28 * _lower_tail_penalty(features["sustained_to_median_ratio"], sustained_stats)
        if terminal_height_stats is not None:
            # 97/99 truth trunks terminate below the tree top. Use the empirical
            # terminal-height distribution strongly enough to stop a thick but
            # prematurely ending side arm from winning (tree_040 guardrail).
            score -= 0.12 * _robust_distance(
                features["terminal_below_component_top_norm_height"], terminal_height_stats
            )
        # Internal termination is legitimate in 97/99 truth views, so a junction is not penalized.
        candidate_type = "junction" if candidate in junction_set else "endpoint"
        replacement_safety = _replacement_safety(
            legacy_path, path, candidate_type, skeleton.shape
        )
        record = {
            "candidate_rc": [int(candidate[0]), int(candidate[1])],
            "candidate_type": candidate_type,
            "score": float(score), "base_score": float(base_score),
            "replacement_safety": replacement_safety,
            **features,
        }
        ranked.append(record)
        if replacement_safety["safe"] and score > best_score:
            best_score, best_path = score, path
    ranked.sort(key=lambda item: item["score"], reverse=True)
    gate_record = {
        "selection_gate": "allow_thickness_reroute" if gate_reasons else "retain_legacy_trunk",
        "selection_gate_reasons": gate_reasons,
        "legacy_score": float(legacy_score),
        "legacy_features": legacy_features,
    }
    if ranked:
        ranked[0]["selection_gate"] = gate_record
    if not gate_reasons:
        return legacy_path, legacy_candidates, float(legacy_score), legacy_cost, ranked
    if best_path is None:
        gate_record["selection_gate"] = "retain_legacy_no_safe_replacement"
        if ranked:
            ranked[0]["selection_gate"] = gate_record
        return legacy_path, legacy_candidates, float(legacy_score), legacy_cost, ranked
    return best_path, candidates, float(best_score), skeleton_cost, ranked


def make_router_selector(
    thickness_rulebook_path: str | Path,
    topology_rulebook_path: str | Path | None = None,
):
    thickness_rulebook = load_thickness_rulebook(thickness_rulebook_path)
    topology_rulebook = load_thickness_rulebook(topology_rulebook_path) if topology_rulebook_path else None

    def selector(skeleton, dt_map, dt_norm, root_rc, outside_cost):
        path, candidates, score, cost, ranked = choose_trunk_path_thickness_aware(
            skeleton, dt_map, dt_norm, root_rc, outside_cost,
            thickness_rulebook=thickness_rulebook,
            topology_rulebook=topology_rulebook,
        )
        selector.last_ranked_candidates = ranked
        return path, candidates, score, cost

    selector.last_ranked_candidates = []
    return selector
