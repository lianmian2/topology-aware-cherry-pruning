from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from utils import DEFAULT_JUNCTION_PRIOR_PATH, DEFAULT_PROCESSED_ROOT, load_json, save_json


def _normalize(vec_xy: np.ndarray) -> np.ndarray:
    vec_xy = np.asarray(vec_xy, dtype=np.float32)
    norm = float(np.linalg.norm(vec_xy))
    if norm < 1e-6:
        return np.zeros((2,), dtype=np.float32)
    return vec_xy / norm


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    unit_a = _normalize(vec_a)
    unit_b = _normalize(vec_b)
    dot = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _group_graph(group: Dict) -> Tuple[List[Tuple[float, float]], nx.Graph]:
    points = [tuple(map(float, point)) for point in group.get("points", [])]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    for edge in group.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = int(edge[0]), int(edge[1])
        if 0 <= src < len(points) and 0 <= dst < len(points) and src != dst:
            graph.add_edge(src, dst)
    return points, graph


def collect_gt_angles(raw_json_dir: Path) -> Dict[str, List[float]]:
    branch_angles: List[float] = []
    crossing_angles: List[float] = []
    all_angles: List[float] = []

    for json_path in sorted(raw_json_dir.glob("*_skeleton.json")):
        annotation = load_json(json_path)
        for group in annotation.get("groups", []):
            points, graph = _group_graph(group)
            if graph.number_of_edges() == 0:
                continue
            for node_id, degree in graph.degree():
                if degree < 2:
                    continue
                center = np.asarray(points[node_id], dtype=np.float32)
                neighbors = list(graph.neighbors(node_id))
                vectors = [_normalize(np.asarray(points[nbr], dtype=np.float32) - center) for nbr in neighbors]
                for idx_a in range(len(vectors)):
                    for idx_b in range(idx_a + 1, len(vectors)):
                        angle = _angle_deg(vectors[idx_a], vectors[idx_b])
                        all_angles.append(angle)
                        if angle >= 120.0:
                            crossing_angles.append(angle)
                        elif 15.0 <= angle <= 120.0:
                            branch_angles.append(angle)

    return {"all_angles": all_angles, "branch_angles": branch_angles, "crossing_angles": crossing_angles}


def summarize_priors(angle_dict: Dict[str, List[float]]) -> Dict[str, float]:
    def mean_std(values: Sequence[float], default_mean: float, default_std: float) -> Tuple[float, float]:
        if not values:
            return float(default_mean), float(default_std)
        arr = np.asarray(values, dtype=np.float32)
        return float(arr.mean()), float(max(arr.std(ddof=0), 1.0))

    branch_mean, branch_std = mean_std(angle_dict["branch_angles"], 46.0, 14.0)
    crossing_mean, crossing_std = mean_std(angle_dict["crossing_angles"], 176.0, 8.0)
    priors = {
        "branching_angle_mean": branch_mean,
        "branching_angle_std": branch_std,
        "crossing_angle_mean": crossing_mean,
        "crossing_angle_std": crossing_std,
        "crossing_radius_ratio_max": 1.7,
        "branching_radius_ratio_min": 1.15,
        "junction_cluster_radius": 12.0,
        "crossing_cost_threshold": 1.35,
        "num_all_angles": float(len(angle_dict["all_angles"])),
        "num_branch_angles": float(len(angle_dict["branch_angles"])),
        "num_crossing_angles": float(len(angle_dict["crossing_angles"])),
    }
    priors["crossing_angle_min"] = max(150.0, priors["crossing_angle_mean"] - 2.5 * priors["crossing_angle_std"])
    priors["branching_angle_low"] = max(15.0, priors["branching_angle_mean"] - 1.5 * priors["branching_angle_std"])
    priors["branching_angle_high"] = min(95.0, priors["branching_angle_mean"] + 1.5 * priors["branching_angle_std"])
    return priors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 GT 统计交汇区角度先验")
    parser.add_argument(
        "--raw-json-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_ROOT / "annotations" / "skeleton_prediction" / "raw_json"),
    )
    parser.add_argument("--prior-output", type=str, default=str(DEFAULT_JUNCTION_PRIOR_PATH))
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-script", type=str, default=str(Path(__file__).resolve().parent / "evaluate_visualize.py"))
    parser.add_argument("--num-samples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_json_dir = Path(args.raw_json_dir)
    angle_dict = collect_gt_angles(raw_json_dir)
    priors = summarize_priors(angle_dict)
    prior_output = Path(args.prior_output)
    save_json(priors, prior_output)

    print(json.dumps({"prior_output": str(prior_output), **priors}, ensure_ascii=False, indent=2))

    if args.run_eval:
        cmd = [
            str(Path(r"D:\app\anna\envs\cherry\python.exe")),
            str(Path(args.eval_script)),
            "--num-samples",
            str(args.num_samples),
            "--junction-prior-path",
            str(prior_output),
        ]
        print("\n[Run Eval]", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
