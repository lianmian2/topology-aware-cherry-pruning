"""Honest metric suite for pruning-decision OOF predictions.

Runs on the frozen 5-fold nested tree-level CV OOF predictions
(tree_cv_v1/out_of_fold_predictions.csv) and produces a paper-facing
metric bundle:

- pooled PR-AUC with the random baseline (positive rate) and relative multiple
- AUROC as a secondary ranking metric (less pessimistic on 1.5% positives)
- precision@recall operating points (e.g. "at 25% recall, precision = X%")
- tree-macro Success@1 / Success@3 / MRR, each with an analytic random baseline
- truth-budget coverage (recommend exactly as many cuts as the ground truth)
- tree-clustered bootstrap 95% CI for pooled PR-AUC and AUROC
- PR / ROC curve data for figure generation

Positive-class precision / recall / F1 are reported (the majority class dominates
macro-F1 and must not be used as the headline).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def random_success_at_k(n_candidates: int, positives: int, k: int) -> float:
    """Probability that >=1 of `positives` true positives lands in a random top-k."""
    if positives <= 0 or n_candidates <= 0:
        return 0.0
    take = min(k, n_candidates)
    if take >= n_candidates:
        return 1.0 if positives > 0 else 0.0
    if take > n_candidates - positives:
        return 1.0
    return 1.0 - comb(n_candidates - positives, take) / comb(n_candidates, take)


def random_mrr(n_candidates: int, positives: int) -> float:
    """Expected 1/rank of the first positive under a uniform random ordering."""
    if positives <= 0 or n_candidates <= 0:
        return 0.0
    return (positives + 1.0) / (n_candidates + 1.0)


def tree_macro_ranking(frame: pd.DataFrame) -> dict[str, float]:
    """Tree-macro Success@1/@3/MRR exactly as ranking_metrics() in the closeout scripts."""
    view_rows = []
    for (tree_id, sample_id), group in frame.groupby(["tree_id", "sample_id"], sort=False):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        positive = np.flatnonzero(ordered["is_cut_segment"].to_numpy(dtype=int) == 1)
        if len(positive) == 0:
            continue
        first_rank = int(positive[0]) + 1
        view_rows.append(
            {
                "tree_id": tree_id,
                "success_at_1": float(first_rank <= 1),
                "success_at_3": float(first_rank <= 3),
                "mrr": 1.0 / first_rank,
            }
        )
    if not view_rows:
        return {"success_at_1": float("nan"), "success_at_3": float("nan"), "mrr": float("nan")}
    view_frame = pd.DataFrame(view_rows)
    tree_frame = view_frame.groupby("tree_id")[["success_at_1", "success_at_3", "mrr"]].mean()
    return {
        "success_at_1": float(tree_frame["success_at_1"].mean()),
        "success_at_3": float(tree_frame["success_at_3"].mean()),
        "mrr": float(tree_frame["mrr"].mean()),
    }


def tree_macro_budget(frame: pd.DataFrame) -> dict[str, float]:
    """Truth-budget coverage: recommend as many cuts as the view's ground truth.

    coverage = true positives found among the top-B / B (B = #true positives).
    any_hit   = 1 if the top-B contains at least one true positive.
    Aggregated as tree macro (mean over trees of view means).
    """
    view_rows = []
    for (tree_id, sample_id), group in frame.groupby(["tree_id", "sample_id"], sort=False):
        labels = group["is_cut_segment"].to_numpy(dtype=int)
        if labels.sum() == 0:
            continue
        budget = int(labels.sum())
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        top = ordered.head(budget)
        hits = int(top["is_cut_segment"].sum())
        view_rows.append(
            {
                "tree_id": tree_id,
                "coverage": hits / budget,
                "any_hit": float(hits > 0),
            }
        )
    if not view_rows:
        return {"budget_coverage": float("nan"), "budget_any_hit": float("nan")}
    view_frame = pd.DataFrame(view_rows)
    tree_frame = view_frame.groupby("tree_id")[["coverage", "any_hit"]].mean()
    return {
        "budget_coverage": float(tree_frame["coverage"].mean()),
        "budget_any_hit": float(tree_frame["any_hit"].mean()),
    }


def operating_points(y_true: np.ndarray, scores: np.ndarray, targets: list[float]) -> dict[str, float]:
    """Precision at target recalls, interpolated on the PR curve.

    precision_recall_curve returns recall sorted by decreasing threshold (recall
    starts at 1.0); np.interp needs an increasing xp, so reverse both arrays.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    recall = recall[::-1]
    precision = precision[::-1]
    result = {}
    for target in targets:
        result[f"precision_at_recall_{int(target * 100):03d}"] = float(
            np.interp(target, recall, precision)
        )
    return result


def random_tree_macro_ranking(frame: pd.DataFrame) -> dict[str, float]:
    """Analytic random baseline for Success@1/@3/MRR, aggregated exactly like the model."""
    view_rows = []
    for (tree_id, sample_id), group in frame.groupby(["tree_id", "sample_id"], sort=False):
        labels = group["is_cut_segment"].to_numpy(dtype=int)
        positives = int(labels.sum())
        n = int(len(group))
        if positives == 0:
            continue
        view_rows.append(
            {
                "tree_id": tree_id,
                "success_at_1": float(random_success_at_k(n, positives, 1)),
                "success_at_3": float(random_success_at_k(n, positives, 3)),
                "mrr": float(random_mrr(n, positives)),
            }
        )
    if not view_rows:
        return {"success_at_1": float("nan"), "success_at_3": float("nan"), "mrr": float("nan")}
    view_frame = pd.DataFrame(view_rows)
    tree_frame = view_frame.groupby("tree_id")[["success_at_1", "success_at_3", "mrr"]].mean()
    return {
        "success_at_1": float(tree_frame["success_at_1"].mean()),
        "success_at_3": float(tree_frame["success_at_3"].mean()),
        "mrr": float(tree_frame["mrr"].mean()),
    }


def tree_bootstrap_ci(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    tree_groups = {tree: group for tree, group in frame.groupby("tree_id")}
    trees = np.asarray(list(tree_groups))
    pr_auc_values = []
    auroc_values = []
    for _ in range(repeats):
        draw = rng.choice(trees, size=len(trees), replace=True)
        labels = np.concatenate([tree_groups[item]["is_cut_segment"].to_numpy() for item in draw])
        scores = np.concatenate([tree_groups[item]["score"].to_numpy() for item in draw])
        if labels.sum() > 0 and len(np.unique(labels)) > 1:
            pr_auc_values.append(average_precision_score(labels, scores))
            auroc_values.append(roc_auc_score(labels, scores))
    return {
        "pooled_pr_auc": [
            float(np.quantile(pr_auc_values, 0.025)),
            float(np.quantile(pr_auc_values, 0.975)),
        ],
        "auroc": [
            float(np.quantile(auroc_values, 0.025)),
            float(np.quantile(auroc_values, 0.975)),
        ],
    }


def evaluate_model(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, Any]:
    y_true = frame["is_cut_segment"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    positive_rate = float(y_true.mean())
    pooled_pr_auc = float(average_precision_score(y_true, scores))
    auroc = float(roc_auc_score(y_true, scores))
    threshold = float(frame["threshold"].iloc[0])
    predictions = frame["prediction"].to_numpy(dtype=int)
    tp = int(((predictions == 1) & (y_true == 1)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    pruned_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    model_ranking = tree_macro_ranking(frame)
    random_ranking = random_tree_macro_ranking(frame)
    budget = tree_macro_budget(frame)
    false_per_view = (
        frame.assign(false_positive=predictions * (1 - y_true))
        .groupby("sample_id")["false_positive"]
        .sum()
        .mean()
    )
    return {
        "n_segments": int(len(frame)),
        "n_positive_segments": int(y_true.sum()),
        "positive_rate": positive_rate,
        "val_selected_threshold": threshold,
        "pooled_pr_auc": pooled_pr_auc,
        "random_pr_auc": positive_rate,
        "pr_auc_multiple_over_random": pooled_pr_auc / positive_rate if positive_rate else None,
        "auroc": auroc,
        "auroc_multiple_over_random": auroc / 0.5,
        "pruned_precision_at_threshold": float(precision),
        "pruned_recall_at_threshold": float(recall),
        "pruned_f1_at_threshold": float(pruned_f1),
        "false_recommendations_per_view": float(false_per_view),
        "success_at_1": model_ranking["success_at_1"],
        "success_at_3": model_ranking["success_at_3"],
        "mrr": model_ranking["mrr"],
        "random_success_at_1": random_ranking["success_at_1"],
        "random_success_at_3": random_ranking["success_at_3"],
        "random_mrr": random_ranking["mrr"],
        "success_at_3_multiple_over_random": (
            model_ranking["success_at_3"] / random_ranking["success_at_3"]
            if random_ranking["success_at_3"] > 0
            else None
        ),
        "mrr_multiple_over_random": (
            model_ranking["mrr"] / random_ranking["mrr"] if random_ranking["mrr"] > 0 else None
        ),
        "budget_coverage": budget["budget_coverage"],
        "budget_any_hit": budget["budget_any_hit"],
        "operating_points": operating_points(y_true, scores, [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]),
        "tree_bootstrap_ci": tree_bootstrap_ci(frame, repeats, seed),
    }


def curve_rows(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    y_true = frame["is_cut_segment"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)

    def align_threshold(curve_len: int, thr: np.ndarray) -> np.ndarray:
        if len(thr) == curve_len:
            return thr
        return np.append(thr, [np.nan] * (curve_len - len(thr)))

    pr = pd.DataFrame(
        {
            "model": model,
            "precision": precision,
            "recall": recall,
            "threshold": align_threshold(len(precision), thresholds),
        }
    )
    roc = pd.DataFrame(
        {
            "model": model,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": align_threshold(len(fpr), roc_thresholds),
        }
    )
    return pr, roc


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper metric suite over frozen OOF pruning predictions")
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--paper-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    oof_path = args.oof if args.oof.is_absolute() else PROJECT_ROOT / args.oof
    summary_path = args.paper_summary if args.paper_summary.is_absolute() else PROJECT_ROOT / args.paper_summary
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)
    oof = pd.read_csv(oof_path)
    paper = json.loads(summary_path.read_text(encoding="utf-8"))
    pr_rows: list[pd.DataFrame] = []
    roc_rows: list[pd.DataFrame] = []
    models = {}
    for model, group in oof.groupby("model", sort=True):
        models[model] = evaluate_model(group.reset_index(drop=True), args.bootstrap, args.seed)
        pr_frame, roc_frame = curve_rows(group.reset_index(drop=True), model)
        pr_rows.append(pr_frame)
        roc_rows.append(roc_frame)
    cnn_test = paper["local_cnn_exploratory"]["test"]
    models["local_CNN_exploratory"] = {
        "note": "Exploratory 64x64 local patch CNN from the frozen fixed split; no full-tree structure.",
        "pooled_pr_auc": cnn_test["pooled_pr_auc"]["mean"],
        "auroc": None,
        "pruned_precision_at_threshold": cnn_test["pruned_precision"]["mean"],
        "pruned_recall_at_threshold": cnn_test["pruned_recall"]["mean"],
        "success_at_3": cnn_test["tree_macro_success_at_3"]["mean"],
        "positive_rate": paper["data"]["positive_segments"] / paper["data"]["candidate_segments"],
    }
    positive_rate = paper["data"]["positive_segments"] / paper["data"]["candidate_segments"]
    graph_sage_pr = models.get("GraphSAGE-branch", {}).get("pooled_pr_auc")
    hgb_pr = models.get("HGB-full", {}).get("pooled_pr_auc")
    comparison = {
        "positive_rate": positive_rate,
        "random_pr_auc": positive_rate,
        "local_cnn_pr_auc": cnn_test["pooled_pr_auc"]["mean"],
        "gnn_vs_cnn_pr_auc_ratio": (
            graph_sage_pr / cnn_test["pooled_pr_auc"]["mean"] if graph_sage_pr else None
        ),
        "hgb_vs_cnn_pr_auc_ratio": hgb_pr / cnn_test["pooled_pr_auc"]["mean"] if hgb_pr else None,
    }
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "model_evaluation",
        "source": str(oof_path),
        "independent_unit": "tree",
        "protocol": "pooled OOF over 5-fold nested tree-level CV; val-selected threshold per fold",
        "base_rate": positive_rate,
        "comparison_vs_local_cnn": comparison,
        "models": models,
        "claim_boundary": (
            "Metrics characterize learnable pruning-decision signal and candidate ranking on "
            "accepted automatic-structure views. They do not establish universal horticultural "
            "correctness; AUROC is a secondary ranking metric reported alongside PR-AUC."
        ),
    }
    atomic_json(output / "metric_suite_summary.json", payload)
    pd.concat(pr_rows, ignore_index=True).to_csv(
        output / "pr_curve_data.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(roc_rows, ignore_index=True).to_csv(
        output / "roc_curve_data.csv", index=False, encoding="utf-8-sig"
    )
    for model, result in models.items():
        if result.get("pooled_pr_auc") is None or result.get("auroc") is None:
            continue
        print(
            f"{model:>28} PR-AUC={result['pooled_pr_auc']:.4f} ({result.get('pr_auc_multiple_over_random', 0):.1f}x random) "
            f"AUROC={result['auroc']:.3f} "
            f"prunedF1={result.get('pruned_f1_at_threshold', 0):.3f} "
            f"S@3={result.get('success_at_3', 0):.3f} "
            f"(random {result.get('random_success_at_3', 0):.3f}, {result.get('success_at_3_multiple_over_random', 0):.1f}x) "
            f"budget_cov={result.get('budget_coverage', 0):.3f} "
            f"false/vw={result.get('false_recommendations_per_view', 0):.2f}"
        )
    print("comparison_vs_local_cnn:", json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
