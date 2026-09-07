from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = PROJECT_ROOT / "02_code" / "03_training"
sys.path.insert(0, str(TRAINING_DIR))
from train_pruning_segment_gnn_v2 import (  # noqa: E402
    feature_groups,
    load_split,
    predict,
    train_seed,
)
from evaluate_pruning_segment_models import (  # noqa: E402
    calculate_metrics,
    choose_threshold,
    make_models,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested tree-level CV for frozen pruning models")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=20260809)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = args.features if args.features.is_absolute() else PROJECT_ROOT / args.features
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)
    metadata = pd.read_csv(feature_dir / "features.csv")
    schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    columns = list(schema["feature_columns"])
    indices = list(range(len(columns)))
    tree_ids = np.asarray(sorted(metadata.tree_id.unique()))
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    fold_trees = [set(tree_ids[test_indices]) for _, test_indices in splitter.split(tree_ids)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    fold_metrics = []
    fold_manifest = []
    for fold in range(args.folds):
        test_trees = fold_trees[fold]
        val_trees = fold_trees[(fold + 1) % args.folds]
        train_trees = set(tree_ids) - test_trees - val_trees
        fold_meta = metadata.copy()
        fold_meta["split"] = np.select(
            [fold_meta.tree_id.isin(train_trees), fold_meta.tree_id.isin(val_trees)],
            ["train", "val"],
            default="test",
        )
        fold_manifest.append(
            {
                "fold": fold,
                "train_trees": sorted(train_trees),
                "val_trees": sorted(val_trees),
                "test_trees": sorted(test_trees),
            }
        )
        candidate = fold_meta.loc[fold_meta.label_mask == 1]
        train_frame = candidate.loc[candidate.split == "train"]
        val_frame = candidate.loc[candidate.split == "val"]
        test_frame = candidate.loc[candidate.split == "test"]
        hgb = clone(make_models(args.seeds[0])["hgb_full"][1])
        hgb.fit(train_frame[columns], train_frame.is_cut_segment)
        val_score = hgb.predict_proba(val_frame[columns])[:, 1]
        hgb_threshold, _ = choose_threshold(val_frame.is_cut_segment.to_numpy(), val_score)
        hgb_prediction = test_frame[
            ["sample_id", "tree_id", "view", "segment_id", "candidate_type", "is_cut_segment"]
        ].copy()
        hgb_prediction["score"] = hgb.predict_proba(test_frame[columns])[:, 1]
        hgb_prediction["prediction"] = (hgb_prediction.score >= hgb_threshold).astype(int)
        hgb_prediction["threshold"] = hgb_threshold
        hgb_prediction["model"] = "HGB-full"
        hgb_prediction["outer_fold"] = fold
        predictions.append(hgb_prediction)
        hgb_metrics = calculate_metrics(hgb_prediction, hgb_threshold)
        hgb_metrics.update({"model": "HGB-full", "outer_fold": fold})
        fold_metrics.append(hgb_metrics)
        train_graph, train_rows = load_split(feature_dir, fold_meta, "train", indices, "branch_group")
        val_graph, val_rows = load_split(feature_dir, fold_meta, "val", indices, "branch_group")
        test_graph, test_rows = load_split(feature_dir, fold_meta, "test", indices, "branch_group")
        mean = train_graph["x"].mean(0)
        std = train_graph["x"].std(0).clamp_min(1e-6)
        for graph in (train_graph, val_graph, test_graph):
            graph["x"] = (graph["x"] - mean) / std
            for key in graph:
                graph[key] = graph[key].to(device)
        val_seed_predictions = []
        test_seed_predictions = []
        for seed in args.seeds:
            model, _ = train_seed(
                seed,
                "sage",
                "full",
                columns,
                train_graph,
                val_graph,
                val_rows,
                device,
                64,
                0.25,
                0.003,
                0.0001,
                args.epochs,
                args.patience,
            )
            val_seed_predictions.append(predict(model, val_graph, val_rows).score.to_numpy())
            test_seed_predictions.append(predict(model, test_graph, test_rows).score.to_numpy())
        val_candidates = val_rows.loc[val_rows.label_mask == 1].copy()
        val_candidates["score"] = np.mean(val_seed_predictions, axis=0)
        gnn_threshold, _ = choose_threshold(
            val_candidates.is_cut_segment.to_numpy(), val_candidates.score.to_numpy()
        )
        gnn_prediction = test_rows.loc[test_rows.label_mask == 1, [
            "sample_id", "tree_id", "view", "segment_id", "candidate_type", "is_cut_segment"
        ]].copy()
        gnn_prediction["score"] = np.mean(test_seed_predictions, axis=0)
        gnn_prediction["prediction"] = (gnn_prediction.score >= gnn_threshold).astype(int)
        gnn_prediction["threshold"] = gnn_threshold
        gnn_prediction["model"] = "GraphSAGE-branch"
        gnn_prediction["outer_fold"] = fold
        predictions.append(gnn_prediction)
        gnn_metrics = calculate_metrics(gnn_prediction, gnn_threshold)
        gnn_metrics.update({"model": "GraphSAGE-branch", "outer_fold": fold})
        fold_metrics.append(gnn_metrics)
        print(
            f"fold={fold} HGB_AP={hgb_metrics['pooled_pr_auc']:.5f} "
            f"GNN_AP={gnn_metrics['pooled_pr_auc']:.5f}"
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.DataFrame(fold_metrics)
    prediction_frame.to_csv(output / "out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    metric_frame.to_csv(output / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    atomic_json(output / "fold_manifest.json", fold_manifest)
    summaries = []
    for model, group in metric_frame.groupby("model"):
        model_predictions = prediction_frame.loc[prediction_frame.model == model]
        summaries.append(
            {
                "model": model,
                "folds": args.folds,
                "trees": int(model_predictions.tree_id.nunique()),
                "pooled_out_of_fold_pr_auc": float(
                    average_precision_score(model_predictions.is_cut_segment, model_predictions.score)
                ),
                "fold_pr_auc_mean": float(group.pooled_pr_auc.mean()),
                "fold_pr_auc_std": float(group.pooled_pr_auc.std(ddof=1)),
                "fold_macro_f1_mean": float(group.macro_f1.mean()),
                "fold_macro_f1_std": float(group.macro_f1.std(ddof=1)),
                "fold_pruned_precision_mean": float(group.pruned_precision.mean()),
                "fold_pruned_recall_mean": float(group.pruned_recall.mean()),
                "fold_tree_macro_success_at_3_mean": float(group.tree_macro_success_at_3.mean()),
                "fold_tree_macro_success_at_3_std": float(group.tree_macro_success_at_3.std(ddof=1)),
                "fold_false_recommendations_per_view_mean": float(group.mean_false_recommendations_per_view.mean()),
            }
        )
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "model_evaluation",
        "protocol": "5-fold nested tree-level CV; outer test fold, next fold validation, remaining three folds training",
        "models_frozen_before_cv": ["HGB-full", "GraphSAGE-branch"],
        "gnn_seeds_per_fold": args.seeds,
        "independent_unit": "tree",
        "device": str(device),
        "summaries": summaries,
        "claim_boundary": (
            "Primary robustness comparison after exploratory route selection; all views from a tree remain in one fold."
        ),
    }
    atomic_json(output / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
