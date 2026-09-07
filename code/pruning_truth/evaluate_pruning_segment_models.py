from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ID_COLUMNS = {
    "sample_id",
    "tree_id",
    "view",
    "split",
    "segment_id",
    "candidate_type",
    "group_ids",
    "is_cut_segment",
    "label_mask",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    bud = {
        "bud_count_norm",
        "bud_density_norm",
        "flower_bud_ratio",
        "bud_score_mean",
        "bud_attachment_distance_norm",
        "reliable_direction_ratio",
        "bud_direction_alignment",
        "local_crowding",
        "group_relative_height",
        "group_bud_density",
    }
    topology = {
        "root_depth_norm",
        "neighbor_count",
        "candidate_neighbor_count",
        "neighbor_length_mean_norm",
        "same_group_segment_count_norm",
        "start_degree",
        "end_degree",
    }
    geometry = [
        item
        for item in columns
        if item
        in {
            "length_norm",
            "chord_norm",
            "tortuosity",
            "curvature_mean",
            "curvature_max",
            "mid_x_norm",
            "relative_height",
            "orientation_x",
            "orientation_y",
            "image_edge_distance_norm",
            "roi_edge_distance_norm",
            "start_bud",
            "end_bud",
            "start_junction",
            "end_junction",
            "start_endpoint",
            "end_endpoint",
            "candidate_bud_bud",
            "candidate_junction_bud",
            "candidate_bud_endpoint",
            "candidate_junction_endpoint",
            "candidate_junction_junction",
        }
    ]
    return {
        "full": columns,
        "geometry": geometry,
        "no_bud": [item for item in columns if item not in bud],
        "no_topology": [item for item in columns if item not in topology],
    }


def make_models(seed: int) -> dict[str, tuple[str, Any]]:
    scaled_logistic = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=3000, C=1.0, random_state=seed
                ),
            ),
        ]
    )
    scaled_mlp = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=250,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    n_iter_no_change=18,
                    random_state=seed,
                ),
            ),
        ]
    )
    return {
        "rule_tree": (
            "geometry",
            DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=25, class_weight="balanced", random_state=seed
            ),
        ),
        "logistic_full": ("full", scaled_logistic),
        "rf_full": (
            "full",
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "hgb_full": (
            "full",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "mlp_full": ("full", scaled_mlp),
        "hgb_geometry": (
            "geometry",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "hgb_no_bud": (
            "no_bud",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "hgb_no_topology": (
            "no_topology",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    quantiles = np.linspace(0.01, 0.99, 300)
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, np.quantile(scores, quantiles)])
    best_threshold, best_value = 0.5, -1.0
    for threshold in candidates:
        value = f1_score(y_true, scores >= threshold, average="macro", zero_division=0)
        if value > best_value or (value == best_value and threshold > best_threshold):
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold, best_value


def ranking_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    positive_views = 0
    zero_positive_views = 0
    view_rows = []
    for (tree_id, sample_id), group in frame.groupby(["tree_id", "sample_id"], sort=False):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        positive = np.flatnonzero(ordered["is_cut_segment"].to_numpy(dtype=int) == 1)
        if len(positive) == 0:
            zero_positive_views += 1
            continue
        positive_views += 1
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
        return {
            "positive_views": 0,
            "zero_positive_views": zero_positive_views,
            "tree_macro_success_at_1": float("nan"),
            "tree_macro_success_at_3": float("nan"),
            "tree_macro_mrr": float("nan"),
        }
    view_frame = pd.DataFrame(view_rows)
    tree_frame = view_frame.groupby("tree_id")[["success_at_1", "success_at_3", "mrr"]].mean()
    return {
        "positive_views": positive_views,
        "zero_positive_views": zero_positive_views,
        "tree_macro_success_at_1": float(tree_frame["success_at_1"].mean()),
        "tree_macro_success_at_3": float(tree_frame["success_at_3"].mean()),
        "tree_macro_mrr": float(tree_frame["mrr"].mean()),
    }


def calculate_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    y_true = frame["is_cut_segment"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    per_tree_ap = []
    for _, group in frame.groupby("tree_id"):
        if group["is_cut_segment"].sum() > 0:
            per_tree_ap.append(average_precision_score(group["is_cut_segment"], group["score"]))
    per_view = frame.assign(prediction=predictions).groupby("sample_id").agg(
        false_recommendations=("prediction", lambda values: 0.0),
        recommendations=("prediction", "sum"),
    )
    false_by_view = (
        frame.assign(prediction=predictions)
        .assign(false_positive=lambda item: item.prediction * (1 - item.is_cut_segment))
        .groupby("sample_id")["false_positive"]
        .sum()
    )
    metrics = {
        "n": int(len(frame)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "pooled_pr_auc": float(average_precision_score(y_true, scores)),
        "tree_macro_pr_auc": float(np.mean(per_tree_ap)) if per_tree_ap else float("nan"),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "pruned_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pruned_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "pruned_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, scores)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
        "mean_false_recommendations_per_view": float(false_by_view.mean()),
        "mean_recommendations_per_view": float(per_view["recommendations"].mean()),
    }
    metrics.update(ranking_metrics(frame))
    return metrics


def bootstrap_ap(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    tree_groups = {tree: group for tree, group in frame.groupby("tree_id")}
    trees = np.asarray(list(tree_groups))
    values = []
    for _ in range(repeats):
        draw = rng.choice(trees, size=len(trees), replace=True)
        labels = np.concatenate([tree_groups[item]["is_cut_segment"].to_numpy() for item in draw])
        scores = np.concatenate([tree_groups[item]["score"].to_numpy() for item in draw])
        if labels.sum() > 0:
            values.append(average_precision_score(labels, scores))
    return {
        "repeats": len(values),
        "pooled_pr_auc_ci_low": float(np.quantile(values, 0.025)),
        "pooled_pr_auc_ci_high": float(np.quantile(values, 0.975)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pruning segment baselines")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = args.features if args.features.is_absolute() else PROJECT_ROOT / args.features
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)
    frame = pd.read_csv(feature_dir / "features.csv")
    frame = frame.loc[frame["label_mask"] == 1].reset_index(drop=True)
    schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    columns = list(schema["feature_columns"])
    groups = feature_groups(columns)
    train = frame.loc[frame.split == "train"].copy()
    val = frame.loc[frame.split == "val"].copy()
    test = frame.loc[frame.split == "test"].copy()
    models = make_models(args.seed)
    predictions = []
    metrics = []
    validation_scores = {}
    rules = {}
    for name, (feature_set, estimator) in models.items():
        selected_columns = groups[feature_set]
        fit_frame = train
        if name == "mlp_full":
            positives = train.loc[train.is_cut_segment == 1]
            negatives = train.loc[train.is_cut_segment == 0].sample(
                n=min(len(train) - len(positives), len(positives) * 8), random_state=args.seed
            )
            fit_frame = pd.concat([positives, negatives]).sample(frac=1.0, random_state=args.seed)
        estimator.fit(fit_frame[selected_columns], fit_frame["is_cut_segment"])
        model_dir = ensure_dir(output / "models")
        joblib.dump(
            {"model": estimator, "feature_columns": selected_columns, "feature_set": feature_set},
            model_dir / f"{name}.joblib",
        )
        val_scores = estimator.predict_proba(val[selected_columns])[:, 1]
        threshold, validation_macro_f1 = choose_threshold(
            val["is_cut_segment"].to_numpy(dtype=int), val_scores
        )
        validation_scores[name] = float(average_precision_score(val["is_cut_segment"], val_scores))
        for split_name, split_frame in (("val", val), ("test", test)):
            scored = split_frame[
                ["sample_id", "tree_id", "view", "segment_id", "candidate_type", "is_cut_segment"]
            ].copy()
            scored["model"] = name
            scored["feature_set"] = feature_set
            scored["score"] = estimator.predict_proba(split_frame[selected_columns])[:, 1]
            scored["threshold"] = threshold
            scored["prediction"] = (scored.score >= threshold).astype(int)
            scored["split"] = split_name
            predictions.append(scored)
            result = calculate_metrics(scored, threshold)
            result.update(
                {
                    "model": name,
                    "feature_set": feature_set,
                    "split": split_name,
                    "validation_selected_threshold": threshold,
                    "validation_macro_f1_at_threshold": validation_macro_f1,
                }
            )
            metrics.append(result)
        if name == "rule_tree":
            rules[name] = export_text(estimator, feature_names=selected_columns)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    selected_name = max(validation_scores, key=validation_scores.get)
    selected_test = prediction_frame.loc[
        (prediction_frame.model == selected_name) & (prediction_frame.split == "test")
    ]
    selected_threshold = float(selected_test.threshold.iloc[0])
    bootstrap = bootstrap_ap(selected_test, args.bootstrap, args.seed)
    prediction_frame.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    metric_frame.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    (output / "transparent_rules.txt").write_text(rules.get("rule_tree", ""), encoding="utf-8")
    selection = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "model_evaluation",
        "selection_basis": "highest validation pooled PR-AUC among predeclared baselines",
        "selected_model": selected_name,
        "selected_validation_pr_auc": validation_scores[selected_name],
        "selected_threshold": selected_threshold,
        "validation_pr_auc_by_model": validation_scores,
        "test_tree_cluster_bootstrap": bootstrap,
        "independent_unit": "tree",
        "test_trees": int(test.tree_id.nunique()),
        "test_views": int(test.sample_id.nunique()),
        "test_candidates": int(len(test)),
        "test_positives": int(test.is_cut_segment.sum()),
        "feature_source": str(feature_dir),
        "claim_boundary": (
            "Exploratory automatic-structure pruning-decision baseline; test results do not establish "
            "horticultural equivalence or universal pruning correctness."
        ),
    }
    atomic_json(output / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    print(metric_frame[["model", "split", "pooled_pr_auc", "macro_f1", "pruned_precision", "pruned_recall", "tree_macro_success_at_3"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
