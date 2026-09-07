from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_MAP = {"retained": 0, "pruned": 1}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_table(frame: pd.DataFrame, schema: dict[str, Any]) -> None:
    missing = [column for column in schema["required_columns"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    invalid_labels = sorted(set(frame["label"].dropna()) - set(schema["label_values"]))
    if invalid_labels:
        raise ValueError(f"Invalid labels: {invalid_labels}")
    invalid_representations = sorted(
        set(frame["representation"].dropna()) - set(schema["representation_values"])
    )
    if invalid_representations:
        raise ValueError(f"Invalid representations: {invalid_representations}")
    duplicated = frame.duplicated(["tree_id", "view", "branch_id", "representation"], keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, ["tree_id", "view", "branch_id", "representation"]]
        raise ValueError(f"Duplicate branch rows: {rows.head(10).to_dict(orient='records')}")
    if frame["tree_id"].isna().any() or frame["branch_id"].isna().any():
        raise ValueError("tree_id and branch_id cannot be missing")
    invalid_rule = ~frame["rule_prediction"].isin([0, 1, True, False])
    if invalid_rule.any():
        raise ValueError("rule_prediction must be frozen binary predictions")


def model_factories(seed: int) -> dict[str, Callable[[], Any]]:
    return {
        "logistic_regression": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "mlp": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        early_stopping=True,
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def splitter_for(groups: np.ndarray, requested_splits: int):
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("At least two trees are required")
    if n_groups < 10:
        return LeaveOneGroupOut(), "leave_one_tree_out"
    n_splits = min(requested_splits, n_groups)
    return GroupKFold(n_splits=n_splits), f"group_kfold_{n_splits}"


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_true, probabilities >= threshold, average="macro", labels=[0, 1], zero_division=0)
        for threshold in thresholds
    ]
    best_score = max(scores)
    candidates = [threshold for threshold, score in zip(thresholds, scores) if score == best_score]
    return float(min(candidates, key=lambda value: abs(value - 0.5)))


def inner_threshold(
    estimator: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return 0.5
    n_splits = min(max(2, requested_splits - 1), len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    probabilities = np.full(len(y), np.nan, dtype=float)
    for train_index, valid_index in splitter.split(x, y, groups):
        if len(np.unique(y[train_index])) < 2:
            return 0.5
        fold_estimator = clone(estimator)
        fold_estimator.fit(x.iloc[train_index], y[train_index])
        probabilities[valid_index] = fold_estimator.predict_proba(x.iloc[valid_index])[:, 1]
    valid = np.isfinite(probabilities)
    if not valid.all() or len(np.unique(y[valid])) < 2:
        return 0.5
    return choose_threshold(y[valid], probabilities[valid])


def metric_values(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    pr_auc = float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None
    return {
        "pr_auc": pr_auc,
        "macro_f1": float(f1_score(y_true, prediction, average="macro", labels=[0, 1], zero_division=0)),
        "precision": float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def available_feature_sets(frame: pd.DataFrame, schema: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    missing_groups: dict[str, list[str]] = {}
    for group_name, columns in schema["feature_groups"].items():
        present = [column for column in columns if column in frame.columns]
        groups[group_name] = present
        missing_groups[group_name] = [column for column in columns if column not in frame.columns]
    feature_sets: dict[str, list[str]] = {}
    all_present = [column for columns in groups.values() for column in columns]
    if not all_present:
        raise ValueError("No schema feature columns are present")
    for ablation_name, removed_groups in schema["ablations"].items():
        columns = [
            column
            for group_name, group_columns in groups.items()
            if group_name not in removed_groups
            for column in group_columns
        ]
        if columns:
            feature_sets[ablation_name] = columns
    return feature_sets, {"present": groups, "missing": missing_groups}


def bootstrap_tree_means(
    tree_metrics: pd.DataFrame,
    seed: int,
    iterations: int,
) -> dict[str, dict[str, float | None]]:
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float | None]] = {}
    for metric in ("pr_auc", "macro_f1", "precision", "recall"):
        values = tree_metrics[metric].dropna().to_numpy(dtype=float)
        if not len(values):
            result[metric] = {"mean": None, "ci95_low": None, "ci95_high": None}
            continue
        means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(iterations)])
        result[metric] = {
            "mean": float(values.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
        }
    return result


def evaluate_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    estimator_factory: Callable[[], Any],
    splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    x = frame[feature_columns]
    y = frame["target"].to_numpy(dtype=int)
    groups = frame["tree_id"].astype(str).to_numpy()
    splitter, splitter_name = splitter_for(groups, splits)
    records: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        if len(np.unique(y[train_index])) < 2:
            raise ValueError(f"Fold {fold_index} training trees contain only one class")
        estimator = estimator_factory()
        threshold = inner_threshold(
            estimator,
            x.iloc[train_index],
            y[train_index],
            groups[train_index],
            splits,
        )
        estimator.fit(x.iloc[train_index], y[train_index])
        probabilities = estimator.predict_proba(x.iloc[test_index])[:, 1]
        fold_metric = metric_values(y[test_index], probabilities, threshold)
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold_index,
                "threshold": threshold,
                "train_trees": len(np.unique(groups[train_index])),
                "test_trees": len(np.unique(groups[test_index])),
                "train_branches": len(train_index),
                "test_branches": len(test_index),
                **fold_metric,
            }
        )
        for row_index, probability in zip(test_index, probabilities):
            records.append(
                {
                    "row_index": int(row_index),
                    "model": model_name,
                    "fold": fold_index,
                    "threshold": threshold,
                    "probability_pruned": float(probability),
                    "prediction": int(probability >= threshold),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(fold_rows), splitter_name


def evaluate_rule(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    probabilities = frame["rule_prediction"].to_numpy(dtype=float)
    records = pd.DataFrame(
        {
            "row_index": np.arange(len(frame), dtype=int),
            "model": "horticultural_rule",
            "fold": 0,
            "threshold": 0.5,
            "probability_pruned": probabilities,
            "prediction": probabilities.astype(int),
        }
    )
    metrics = metric_values(frame["target"].to_numpy(dtype=int), probabilities, 0.5)
    folds = pd.DataFrame(
        [
            {
                "model": "horticultural_rule",
                "fold": 0,
                "threshold": 0.5,
                "train_trees": 0,
                "test_trees": frame["tree_id"].nunique(),
                "train_branches": 0,
                "test_branches": len(frame),
                **metrics,
            }
        ]
    )
    return records, folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe pruning decision baselines")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("feature_schema.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("gt", "pred"), required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = load_schema(args.schema)
    frame = pd.read_csv(args.features)
    validate_table(frame, schema)
    source_counts = Counter(frame["label"].astype(str))
    frame = frame.loc[
        (frame["representation"] == args.representation) & frame["label"].isin(LABEL_MAP)
    ].copy()
    if frame.empty:
        raise ValueError(f"No eligible rows for representation={args.representation}")
    frame["target"] = frame["label"].map(LABEL_MAP).astype(int)
    frame = frame.reset_index(drop=True)
    if frame["tree_id"].nunique() < 2:
        raise ValueError("At least two eligible trees are required")

    feature_sets, feature_availability = available_feature_sets(frame, schema)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[pd.DataFrame] = []
    rule_predictions, rule_folds = evaluate_rule(frame)
    predictions.append(rule_predictions.assign(ablation="full"))
    fold_metrics.append(rule_folds.assign(ablation="full"))
    splitter_names: set[str] = set()
    for ablation_name, feature_columns in feature_sets.items():
        for model_name, factory in model_factories(args.seed).items():
            prediction, folds, splitter_name = evaluate_model(
                frame,
                feature_columns,
                model_name,
                factory,
                args.splits,
            )
            predictions.append(prediction.assign(ablation=ablation_name))
            fold_metrics.append(folds.assign(ablation=ablation_name))
            splitter_names.add(splitter_name)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_frame = pd.concat(fold_metrics, ignore_index=True)
    identity = frame[["tree_id", "view", "branch_id", "label", "target", "representation"]].copy()
    prediction_frame = prediction_frame.merge(identity, left_on="row_index", right_index=True, validate="many_to_one")

    tree_rows: list[dict[str, Any]] = []
    for (model_name, ablation, tree_id), group in prediction_frame.groupby(
        ["model", "ablation", "tree_id"], sort=True
    ):
        metrics = metric_values(
            group["target"].to_numpy(dtype=int),
            group["probability_pruned"].to_numpy(dtype=float),
            float(group["threshold"].iloc[0]),
        )
        tree_rows.append(
            {
                "model": model_name,
                "ablation": ablation,
                "tree_id": tree_id,
                "branches": len(group),
                "pruned": int(group["target"].sum()),
                "retained": int((group["target"] == 0).sum()),
                **metrics,
            }
        )
    tree_frame = pd.DataFrame(tree_rows)

    summaries: list[dict[str, Any]] = []
    for (model_name, ablation), group in tree_frame.groupby(["model", "ablation"], sort=True):
        summaries.append(
            {
                "model": model_name,
                "ablation": ablation,
                "tree_count": int(group["tree_id"].nunique()),
                "branch_count": int(group["branches"].sum()),
                "tree_bootstrap": bootstrap_tree_means(
                    group,
                    args.seed,
                    args.bootstrap_iterations,
                ),
            }
        )
    summary = {
        "status": "completed",
        "representation": args.representation,
        "independent_unit": "tree",
        "splitters": sorted(splitter_names),
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "source_label_counts": dict(sorted(source_counts.items())),
        "eligible_label_counts": dict(sorted(Counter(frame["label"]).items())),
        "tree_count": int(frame["tree_id"].nunique()),
        "branch_count": len(frame),
        "feature_availability": feature_availability,
        "models": summaries,
        "gnn_status": "not_evaluated",
    }

    output = ensure_dir(args.output)
    prediction_frame.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    fold_frame.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    tree_frame.to_csv(output / "tree_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

