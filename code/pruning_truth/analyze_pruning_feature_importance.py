"""Permutation feature importance for the frozen pruning-decision HGB head.

Runs on the frozen feature export (features_v1/features.csv) with the saved
hgb_full decision model (baselines_v1/models/hgb_full.joblib) and reports:

- permutation importance (scoring = average precision) on the held-out test split,
  i.e. trees never used to fit the model
- native contrast: mean positive vs mean negative value per feature (direction)
- feature-group aggregation (geometry / topology / bud) for the mechanism story

HistGradientBoosting has no native feature_importances_, so permutation
importance is the honest global attribution used here. This is an
interpretability companion to the metric suite, not a new predictive result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GROUPS = {
    "geometry": {
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
    },
    "topology": {
        "root_depth_norm",
        "neighbor_count",
        "candidate_neighbor_count",
        "neighbor_length_mean_norm",
        "same_group_segment_count_norm",
        "start_degree",
        "end_degree",
    },
    "bud": {
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
    },
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def mean_contrast(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    """Mean value among true cuts vs candidates, with the base-rate-normalized lift."""
    positive = frame.loc[frame["is_cut_segment"] == 1]
    negative = frame.loc[frame["is_cut_segment"] == 0]
    contrast = {}
    for column in columns:
        pos_mean = float(positive[column].mean())
        neg_mean = float(negative[column].mean())
        scale = max(float(negative[column].std()), 1e-9)
        contrast[column] = {
            "mean_positive": pos_mean,
            "mean_negative": neg_mean,
            "std_negative": float(negative[column].std()),
            "standardized_shift": float((pos_mean - neg_mean) / scale),
        }
    return contrast


def main() -> int:
    parser = argparse.ArgumentParser(description="Permutation feature importance for pruning-decision HGB")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    feature_dir = args.features if args.features.is_absolute() else PROJECT_ROOT / args.features
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)

    frame = pd.read_csv(feature_dir / "features.csv")
    frame = frame.loc[frame["label_mask"] == 1].reset_index(drop=True)
    schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    columns = list(schema["feature_columns"])
    bundle = joblib.load(model_path)
    model = bundle["model"]
    used_columns = list(bundle["feature_columns"])

    test = frame.loc[frame.split == "test"].copy()
    x_test = test[used_columns]
    y_test = test["is_cut_segment"].to_numpy(dtype=int)

    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=args.repeats,
        random_state=args.seed,
        n_jobs=-1,
    )
    rank = np.argsort(result.importances_mean)[::-1]
    importance_rows = []
    for i in rank:
        feature = used_columns[i]
        importance_rows.append(
            {
                "feature": feature,
                "group": next((g for g, feats in GROUPS.items() if feature in feats), "other"),
                "importance_mean": float(result.importances_mean[i]),
                "importance_std": float(result.importances_std[i]),
                "n_repeats": args.repeats,
            }
        )
    importance_frame = pd.DataFrame(importance_rows)
    importance_frame.to_csv(output / "feature_importance_permutation.csv", index=False, encoding="utf-8-sig")

    group_importance = (
        importance_frame.groupby("group")["importance_mean"].sum().sort_values(ascending=False)
    )
    contrast = mean_contrast(test, used_columns)
    top_features = [row["feature"] for row in importance_rows[:12]]

    print(f"{'feature':>34} {'group':>9} {'imp_mean':>10} {'imp_std':>10}  dir(shift)")
    for row in importance_rows:
        shift = contrast[row["feature"]]["standardized_shift"]
        direction = "+" if shift > 0 else "-"
        print(
            f"{row['feature']:>34} {row['group']:>9} {row['importance_mean']:>10.4f} "
            f"{row['importance_std']:>10.4f}   {direction} ({shift:+.2f})"
        )
    print("\ngroup_importance (sum of mean perm-importance):")
    for group, value in group_importance.items():
        print(f"  {group:>10}: {value:.4f}")

    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "feature_interpretation",
        "method": "sklearn permutation importance, scoring=average_precision",
        "model": "hgb_full (frozen, full 39-feature)",
        "data": {
            "test_candidates": int(len(test)),
            "test_positives": int(y_test.sum()),
            "test_trees": int(test.tree_id.nunique()),
            "n_repeats": args.repeats,
        },
        "top_features": top_features,
        "group_importance": {k: float(v) for k, v in group_importance.items()},
        "feature_importance": importance_rows,
        "positive_vs_negative_contrast": contrast,
        "claim_boundary": (
            "Permutation importance is a global attribution on the frozen held-out test split; "
            "it indicates which segment attributes the decision model relies on, not causal pruning "
            "rules or horticultural correctness."
        ),
    }
    atomic_json(output / "feature_importance_permutation.json", payload)
    print(f"\nwrote -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
