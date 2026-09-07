from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def maximum_match(predictions: list[int], truths: set[int], adjacency: dict[int, set[int]]) -> int:
    options = {
        prediction: [truth for truth in truths if truth == prediction or truth in adjacency.get(prediction, set())]
        for prediction in predictions
    }
    matched: dict[int, int] = {}

    def augment(prediction: int, visited: set[int]) -> bool:
        for truth in options[prediction]:
            if truth in visited:
                continue
            visited.add(truth)
            if truth not in matched or augment(matched[truth], visited):
                matched[truth] = prediction
                return True
        return False

    return sum(augment(prediction, set()) for prediction in predictions)


def graph_adjacency(feature_dir: Path, sample_id: str) -> dict[int, set[int]]:
    tensor = torch.load(feature_dir / "graphs" / sample_id / "graph.pt", map_location="cpu")
    adjacency = {index: set() for index in range(len(tensor["y"]))}
    for source, target in tensor["edge_index"].t().tolist():
        adjacency[int(source)].add(int(target))
    return adjacency


def evaluate_model(
    prediction: pd.DataFrame,
    feature_dir: Path,
    model_name: str,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    view_results = []
    for (tree_id, sample_id), group in prediction.groupby(["tree_id", "sample_id"], sort=False):
        ordered = group.sort_values("score", ascending=False)
        truths = set(ordered.loc[ordered.is_cut_segment == 1, "segment_id"].astype(int))
        if not truths:
            continue
        adjacency = graph_adjacency(feature_dir, sample_id)
        ranked = ordered.segment_id.astype(int).tolist()
        row: dict[str, Any] = {
            "model": model_name,
            "split": split,
            "tree_id": tree_id,
            "sample_id": sample_id,
            "truth_cuts": len(truths),
        }
        for budget in (1, 3):
            selected = ranked[:budget]
            exact = len(set(selected) & truths)
            relaxed = maximum_match(selected, truths, adjacency)
            row[f"exact_hit_at_{budget}"] = float(exact > 0)
            row[f"hop1_hit_at_{budget}"] = float(relaxed > 0)
            row[f"exact_recall_at_{budget}"] = exact / len(truths)
            row[f"hop1_recall_at_{budget}"] = relaxed / len(truths)
        budget = len(truths)
        selected = ranked[:budget]
        exact = len(set(selected) & truths)
        relaxed = maximum_match(selected, truths, adjacency)
        row["budget_exact_coverage"] = exact / budget
        row["budget_hop1_coverage"] = relaxed / budget
        row["budget_exact_any"] = float(exact > 0)
        row["budget_hop1_any"] = float(relaxed > 0)
        view_results.append(row)
    views = pd.DataFrame(view_results)
    metric_columns = [
        column
        for column in views.columns
        if column.startswith("exact_") or column.startswith("hop1_") or column.startswith("budget_")
    ]
    tree_metrics = views.groupby("tree_id")[metric_columns].mean()
    summary = {
        "model": model_name,
        "split": split,
        "trees": int(views.tree_id.nunique()),
        "positive_views": int(len(views)),
        "tree_macro": {column: float(tree_metrics[column].mean()) for column in metric_columns},
        "view_pooled": {column: float(views[column].mean()) for column in metric_columns},
    }
    return views, summary


def load_baseline(path: Path, model: str, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[(frame.model == model) & (frame.split == split)].copy()


def load_gnn_ensemble(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame.split == split]
    keys = ["sample_id", "tree_id", "view", "segment_id", "candidate_type", "is_cut_segment"]
    return frame.groupby(keys, as_index=False).score.mean()


def parse_spec(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected NAME=TYPE=CSV_PATH")
    name, kind, path = parts
    if kind not in {"baseline", "gnn"}:
        raise argparse.ArgumentTypeError("TYPE must be baseline or gnn")
    return name, Path(path), kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranked pruning recommendations")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_spec, required=True)
    parser.add_argument("--baseline-model", default="hgb_full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = args.features if args.features.is_absolute() else PROJECT_ROOT / args.features
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)
    all_views = []
    summaries = []
    for name, path, kind in args.model:
        path = path if path.is_absolute() else PROJECT_ROOT / path
        for split in ("val", "test"):
            prediction = (
                load_baseline(path, args.baseline_model, split)
                if kind == "baseline"
                else load_gnn_ensemble(path, split)
            )
            views, summary = evaluate_model(prediction, feature_dir, name, split)
            all_views.append(views)
            summaries.append(summary)
    pd.concat(all_views, ignore_index=True).to_csv(
        output / "per_view_recommendation_metrics.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "model_evaluation",
        "independent_unit": "tree",
        "strict_match": "predicted atomic segment equals a frozen truth segment",
        "relaxed_match": "one-to-one match to the truth segment itself or an immediately adjacent segment in the original line graph",
        "recommendation_budgets": [1, 3, "number_of_truth_cuts"],
        "summaries": summaries,
        "claim_boundary": (
            "Hop-1 metrics are operational recommendation metrics for local segment ambiguity; "
            "they are reported beside, not instead of, exact PR-AUC and exact classification metrics."
        ),
    }
    atomic_json(output / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
