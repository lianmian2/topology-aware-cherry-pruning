"""Train a DeepSets set-pooling baseline for atomic pruning-decision segments.

The GraphSAGE segment model adds local neighbor message passing on top of a
per-segment MLP. To isolate that specific contribution, this DeepSets model
replaces local message passing with a permutation-invariant *global* pool over
all segments of a view: each candidate score is a function of its own features
plus a view-level context vector. No edge_index is used.

Model ordering for the ablation ladder (same 5-fold nested tree-level CV as the
cross-validation protocol):

- HGB-full / MLP-full: no shared context at all.
- DeepSets: self features + global view context.
- GraphSAGE-branch: self features + local neighbor messaging (plus implicit
  propagation). This is the only member of the ladder that uses graph structure.

If GraphSAGE improves over DeepSets, the lift is attributable to *local* graph
message passing, not merely to having any view-level context.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = PROJECT_ROOT / "02_code" / "03_training"
EVALUATOR_DIR = PROJECT_ROOT / "02_code" / "05_utils" / "pruning_truth"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(EVALUATOR_DIR))
from evaluate_pruning_segment_models import (  # noqa: E402
    bootstrap_ap,
    calculate_metrics,
    choose_threshold,
)
from train_pruning_segment_gnn_v2 import load_split  # noqa: E402


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def view_id_per_node(rows: pd.DataFrame) -> torch.Tensor:
    """Map each global node index to the index of its view in the graph batch."""
    ids = np.zeros(int(rows["node_index"].max()) + 1, dtype=np.int64)
    view_lookup = {}
    for view, group in rows.groupby("sample_id", sort=False):
        view_id = len(view_lookup)
        view_lookup[view] = view_id
        ids[group["node_index"].to_numpy(dtype=np.int64)] = view_id
    return torch.as_tensor(ids, dtype=torch.long)


class DeepSetsModel(nn.Module):
    """Self MLP + view-global pooled context, no neighbor messaging."""

    def __init__(self, in_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, view_id: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        pooled = torch.zeros_like(hidden)
        pooled.index_add_(0, view_id, hidden)
        counts = torch.zeros(hidden.shape[0], device=x.device, dtype=x.dtype)
        counts.index_add_(0, view_id, torch.ones_like(view_id, dtype=x.dtype))
        context = pooled[view_id] / counts[view_id].clamp_min(1.0).unsqueeze(1)
        context = self.context_proj(context)
        return self.classifier(torch.cat([hidden, context], dim=1)).squeeze(1)


@torch.no_grad()
def predict(
    model: nn.Module,
    graph: dict[str, torch.Tensor],
    rows: pd.DataFrame,
) -> pd.DataFrame:
    model.eval()
    scores = torch.sigmoid(model(graph["x"], graph["view_id"])).cpu().numpy()
    candidates = rows.loc[rows.label_mask == 1].copy()
    candidates["score"] = scores[candidates.node_index.to_numpy(dtype=int)]
    return candidates


def train_seed(
    seed: int,
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    val_rows: pd.DataFrame,
    device: torch.device,
    in_features: int,
    hidden: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    model = DeepSetsModel(in_features, hidden, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_labels = train["y"][train["mask"]]
    positive = float(train_labels.sum().item())
    negative = float(len(train_labels) - positive)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / max(positive, 1.0), device=device)
    )
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train["x"], train["view_id"])
        loss = criterion(logits[train["mask"]], train["y"][train["mask"]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        val_prediction = predict(model, val, val_rows)
        score = average_precision_score(val_prediction.is_cut_segment, val_prediction.score)
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_pr_auc": float(score)})
        if score > best_score + 1e-6:
            best_score = float(score)
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_score,
        "epochs_ran": len(history),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepSets set-pooling baseline (nested tree-level CV)")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260809)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--bootstrap", type=int, default=1000)
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
    histories = []
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
        train_graph, train_rows = load_split(feature_dir, fold_meta, "train", indices, "branch_group")
        val_graph, val_rows = load_split(feature_dir, fold_meta, "val", indices, "branch_group")
        test_graph, test_rows = load_split(feature_dir, fold_meta, "test", indices, "branch_group")
        train_graph["view_id"] = view_id_per_node(train_rows)
        val_graph["view_id"] = view_id_per_node(val_rows)
        test_graph["view_id"] = view_id_per_node(test_rows)
        mean = train_graph["x"].mean(0)
        std = train_graph["x"].std(0).clamp_min(1e-6)
        for graph in (train_graph, val_graph, test_graph):
            graph["x"] = (graph["x"] - mean) / std
            for key in graph:
                graph[key] = graph[key].to(device)
        val_seed_predictions = []
        test_seed_predictions = []
        for seed in args.seeds:
            model, history = train_seed(
                seed,
                train_graph,
                val_graph,
                val_rows,
                device,
                len(columns),
                args.hidden,
                0.25,
                0.003,
                0.0001,
                args.epochs,
                args.patience,
            )
            val_seed_predictions.append(predict(model, val_graph, val_rows).score.to_numpy())
            test_seed_predictions.append(predict(model, test_graph, test_rows).score.to_numpy())
            histories.append(history)
        val_candidates = val_rows.loc[val_rows.label_mask == 1].copy()
        val_candidates["score"] = np.mean(val_seed_predictions, axis=0)
        threshold, _ = choose_threshold(
            val_candidates.is_cut_segment.to_numpy(), val_candidates.score.to_numpy()
        )
        prediction = test_rows.loc[test_rows.label_mask == 1, [
            "sample_id", "tree_id", "view", "segment_id", "candidate_type", "is_cut_segment"
        ]].copy()
        prediction["score"] = np.mean(test_seed_predictions, axis=0)
        prediction["prediction"] = (prediction.score >= threshold).astype(int)
        prediction["threshold"] = threshold
        prediction["model"] = "DeepSets"
        prediction["outer_fold"] = fold
        predictions.append(prediction)
        metrics = calculate_metrics(prediction, threshold)
        metrics.update({"model": "DeepSets", "outer_fold": fold})
        fold_metrics.append(metrics)
        print(
            f"fold={fold} DeepSets_AP={metrics['pooled_pr_auc']:.5f} "
            f"threshold={threshold:.4f}"
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.DataFrame(fold_metrics)
    prediction_frame.to_csv(output / "out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    metric_frame.to_csv(output / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    bootstrap = bootstrap_ap(prediction_frame, args.bootstrap, args.seeds[0])
    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "model_evaluation",
        "model": "DeepSets",
        "protocol": f"{args.folds}-fold nested tree-level CV; outer test fold, next fold validation, remaining folds training",
        "independent_unit": "tree",
        "device": str(device),
        "seeds_per_fold": args.seeds,
        "pooled_out_of_fold_pr_auc": float(
            average_precision_score(prediction_frame.is_cut_segment, prediction_frame.score)
        ),
        "fold_pr_auc_mean": float(metric_frame.pooled_pr_auc.mean()),
        "fold_pr_auc_std": float(metric_frame.pooled_pr_auc.std(ddof=1)),
        "fold_macro_f1_mean": float(metric_frame.macro_f1.mean()),
        "fold_tree_macro_success_at_3_mean": float(metric_frame.tree_macro_success_at_3.mean()),
        "tree_cluster_bootstrap": bootstrap,
        "claim_boundary": (
            "DeepSets set-pooling control: self features + view-global context, no neighbor messaging. "
            "Isolates whether GraphSAGE's local message passing adds signal beyond any shared context."
        ),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "training_history.json", histories)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
