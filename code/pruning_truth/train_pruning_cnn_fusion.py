"""CNN+GNN fusion: visual patch embeddings concatenated onto GNN node features.

The fair wider-context CNN (cnn_candidate_v2) shows local/group visual crops
alone are weaker than structural models. This fusion tests whether the CNN's
patch embeddings add signal *on top of* the structural node features that the
GraphSAGE model already consumes.

Protocol (identical to the segment cross-validation):
- 5-fold nested tree-level CV, outer test fold, next fold validation, rest train.
- Per fold a CNN is trained ONLY on that fold's train trees, then its
  penultimate-layer embeddings are extracted for every candidate crop.
- Candidate nodes get [structural features | CNN embedding] concatenated;
  non-candidate nodes get zero-filled CNN embeddings (they are not scored but
  participate in message passing).
- GraphSAGE is trained on the augmented node features and scored on the outer
  test fold.

If fused > structural-only GraphSAGE, the wider-context visual patch adds
signal beyond the structural representation; if not, the structural graph
features alone carry the decision signal.
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
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

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
from train_pruning_candidate_cnn_v2 import (  # noqa: E402
    CandidateCNN,
    CropDataset,
    build_cache,
)
from train_pruning_segment_gnn_v2 import (  # noqa: E402
    SegmentGNN,
    load_split,
    predict,
    train_seed,
)


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


def train_cnn_embeddings(
    args: argparse.Namespace,
    array_path: Path,
    split_records: dict[str, pd.DataFrame],
    seed: int,
    device: torch.device,
    embedding_dim: int = 128,
) -> dict[str, np.ndarray]:
    """Train one CNN on the fold's train crops and embed all candidate crops.

    Returns {split: (records, embeddings)} aligned by crop_index.
    """
    set_seed(seed)
    datasets = {split: CropDataset(array_path, records) for split, records in split_records.items()}
    train_labels = split_records["train"].is_cut_segment.to_numpy(dtype=int)
    positive_weight = len(train_labels) / max(2 * train_labels.sum(), 1)
    negative_weight = len(train_labels) / max(2 * (len(train_labels) - train_labels.sum()), 1)
    weights = np.where(train_labels == 1, positive_weight, negative_weight)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True
    )
    model = CandidateCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_score, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for crops, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(crops.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0)
        val_scores = np.zeros(len(split_records["val"]), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for crops, _, indices in val_loader:
                val_scores[indices.numpy()] = torch.sigmoid(model(crops.to(device))).cpu().numpy()
        score = average_precision_score(split_records["val"].is_cut_segment, val_scores)
        if score > best_score + 1e-6:
            best_score = float(score)
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("CNN embedding training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for split, records in split_records.items():
            loader = DataLoader(CropDataset(array_path, records), batch_size=args.batch_size, shuffle=False, num_workers=0)
            embed = np.zeros((len(records), embedding_dim), dtype=np.float32)
            for crops, _, indices in loader:
                hidden = model.features(crops.to(device, non_blocking=True)).squeeze(-1).squeeze(-1)
                embed[indices.numpy()] = hidden.cpu().numpy()
            embeddings[split] = (records, embed)
    return embeddings


def augment_graph_x(
    graph: dict[str, torch.Tensor],
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    embed_records: pd.DataFrame,
    embedding_dim: int,
) -> torch.Tensor:
    """Concatenate CNN embeddings onto node features.

    Candidate nodes receive their crop embedding; non-candidate nodes receive
    zeros (they are message-passing context, not scored).
    """
    augmented = np.zeros((len(rows), graph["x"].shape[1] + embedding_dim), dtype=np.float32)
    augmented[:, : graph["x"].shape[1]] = graph["x"].numpy()
    if len(embed_records) != len(embeddings):
        raise ValueError("embedding length mismatch")
    merged = embed_records.copy()
    merged["embedding"] = list(embeddings)
    joined = rows.merge(merged, on=["sample_id", "segment_id"], how="left", suffixes=("", "_crop"))
    embed_matrix = np.zeros((len(rows), embedding_dim), dtype=np.float32)
    valid = joined["embedding"].notna()
    embed_matrix[valid.to_numpy()] = np.stack(joined.loc[valid, "embedding"].to_numpy())
    augmented[:, graph["x"].shape[1] :] = embed_matrix
    return torch.as_tensor(augmented, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN+GNN fusion under tree-level CV")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=False)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260809)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--task", choices=["segment"], default="segment")
    parser.add_argument("--fov-multiplier", type=float, default=6.0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gnn-epochs", type=int, default=300)
    parser.add_argument("--gnn-patience", type=int, default=35)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = args.features if args.features.is_absolute() else PROJECT_ROOT / args.features
    source_dataset = args.source_dataset if args.source_dataset.is_absolute() else PROJECT_ROOT / args.source_dataset
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)
    schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    columns = list(schema["feature_columns"])
    indices = list(range(len(columns)))
    feature_frame = pd.read_csv(feature_dir / "features.csv")
    candidates = feature_frame.loc[feature_frame.label_mask == 1].reset_index(drop=True)
    cache_dir = args.cache if args.cache is not None else output
    if not args.cache.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    array_path, record_path = build_cache(candidates, source_dataset, cache_dir, args)
    augmented_columns = columns + [f"cnn_embed_{i}" for i in range(128)]
    all_records = pd.read_csv(record_path)
    metadata = pd.read_csv(feature_dir / "features.csv")
    tree_ids = np.asarray(sorted(metadata.tree_id.unique()))

    from sklearn.model_selection import KFold
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    fold_trees = [set(tree_ids[test_indices]) for _, test_indices in splitter.split(tree_ids)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_predictions = []
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
        crop_meta = all_records.copy()
        crop_meta["split"] = np.select(
            [crop_meta.tree_id.isin(train_trees), crop_meta.tree_id.isin(val_trees)],
            ["train", "val"],
            default="test",
        )
        split_records = {split: group.reset_index(drop=True) for split, group in crop_meta.groupby("split", sort=False)}
        print(f"fold={fold} training CNN embeddings on {len(split_records['train'])} train crops")
        embed_data = train_cnn_embeddings(args, array_path, split_records, args.seeds[0], device)

        train_graph, train_rows = load_split(feature_dir, fold_meta, "train", indices, "branch_group")
        val_graph, val_rows = load_split(feature_dir, fold_meta, "val", indices, "branch_group")
        test_graph, test_rows = load_split(feature_dir, fold_meta, "test", indices, "branch_group")
        train_x = augment_graph_x(train_graph, train_rows, embed_data["train"][1], embed_data["train"][0], 128)
        val_x = augment_graph_x(val_graph, val_rows, embed_data["val"][1], embed_data["val"][0], 128)
        test_x = augment_graph_x(test_graph, test_rows, embed_data["test"][1], embed_data["test"][0], 128)
        train_graph["x"] = train_x
        val_graph["x"] = val_x
        test_graph["x"] = test_x
        mean = train_x.mean(0)
        std = train_x.std(0).clamp_min(1e-6)
        for graph in (train_graph, val_graph, test_graph):
            graph["x"] = (graph["x"] - mean) / std
            for key in graph:
                graph[key] = graph[key].to(device)

        val_seed_predictions = []
        test_seed_predictions = []
        for seed in args.seeds:
            model, history = train_seed(
                seed, "sage", "full", augmented_columns, train_graph, val_graph, val_rows, device,
                args.hidden, 0.25, 0.003, 0.0001, args.gnn_epochs, args.gnn_patience,
            )
            val_seed_predictions.append(predict(model, val_graph, val_rows).score.to_numpy())
            test_seed_predictions.append(predict(model, test_graph, test_rows).score.to_numpy())
            histories.append(history)
        val_candidates = val_rows.loc[val_rows.label_mask == 1].copy()
        val_candidates["score"] = np.mean(val_seed_predictions, axis=0)
        threshold, _ = choose_threshold(val_candidates.is_cut_segment.to_numpy(), val_candidates.score.to_numpy())
        prediction = test_rows.loc[test_rows.label_mask == 1, [
            "sample_id", "tree_id", "view", "segment_id", "candidate_type", "is_cut_segment"
        ]].copy()
        prediction["score"] = np.mean(test_seed_predictions, axis=0)
        prediction["prediction"] = (prediction.score >= threshold).astype(int)
        prediction["threshold"] = threshold
        prediction["model"] = "GraphSAGE-CNN-fusion"
        prediction["outer_fold"] = fold
        oof_predictions.append(prediction)
        metrics = calculate_metrics(prediction, threshold)
        metrics.update({"model": "GraphSAGE-CNN-fusion", "outer_fold": fold})
        fold_metrics.append(metrics)
        print(f"fold={fold} Fusion_AP={metrics['pooled_pr_auc']:.5f}")

    prediction_frame = pd.concat(oof_predictions, ignore_index=True)
    metric_frame = pd.DataFrame(fold_metrics)
    prediction_frame.to_csv(output / "out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    metric_frame.to_csv(output / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    pooled_oof_pr_auc = float(average_precision_score(prediction_frame.is_cut_segment, prediction_frame.score))
    bootstrap = bootstrap_ap(prediction_frame, args.bootstrap, args.seeds[0])
    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "model_evaluation",
        "model": "GraphSAGE-CNN-fusion",
        "protocol": f"{args.folds}-fold nested tree-level CV; CNN embeddings trained on outer-train trees only",
        "independent_unit": "tree",
        "device": str(device),
        "pooled_out_of_fold_pr_auc": pooled_oof_pr_auc,
        "fold_pr_auc_mean": float(metric_frame.pooled_pr_auc.mean()),
        "fold_pr_auc_std": float(metric_frame.pooled_pr_auc.std(ddof=1)),
        "tree_cluster_bootstrap": bootstrap,
        "claim_boundary": (
            "CNN patch embeddings concatenated onto GraphSAGE node features; isolates whether wider-context "
            "visual patches add signal beyond structural node features. CNN trained per fold on train trees only."
        ),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "training_history.json", histories)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
