"""Train region/branch-group level pruning-decision models with tree-level CV.

Reuses the frozen region-level dataset (region_level_v1/region_features.csv) and
the segment-level graph contractions (features_v1/graphs/<view>/graph.pt) to
build a group-level graph. Two models share the same tree-level nested CV
protocol as the atomic decision experiments:

- HGB-region: gradient boosting on the aggregated group features (non-graph).
- GraphSAGE-region: pure-PyTorch GraphSAGE over the contracted group graph, so
  group-level topology messaging can be compared against the plain aggregation.

OOF predictions, fold metrics and a summary are written under the output dir.
Metrics reuse the metric-suite functions so the region bundle matches the atomic
bundle (PR-AUC with random base rate, AUROC, operating points, Success@K with
analytic random baselines, tree-clustered bootstrap CI).
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_DIR = PROJECT_ROOT / "02_code" / "05_utils" / "pruning_truth"
sys.path.insert(0, str(EVALUATOR_DIR))
from evaluate_pruning_segment_models import choose_threshold  # noqa: E402
from evaluate_pruning_metric_suite import evaluate_model, tree_macro_ranking  # noqa: E402


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


def split_groups(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text == "nan":
        return []
    return [item for item in text.split(";") if item and item != "trunk"]


class SageLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, target, x[source])
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(1)
        return self.self_linear(x) + self.neighbor_linear(aggregated)


class GroupGNN(nn.Module):
    def __init__(self, in_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = SageLayer(in_features, hidden)
        self.layer2 = SageLayer(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(torch.relu(self.norm1(self.layer1(x, edge_index))))
        residual = hidden
        hidden = self.dropout(torch.relu(self.norm2(self.layer2(hidden, edge_index))))
        return self.output(hidden + residual).squeeze(1)


def build_group_graph_for_view(
    sample_id: str,
    rows: pd.DataFrame,
    region_features: pd.DataFrame,
    feature_columns: list[str],
    graph_dir: Path,
) -> dict[str, torch.Tensor] | None:
    """Contract a view's segment line graph to a group graph.

    Nodes are the branch groups that contain at least one candidate segment;
    edges connect groups that share a segment-level adjacency or a multi-group
    segment. Node features and labels come from the region-level dataset.
    """
    graph_path = graph_dir / sample_id / "graph.pt"
    if not graph_path.exists():
        return None
    tensor = torch.load(graph_path, map_location="cpu")
    if tensor["x"].shape[0] != len(rows):
        raise ValueError(f"Graph/metadata size mismatch for {sample_id}")
    segment_group: list[list[str]] = []
    for value in rows["group_ids"]:
        segment_group.append(split_groups(value))

    view_regions = region_features.loc[region_features.sample_id == sample_id].copy()
    node_order: list[str] = list(view_regions["group_id"])
    node_index = {group_id: i for i, group_id in enumerate(node_order)}
    if not node_index:
        return None

    edge_set: set[tuple[int, int]] = set()
    for multi in segment_group:
        for i in range(len(multi)):
            for j in range(i + 1, len(multi)):
                if multi[i] in node_index and multi[j] in node_index:
                    a, b = node_index[multi[i]], node_index[multi[j]]
                    edge_set.add((a, b))
                    edge_set.add((b, a))
    edge_index = tensor["edge_index"].long()
    if edge_index.numel() > 0:
        for a_node, b_node in edge_index.t().tolist():
            for group_a in segment_group[a_node]:
                if group_a not in node_index:
                    continue
                for group_b in segment_group[b_node]:
                    if group_b not in node_index:
                        continue
                    if group_a != group_b:
                        a, b = node_index[group_a], node_index[group_b]
                        edge_set.add((a, b))
                        edge_set.add((b, a))

    x = view_regions[feature_columns].to_numpy(dtype=float)
    contracted_edges = (
        torch.as_tensor(sorted(edge_set), dtype=torch.long).t().contiguous()
        if edge_set
        else torch.empty((2, 0), dtype=torch.long)
    )
    return {
        "x": torch.as_tensor(x, dtype=torch.float32),
        "y": torch.as_tensor(view_regions["region_label"].to_numpy(dtype=float)),
        "mask": torch.ones(len(node_order), dtype=torch.bool),
        "edge_index": contracted_edges,
        "group_id": node_order,
        "sample_id": sample_id,
    }


def build_fold_graphs(
    region_features: pd.DataFrame,
    features_csv: pd.DataFrame,
    trees: list[str],
    feature_columns: list[str],
    graph_dir: Path,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    graphs = []
    rows_all = []
    offset = 0
    for tree_id in sorted(trees):
        view_rows = features_csv.loc[features_csv.tree_id == tree_id]
        for sample_id, group in view_rows.groupby("sample_id", sort=False):
            graph = build_group_graph_for_view(
                sample_id, group, region_features, feature_columns, graph_dir
            )
            if graph is None:
                continue
            graph["edge_index"] = graph["edge_index"] + offset
            graphs.append(graph)
            rows_all.append(
                pd.DataFrame(
                    {
                        "node_index": np.arange(len(graph["group_id"])) + offset,
                        "tree_id": tree_id,
                        "sample_id": sample_id,
                        "view": group["view"].iloc[0],
                        "group_id": graph["group_id"],
                        "region_label": graph["y"].numpy(),
                    }
                )
            )
            offset += len(graph["group_id"])
    if not graphs:
        raise ValueError("No group graphs built for fold")
    combined = {
        "x": torch.cat([g["x"] for g in graphs]),
        "y": torch.cat([g["y"] for g in graphs]),
        "mask": torch.cat([g["mask"] for g in graphs]),
        "edge_index": torch.cat([g["edge_index"] for g in graphs], dim=1),
    }
    return combined, pd.concat(rows_all, ignore_index=True)


@torch.no_grad()
def predict_region(model: nn.Module, graph: dict[str, torch.Tensor], rows: pd.DataFrame) -> pd.DataFrame:
    model.eval()
    scores = torch.sigmoid(model(graph["x"], graph["edge_index"])).cpu().numpy()
    out = rows.copy()
    out["score"] = scores[rows["node_index"].to_numpy(dtype=int)]
    return out


def train_region_gnn(
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
    model = GroupGNN(in_features, hidden, dropout).to(device)
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
        logits = model(train["x"], train["edge_index"])
        loss = criterion(logits[train["mask"]], train["y"][train["mask"]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        val_prediction = predict_region(model, val, val_rows)
        labels = val_rows["region_label"].to_numpy(dtype=int)
        from sklearn.metrics import average_precision_score

        score = float(average_precision_score(labels, val_prediction["score"]))
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_pr_auc": score})
        if score > best_score + 1e-6:
            best_score = score
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
    return model, {"seed": seed, "best_epoch": best_epoch, "best_validation_pr_auc": best_score, "epochs_ran": len(history), "history": history}


def region_frame_for_oof(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = rows[["tree_id", "sample_id", "view", "group_id", "region_label", "node_index", "score"]].copy()
    out["is_cut_segment"] = out["region_label"]
    out["threshold"] = threshold
    out["prediction"] = (out["score"] >= threshold).astype(int)
    out = out.rename(columns={"score": "score"})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train region-level pruning-decision models (tree-level CV)")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--atomic-features", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["hgb", "sage"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260809)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    feature_dir = args.features if args.features.is_absolute() else PROJECT_ROOT / args.features
    atomic_dir = args.atomic_features if args.atomic_features.is_absolute() else PROJECT_ROOT / args.atomic_features
    graph_dir = args.graph_dir if args.graph_dir.is_absolute() else PROJECT_ROOT / args.graph_dir
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)

    regions = pd.read_csv(feature_dir / "region_features.csv")
    schema = json.loads((feature_dir / "region_schema.json").read_text(encoding="utf-8"))
    feature_columns = list(schema["feature_columns"])
    atomic = pd.read_csv(atomic_dir / "features.csv")
    forbidden_features = sorted(
        set(feature_columns)
        & {
            "n_positive_segments",
            "region_label",
            "is_cut_segment",
            "normal_pruned_side",
        }
    )
    if forbidden_features:
        raise ValueError(f"Leakage-prone region features: {forbidden_features}")
    if len(feature_columns) != 84:
        raise ValueError(f"Expected 84 leakage-safe region features, found {len(feature_columns)}")
    if regions.duplicated(["sample_id", "group_id"]).any():
        raise ValueError("Duplicate (sample_id, group_id) region keys")

    trees = np.asarray(sorted(regions.tree_id.unique()))
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    fold_assignments = list(splitter.split(trees))
    fold_trees = [set(str(item) for item in trees[test_idx]) for _, test_idx in fold_assignments]
    oof_frames = {model: [] for model in args.models}
    summaries = {model: [] for model in args.models}
    fold_manifest = []

    for fold in range(args.folds):
        test_trees = fold_trees[fold]
        val_trees = fold_trees[(fold + 1) % args.folds]
        train_trees = set(str(item) for item in trees) - test_trees - val_trees
        intersections = {
            "train_val": sorted(train_trees & val_trees),
            "train_test": sorted(train_trees & test_trees),
            "val_test": sorted(val_trees & test_trees),
        }
        if any(intersections.values()):
            raise ValueError(f"Tree leakage in fold {fold}: {intersections}")
        fold_manifest.append(
            {
                "fold": fold,
                "train_trees": sorted(train_trees),
                "validation_trees": sorted(val_trees),
                "test_trees": sorted(test_trees),
                "intersections": intersections,
            }
        )
        train_regions = regions.loc[regions.tree_id.isin(train_trees)].copy()
        val_regions = regions.loc[regions.tree_id.isin(val_trees)].copy()
        test_regions = regions.loc[regions.tree_id.isin(test_trees)].copy()

        if "hgb" in args.models:
            model = HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
                l2_regularization=1.0, class_weight="balanced", random_state=args.seeds[0],
            )
            model.fit(train_regions[feature_columns], train_regions["region_label"])
            val_scores = model.predict_proba(val_regions[feature_columns])[:, 1]
            threshold, _ = choose_threshold(val_regions["region_label"].to_numpy(dtype=int), val_scores)
            test_scores = model.predict_proba(test_regions[feature_columns])[:, 1]
            test_out = test_regions[["tree_id", "sample_id", "view", "group_id", "region_label"]].copy()
            test_out["is_cut_segment"] = test_out["region_label"]
            test_out["score"] = test_scores
            test_out["threshold"] = threshold
            test_out["prediction"] = (test_scores >= threshold).astype(int)
            test_out["model"] = "HGB-region"
            test_out["fold"] = fold
            oof_frames["hgb"].append(test_out)
            summaries["hgb"].append(
                {
                    "fold": fold,
                    "threshold": float(threshold),
                    "test_regions": int(len(test_out)),
                    "test_positive_rate": float(test_out["region_label"].mean()),
                }
            )

        if "sage" in args.models:
            train_graph, train_rows = build_fold_graphs(
                regions, atomic, [str(item) for item in train_trees], feature_columns, graph_dir
            )
            val_graph, val_rows = build_fold_graphs(
                regions, atomic, [str(item) for item in val_trees], feature_columns, graph_dir
            )
            test_graph, test_rows = build_fold_graphs(
                regions, atomic, [str(item) for item in test_trees], feature_columns, graph_dir
            )
            mean = train_graph["x"].mean(0)
            std = train_graph["x"].std(0).clamp_min(1e-6)
            for graph in (train_graph, val_graph, test_graph):
                graph["x"] = (graph["x"] - mean) / std
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for graph in (train_graph, val_graph, test_graph):
                for key in ("x", "y", "mask", "edge_index"):
                    graph[key] = graph[key].to(device)
            seed_predictions = []
            seed_val_predictions = []
            for seed in args.seeds:
                model, history = train_region_gnn(
                    seed, train_graph, val_graph, val_rows, device, len(feature_columns),
                    args.hidden, 0.25, 0.003, 0.0001, args.epochs, args.patience,
                )
                seed_val_predictions.append(predict_region(model, val_graph, val_rows))
                seed_predictions.append(predict_region(model, test_graph, test_rows))
            val_ensembled = (
                pd.concat(seed_val_predictions)
                .groupby(["tree_id", "sample_id", "view", "group_id", "region_label"], as_index=False)["score"]
                .mean()
            )
            val_threshold, _ = choose_threshold(
                val_ensembled["region_label"].to_numpy(dtype=int),
                val_ensembled["score"].to_numpy(),
            )
            ensembled = (
                pd.concat(seed_predictions)
                .groupby(["tree_id", "sample_id", "view", "group_id", "region_label"], as_index=False)["score"]
                .mean()
            )
            ensembled["threshold"] = val_threshold
            ensembled["prediction"] = (ensembled["score"] >= val_threshold).astype(int)
            ensembled["model"] = "GraphSAGE-region"
            ensembled["fold"] = fold
            ensembled["is_cut_segment"] = ensembled["region_label"]
            oof_frames["sage"].append(ensembled)
            summaries["sage"].append(
                {
                    "fold": fold,
                    "threshold": val_threshold,
                    "test_regions": int(len(ensembled)),
                    "test_positive_rate": float(ensembled["region_label"].mean()),
                    "seeds": args.seeds,
                }
            )

    results = {}
    fold_metric_frames = []
    expected_keys = set(zip(regions["sample_id"], regions["group_id"]))
    for model in args.models:
        frame = pd.concat(oof_frames[model], ignore_index=True)
        observed_keys = set(zip(frame["sample_id"], frame["group_id"]))
        if len(frame) != len(regions) or observed_keys != expected_keys:
            raise ValueError(
                f"{model} region coverage mismatch: rows={len(frame)}/{len(regions)}, "
                f"missing={len(expected_keys - observed_keys)}, extra={len(observed_keys - expected_keys)}"
            )
        if frame.duplicated(["sample_id", "group_id"]).any():
            raise ValueError(f"{model} has duplicate OOF region keys")
        frame.to_csv(output / f"oof_predictions_{model}.csv", index=False, encoding="utf-8-sig")
        metric_rows = []
        for fold, group in frame.groupby("fold"):
            metric = evaluate_model(group.reset_index(drop=True), args.bootstrap, args.seeds[0])
            metric["fold"] = int(fold)
            metric_rows.append(metric)
        pooled_metrics = evaluate_model(frame.reset_index(drop=True), args.bootstrap, args.seeds[0])
        pooled_metrics.pop("val_selected_threshold", None)
        pooled_metrics["threshold_policy"] = "validation-selected independently within each outer fold"
        pooled_metrics["fold_thresholds"] = [float(item["threshold"]) for item in summaries[model]]
        pooled_pr_auc = float(average_precision_score(frame["region_label"], frame["score"]))
        pooled_auroc = float(roc_auc_score(frame["region_label"], frame["score"]))
        results[model] = {
            "model": model,
            "n_regions": int(len(frame)),
            "n_positive_regions": int(frame["region_label"].sum()),
            "n_trees": int(frame["tree_id"].nunique()),
            "n_views": int(frame["sample_id"].nunique()),
            "oof_positive_rate": float(frame["region_label"].mean()),
            "pooled_oof_pr_auc": pooled_pr_auc,
            "pooled_oof_auroc": pooled_auroc,
            "fold_mean": {
                "pr_auc": float(np.mean([row["pooled_pr_auc"] for row in metric_rows])),
                "auroc": float(np.mean([row["auroc"] for row in metric_rows])),
            },
            "fold_std": {
                "pr_auc": float(np.std([row["pooled_pr_auc"] for row in metric_rows])),
                "auroc": float(np.std([row["auroc"] for row in metric_rows])),
            },
            "tree_bootstrap_ci": pooled_metrics["tree_bootstrap_ci"],
            "pooled_metrics": pooled_metrics,
            "metrics_by_fold": metric_rows,
            "fold_summaries": summaries[model],
        }
        fold_metric_frames.append(
            pd.DataFrame(metric_rows).assign(model=model)
        )
        print(
            f"{model:>18} pooled OOF PR-AUC={pooled_pr_auc:.4f}, AUROC={pooled_auroc:.4f} "
            f"(fold PR-AUC {results[model]['fold_mean']['pr_auc']:.4f}±"
            f"{results[model]['fold_std']['pr_auc']:.4f})"
        )

    key_coverage_audit = {
        "expected_regions": int(len(regions)),
        "expected_positive_regions": int(regions["region_label"].sum()),
        "expected_trees": int(regions["tree_id"].nunique()),
        "expected_views": int(regions["sample_id"].nunique()),
        "models_have_identical_keys": all(
            set(zip(pd.concat(oof_frames[model])["sample_id"], pd.concat(oof_frames[model])["group_id"]))
            == expected_keys
            for model in args.models
        ),
    }
    split_disjoint_audit = {
        "all_folds_disjoint": all(not any(item["intersections"].values()) for item in fold_manifest),
        "folds": args.folds,
        "tree_count": int(len(trees)),
    }
    atomic_json(output / "fold_manifest.json", {"folds": fold_manifest})
    pd.concat(fold_metric_frames, ignore_index=True).to_csv(
        output / "metrics_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    atomic_json(
        output / "run_config.json",
        {
            "models": args.models,
            "folds": args.folds,
            "split_seed": args.split_seed,
            "seeds": args.seeds,
            "hidden": args.hidden,
            "epochs": args.epochs,
            "patience": args.patience,
            "bootstrap": args.bootstrap,
            "feature_count": len(feature_columns),
            "forbidden_features_present": forbidden_features,
        },
    )
    summary = {
        "schema_version": "2.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "model_evaluation",
        "protocol": f"{args.folds}-fold nested tree-level CV; outer test fold, next fold validation, remaining folds training",
        "independent_unit": "tree",
        "task": "region/branch-group pruning-decision (group contains a truth cut)",
        "base_rate": float(regions["region_label"].mean()),
        "split_disjoint_audit": split_disjoint_audit,
        "key_coverage_audit": key_coverage_audit,
        "results": results,
        "claim_boundary": (
            "Region-level aggregation over frozen atomic features; a region is a branch group within "
            "a view. Metrics characterize learnable region-level pruning signal; not horticultural correctness."
        ),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "audit_summary.json",
        {
            "schema_version": "1.0",
            "split_disjoint_audit": split_disjoint_audit,
            "key_coverage_audit": key_coverage_audit,
            "feature_count": len(feature_columns),
            "forbidden_features_present": forbidden_features,
            "passed": (
                split_disjoint_audit["all_folds_disjoint"]
                and key_coverage_audit["models_have_identical_keys"]
                and len(feature_columns) == 84
                and not forbidden_features
            ),
        },
    )
    print("wrote ->", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
