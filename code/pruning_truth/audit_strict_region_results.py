"""Regression audit for strict region-level pruning-decision artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {"n_regions": 3333, "n_positive_regions": 442, "n_trees": 118, "n_views": 185}
REQUIRED_RESULT_FIELDS = {
    "pooled_oof_pr_auc",
    "pooled_oof_auroc",
    "fold_mean",
    "fold_std",
    "tree_bootstrap_ci",
    "n_regions",
    "n_positive_regions",
    "n_trees",
    "n_views",
}
FORBIDDEN_FEATURES = {
    "n_positive_segments",
    "positive_segment_count",
    "region_label",
    "is_cut_segment",
    "cut_count",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_oof(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_oof(rows, model):
    keys = [(row["sample_id"], row["group_id"]) for row in rows]
    assert len(rows) == EXPECTED["n_regions"], f"{model}: wrong OOF row count"
    assert len(keys) == len(set(keys)), f"{model}: duplicate (sample_id, group_id)"
    assert sum(int(float(row["region_label"])) for row in rows) == EXPECTED["n_positive_regions"]
    assert len({row["tree_id"] for row in rows}) == EXPECTED["n_trees"]
    assert len({row["sample_id"] for row in rows}) == EXPECTED["n_views"]
    return set(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()

    summary = load_json(args.result_dir / "summary.json")
    audit = load_json(args.result_dir / "audit_summary.json")
    folds = load_json(args.result_dir / "fold_manifest.json")["folds"]
    schema = load_json(args.schema)

    assert summary["split_disjoint_audit"]["all_folds_disjoint"] is True
    assert summary["key_coverage_audit"]["models_have_identical_keys"] is True
    assert audit["passed"] is True
    assert audit["feature_count"] == 84
    assert audit["forbidden_features_present"] == []

    for fold in folds:
        train, val, test = map(
            set, (fold["train_trees"], fold["validation_trees"], fold["test_trees"])
        )
        assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)

    for name, result in summary["results"].items():
        missing = REQUIRED_RESULT_FIELDS - set(result)
        assert not missing, f"{name}: missing result fields {sorted(missing)}"
        for field, expected in EXPECTED.items():
            assert result[field] == expected, f"{name}: {field} mismatch"

    hgb_keys = audit_oof(load_oof(args.result_dir / "oof_predictions_hgb.csv"), "hgb")
    sage_keys = audit_oof(load_oof(args.result_dir / "oof_predictions_sage.csv"), "sage")
    assert hgb_keys == sage_keys, "HGB/GraphSAGE key sets differ"

    features = schema["feature_columns"]
    assert len(features) == 84
    assert not (set(features) & FORBIDDEN_FEATURES)

    print(json.dumps({"passed": True, **EXPECTED, "feature_count": len(features)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
