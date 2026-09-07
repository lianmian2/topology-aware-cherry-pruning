"""Build the region/branch-group level pruning-decision dataset.

Aggregates the frozen atomic-segment features (features_v1/features.csv) up to
the branch-group level. A region is one branch group within one view. The region
label is 1 if the group contains at least one ground-truth cut segment, which is
the natural horticultural unit: a pruning cut removes a whole branch at its base,
not an isolated atomic segment.

Region features are the mean/max of the 39 atomic features over the group's
candidate segments, plus the group size, plus candidate-type composition counts.
Tree membership is preserved so every downstream model can reuse the tree-level
nested CV protocol with zero leakage.

Writes region_features.csv, region_schema.json and region_summary.json under the
output directory. Never writes into the frozen dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GROUP_DELIMITERS = re.compile(r"[;,|]")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def split_group_ids(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text == "nan":
        return []
    return [item for item in GROUP_DELIMITERS.split(text) if item and item != "trunk"]


def build_region_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Expand multi-group segments and aggregate per (tree_id, view, group_id)."""
    rows = []
    expanded = []
    for _, row in frame.iterrows():
        groups = split_group_ids(row["group_ids"])
        if not groups:
            continue
        for group_id in groups:
            item = row.copy()
            item["group_id"] = group_id
            expanded.append(item)
    expanded_frame = pd.DataFrame(expanded)

    for (tree_id, view, group_id), group in expanded_frame.groupby(
        ["tree_id", "view", "group_id"], sort=False
    ):
        label = int(group["is_cut_segment"].max())
        record = {
            "tree_id": tree_id,
            "view": view,
            "sample_id": group["sample_id"].iloc[0],
            "group_id": group_id,
            "split": group["split"].iloc[0],
            "region_label": label,
            "n_segments": int(len(group)),
            "n_positive_segments": int(group["is_cut_segment"].sum()),
        }
        for column in feature_columns:
            values = group[column].to_numpy(dtype=float)
            record[f"{column}_mean"] = float(np.mean(values))
            record[f"{column}_max"] = float(np.max(values))
        candidate_counts = group["candidate_type"].value_counts()
        for candidate_type in ("junction--bud", "bud--bud", "junction--junction", "bud--endpoint",
                               "junction--endpoint"):
            record[f"cand_{candidate_type.replace('--', '_')}"] = int(
                candidate_counts.get(candidate_type, 0)
            )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build region-level pruning-decision dataset")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feature_dir = args.features if args.features.is_absolute() else PROJECT_ROOT / args.features
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)

    frame = pd.read_csv(feature_dir / "features.csv")
    frame = frame.loc[frame["label_mask"] == 1].reset_index(drop=True)
    schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    feature_columns = list(schema["feature_columns"])
    regions = build_region_frame(frame, feature_columns)

    id_columns = ["tree_id", "view", "sample_id", "group_id", "split", "region_label"]
    label_derived = {"n_positive_segments"}
    feature_columns_out = [
        column
        for column in regions.columns
        if column not in id_columns and column not in label_derived
    ]
    schema_out = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "dataset_build",
        "source": str(feature_dir),
        "unit": "branch group within view (tree_id, view, group_id)",
        "label": "region_label = 1 if group contains >=1 truth cut segment",
        "id_columns": id_columns,
        "feature_columns": feature_columns_out,
        "atomic_feature_count": len(feature_columns),
    }
    regions.to_csv(output / "region_features.csv", index=False, encoding="utf-8-sig")
    atomic_json(output / "region_schema.json", schema_out)

    positive_rate = float(regions["region_label"].mean())
    view_key = regions["tree_id"].astype(str) + "_" + regions["view"]
    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "dataset_summary",
        "regions": int(len(regions)),
        "trees": int(regions.tree_id.nunique()),
        "views": int(view_key.nunique()),
        "positive_regions": int(regions["region_label"].sum()),
        "positive_rate": positive_rate,
        "segments_expanded": int(frame.shape[0]),
        "regions_per_view_mean": float(regions.groupby(view_key).size().mean()),
        "claim_boundary": (
            "Region-level aggregation over frozen atomic features; label = group contains a truth "
            "cut, which is the horticultural pruning unit. Not botanical correctness."
        ),
    }
    atomic_json(output / "region_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("region schema feature count:", len(feature_columns_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
