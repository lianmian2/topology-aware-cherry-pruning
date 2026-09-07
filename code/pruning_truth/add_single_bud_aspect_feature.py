from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pruning_decision_features import assign_buds, read_json  # noqa: E402


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add one leakage-safe single-bud aspect-ratio feature to frozen pruning features"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aspect_ratio_by_segment(graph_path: Path) -> dict[tuple[str, int], float]:
    graph = read_json(graph_path)
    sample_id = str(graph["sample_id"])
    source = Path(graph["source"]["attachments"])
    attachments = read_json(source)
    directions_path = source.parent / "bud_directions.json"
    directions = read_json(directions_path) if directions_path.exists() else []
    buds_by_segment = assign_buds(graph["segment_nodes"], attachments, directions)
    output: dict[tuple[str, int], float] = {}
    for segment in graph["segment_nodes"]:
        segment_id = int(segment["id"])
        values = []
        for bud in buds_by_segment.get(segment_id, []):
            if int(bud.get("bud_label", -1)) != 1:
                continue
            direction = bud.get("direction") or {}
            value = float(direction.get("aspect_ratio", 1.0))
            if np.isfinite(value) and value >= 1.0:
                values.append(value)
        output[(sample_id, segment_id)] = float(np.mean(values)) if values else 1.0
    return output


def main() -> int:
    args = parse_args()
    feature_dir = resolve_path(args.features)
    graph_dir = resolve_path(args.graphs)
    output_dir = ensure_dir(resolve_path(args.output))
    frame = pd.read_csv(feature_dir / "features.csv")
    schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    feature_name = "single_bud_aspect_ratio_mean"
    if feature_name in frame.columns or feature_name in schema["feature_columns"]:
        raise ValueError(f"Feature already exists: {feature_name}")

    values: dict[tuple[str, int], float] = {}
    graph_count = 0
    for graph_path in sorted(graph_dir.glob("*/decision_graph.json")):
        values.update(aspect_ratio_by_segment(graph_path))
        graph_count += 1

    keys = list(zip(frame["sample_id"].astype(str), frame["segment_id"].astype(int)))
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"Missing aspect-ratio values for {len(missing)} feature rows")
    frame[feature_name] = [values[key] for key in keys]
    schema["feature_columns"] = list(schema["feature_columns"]) + [feature_name]
    frame.to_csv(output_dir / "features.csv", index=False, encoding="utf-8-sig")
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = Counter(value == 1.0 for value in values.values())
    audit = {
        "schema_version": "1.0",
        "source_feature_dir": str(feature_dir),
        "source_graph_dir": str(graph_dir),
        "output_feature_dir": str(output_dir),
        "feature_name": feature_name,
        "base_feature_count": len(schema["feature_columns"]) - 1,
        "output_feature_count": len(schema["feature_columns"]),
        "graph_count": graph_count,
        "feature_rows": int(len(frame)),
        "rows_with_neutral_no_single_bud_value": int(counts[True]),
        "rows_with_non_neutral_single_bud_value": int(counts[False]),
        "neutral_value": 1.0,
        "claim_boundary": (
            "Exploratory one-feature ablation. The added value is derived from predicted bud masks and "
            "directions; it does not alter labels, splits, raw annotations, or the frozen 39-feature result."
        ),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
