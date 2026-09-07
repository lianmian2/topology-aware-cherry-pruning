from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def resolve_path(path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return path if path.is_absolute() else project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a derived feature to frozen graph tensors")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = resolve_path(args.features)
    frame = pd.read_csv(feature_dir / "features.csv")
    if args.feature_name not in frame.columns:
        raise ValueError(f"Feature missing from CSV: {args.feature_name}")
    for sample_id, sample_rows in frame.groupby("sample_id", sort=False):
        graph_path = feature_dir / "graphs" / str(sample_id) / "graph.pt"
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        graph = torch.load(graph_path, map_location="cpu")
        values = torch.as_tensor(sample_rows[args.feature_name].to_numpy(dtype="float32"))
        if values.shape[0] != graph["x"].shape[0]:
            raise ValueError(f"Metadata/tensor size mismatch: {sample_id}")
        if graph["x"].shape[1] != 39:
            raise ValueError(f"Expected 39 base tensor features, found {graph['x'].shape[1]}")
        graph["x"] = torch.cat([graph["x"].float(), values[:, None]], dim=1)
        torch.save(graph, graph_path)
    print(f"updated_graphs={frame.sample_id.nunique()} feature_width=40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
