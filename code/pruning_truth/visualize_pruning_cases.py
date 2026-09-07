"""Case-grid visualization for the pruning-decision OOF ranking.

Renders a 3x3 tile grid over candidate crops (RGB + atomic segment mask from
cnn_candidate_v1) for one OOF model:

- successes: true-positive candidates the model recommended with highest score
- false positives: recommended-but-not-cut candidates with highest score
- missed: truth cuts the model ranked lowest (never recommended)

This is a method-illustration figure (deterministic ordering by score), not a
case-selection made to flatter the method. Tiles show the atomic segment
overlay, the candidate type, the model score and the prediction vs truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MASK_CHANNEL = 3


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_frame(args) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    oof = pd.read_csv(args.oof)
    model_frame = oof.loc[oof.model == args.model].reset_index(drop=True)
    crop_index = pd.read_csv(args.crop_index, dtype={"sample_id": str, "segment_id": int})
    crop_index = crop_index[["crop_index", "sample_id", "segment_id", "candidate_type", "is_cut_segment"]]
    merged = model_frame.merge(
        crop_index, on=["sample_id", "segment_id", "candidate_type", "is_cut_segment"], how="left"
    )
    missing = int(merged["crop_index"].isna().sum())
    if missing:
        raise ValueError(f"{missing} OOF rows had no crop match; aborting to avoid mislabeled tiles")
    merged = merged.sort_values("crop_index").reset_index(drop=True)
    crops = np.load(args.crops, mmap_mode="r")
    rows = merged["crop_index"].to_numpy(dtype=int)
    selected = np.take(crops, rows, axis=0)
    return model_frame, merged, selected


def pick_cases(frame: pd.DataFrame, k: int) -> list[tuple[str, pd.DataFrame]]:
    truth = frame["is_cut_segment"].to_numpy(dtype=int)
    prediction = frame["prediction"].to_numpy(dtype=int)
    score = frame["score"].to_numpy(dtype=float)
    successes = (truth == 1) & (prediction == 1)
    false_pos = (truth == 0) & (prediction == 1)
    missed = (truth == 1) & (prediction == 0)
    return [
        ("success (TP)", frame.loc[successes].sort_values("score", ascending=False).head(k)),
        ("false positive", frame.loc[false_pos].sort_values("score", ascending=False).head(k)),
        ("missed (FN)", frame.loc[missed].sort_values("score", ascending=True).head(k)),
    ]


def render_tile(ax, crop: np.ndarray, title: str, border_color: str) -> None:
    rgb = np.transpose(crop[:3], (1, 2, 0)).astype(float) / 255.0
    mask = crop[MASK_CHANNEL] > 0
    ax.imshow(rgb)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = (0.85, 0.25, 0.15, 0.35)
    ax.imshow(overlay)
    ax.set_title(title, fontsize=7)
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(2.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser(description="Pruning-decision case-grid figure")
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--crop-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="HGB-full")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    oof_path = args.oof if args.oof.is_absolute() else PROJECT_ROOT / args.oof
    crops_path = args.crops if args.crops.is_absolute() else PROJECT_ROOT / args.crops
    index_path = args.crop_index if args.crop_index.is_absolute() else PROJECT_ROOT / args.crop_index
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)

    _, merged, crops = load_frame(
        argparse.Namespace(oof=oof_path, crops=crops_path, crop_index=index_path, model=args.model)
    )
    colors = {"success (TP)": "#2e7d32", "false positive": "#c62828", "missed (FN)": "#ef6c00"}
    categories = pick_cases(merged, args.top_k)
    tiles = [(category, row) for category, cat_frame in categories for _, row in cat_frame.iterrows()]
    total_tiles = len(tiles)
    cols = max(1, min(3, total_tiles))
    rows = int(np.ceil(total_tiles / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (category, row) in zip(axes, tiles):
        crop = crops[int(row["crop_index"])]
        truth = "cut" if row["is_cut_segment"] == 1 else "keep"
        prediction = "cut" if row["prediction"] == 1 else "keep"
        title = (
            f"{row['candidate_type']}\n"
            f"score={row['score']:.3f}  pred={prediction}  truth={truth}\n"
            f"{row['sample_id']} seg{row['segment_id']}"
        )
        render_tile(ax, crop, title, colors[category])

    for ax in axes[total_tiles:]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Pruning-decision cases — {args.model} (OOF, candidate-crop + atomic segment)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure_path = output / f"case_grid_{args.model.replace(' ', '_').lower()}.png"
    fig.savefig(figure_path, dpi=170)
    print(f"wrote -> {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
