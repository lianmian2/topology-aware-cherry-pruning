import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Replace only the original-image panel, keeping exact matplotlib styling")
    parser.add_argument("--vis", type=str, required=True, help="Path to the existing 4-panel visualization PNG")
    parser.add_argument("--new-image", type=str, required=True, help="Path to the new original image")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.vis)
        args.output = f"{base}_new_orig{ext}"

    vis_img = cv2.imread(args.vis)
    if vis_img is None:
        raise FileNotFoundError(f"Cannot read visualization: {args.vis}")
    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    new_orig_bgr = cv2.imread(args.new_image)
    if new_orig_bgr is None:
        raise FileNotFoundError(f"Cannot read new image: {args.new_image}")
    new_orig_rgb = cv2.cvtColor(new_orig_bgr, cv2.COLOR_BGR2RGB)

    h, w = vis_rgb.shape[:2]
    panel_w = w // 4

    panels = [
        vis_rgb[:, i * panel_w : (i + 1) * panel_w] for i in range(4)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(new_orig_rgb)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(panels[1])
    axes[1].set_title("Ground Truth", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(panels[2])
    axes[2].set_title("Prediction", fontsize=14)
    axes[2].axis("off")

    axes[3].imshow(panels[3])
    axes[3].set_title("Overlay (Green=GT, Red=Pred)", fontsize=14)
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
