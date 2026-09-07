import os
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mmdet_data"
OUTPUT_DIR = BASE_DIR / "outputs"
VIS_DIR = OUTPUT_DIR / "vis"
MODEL_PATH = OUTPUT_DIR / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (512, 512)
PRED_THRESHOLD = 0.5
OVERLAY_ALPHA = 0.35

VIS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    if not MODEL_PATH.exists():
        print(f"Error: Model weights not found at {MODEL_PATH}")
        return None

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def visualize():
    model = load_model()
    if model is None:
        return

    test_img_dir = DATA_DIR / "img_dir" / "test"
    test_mask_dir = DATA_DIR / "ann_dir" / "test"

    if not test_img_dir.exists():
        print(f"No test data found at {test_img_dir}")
        return

    test_images = sorted(os.listdir(test_img_dir))
    print(f"Found {len(test_images)} test images. Generating visualizations...")
    print(f"Using prediction threshold: {PRED_THRESHOLD}")

    for img_name in tqdm(test_images):
        img_path = test_img_dir / img_name
        mask_path = test_mask_dir / f"{Path(img_name).stem}.png"

        original_image = Image.open(img_path).convert("RGB")
        resized_image = original_image.resize(IMAGE_SIZE, Image.BILINEAR)
        img_array = np.array(resized_image, dtype=np.float32) / 255.0

        img_tensor = (
            torch.tensor(img_array, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

        if mask_path.exists():
            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = gt_mask.resize(IMAGE_SIZE, Image.NEAREST)
            gt_array = np.array(gt_mask)
        else:
            gt_array = np.zeros(IMAGE_SIZE, dtype=np.uint8)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > PRED_THRESHOLD).float().cpu().numpy()[0, 0]
            pred_mask = (pred_mask * 255).astype(np.uint8)

        img_vis = (img_array * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        gt_vis = cv2.cvtColor(gt_array, cv2.COLOR_GRAY2BGR)
        pred_vis = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
        overlay_vis = img_vis.copy()
        overlay_vis[pred_mask == 255] = (0, 0, 255)
        overlay_vis = cv2.addWeighted(overlay_vis, OVERLAY_ALPHA, img_vis, 1 - OVERLAY_ALPHA, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_vis, "Original", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(gt_vis, "Ground Truth", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(pred_vis, "Prediction", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay_vis, "Overlay", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        combined = np.hstack((img_vis, gt_vis, pred_vis, overlay_vis))
        save_path = VIS_DIR / img_name
        cv2.imwrite(str(save_path), combined)

    print(f"Visualization complete. Please check '{VIS_DIR}'.")


if __name__ == "__main__":
    visualize()
