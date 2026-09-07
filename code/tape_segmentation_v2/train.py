import json
import os
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mmdet_data"
OUTPUT_DIR = BASE_DIR / "outputs"
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (512, 512)
LABEL_SMOOTHING = 0.1
NUM_WORKERS = min(4, os.cpu_count() or 1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ReferenceTapeDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.img_dir = data_dir / "img_dir" / split
        self.mask_dir = data_dir / "ann_dir" / split

        self.samples = []
        for img_name in sorted(os.listdir(self.img_dir)):
            img_path = self.img_dir / img_name
            mask_path = self.mask_dir / f"{Path(img_name).stem}.png"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No valid samples found in split '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(IMAGE_SIZE, Image.BILINEAR)
        mask = mask.resize(IMAGE_SIZE, Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, label_smoothing=0.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed_targets = targets

        bce_loss = self.bce(logits, smoothed_targets)

        probs = torch.sigmoid(logits)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score.mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def train_model():
    print(f"Using device: {DEVICE}")
    print(f"Using data dir: {DATA_DIR}")
    print(f"Saving outputs to: {OUTPUT_DIR}")

    train_dataset = ReferenceTapeDataset(DATA_DIR, "train")
    val_dataset = ReferenceTapeDataset(DATA_DIR, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = BCEDiceLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_loss = float("inf")
    metrics_log = []

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train")

        for images, masks in train_loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Val  ")
            for images, masks in val_loop:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        metrics_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            print(f"--> Saved best model with Val Loss: {best_val_loss:.4f}")

    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=4, ensure_ascii=False)

    print(f"Training complete. Metrics saved to '{metrics_path}'.")


if __name__ == "__main__":
    train_model()
