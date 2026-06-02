"""Train U-Net for deforestation segmentation.

Usage:
    python scripts/train_unet.py \
        --manifest services/gee-export/src/data/training/unet/manifest.json \
        --output models/unet_deforest \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3
"""

import argparse
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DeforestDataset(Dataset):
    """Load chips + masks from manifest."""

    def __init__(self, manifest_entries, transforms=None):
        self.entries = manifest_entries
        self.transforms = transforms

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        data = np.load(entry["image"])
        rgb = data["rgb"].astype(np.float32) / 255.0  # (3, 64, 64)
        nir = data["nir"]                              # (64, 64)
        ndvi = data["ndvi"]                            # (64, 64)

        # Stack input: RGB + NIR + NDVI = 5 channels
        image = np.concatenate([
            rgb,
            nir[np.newaxis, :, :],
            ndvi[np.newaxis, :, :],
        ], axis=0)  # (5, 64, 64)

        mask = np.load(entry["mask"]).astype(np.int64)  # (64, 64)

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask),
        )


class SimpleUNet(nn.Module):
    """Lightweight U-Net for 5-channel input, 2-class output."""

    def __init__(self, in_channels=5, out_channels=2, base_filters=32):
        super().__init__()
        self.enc1 = self._block(in_channels, base_filters)
        self.enc2 = self._block(base_filters, base_filters * 2)
        self.enc3 = self._block(base_filters * 2, base_filters * 4)
        self.enc4 = self._block(base_filters * 4, base_filters * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._block(base_filters * 8, base_filters * 16)

        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, 2)
        self.dec4 = self._block(base_filters * 16, base_filters * 8)
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, 2)
        self.dec3 = self._block(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, 2)
        self.dec2 = self._block(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, 2)
        self.dec1 = self._block(base_filters * 2, base_filters)

        self.out = nn.Conv2d(base_filters, out_channels, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


class DiceLoss(nn.Module):
    """Dice loss for imbalanced segmentation."""

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        mask = targets != self.ignore_index
        if not mask.any():
            return logits.sum() * 0

        targets = targets.clone()
        targets[~mask] = 0

        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        targets_onehot = targets_onehot * mask.unsqueeze(1).float()

        smooth = 1.0
        loss = 0
        for c in range(logits.shape[1]):
            intersection = (probs[:, c] * targets_onehot[:, c]).sum()
            denom = probs[:, c].sum() + targets_onehot[:, c].sum()
            loss += 1 - (2.0 * intersection + smooth) / (denom + smooth)

        return loss / logits.shape[1]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Train"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    ious = []
    for images, masks in tqdm(loader, desc="Val"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        # IoU for deforest class (class 1)
        preds = logits.argmax(dim=1)
        valid = masks != 255
        intersection = ((preds == 1) & (masks == 1) & valid).sum().float()
        union = ((preds == 1) | (masks == 1)) & valid
        union = union.sum().float()
        iou = (intersection / (union + 1e-6)).item() if union > 0 else 0
        ious.append(iou)

    return total_loss / len(loader), np.mean(ious)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="models/unet_deforest")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    train_dataset = DeforestDataset(manifest["train"])
    val_dataset = DeforestDataset(manifest["val"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    print(f"Device: {device}")

    model = SimpleUNet(in_channels=5, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceLoss(ignore_index=255)

    best_iou = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU (deforest): {val_iou:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), output_dir / "best.pth")
            print(f"  → New best model saved (IoU: {val_iou:.4f})")

    torch.save(model.state_dict(), output_dir / "final.pth")
    print(f"\nDone. Best IoU: {best_iou:.4f}")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
