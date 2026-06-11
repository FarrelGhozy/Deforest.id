"""Train U-Net for deforestation segmentation.

Usage:
    # Train with GFW labels (default)
    python scripts/train_unet.py \
        --manifest data/training/unet/manifest.json \
        --output models/unet_deforest_v2 \
        --epochs 50 --batch-size 32 --lr 1e-3

    # Ablation: train with NDVI threshold labels instead
    python scripts/train_unet.py \
        --manifest data/training/unet/manifest.json \
        --output models/unet_deforest_ndvi \
        --epochs 50 --batch-size 32 --lr 1e-3 \
        --label-key label_ndvi
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from rich.live import Live
from rich.table import Table
from rich.console import Console


class RandomFlip:
    """Safe augmentation for satellite imagery: flips image + mask together."""

    def __call__(self, image, mask):
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        return image, mask


class DeforestDataset(Dataset):
    """Load chips + masks from manifest. Optionally caches all data in RAM.

    Args:
        label_key: "mask" (GFW) or "label_ndvi" (NDVI threshold for ablation).
        apply_cloud_mask: set target=255 (ignore) where cloud=1.
        augment: apply random flips during training.
        cache_ram: pre-load all data into memory (eliminates I/O bottleneck).
    """

    def __init__(self, manifest_entries, label_key="mask", apply_cloud_mask=True, augment=False, cache_ram=False):
        self.entries = manifest_entries
        self.label_key = label_key
        self.apply_cloud_mask = apply_cloud_mask
        self.augment = augment
        self.flip = RandomFlip() if augment else None
        self.cache_ram = cache_ram
        self._cache = None

        if cache_ram:
            self._cache = self._build_cache()

    def _build_cache(self):
        cache = []
        for entry in tqdm(self.entries, desc="Caching to RAM"):
            data = np.load(entry["image"])
            rgb = data["rgb"].astype(np.float32) / 255.0
            nir = np.nan_to_num(data["nir"].astype(np.float32) / 10000.0, 0.0)
            ndvi = np.nan_to_num(data["ndvi"].astype(np.float32), 0.0)
            image = np.concatenate([
                rgb,
                nir[np.newaxis, :, :],
                ndvi[np.newaxis, :, :],
            ], axis=0)

            mask_npz = np.load(entry["mask"])
            mask = mask_npz[self.label_key].astype(np.int64)
            if self.apply_cloud_mask and "cloud" in mask_npz:
                cloud = mask_npz["cloud"].astype(bool)
                mask[cloud] = 255

            cache.append((image, mask))
        return cache

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if self._cache is not None:
            image, mask = self._cache[idx]
            image = image.copy()
            mask = mask.copy()
        else:
            entry = self.entries[idx]
            data = np.load(entry["image"])
            rgb = data["rgb"].astype(np.float32) / 255.0
            nir = np.nan_to_num(data["nir"].astype(np.float32) / 10000.0, 0.0)
            ndvi = np.nan_to_num(data["ndvi"].astype(np.float32), 0.0)
            image = np.concatenate([
                rgb,
                nir[np.newaxis, :, :],
                ndvi[np.newaxis, :, :],
            ], axis=0)

            mask_npz = np.load(entry["mask"])
            mask = mask_npz[self.label_key].astype(np.int64)
            if self.apply_cloud_mask and "cloud" in mask_npz:
                cloud = mask_npz["cloud"].astype(bool)
                mask[cloud] = 255

        if self.augment:
            image, mask = self.flip(image, mask)

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
    """Dice loss for imbalanced segmentation (fp32-stable)."""

    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.float()
        mask = targets != self.ignore_index
        if not mask.any():
            return logits.sum() * 0

        targets = targets.clone()
        targets[~mask] = 0

        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        targets_onehot = targets_onehot * mask.unsqueeze(1).float()

        loss = 0
        for c in range(logits.shape[1]):
            intersection = (probs[:, c] * targets_onehot[:, c]).sum()
            denom = probs[:, c].sum() + targets_onehot[:, c].sum()
            loss += 1 - (2.0 * intersection + self.smooth) / (denom + self.smooth)

        return loss / logits.shape[1]


def train_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=False, clip_norm=1.0):
    model.train()
    total_loss = 0
    total_norm = 0
    for images, masks in tqdm(loader, desc="Train"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        total_loss += loss.item()
        total_norm += gn.item()

    return total_loss / len(loader), total_norm / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    tp, fp, fn = 0, 0, 0
    correct, total_px = 0, 0
    for images, masks in tqdm(loader, desc="Val"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        valid = masks != 255
        tp += ((preds == 1) & (masks == 1) & valid).sum().item()
        fp += ((preds == 1) & (masks == 0) & valid).sum().item()
        fn += ((preds == 0) & (masks == 1) & valid).sum().item()
        correct += ((preds == masks) & valid).sum().item()
        total_px += valid.sum().item()

    iou = tp / (tp + fp + fn + 1e-6)
    tn = total_px - tp - fp - fn
    iou_bg = tn / (tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    acc = correct / (total_px + 1e-6)
    return total_loss / len(loader), {
        "IoU": iou, "IoU_bg": iou_bg, "Precision": precision,
        "Recall": recall, "F1": f1, "Accuracy": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="models/unet_deforest")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--label-key", default="mask",
                        choices=["mask", "label_ndvi"],
                        help="Label source: 'mask' (GFW) or 'label_ndvi' (NDVI ablation)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-ram", action="store_true",
                        help="Pre-load all data into RAM (eliminates I/O bottleneck)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log metrics to Weights & Biases")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (safe with --cache-ram)")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    train_dataset = DeforestDataset(manifest["train"], label_key=args.label_key, augment=True, cache_ram=args.cache_ram)
    val_dataset = DeforestDataset(manifest["val"], label_key=args.label_key, augment=False, cache_ram=args.cache_ram)

    if args.wandb:
        import wandb
        wandb.init(project="deforest-id", name=Path(args.output).name, config=vars(args))
        wandb.config.arch = "SimpleUNet"
        wandb.config.base_filters = 64
        wandb.config.loss = "DiceLoss(ignore_index=255)"
        wandb.config.optimizer = "AdamW"
        wandb.config.scheduler = "CosineAnnealingLR"
        wandb.config.input_channels = 5
        wandb.config.input_bands = "rgb+nir+ndvi"
        wandb.config.dataset_size = len(train_dataset)
        wandb.config.val_size = len(val_dataset)

    # Oversample positive chips (has deforestation) to combat 0.38% class imbalance
    train_weights = []
    for e in tqdm(manifest["train"], desc="Computing sampler weights"):
        m = np.load(e["mask"])
        has_deforest = m[args.label_key].sum() > 0
        train_weights.append(3.0 if has_deforest else 1.0)
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4 if args.num_workers > 0 else None)

    pos_pct = sum(1 for w in train_weights if w > 1) / len(train_weights) * 100
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    print(f"Device: {device}")
    print(f"Label key: {args.label_key}")
    print(f"RAM cache: {args.cache_ram}")
    print(f"Chips with deforestation: {pos_pct:.1f}% (oversampled 3x)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SimpleUNet(in_channels=5, out_channels=2, base_filters=64).to(device)

    if device.type == "cuda":
        try:
            model = torch.compile(model, mode="default")
            print("torch.compile enabled")
        except Exception:
            pass

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    if args.wandb:
        wandb.config.amp = use_amp
        wandb.config.compile = hasattr(model, "_orig_mod")
        wandb.config.tf32 = device.type == "cuda"
        wandb.watch(model, log="all", log_freq=100, log_graph=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceLoss(ignore_index=255)

    log_path = output_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "IoU", "IoU_bg", "Precision", "Recall", "F1", "Accuracy", "grad_norm", "lr"])

    console = Console()
    best_iou = 0
    log_images_freq = 25  # log sample predictions every N epochs
    table = Table(title=f"Training — {args.label_key}", expand=True)
    table.add_column("Epoch", justify="center")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("IoU", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Acc", justify="right")
    table.add_column("LR", justify="right")
    table.add_column("Best IoU", justify="center")

    with Live(table, console=console, refresh_per_second=0.5, vertical_overflow="visible") as live:
        for epoch in range(1, args.epochs + 1):
            train_loss, grad_norm = train_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp, clip_norm=1.0)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()

            lr = scheduler.get_last_lr()[0]
            is_best = val_metrics["IoU"] > best_iou
            if is_best:
                best_iou = val_metrics["IoU"]
                torch.save(model.state_dict(), output_dir / "best.pth")

            if args.wandb:
                log_dict = {
                    "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "IoU": val_metrics["IoU"], "IoU_bg": val_metrics["IoU_bg"],
                    "F1": val_metrics["F1"], "Precision": val_metrics["Precision"],
                    "Recall": val_metrics["Recall"], "Accuracy": val_metrics["Accuracy"],
                    "grad_norm": grad_norm, "lr": lr,
                }
                if epoch % log_images_freq == 0 or epoch == 1:
                    model.eval()
                    sample_images, sample_masks = next(iter(val_loader))
                    imgs = []
                    gts = []
                    preds = []
                    for i in range(min(4, len(sample_images))):
                        img_rgb = sample_images[i, :3]
                        gt = sample_masks[i].float().unsqueeze(0)
                        pred = model(sample_images[i:i+1].to(device)).argmax(dim=1).cpu()[0].float().unsqueeze(0)
                        imgs.append(wandb.Image(img_rgb, caption=f"RGB @ epoch {epoch}"))
                        gts.append(wandb.Image(gt, caption=f"GT @ epoch {epoch}"))
                        preds.append(wandb.Image(pred, caption=f"Pred @ epoch {epoch}"))
                    log_dict["sample_rgb"] = imgs
                    log_dict["sample_gt"] = gts
                    log_dict["sample_pred"] = preds
                wandb.log(log_dict)

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                f"{val_metrics['IoU']:.6f}", f"{val_metrics['IoU_bg']:.6f}",
                                f"{val_metrics['Precision']:.6f}", f"{val_metrics['Recall']:.6f}",
                                f"{val_metrics['F1']:.6f}", f"{val_metrics['Accuracy']:.6f}",
                                f"{grad_norm:.4f}", f"{lr:.2e}"])

            marker = "★" if is_best else ""
            table.add_row(
                f"{epoch}/{args.epochs}",
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_metrics['IoU']:.4f}",
                f"{val_metrics['F1']:.4f}",
                f"{val_metrics['Precision']:.4f}",
                f"{val_metrics['Recall']:.4f}",
                f"{val_metrics['Accuracy']:.4f}",
                f"{lr:.2e}",
                f"{best_iou:.4f} {marker}",
            )

    torch.save(model.state_dict(), output_dir / "final.pth")
    console.print(f"\n[bold green]Done![/] Best IoU: {best_iou:.4f}")
    console.print(f"Model saved to [bold]{output_dir}[/]")
    console.print(f"Log: [bold]{log_path}[/]")


if __name__ == "__main__":
    main()
