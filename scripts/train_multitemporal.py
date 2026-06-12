"""
Multi-temporal U-Net with ResNet34 encoder for deforestation segmentation.

Usage:
    python scripts/train_multitemporal.py \
        --manifest data/training/unet/manifest.json \
        --chips-dir data/training/unet/chips \
        --output models/deforest_multitemporal \
        --epochs 150 --batch-size 64 --lr 1e-4
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
from torchvision import models
from tqdm import tqdm
from rich.live import Live
from rich.table import Table
from rich.console import Console


class CorruptedFile(Exception):
    pass


# ─── Datasets ────────────────────────────────────────────────────────────────

class MultiTemporalDeforestDataset(Dataset):
    """Loads paired pre (baseline) + post (deforest) chips + GFW mask.
    Each chip is 64x64 with 5 bands (rgb+nir+ndvi). Stacked -> 10 channels.
    Only chips with BOTH pre and post available are kept.
    Corrupted files are filtered during init.
    """

    def __init__(self, manifest_entries, chips_dir, apply_cloud_mask=True,
                 augment=False, cache_ram=False):
        self.chips_dir = Path(chips_dir)
        self.apply_cloud_mask = apply_cloud_mask
        self.augment = augment

        self.pairs = []
        for entry in tqdm(manifest_entries, desc="Linking pre/post chips"):
            pre_path = Path(entry["image"])
            stem_pre = pre_path.stem
            parts = stem_pre.rsplit("_", 2)
            if len(parts) < 3:
                continue
            for suffix in ["baseline", "deforest"]:
                if suffix in stem_pre:
                    post_stem = stem_pre.replace(suffix, "deforest" if suffix == "baseline" else "baseline")
                    break
            else:
                continue
            post_path = self.chips_dir / f"{post_stem}.npz"
            if not post_path.exists():
                continue
            self.pairs.append({
                "pre": str(pre_path),
                "post": str(post_path),
                "mask": str(entry["mask"]),
            })

        self.pos_weights = None

        self._cache = None
        if cache_ram:
            self._cache = self._build_cache()

    @staticmethod
    def _valid_npz(path):
        try:
            with np.load(path) as data:
                for key in ["rgb", "nir", "ndvi"]:
                    if key not in data:
                        return False
                return True
        except Exception:
            return False

    @staticmethod
    def _check_mask(path):
        try:
            with np.load(path) as data:
                m = data["mask"]
                return True, m.sum() > 0
        except Exception:
            return False, False

    def _load_chip(self, path):
        try:
            data = np.load(path)
        except Exception as e:
            raise CorruptedFile(str(e))
        rgb = data["rgb"].astype(np.float32) / 255.0
        nir = np.nan_to_num(data["nir"].astype(np.float32) / 10000.0, 0.0)
        ndvi = np.nan_to_num(data["ndvi"].astype(np.float32), 0.0)
        cloud = data.get("cloud", None)
        return np.concatenate([
            rgb, nir[np.newaxis, :, :], ndvi[np.newaxis, :, :],
        ], axis=0), cloud

    def _build_cache(self):
        cache = []
        for pair in tqdm(self.pairs, desc="Caching to RAM"):
            try:
                pre_img, pre_cloud = self._load_chip(pair["pre"])
                post_img, post_cloud = self._load_chip(pair["post"])
                image = np.concatenate([pre_img, post_img], axis=0)
                mask_npz = np.load(pair["mask"])
                mask = mask_npz["mask"].astype(np.int64)
                if self.apply_cloud_mask:
                    cloud_pre = pre_cloud.astype(bool) if pre_cloud is not None else np.zeros((64, 64), dtype=bool)
                    cloud_post = post_cloud.astype(bool) if post_cloud is not None else np.zeros((64, 64), dtype=bool)
                    mask[cloud_pre | cloud_post] = 255
                cache.append((image, mask))
            except Exception:
                continue
        return cache

    def compute_weights(self, cache_path=None, oversample_factor=3.0):
        """Scan masks to compute per-sample weights for oversampling.
        Saves to cache_path for fast reuse. Returns weights list.
        """
        if cache_path and Path(cache_path).exists():
            self.pos_weights = np.load(cache_path).tolist()
            if len(self.pos_weights) == len(self.pairs):
                return self.pos_weights
        self.pos_weights = []
        for pair in tqdm(self.pairs, desc="Computing sample weights"):
            has_pos = False
            try:
                with np.load(pair["mask"]) as data:
                    has_pos = data["mask"].sum() > 0
            except Exception:
                pass
            self.pos_weights.append(oversample_factor if has_pos else 1.0)
        if cache_path:
            np.save(cache_path, np.array(self.pos_weights))
        return self.pos_weights

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self._cache is not None:
            image, mask = self._cache[idx]
            return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy())

        pair = self.pairs[idx]
        try:
            pre_img, pre_cloud = self._load_chip(pair["pre"])
            post_img, post_cloud = self._load_chip(pair["post"])
        except CorruptedFile:
            return self[(idx + 1) % len(self)]

        image = np.concatenate([pre_img, post_img], axis=0)
        mask_npz = np.load(pair["mask"])
        mask = mask_npz["mask"].astype(np.int64)
        if self.apply_cloud_mask:
            cloud_pre = pre_cloud.astype(bool) if pre_cloud is not None else np.zeros((64, 64), dtype=bool)
            cloud_post = post_cloud.astype(bool) if post_cloud is not None else np.zeros((64, 64), dtype=bool)
            mask[cloud_pre | cloud_post] = 255

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

        return torch.from_numpy(image), torch.from_numpy(mask)


# ─── Dice Loss (symmetric Tversky) ──────────────────────────────────────────

class DiceLoss(nn.Module):
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
        onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        onehot = onehot * mask.unsqueeze(1).float()
        loss = 0
        for c in range(logits.shape[1]):
            inter = (probs[:, c] * onehot[:, c]).sum()
            denom = probs[:, c].sum() + onehot[:, c].sum()
            loss += 1 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        return loss / logits.shape[1]


# ─── ResNet34 U-Net ─────────────────────────────────────────────────────────

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=2):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        old_conv = resnet.conv1
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            extra = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
            new_conv.weight[:, 3:] = extra
        self.enc_conv1 = nn.Sequential(new_conv, resnet.bn1, resnet.relu)
        self.enc_maxpool = resnet.maxpool

        self.enc_layer1 = resnet.layer1
        self.enc_layer2 = resnet.layer2
        self.enc_layer3 = resnet.layer3
        self.enc_layer4 = resnet.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = self._dec_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = self._dec_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self._dec_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self._dec_block(64 + 32, 32)
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def _dec_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e0 = self.enc_conv1(x)           # 32x32
        e0p = self.enc_maxpool(e0)        # 16x16
        e1 = self.enc_layer1(e0p)         # 16x16
        e2 = self.enc_layer2(e1)          # 8x8
        e3 = self.enc_layer3(e2)          # 4x4
        e4 = self.enc_layer4(e3)          # 2x2

        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.dec1(d1)
        return self.out_conv(d1)


# ─── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler=None,
                use_amp=False, clip_norm=1.0, accum_steps=1):
    model.train()
    total_loss = 0
    total_norm = 0
    opt_steps = 0
    optimizer.zero_grad()
    for i, (images, masks) in enumerate(tqdm(loader, desc="Train")):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            optimizer.zero_grad()
            total_norm += gn.item()
            opt_steps += 1

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader), total_norm / max(opt_steps, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    tp, fp, fn = 0, 0, 0
    correct, total_px = 0, 0
    for images, masks in tqdm(loader, desc="Val"):
        images = images.to(device)
        masks = masks.to(device)

        with torch.amp.autocast("cuda", enabled=True):
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
    parser.add_argument("--manifest", default="data/training/unet/manifest.json")
    parser.add_argument("--chips-dir", default="data/training/unet/chips")
    parser.add_argument("--output", default="models/deforest_multitemporal")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-ram", action="store_true")
    parser.add_argument("--no-wandb", action="store_true", dest="no_wandb")
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()
    args.wandb = not args.no_wandb

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    train_dataset = MultiTemporalDeforestDataset(
        manifest["train"], args.chips_dir, augment=True, cache_ram=args.cache_ram
    )
    val_dataset = MultiTemporalDeforestDataset(
        manifest["val"], args.chips_dir, augment=False, cache_ram=args.cache_ram
    )

    # Compute sample weights for oversampling (cached on disk)
    weights_cache = output_dir / "pos_weights.npy"
    train_dataset.compute_weights(cache_path=str(weights_cache), oversample_factor=10.0)
    pos_count = sum(1 for w in train_dataset.pos_weights if w > 1)
    print(f"Positive chips: {pos_count}/{len(train_dataset)} ({pos_count/len(train_dataset)*100:.1f}%)")

    if args.wandb:
        import wandb
        wandb.init(project="deforest-id", name="multitemporal",
                   config={k: v for k, v in vars(args).items() if k != "no_wandb"})
        wandb.config.arch = "ResNet34_UNet"
        wandb.config.input_channels = 10
        wandb.config.loss = "Dice"
        wandb.config.model = "ResNet34"
        wandb.config.accum = 2
        wandb.config.accum_steps = 2
        wandb.config.train_samples = len(train_dataset)
        wandb.config.val_samples = len(val_dataset)

    sampler = WeightedRandomSampler(
        train_dataset.pos_weights, len(train_dataset.pos_weights), replacement=True
    )

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset, sampler=sampler, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, **loader_kwargs,
    )

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    print(f"Device: {device}")
    print(f"Cache RAM: {args.cache_ram}")
    print(f"Loss: Dice")
    print(f"Model: ResNet34, accum=2")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = ResNetUNet(in_channels=10, out_channels=2).to(device)
    if args.wandb:
        wandb.config.params = sum(p.numel() for p in model.parameters())

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceLoss()

    if args.wandb:
        wandb.watch(model, log="all", log_freq=100, log_graph=True)

    log_path = output_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "IoU", "IoU_bg",
                         "Precision", "Recall", "F1", "Accuracy", "grad_norm", "lr"])

    console = Console()
    best_iou = 0
    table = Table(title="Multi-temporal U-Net (ResNet34)", expand=True)
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
            train_loss, grad_norm = train_epoch(
                model, train_loader, optimizer, criterion, device,
                scaler, use_amp, clip_norm=1.0, accum_steps=2,
            )
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()

            lr = scheduler.get_last_lr()[0]
            is_best = val_metrics["IoU"] > best_iou
            if is_best:
                best_iou = val_metrics["IoU"]
                torch.save(model.state_dict(), output_dir / "best.pth")

            if args.wandb:
                wandb.log({
                    "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "IoU": val_metrics["IoU"], "IoU_bg": val_metrics["IoU_bg"],
                    "F1": val_metrics["F1"], "Precision": val_metrics["Precision"],
                    "Recall": val_metrics["Recall"], "Accuracy": val_metrics["Accuracy"],
                    "grad_norm": grad_norm, "lr": lr,
                })

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                    f"{val_metrics['IoU']:.6f}", f"{val_metrics['IoU_bg']:.6f}",
                    f"{val_metrics['Precision']:.6f}", f"{val_metrics['Recall']:.6f}",
                    f"{val_metrics['F1']:.6f}", f"{val_metrics['Accuracy']:.6f}",
                    f"{grad_norm:.4f}", f"{lr:.2e}",
                ])

            marker = " *" if is_best else ""
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
                f"{best_iou:.4f}{marker}",
            )

    torch.save(model.state_dict(), output_dir / "final.pth")
    console.print(f"\n[bold green]Done![/] Best IoU: {best_iou:.4f}")
    console.print(f"Model saved to [bold]{output_dir}[/]")


if __name__ == "__main__":
    main()
