"""
Siamese U-Net with shared ResNet34 encoder + change detection head.

Architecture (FC-Siam-diff):
- Shared encoder processes pre (5ch) and post (5ch) independently
- Decoder fuses [post_feat, pre_feat, post_feat-pre_feat] at each level
- Explicit change features instead of naive channel stacking

Usage:
    python scripts/train_siamese.py \
        --manifest data/training/unet/manifest.json \
        --chips-dir data/training/unet/chips \
        --output models/deforest_siamese \
        --epochs 30 --batch-size 64 --lr 1e-4
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


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SiameseDeforestDataset(Dataset):
    """Loads pre + post chips separately (not stacked) for siamese encoder."""

    def __init__(self, manifest_entries, chips_dir, apply_cloud_mask=True,
                 augment=False, cache_ram=False):
        self.chips_dir = Path(chips_dir)
        self.apply_cloud_mask = apply_cloud_mask
        self.augment = augment

        self.pairs = []
        for entry in tqdm(manifest_entries, desc="Linking pre/post chips"):
            pre_path = Path(entry["image"])
            stem_pre = pre_path.stem
            for suffix in ["baseline", "deforest"]:
                if suffix in stem_pre:
                    post_stem = stem_pre.replace(suffix,
                        "deforest" if suffix == "baseline" else "baseline")
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
        valid_pairs = []
        for pair in tqdm(self.pairs, desc="Caching to RAM"):
            try:
                pre_img, pre_cloud = self._load_chip(pair["pre"])
                post_img, post_cloud = self._load_chip(pair["post"])
                mask_npz = np.load(pair["mask"])
                mask = mask_npz["mask"].astype(np.int64)
                if self.apply_cloud_mask:
                    cpre = pre_cloud.astype(bool) if pre_cloud is not None else np.zeros((64, 64), dtype=bool)
                    cpost = post_cloud.astype(bool) if post_cloud is not None else np.zeros((64, 64), dtype=bool)
                    mask[cpre | cpost] = 255
                cache.append((pre_img, post_img, mask))
                valid_pairs.append(pair)
            except Exception:
                continue
        self.pairs = valid_pairs
        return cache

    def compute_weights(self, cache_path=None, oversample_factor=3.0):
        if cache_path and Path(cache_path).exists():
            cached = np.load(cache_path).tolist()
            if len(cached) == len(self.pairs):
                self.pos_weights = cached
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
            pre_img, post_img, mask = self._cache[idx]
            pre_img = pre_img.copy()
            post_img = post_img.copy()
            mask = mask.copy()
        else:
            pair = self.pairs[idx]
            try:
                pre_img, pre_cloud = self._load_chip(pair["pre"])
                post_img, post_cloud = self._load_chip(pair["post"])
            except CorruptedFile:
                return self[(idx + 1) % len(self)]
            mask_npz = np.load(pair["mask"])
            mask = mask_npz["mask"].astype(np.int64)
            if self.apply_cloud_mask:
                cpre = pre_cloud.astype(bool) if pre_cloud is not None else np.zeros((64, 64), dtype=bool)
                cpost = post_cloud.astype(bool) if post_cloud is not None else np.zeros((64, 64), dtype=bool)
                mask[cpre | cpost] = 255

        if self.augment:
            if np.random.rand() > 0.5:
                pre_img = np.flip(pre_img, axis=2).copy()
                post_img = np.flip(post_img, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                pre_img = np.flip(pre_img, axis=1).copy()
                post_img = np.flip(post_img, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

        return (torch.from_numpy(pre_img), torch.from_numpy(post_img),
                torch.from_numpy(mask))


# ─── Loss ────────────────────────────────────────────────────────────────────

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


# ─── Siamese ResNet34 U-Net (FC-Siam-diff) ────────────────────────────────────

class SiameseUNet(nn.Module):
    """Shared ResNet34 encoder + change detection decoder.

    Encoder processes pre (5ch) and post (5ch) with shared weights.
    Decoder fuses [post_feat, pre_feat, post_feat-pre_feat] at each level.
    """

    def __init__(self, in_channels=5, out_channels=2):
        super().__init__()

        # Shared encoder — ResNet34 backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        old_conv = resnet.conv1
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                             padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            extra = old_conv.weight.mean(dim=1, keepdim=True).repeat(
                1, in_channels - 3, 1, 1)
            new_conv.weight[:, 3:] = extra
        self.enc_conv1 = nn.Sequential(new_conv, resnet.bn1, resnet.relu)
        self.enc_maxpool = resnet.maxpool

        self.enc_layer1 = resnet.layer1   # 64 -> 64
        self.enc_layer2 = resnet.layer2   # 64 -> 128
        self.enc_layer3 = resnet.layer3   # 128 -> 256
        self.enc_layer4 = resnet.layer4   # 256 -> 512

        # Decoder — fuses [post, pre, diff] at each level
        # up4 -> dec4: cat(up=256, post_e3=256, pre_e3=256, diff=256) = 1024
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = self._dec_block(1024, 256)

        # up3 -> dec3: cat(up=128, post_e2=128, pre_e2=128, diff=128) = 512
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = self._dec_block(512, 128)

        # up2 -> dec2: cat(up=64, post_e1=64, pre_e1=64, diff=64) = 256
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self._dec_block(256, 64)

        # up1 -> dec1: cat(up=32, post_e0=64, pre_e0=64, diff=64) = 224
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self._dec_block(224, 32)

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

    def _encode(self, x):
        """Encode a single timestamp. Returns features at each level."""
        e0 = self.enc_conv1(x)
        e0p = self.enc_maxpool(e0)
        e1 = self.enc_layer1(e0p)
        e2 = self.enc_layer2(e1)
        e3 = self.enc_layer3(e2)
        e4 = self.enc_layer4(e3)
        return e0, e1, e2, e3, e4

    def forward(self, pre, post):
        # Encode both timestamps with shared weights
        pre_e0, pre_e1, pre_e2, pre_e3, pre_e4 = self._encode(pre)
        post_e0, post_e1, post_e2, post_e3, post_e4 = self._encode(post)

        # Decoder with difference fusion
        d4 = self.up4(post_e4)
        diff3 = post_e3 - pre_e3
        d4 = torch.cat([d4, post_e3, pre_e3, diff3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        diff2 = post_e2 - pre_e2
        d3 = torch.cat([d3, post_e2, pre_e2, diff2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        diff1 = post_e1 - pre_e1
        d2 = torch.cat([d2, post_e1, pre_e1, diff1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        diff0 = post_e0 - pre_e0
        d1 = torch.cat([d1, post_e0, pre_e0, diff0], dim=1)
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
    for i, (pre, post, masks) in enumerate(tqdm(loader, desc="Train")):
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(pre, post)
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
    for pre, post, masks in tqdm(loader, desc="Val"):
        pre = pre.to(device)
        post = post.to(device)
        masks = masks.to(device)

        with torch.amp.autocast("cuda", enabled=True):
            logits = model(pre, post)
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
    edge = 1e-6
    precision = tp / (tp + fp + edge)
    recall = tp / (tp + fn + edge)
    f1 = 2 * precision * recall / (precision + recall + edge)
    acc = correct / (total_px + edge)
    return total_loss / len(loader), {
        "IoU": iou, "IoU_bg": tn / (tn + fp + fn + edge),
        "Precision": precision, "Recall": recall, "F1": f1, "Accuracy": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/training/unet/manifest.json")
    parser.add_argument("--chips-dir", default="data/training/unet/chips")
    parser.add_argument("--output", default="models/deforest_siamese")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-ram", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    train_dataset = SiameseDeforestDataset(
        manifest["train"], args.chips_dir, augment=True, cache_ram=args.cache_ram
    )
    val_dataset = SiameseDeforestDataset(
        manifest["val"], args.chips_dir, augment=False, cache_ram=args.cache_ram
    )

    weights_cache = output_dir / "pos_weights.npy"
    train_dataset.compute_weights(cache_path=str(weights_cache), oversample_factor=10.0)
    pos_count = sum(1 for w in train_dataset.pos_weights if w > 1)
    print(f"Positive chips: {pos_count}/{len(train_dataset)} ({pos_count/len(train_dataset)*100:.1f}%)")

    sampler = WeightedRandomSampler(
        train_dataset.pos_weights, len(train_dataset.pos_weights), replacement=True
    )

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    print(f"Device: {device}")
    print(f"Model: Siamese ResNet34 (FC-Siam-diff)")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SiameseUNet(in_channels=5, out_channels=2).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = DiceLoss()

    log_path = output_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "IoU", "IoU_bg",
                         "Precision", "Recall", "F1", "Accuracy", "grad_norm", "lr"])

    console = Console()
    best_iou = 0
    table = Table(title="Siamese U-Net (FC-Siam-diff)", expand=True)
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

    with Live(table, console=console, refresh_per_second=0.5,
              vertical_overflow="visible") as live:
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

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                    f"{val_metrics['IoU']:.6f}", f"{val_metrics['IoU_bg']:.6f}",
                    f"{val_metrics['Precision']:.6f}",
                    f"{val_metrics['Recall']:.6f}",
                    f"{val_metrics['F1']:.6f}",
                    f"{val_metrics['Accuracy']:.6f}",
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
