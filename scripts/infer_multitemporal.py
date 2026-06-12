"""
Multi-temporal inference with TTA + visualization.

Usage:
    python scripts/infer_multitemporal.py \
        --checkpoint models/deforest_multitemporal/best.pth \
        --manifest data/training/unet/manifest.json \
        --chips-dir data/training/unet/chips \
        --output models/deforest_multitemporal/inference

Options:
    --split test         (default) evaluate on test set
    --no-tta             disable test-time augmentation
"""

import argparse
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_multitemporal import ResNetUNet, MultiTemporalDeforestDataset


class InferenceDataset(Dataset):
    """Wraps MultiTemporalDeforestDataset but returns metadata for saving."""

    def __init__(self, base_dataset, shuffle_rng=None):
        self.base = base_dataset
        self.indices = list(range(len(base_dataset)))
        if shuffle_rng is not None:
            shuffle_rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, mask = self.base[actual_idx]
        pair = self.base.pairs[actual_idx]
        return img, mask, actual_idx, pair["pre"], pair["mask"]


@torch.no_grad()
def predict_tta(model, image, device, use_flip=True):
    """Predict with test-time augmentation (horizontal + vertical flip)."""
    model.eval()
    image = image.unsqueeze(0).to(device)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(image)
        if use_flip:
            logits += model(image.flip(-1)).flip(-1)
            logits += model(image.flip(-2)).flip(-2)
        logits /= (1 + 2 * use_flip)

    return logits.softmax(dim=1).squeeze(0).cpu()


def compute_metrics(pred_mask, gt_mask, ignore_val=255):
    valid = gt_mask != ignore_val
    tp = ((pred_mask == 1) & (gt_mask == 1) & valid).sum().item()
    fp = ((pred_mask == 1) & (gt_mask == 0) & valid).sum().item()
    fn = ((pred_mask == 0) & (gt_mask == 1) & valid).sum().item()
    tn = ((pred_mask == 0) & (gt_mask == 0) & valid).sum().item()

    iou = tp / (tp + fp + fn + 1e-8)
    iou_bg = tn / (tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return {"IoU": iou, "IoU_bg": iou_bg, "Precision": precision,
            "Recall": recall, "F1": f1, "Accuracy": acc, "TP": tp, "FP": fp,
            "FN": fn, "TN": tn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/deforest_multitemporal/best.pth")
    parser.add_argument("--manifest", default="data/training/unet/manifest.json")
    parser.add_argument("--chips-dir", default="data/training/unet/chips")
    parser.add_argument("--output", default="models/deforest_multitemporal/inference")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    base_ds = MultiTemporalDeforestDataset(
        manifest[args.split], args.chips_dir, augment=False, cache_ram=False
    )
    ds = InferenceDataset(base_ds, shuffle_rng=np.random.RandomState(42) if args.max_samples else None)
    if args.max_samples:
        ds.indices = ds.indices[:args.max_samples]

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = ResNetUNet(in_channels=10, out_channels=2).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    use_tta = not args.no_tta
    all_metrics = []
    n_vis = min(len(ds), 20)

    print(f"Running {args.split} inference on {len(ds)} samples")
    print(f"TTA: {use_tta}")

    for batch_idx, (img, mask, actual_idx, pre_path, mask_path) in enumerate(
        tqdm(loader, desc="Inference")
    ):
        probs = predict_tta(model, img[0], device, use_flip=use_tta)
        pred = probs.argmax(dim=0).numpy().astype(np.int64)
        gt = mask[0].numpy().astype(np.int64)

        metrics = compute_metrics(pred, gt)
        all_metrics.append(metrics)

        if batch_idx < n_vis:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            ch = img[0].numpy()

            # Pre (first 3 bands)
            pre_rgb = np.clip(ch[:3].transpose(1, 2, 0), 0, 1)
            axes[0, 0].imshow(pre_rgb)
            axes[0, 0].set_title(f"Pre (RGB) | IoU={metrics['IoU']:.3f}")
            axes[0, 0].axis("off")

            # Post (channels 5-7, i.e. pre's 3 bands shifted)
            post_rgb = np.clip(ch[5:8].transpose(1, 2, 0), 0, 1)
            axes[0, 1].imshow(post_rgb)
            axes[0, 1].set_title("Post (RGB)")
            axes[0, 1].axis("off")

            # NDVI pre vs post
            ndvi_pre = ch[4]
            ndvi_post = ch[9]
            vmin, vmax = -1, 1
            axes[0, 2].imshow(ndvi_pre, cmap="RdYlGn", vmin=vmin, vmax=vmax)
            axes[0, 2].set_title("NDVI Pre")
            axes[0, 2].axis("off")
            axes[0, 3].imshow(ndvi_post, cmap="RdYlGn", vmin=vmin, vmax=vmax)
            axes[0, 3].set_title("NDVI Post")
            axes[0, 3].axis("off")

            # NDVI difference
            ndvi_diff = ndvi_post - ndvi_pre
            axes[1, 0].imshow(ndvi_diff, cmap="RdBu", vmin=-1, vmax=1)
            axes[1, 0].set_title("NDVI Post - Pre")
            axes[1, 0].axis("off")

            # Ground truth
            axes[1, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[1, 1].set_title("GT Mask")
            axes[1, 1].axis("off")

            # Prediction
            axes[1, 2].imshow(pred, cmap="Reds", vmin=0, vmax=1)
            axes[1, 2].set_title(f"Pred IoU={metrics['IoU']:.3f}")
            axes[1, 2].axis("off")

            # Overlay
            overlay = np.clip(pre_rgb * 0.5 + pred[:, :, None] * 0.5, 0, 1)
            axes[1, 3].imshow(overlay)
            axes[1, 3].set_title("Overlay")
            axes[1, 3].axis("off")

            plt.tight_layout()
            plt.savefig(out_dir / f"sample_{batch_idx:04d}.png", dpi=150)
            plt.close(fig)

    # Aggregate metrics (per-sample mean)
    agg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0] if k not in ("TP","FP","FN","TN")}
    # Global metrics (sum TP/FP/FN/TN across all samples — same as validation)
    global_tp = sum(m["TP"] for m in all_metrics)
    global_fp = sum(m["FP"] for m in all_metrics)
    global_fn = sum(m["FN"] for m in all_metrics)
    global_tn = sum(m["TN"] for m in all_metrics)
    global_iou = global_tp / (global_tp + global_fp + global_fn + 1e-8)
    global_precision = global_tp / (global_tp + global_fp + 1e-8)
    global_recall = global_tp / (global_tp + global_fn + 1e-8)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-8)
    global_acc = (global_tp + global_tn) / (global_tp + global_fp + global_fn + global_tn + 1e-8)

    print("\n=== Per-Sample Mean Metrics ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.6f}")
    print("\n=== Global Metrics (sum TP/FP/FN — matches val) ===")
    print(f"  IoU:       {global_iou:.6f}")
    print(f"  Precision: {global_precision:.6f}")
    print(f"  Recall:    {global_recall:.6f}")
    print(f"  F1:        {global_f1:.6f}")
    print(f"  Accuracy:  {global_acc:.6f}")
    print(f"  TP={global_tp:.0f}  FP={global_fp:.0f}  FN={global_fn:.0f}  TN={global_tn:.0f}")

    # Save CSV
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        keys = list(all_metrics[0].keys())
        f.write(",".join(keys) + "\n")
        for m in all_metrics:
            f.write(",".join(str(m[k]) for k in keys) + "\n")
    print(f"Saved per-sample metrics to {csv_path}")

    # Summary text
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Split: {args.split}\n")
        f.write(f"Samples: {len(ds)}\n")
        f.write(f"TTA: {use_tta}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write("=== Per-Sample Mean Metrics ===\n")
        for k, v in agg.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n=== Global Metrics ===\n")
        f.write(f"  IoU:       {global_iou:.6f}\n")
        f.write(f"  Precision: {global_precision:.6f}\n")
        f.write(f"  Recall:    {global_recall:.6f}\n")
        f.write(f"  F1:        {global_f1:.6f}\n")
        f.write(f"  Accuracy:  {global_acc:.6f}\n")
    print(f"Saved summary to {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
