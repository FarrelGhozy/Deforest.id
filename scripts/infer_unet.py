"""Run U-Net inference on test set, save predictions & metrics.

Usage:
    uv run python scripts/infer_unet.py \
        --manifest data/training/unet/manifest.json \
        --model models/unet_deforest/best.pth \
        --output data/training/unet/predictions
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_unet import SimpleUNet, DiceLoss, DeforestDataset


@torch.no_grad()
def infer_test(model, loader, device):
    model.eval()
    all_preds = []
    all_gts = []
    stems = []
    for images, masks in tqdm(loader, desc="Infer"):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_gts.append(masks.numpy())
    return np.concatenate(all_preds), np.concatenate(all_gts)


def compute_metrics(preds, gts, ignore_label=255):
    valid = gts != ignore_label
    tp = ((preds == 1) & (gts == 1) & valid).sum()
    fp = ((preds == 1) & (gts == 0) & valid).sum()
    fn = ((preds == 0) & (gts == 1) & valid).sum()
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {"IoU": float(iou), "Dice": float(dice), "Precision": float(precision), "Recall": float(recall)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="data/training/unet/predictions")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    test_dataset = DeforestDataset(manifest["test"])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    print(f"Model loaded from {args.model}")
    print(f"Test samples: {len(test_dataset)}")

    preds, gts = infer_test(model, test_loader, device)
    metrics = compute_metrics(preds, gts)
    print(f"\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_dir / 'metrics.json'}")

    np.save(out_dir / "predictions.npy", preds)
    np.save(out_dir / "ground_truth.npy", gts)
    print(f"Predictions saved to {out_dir}")


if __name__ == "__main__":
    main()
