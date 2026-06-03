"""Generate comparison grid: RGB | Ground Truth | Prediction.

Usage:
    uv run python scripts/compare_viz.py \
        --manifest data/training/unet/manifest.json \
        --predictions data/training/unet/predictions/predictions.npy \
        --output data/training/unet/comparisons \
        --num-samples 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


COLORS = {
    "deforest": (0, 0, 255),    # blue → red (BGR)
    "background": (0, 0, 0),
    "ignore": (255, 0, 0),      # cyan → blue
}


def load_chip(path):
    """Load RGB, NIR, NDVI from npz and reconstruct 5-channel input."""
    data = np.load(path)
    rgb = (data["rgb"].astype(np.float32) / 255.0).transpose(1, 2, 0)  # (64,64,3)
    return (rgb * 255).astype(np.uint8)


def mask_overlay(rgb, mask, alpha=0.5):
    overlay = rgb.copy().astype(np.float32)
    overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + np.array(COLORS["deforest"]) * alpha)
    overlay[mask == 255] = (overlay[mask == 255] * (1 - alpha) + np.array(COLORS["ignore"]) * alpha)
    return overlay.astype(np.uint8)


def make_comparison_grid(rgb, gt_mask, pred_mask, title=""):
    h, w = rgb.shape[:2]
    gt_viz = mask_overlay(rgb, gt_mask)
    pred_viz = mask_overlay(rgb, pred_mask)

    grid = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    grid[:, :w] = rgb
    grid[:, w + 10:w + 10 + w] = gt_viz
    grid[:, 2 * w + 20:2 * w + 20 + w] = pred_viz

    labels = ["RGB", "Ground Truth", "Prediction"]
    for i, label in enumerate(labels):
        x = i * (w + 10) + 5
        cv2.putText(grid, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="data/training/unet/comparisons")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    preds = np.load(args.predictions)
    entries = manifest["test"]
    assert len(preds) == len(entries), f"Predictions {len(preds)} != test entries {len(entries)}"

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(entries), min(args.num_samples, len(entries)), replace=False)

    for idx in tqdm(indices, desc="Generating comparisons"):
        entry = entries[idx]
        chip_path = entry["image"]
        mask_path = entry["mask"]

        rgb = load_chip(chip_path)
        mask_npz = np.load(mask_path)
        gt_mask = mask_npz["mask"]
        pred_mask = preds[idx]

        valid = gt_mask != 255
        iou = -1
        if valid.sum() > 0:
            tp = ((pred_mask == 1) & (gt_mask == 1) & valid).sum()
            fp = ((pred_mask == 1) & (gt_mask == 0) & valid).sum()
            fn = ((pred_mask == 0) & (gt_mask == 1) & valid).sum()
            iou = tp / (tp + fp + fn + 1e-6)

        grid = make_comparison_grid(rgb, gt_mask, pred_mask, entry["stem"])
        stem = entry["stem"].replace("/", "_")

        iou_text = f"IoU: {iou:.3f}" if iou >= 0 else "IoU: N/A"
        cv2.putText(grid, iou_text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(out_dir / f"{stem}_iou{iou:.3f}.png"), grid)

    print(f"Saved {len(indices)} comparison images to {out_dir}")


if __name__ == "__main__":
    main()
