"""Visualize random chips with mask overlay to validate labels.

Usage:
    python scripts/visualize_masks.py \
        --chips-dir services/gee-export/src/data/training/unet/chips \
        --labels-dir services/gee-export/src/data/training/unet/labels_ndvi \
        --num-samples 9 \
        --output preview_masks.png
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay binary mask on RGB image.

    rgb: (C, H, W) or (H, W, C) uint8
    mask: (H, W) uint8 with 0=forest, 1=deforest, 255=ignore
    """
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)

    overlay = rgb.copy().astype(float)

    # Deforest = red tint
    overlay[mask == 1] = overlay[mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    # Ignore = gray tint
    overlay[mask == 255] = overlay[mask == 255] * (1 - alpha * 0.5) + np.array([128, 128, 128]) * alpha * 0.5

    return overlay.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=9)
    parser.add_argument("--output", default="preview_masks.png")
    args = parser.parse_args()

    chips_path = Path(args.chips_dir)
    labels_path = Path(args.labels_dir)

    npz_files = sorted(chips_path.glob("*.npz"))
    np.random.shuffle(npz_files)

    samples = npz_files[:args.num_samples]
    n = len(samples)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for i, npz_path in enumerate(samples):
        data = np.load(npz_path)
        rgb = data["rgb"]
        ndvi = data["ndvi"]

        stem = npz_path.stem
        mask_path = labels_path / f"{stem}_mask.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
        else:
            mask = np.full((64, 64), 255, dtype=np.uint8)

        # Side-by-side: original, mask only, overlay
        overlay = overlay_mask(rgb, mask)

        combined = np.hstack([
            rgb.transpose(1, 2, 0),
            np.stack([mask * 85] * 3, axis=-1).astype(np.uint8),
            overlay,
        ])

        axes[i].imshow(combined)
        axes[i].set_title(f"{stem[:30]}... NDVI={ndvi.mean():.2f}", fontsize=8)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Preview saved to {args.output}")


if __name__ == "__main__":
    main()
