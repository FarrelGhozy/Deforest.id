import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_chip(path: Path):
    data = np.load(path)
    rgb = np.transpose(data["rgb"], (1, 2, 0)).astype(np.uint8)
    return rgb, data


def make_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy().astype(np.float32)
    mask_bool = mask > 0
    overlay[..., 0][mask_bool] = overlay[..., 0][mask_bool] * 0.3 + 200 * 0.7
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Visualize generated masks')
    parser.add_argument('--chips-dir', required=True, type=Path,
                        help='Path to chips directory')
    parser.add_argument('--labels-dir', required=True, type=Path,
                        help='Path to labels directory with _mask.npz files')
    parser.add_argument('--output', default='preview.png', type=Path,
                        help='Output preview image path')
    parser.add_argument('--samples', type=int, default=16,
                        help='Number of samples to visualize (default: 16)')
    args = parser.parse_args()

    chips_dir = args.chips_dir.resolve()
    labels_dir = args.labels_dir.resolve()

    mask_files = sorted(labels_dir.glob("*_mask.npz"))
    if not mask_files:
        print("No mask files found in", labels_dir)
        return

    samples = min(args.samples, len(mask_files))
    mask_files = mask_files[:samples]

    cols = 4
    rows = int(np.ceil(samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, mask_path in enumerate(tqdm(mask_files, desc="Rendering")):
        mask_data = np.load(mask_path)
        mask = mask_data["mask"]

        chip_name = mask_data.get("scene_baseline", mask_path.name)
        chip_path = chips_dir / chip_name

        if chip_path.exists():
            rgb, _ = load_chip(chip_path)
        elif "deforest" in chip_name:
            chip_path = chips_dir / chip_name.replace("deforest", "baseline")
            if chip_path.exists():
                rgb, _ = load_chip(chip_path)
            else:
                rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)

        overlay = make_overlay(rgb, mask)

        ax = axes[i]
        ax.imshow(overlay)
        area_pct = mask.sum() / mask.size * 100
        ax.set_title(f"{mask_path.stem[:20]}... ({area_pct:.1f}%)", fontsize=8)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(str(args.output), dpi=150, bbox_inches="tight")
    print(f"Preview saved to {args.output}")


if __name__ == "__main__":
    main()
