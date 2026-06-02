import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHIP_PATTERN = re.compile(
    r'(?P<scene>.+?)_(?P<label>baseline|deforest)_(?P<row>\d+)_(?P<col>\d+)\.npz$'
)


def load_chip(path):
    data = np.load(path)
    rgb = np.transpose(data["rgb"], (1, 2, 0)).astype(np.uint8)
    return rgb, data


def save_chip_view(chip_path, mask_path=None, output=None):
    rgb, data = load_chip(chip_path)

    ndvi = data["ndvi"]

    fig, axes = plt.subplots(1, 4 if mask_path else 3, figsize=(16, 4))

    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB — {chip_path.name}")
    axes[0].axis("off")

    im1 = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("NDVI")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    bounds = data["bounds"]
    axes[2].text(0.5, 0.5,
                 f"Scene: {data['scene']}\n"
                 f"Bounds: {bounds}\n"
                 f"NDVI range: {ndvi.min():.3f} ~ {ndvi.max():.3f}",
                 transform=axes[2].transAxes,
                 ha="center", va="center", fontsize=10, fontfamily="monospace")
    axes[2].axis("off")

    if mask_path and mask_path.exists():
        mask_data = np.load(mask_path)
        mask = mask_data["mask"]

        overlay = rgb.copy().astype(np.float32)
        mask_bool = mask > 0
        overlay[..., 0][mask_bool] = overlay[..., 0][mask_bool] * 0.3 + 200 * 0.7
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        axes[3].imshow(overlay)
        area_pct = mask.sum() / mask.size * 100
        axes[3].set_title(f"Mask Overlay ({area_pct:.1f}%)")
        axes[3].axis("off")

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="View individual chip + mask")
    parser.add_argument("--chips-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--index", type=int, default=None,
                        help="Show chip at this index (sorted alphabetically)")
    parser.add_argument("--name", type=str, default=None,
                        help="Show chip by name (partial match)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save to PNG instead of display")
    parser.add_argument("--sample", type=int, default=None,
                        help="Generate a gallery of N sample chips into outputs/ dir")
    args = parser.parse_args()

    chips_dir = args.chips_dir.resolve()
    labels_dir = args.labels_dir.resolve() if args.labels_dir else None

    all_chips = sorted(chips_dir.glob("*.npz"))

    if args.sample:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        np.random.seed(42)
        selected = np.random.choice(all_chips, min(args.sample, len(all_chips)),
                                    replace=False)
        for chip_path in selected:
            mask_path = None
            if labels_dir:
                m = CHIP_PATTERN.match(chip_path.name)
                if m:
                    mask_name = "_".join([
                        m.group("scene"),
                        "deforest",
                        m.group("row"),
                        m.group("col"),
                    ]) + "_mask.npz"
                    mask_path = labels_dir / mask_name
            out_path = out_dir / f"{chip_path.stem}.png"
            save_chip_view(chip_path, mask_path, out_path)
        print(f"\n{len(selected)} images saved to outputs/")
        return

    if args.name:
        matches = [c for c in all_chips if args.name in c.name]
        if not matches:
            print(f"No chips matching '{args.name}'")
            return
        chip_path = matches[0]
        if len(matches) > 1:
            print(f"Multiple matches, showing first: {chip_path.name}")
    elif args.index is not None:
        if args.index < 0 or args.index >= len(all_chips):
            print(f"Index {args.index} out of range (0-{len(all_chips) - 1})")
            return
        chip_path = all_chips[args.index]
    else:
        chip_path = all_chips[0]

    mask_path = None
    if labels_dir:
        m = CHIP_PATTERN.match(chip_path.name)
        if m:
            mask_name = "_".join([
                m.group("scene"),
                "deforest",
                m.group("row"),
                m.group("col"),
            ]) + "_mask.npz"
            mask_path = labels_dir / mask_name
            if not mask_path.exists():
                print(f"Mask not found: {mask_path.name}")
                mask_path = None

    save_chip_view(chip_path, mask_path, args.output)


if __name__ == "__main__":
    main()
