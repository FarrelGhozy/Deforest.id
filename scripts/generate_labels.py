"""Generate pseudo-labels from NDVI threshold for U-Net training.

Usage:
    python scripts/generate_labels.py \
        --chips-dir services/gee-export/src/data/training/unet/chips \
        --labels-dir services/gee-export/src/data/training/unet/labels_ndvi \
        --ndvi-threshold 0.25 \
        --ignore-low 0.15
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def ndvi_to_mask(ndvi: np.ndarray, threshold: float = 0.25, ignore_low: float = 0.15) -> np.ndarray:
    """Convert NDVI array to binary segmentation mask.

    0 = forest (NDVI >= threshold)
    1 = deforest (NDVI < ignore_low)
    255 = uncertain / ignore (ignore_low <= NDVI < threshold)

    The ignore band helps the model focus on confident regions.
    """
    mask = np.full(ndvi.shape, 255, dtype=np.uint8)
    mask[ndvi >= threshold] = 0
    mask[ndvi < ignore_low] = 1
    return mask


def main():
    parser = argparse.ArgumentParser(description="Generate U-Net masks from NDVI")
    parser.add_argument("--chips-dir", required=True, help="Directory with .npz chip files")
    parser.add_argument("--labels-dir", required=True, help="Output directory for mask .npy files")
    parser.add_argument("--ndvi-threshold", type=float, default=0.25, help="NDVI threshold for forest (>=) vs uncertain")
    parser.add_argument("--ignore-low", type=float, default=0.15, help="NDVI below this = confident deforest")
    parser.add_argument("--key", default="ndvi", help="Key in npz file holding NDVI array")
    args = parser.parse_args()

    chips_path = Path(args.chips_dir)
    labels_path = Path(args.labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(chips_path.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {chips_path}")
        return

    print(f"Found {len(npz_files)} chips")
    print(f"Threshold: NDVI >= {args.ndvi_threshold} → forest (0)")
    print(f"           NDVI <  {args.ignore_low} → deforest (1)")
    print(f"           {args.ignore_low} <= NDVI < {args.ndvi_threshold} → ignore (255)")

    stats = {"deforest": 0, "forest": 0, "mixed": 0, "ignore": 0}

    for npz_path in tqdm(npz_files, desc="Generating masks"):
        data = np.load(npz_path)
        ndvi = data[args.key]
        mask = ndvi_to_mask(ndvi, args.ndvi_threshold, args.ignore_low)

        stem = npz_path.stem
        out_path = labels_path / f"{stem}_mask.npy"
        np.save(out_path, mask)

        unique, counts = np.unique(mask, return_counts=True)
        counts_map = dict(zip(unique, counts))

        def_pct = counts_map.get(1, 0) / mask.size * 100
        for_pct = counts_map.get(0, 0) / mask.size * 100
        ign_pct = counts_map.get(255, 0) / mask.size * 100

        if def_pct > 50:
            stats["deforest"] += 1
        elif for_pct > 50:
            stats["forest"] += 1
        else:
            stats["mixed"] += 1
        if ign_pct > 30:
            stats["ignore"] += 1

    print(f"\nDone. {len(npz_files)} masks generated.")
    print(f"  Mostly deforest (>50% pixels): {stats['deforest']}")
    print(f"  Mostly forest (>50% pixels):   {stats['forest']}")
    print(f"  Mixed:                         {stats['mixed']}")
    print(f"  High ignore (>30% pixels):     {stats['ignore']}")


if __name__ == "__main__":
    main()
