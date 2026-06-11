"""Verify cloud mask consistency: mask .npz vs source GeoTIFF.

Samples N random chips and checks per-pixel cloud mask matches
between what's stored in mask .npz and what read_cloud_mask() returns
from the original GeoTIFF.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cloud_utils import read_cloud_mask


def verify(args):
    chips_dir = Path(args.chips_dir)
    labels_dir = Path(args.labels_dir)
    raw_dir = Path(args.raw_dir)

    scratch_geotiffs = {p.name: p for p in raw_dir.glob("*.tif")}
    print(f"Found {len(scratch_geotiffs)} GeoTIFFs in raw/")

    mask_files = sorted(labels_dir.glob("*_mask.npz"))
    print(f"Total mask files: {len(mask_files)}")

    if args.sample > 0:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(mask_files), min(args.sample, len(mask_files)), replace=False)
        mask_files = [mask_files[i] for i in indices]
        print(f"Sampling {len(mask_files)} files for verification")

    mismatches = 0
    missing_cloud = 0
    missing_tif = 0
    total = 0

    for mask_path in mask_files:
        chip_stem = mask_path.stem.replace("_mask", "")
        chip_path = chips_dir / f"{chip_stem}.npz"
        if not chip_path.exists():
            continue

        chip = np.load(chip_path)
        mask = np.load(mask_path)

        if "cloud" not in mask:
            missing_cloud += 1
            continue

        scene_name = str(chip["scene"])
        tif_path = scratch_geotiffs.get(scene_name)
        if tif_path is None:
            missing_tif += 1
            continue

        y0, x0 = int(chip["bounds"][0]), int(chip["bounds"][1])
        expected = read_cloud_mask(tif_path, y0, x0)
        if expected is None:
            missing_cloud += 1
            continue

        actual = mask["cloud"]
        if not np.array_equal(actual, expected):
            mismatches += 1
            if mismatches <= 5:
                diff = (actual != expected).sum()
                print(f"  MISMATCH {chip_stem}: {diff} pixels differ")
        total += 1

    print(f"\nResults:")
    print(f"  Verified: {total}")
    print(f"  Match:    {total - mismatches}")
    print(f"  Mismatch: {mismatches}")
    print(f"  Missing cloud key: {missing_cloud}")
    print(f"  Missing source TIF: {missing_tif}")
    if mismatches == 0:
        print("  ✅ Cloud mask 100% consistent!")
    else:
        print(f"  ❌ {mismatches}/{total} chips have inconsistent cloud masks")


def main():
    parser = argparse.ArgumentParser(description="Verify cloud mask consistency")
    parser.add_argument("--chips-dir", default="data/training/unet/chips", type=Path)
    parser.add_argument("--labels-dir", default="data/training/unet/labels_gfw", type=Path)
    parser.add_argument("--raw-dir", default="data/training/unet/raw", type=Path)
    parser.add_argument("--sample", default=100, type=int, help="Number of chips to check (0=all)")
    args = parser.parse_args()
    verify(args)


if __name__ == "__main__":
    main()
