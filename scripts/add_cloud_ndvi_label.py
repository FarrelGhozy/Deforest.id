"""Add cloud mask + NDVI label to existing GFW mask .npz files.

Usage:
    python scripts/add_cloud_ndvi_label.py \
        --chips-dir data/training/unet/chips \
        --labels-dir data/training/unet/labels_gfw \
        --raw-dir data/training/unet/raw
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cloud_utils import read_cloud_mask


def add_cloud_mask(mask_data, chip, chip_path, scratch_geotiffs):
    scene_name = str(chip["scene"])
    tif_path = scratch_geotiffs.get(scene_name)
    if tif_path is None:
        print(f"  SKIP cloud: {scene_name} not found in raw/")
        return mask_data

    y0, x0, y1, x1 = chip["bounds"]
    cloud = read_cloud_mask(tif_path, int(y0), int(x0))
    if cloud is None:
        return mask_data
    mask_data["cloud"] = cloud
    return mask_data


def add_ndvi_label(mask_data, chip, threshold):
    ndvi = chip["ndvi"]
    label = (ndvi < threshold).astype(np.uint8)
    mask_data["label_ndvi"] = label
    return mask_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--ndvi-threshold", type=float, default=0.3)
    args = parser.parse_args()

    chips_dir = args.chips_dir
    labels_dir = args.labels_dir
    raw_dir = args.raw_dir

    scratch_geotiffs = {p.name: p for p in raw_dir.glob("*.tif")}
    print(f"Found {len(scratch_geotiffs)} GeoTIFFs in raw/")

    mask_files = sorted(labels_dir.glob("*_mask.npz"))
    print(f"Processing {len(mask_files)} mask files...")

    cloud_ok = 0
    cloud_skip = 0

    for mask_path in tqdm(mask_files, desc="Adding cloud+ndvi"):
        chip_stem = mask_path.stem.replace("_mask", "")
        chip_path = chips_dir / f"{chip_stem}.npz"
        if not chip_path.exists():
            cloud_skip += 1
            continue

        chip = np.load(chip_path)
        mask = np.load(mask_path)

        new_data = {}
        for k in mask.keys():
            new_data[k] = mask[k]

        new_data = add_cloud_mask(new_data, chip, chip_path, scratch_geotiffs)
        if "cloud" in new_data:
            cloud_ok += 1
        else:
            cloud_skip += 1

        new_data = add_ndvi_label(new_data, chip, args.ndvi_threshold)

        np.savez_compressed(mask_path, **new_data)

    print(f"\nDone: {cloud_ok} cloud masks added, {cloud_skip} skipped (no CLEAR_COUNT)")


if __name__ == "__main__":
    main()
