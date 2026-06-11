"""Split dataset into train/val/test — stratified by scene.

Usage:
    uv run python scripts/split_dataset.py \
        --chips-dir data/training/unet/chips \
        --labels-dir data/training/unet/labels_gfw \
        --output-dir data/training/unet \
        --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

MASK_PATTERN = re.compile(
    r'(?P<scene>.+?)_(?P<kind>[a-z]+)_(?P<row>\d+)_(?P<col>\d+)_mask\.npz$'
)
CHIP_PATTERN = re.compile(
    r'(?P<scene>.+?)_(?P<kind>[a-z]+)_(?P<row>\d+)_(?P<col>\d+)\.npz$'
)


def main():
    parser = argparse.ArgumentParser(description="Split chips into train/val/test")
    parser.add_argument("--chips-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chips_path = Path(args.chips_dir)
    labels_path = Path(args.labels_dir)
    output_path = Path(args.output_dir)

    # Find all mask files → these define the dataset (one sample per mask)
    mask_files = sorted(labels_path.glob("*_mask.npz"))
    print(f"Splitting {len(mask_files)} mask files...")

    # Group by scene for stratified split
    scene_map = defaultdict(list)
    for mf in mask_files:
        m = MASK_PATTERN.match(mf.name)
        if not m:
            continue
        key = (m.group("scene"), int(m.group("row")), int(m.group("col")))
        scene = m.group("scene")
        scene_map[scene].append(key)

    # For each mask, find the corresponding deforest chip
    matched = 0
    missing_chip = 0
    entries = []  # (scene, key, chip_path, mask_path)
    for mf in mask_files:
        m = MASK_PATTERN.match(mf.name)
        if not m:
            continue
        scene = m.group("scene")
        row, col = int(m.group("row")), int(m.group("col"))
        key = (scene, row, col)

        chip_name = f"{scene}_{m.group('kind')}_{row}_{col}.npz"
        chip_path = chips_path / chip_name
        if not chip_path.exists():
            missing_chip += 1
            continue

        entries.append((scene, key, chip_path, mf))
        matched += 1

    print(f"Matched {matched} chip-mask pairs, {missing_chip} missing chips")

    # Group entries by scene for stratified split
    scene_entries = defaultdict(list)
    for scene, key, chip_path, mask_path in entries:
        scene_entries[scene].append(key)

    scenes = list(scene_entries.keys())
    print(f"Split {len(scenes)} scenes into train/val/test")

    # First split: scenes → train + temp
    train_scenes, temp_scenes = train_test_split(
        scenes,
        test_size=args.val_ratio + args.test_ratio,
        random_state=args.seed,
    )

    # Second split: temp → val + test
    val_ratio_of_temp = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_scenes, test_scenes = train_test_split(
        temp_scenes,
        test_size=1 - val_ratio_of_temp if val_ratio_of_temp < 1 else 0,
        random_state=args.seed,
    )

    split_map = {"train": train_scenes, "val": val_scenes, "test": test_scenes}

    for split_name in ["train", "val", "test"]:
        (output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split_name / "masks").mkdir(parents=True, exist_ok=True)

    manifest = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, scenes_list in split_map.items():
        allowed_scenes = set(scenes_list)
        for scene, key, chip_path, mask_path in entries:
            if scene not in allowed_scenes:
                continue

            img_dst = output_path / split_name / "images" / chip_path.name
            mask_dst = output_path / split_name / "masks" / mask_path.name

            if not img_dst.exists():
                try:
                    shutil.copy2(chip_path, img_dst)
                except PermissionError:
                    print(f"  SKIP: {chip_path.name} (locked)")
                    continue
            if not mask_dst.exists():
                try:
                    shutil.copy2(mask_path, mask_dst)
                except PermissionError:
                    print(f"  SKIP: {mask_path.name} (locked)")
                    continue

            manifest[split_name].append({
                "stem": chip_path.stem,
                "scene": scene,
                "image": str(img_dst),
                "mask": str(mask_dst),
            })
            counts[split_name] += 1

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSplit complete:")
    print(f"  Train: {counts['train']} ({len(train_scenes)} scenes)")
    print(f"  Val:   {counts['val']} ({len(val_scenes)} scenes)")
    print(f"  Test:  {counts['test']} ({len(test_scenes)} scenes)")
    print(f"  Total: {sum(counts.values())}")
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
