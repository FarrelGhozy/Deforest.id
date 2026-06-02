"""Split dataset into train/val/test — stratified by scene.

Usage:
    python scripts/split_dataset.py \
        --chips-dir services/gee-export/src/data/training/unet/chips \
        --labels-dir services/gee-export/src/data/training/unet/labels_ndvi \
        --output-dir services/gee-export/src/data/training/unet \
        --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


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

    npz_files = sorted(chips_path.glob("*.npz"))

    # Group chips by scene for stratified split
    scene_map = defaultdict(list)
    for npz_path in npz_files:
        data = np.load(npz_path)
        scene = str(data["scene"])
        scene_map[scene].append(npz_path.stem)
        data.close()

    print(f"Found {len(npz_files)} chips across {len(scene_map)} scenes")

    scenes = list(scene_map.keys())

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

    # Create output directories
    for split_name in ["train", "val", "test"]:
        (output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split_name / "masks").mkdir(parents=True, exist_ok=True)

    # Copy files (symlink or copy) + create metadata
    manifest = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, scenes_list in split_map.items():
        for scene in scenes_list:
            for stem in scene_map[scene]:
                # Source files
                npz_src = chips_path / f"{stem}.npz"
                mask_src = labels_path / f"{stem}_mask.npy"

                if not mask_src.exists():
                    print(f"  WARNING: mask not found for {stem}, skipping")
                    continue

                # Destination: we symlink to save space
                img_dst = output_path / split_name / "images" / f"{stem}.npz"
                mask_dst = output_path / split_name / "masks" / f"{stem}_mask.npy"

                if not img_dst.exists():
                    img_dst.symlink_to(os.path.relpath(npz_src, img_dst.parent))
                if not mask_dst.exists():
                    mask_dst.symlink_to(os.path.relpath(mask_src, mask_dst.parent))

                manifest[split_name].append({
                    "stem": stem,
                    "scene": scene,
                    "image": str(img_dst),
                    "mask": str(mask_dst),
                })
                counts[split_name] += 1

    # Save manifest
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSplit complete:")
    print(f"  Train: {counts['train']} chips ({len(train_scenes)} scenes)")
    print(f"  Val:   {counts['val']} chips ({len(val_scenes)} scenes)")
    print(f"  Test:  {counts['test']} chips ({len(test_scenes)} scenes)")
    print(f"  Total: {sum(counts.values())} chips")
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    import os
    main()
