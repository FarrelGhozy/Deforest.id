import numpy as np
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from config import PATHS


def split_dataset(
    chip_dir: Path = None,
    label_dir: Path = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    chip_dir = chip_dir or PATHS.chips_dir
    label_dir = label_dir or PATHS.labels_gfw_dir

    train_img_dir = PATHS.train_img_dir
    train_mask_dir = PATHS.train_mask_dir
    val_img_dir = PATHS.val_img_dir
    val_mask_dir = PATHS.val_mask_dir
    test_img_dir = PATHS.test_img_dir
    test_mask_dir = PATHS.test_mask_dir

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
              test_img_dir, test_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    chip_files = sorted(chip_dir.glob("*.npz"))
    if not chip_files:
        print(f"[WARN] No chips in {chip_dir}")
        return

    random.seed(seed)
    random.shuffle(chip_files)

    n = len(chip_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = chip_files[:n_train]
    val_files = chip_files[n_train : n_train + n_val]
    test_files = chip_files[n_train + n_val :]

    def copy_batch(files, img_dir, mask_dir, split_name):
        for chip_path in tqdm(files, desc=f"{split_name}"):
            chip_stem = chip_path.stem

            mask_path = label_dir / f"{chip_stem}_mask.npz"

            rgb_npy = img_dir / f"{chip_stem}.npy"
            mask_npy = mask_dir / f"{chip_stem}.npy"

            if not rgb_npy.exists():
                data = np.load(chip_path)
                np.save(rgb_npy, data["rgb"])

            if mask_path.exists() and not mask_npy.exists():
                mask_data = np.load(mask_path)
                np.save(mask_npy, mask_data["mask"])

        print(f"  → {len(files)} samples in {split_name} set")

    copy_batch(train_files, train_img_dir, train_mask_dir, "train")
    copy_batch(val_files, val_img_dir, val_mask_dir, "val")
    copy_batch(test_files, test_img_dir, test_mask_dir, "test")

    print(f"\n[DONE] Dataset split: {len(train_files)} train / "
          f"{len(val_files)} val / {len(test_files)} test")

    return {
        "total": n,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }
