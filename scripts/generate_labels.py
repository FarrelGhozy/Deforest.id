import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def compute_ndvi_change(ndvi_t1: np.ndarray, ndvi_t2: np.ndarray,
                        threshold: float = -0.15) -> np.ndarray:
    change = ndvi_t2 - ndvi_t1
    return (change < threshold).astype(np.uint8)


def apply_morphology(mask: np.ndarray, kernel_size: int = 5,
                     min_area: int = 64) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 0

    return cleaned


CHIP_PATTERN = re.compile(
    r'(?P<scene>.+?)_(?P<label>baseline|deforest)_(?P<row>\d+)_(?P<col>\d+)\.npz$'
)


def main():
    parser = argparse.ArgumentParser(description='Generate NDVI-based pseudo-labels')
    parser.add_argument('--chips-dir', required=True, type=Path,
                        help='Path to chips directory containing .npz files')
    parser.add_argument('--labels-dir', required=True, type=Path,
                        help='Path to output labels directory')
    parser.add_argument('--threshold', type=float, default=-0.15,
                        help='NDVI change threshold (default: -0.15)')
    parser.add_argument('--kernel-size', type=int, default=5,
                        help='Morphological kernel size (default: 5)')
    parser.add_argument('--min-area', type=int, default=64,
                        help='Minimum connected component area (default: 64)')
    args = parser.parse_args()

    chips_dir = args.chips_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    labels_dir.mkdir(parents=True, exist_ok=True)

    chip_files = sorted(chips_dir.glob("*.npz"))

    baseline_map = {}
    deforest_map = {}

    for f in chip_files:
        m = CHIP_PATTERN.match(f.name)
        if not m:
            continue
        key = (m.group('scene'), int(m.group('row')), int(m.group('col')))
        if m.group('label') == 'baseline':
            baseline_map[key] = f
        else:
            deforest_map[key] = f

    matched = 0
    skipped = 0

    for key, f_baseline in tqdm(baseline_map.items(), desc="Generating labels"):
        f_deforest = deforest_map.get(key)
        if f_deforest is None:
            skipped += 1
            continue

        t1 = np.load(f_baseline)
        t2 = np.load(f_deforest)

        mask = compute_ndvi_change(t1["ndvi"], t2["ndvi"], args.threshold)
        mask = apply_morphology(mask, args.kernel_size, args.min_area)

        out_name = f_baseline.stem.replace("baseline", "deforest") + "_mask.npz"
        out_path = labels_dir / out_name

        np.savez_compressed(
            out_path,
            mask=mask,
            rgb=t1["rgb"],
            ndvi_baseline=t1["ndvi"],
            ndvi_deforest=t2["ndvi"],
            bounds=t1["bounds"],
            scene_baseline=f_baseline.name,
            scene_deforest=f_deforest.name,
        )
        matched += 1

    print(f"\nDone: {matched} masks generated, {skipped} skipped (no matching deforest chip)")


if __name__ == "__main__":
    main()
