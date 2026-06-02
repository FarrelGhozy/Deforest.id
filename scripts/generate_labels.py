import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CLOUD_BRIGHT_THRESH = 180
CLOUD_NDVI_THRESH = 0.10


def detect_clouds(rgb: np.ndarray, ndvi: np.ndarray,
                  bright_thresh: float = 180,
                  ndvi_thresh: float = 0.10) -> np.ndarray:
    """Return binary cloud mask (1=cloud) based on RGB + NDVI heuristic.

    Cloud signature: bright across all bands (R,G,B > bright_thresh),
    NDVI near-zero (< ndvi_thresh), and blue-shifted (B >= G >= R).
    """
    r, g, b = rgb[0], rgb[1], rgb[2]
    bright = (r > bright_thresh) & (g > bright_thresh) & (b > bright_thresh)
    low_ndvi = ndvi < ndvi_thresh
    blue_shift = (b >= g) & (g >= r)
    return (bright & low_ndvi & blue_shift).astype(np.uint8)


IGNORE_LABEL = 255


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
    parser.add_argument('--cloud-bright-thresh', type=float, default=180,
                        help='Cloud brightness threshold (default: 180)')
    parser.add_argument('--cloud-ndvi-thresh', type=float, default=0.10,
                        help='Cloud NDVI threshold (default: 0.10)')
    parser.add_argument('--no-cloud-masking', action='store_true',
                        help='Disable cloud pixel detection (default: cloud masking ON)')
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

        cloud_mask = np.zeros((64, 64), dtype=np.uint8)
        if not args.no_cloud_masking:
            cloud_mask = detect_clouds(t2["rgb"], t2["ndvi"],
                                        args.cloud_bright_thresh, args.cloud_ndvi_thresh)
            mask[cloud_mask > 0] = IGNORE_LABEL

        out_name = f_baseline.stem.replace("baseline", "deforest") + "_mask.npz"
        out_path = labels_dir / out_name

        np.savez_compressed(
            out_path,
            mask=mask,
            cloud_mask=cloud_mask,
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
