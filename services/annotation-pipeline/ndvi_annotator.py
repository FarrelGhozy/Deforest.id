import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List
from config import CONFIG


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


def clean_with_cloud_mask(mask: np.ndarray, cloud: np.ndarray,
                          dilate_px: int = 10) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    cloud_dilated = cv2.dilate(cloud.astype(np.uint8), kernel)
    mask_clean = mask.copy()
    mask_clean[cloud_dilated > 0] = 0
    return mask_clean


def generate_mask(tile_t1: np.ndarray, tile_t2: np.ndarray,
                  cloud_mask: Optional[np.ndarray] = None,
                  config=CONFIG) -> np.ndarray:
    ndvi_t1 = tile_t1["ndvi"]
    ndvi_t2 = tile_t2["ndvi"]

    raw_change = compute_ndvi_change(ndvi_t1, ndvi_t2, config.ndvi_threshold)
    mask = apply_morphology(raw_change, kernel_size=config.change_sensitivity, min_area=64)

    if cloud_mask is not None:
        mask = clean_with_cloud_mask(mask, cloud_mask)

    return mask


def batch_generate(scene_t1: str, scene_t2: str, config=CONFIG) -> List[Path]:
    t1_dir = config.tiles_dir
    t2_dir = config.tiles_dir
    out_dir = config.masks_auto_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_files = sorted(t1_dir.glob(f"{scene_t1}_tile_*.npz"))
    t2_files = sorted(t2_dir.glob(f"{scene_t2}_tile_*.npz"))

    t2_map = {f.name: f for f in t2_files}
    saved = []

    for f_t1 in t1_files:
        f_t2 = t2_map.get(f_t1.name.replace(scene_t1, scene_t2, 1))
        if f_t2 is None:
            continue

        t1 = np.load(f_t1)
        t2 = np.load(f_t2)

        cloud = t1.get("cloud_mask") if "cloud_mask" in t1 else None
        mask = generate_mask(t1, t2, cloud, config)

        out_name = f_t1.name.replace(".npz", "_mask.npz")
        out_path = out_dir / out_name

        np.savez_compressed(
            out_path,
            mask=mask,
            rgb=t1["rgb"],
            ndvi_t1=t1["ndvi"],
            ndvi_t2=t2["ndvi"],
            transform=t1["transform"],
            bounds=t1["bounds"],
            scene_t1=f_t1.name,
            scene_t2=f_t2.name,
        )
        saved.append(out_path)

    return saved
