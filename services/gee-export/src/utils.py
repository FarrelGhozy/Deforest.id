import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import Tuple


def read_geotiff(path: Path) -> Tuple[np.ndarray, dict, rasterio.profiles.Profile]:
    with rasterio.open(path) as src:
        data = src.read()
        meta = src.meta.copy()
        profile = src.profile
    return data, meta, profile


def get_band_names(path: Path) -> list:
    with rasterio.open(path) as src:
        desc = src.descriptions
    if desc and desc[0]:
        return list(desc)
    n_bands = src.count
    return [f"B{i}" for i in range(1, n_bands + 1)]


def cloud_mask_sentinel2(
    data: np.ndarray, band_names: list, threshold: float = 0.3
) -> np.ndarray:
    qa_band = None
    for candidate in ["QA60", "MSK_CLDPRB"]:
        if candidate in band_names:
            qa_band = candidate
            break
    if qa_band is None:
        return np.zeros(data.shape[1:], dtype=bool)

    idx = band_names.index(qa_band)
    qa = data[idx]
    cloud = (qa & (1 << 10)) | (qa & (1 << 11))
    return cloud.astype(bool)


def normalize_band(band: np.ndarray) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    p2, p98 = np.percentile(valid, (2, 98))
    band_clipped = np.clip(band, p2, p98)
    return ((band_clipped - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)


def extract_bands(
    data: np.ndarray, band_names: list, src_band_order: list
) -> np.ndarray:
    indices = [src_band_order.index(b) for b in band_names]
    return data[np.array(indices)]


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir.astype(np.float32) - red.astype(np.float32)) / (nir + red + 1e-8)


def sliding_window_tiles(
    data: np.ndarray, chip_size: int = 64, stride: int = 64
) -> list:
    _, h, w = data.shape
    tiles = []
    for y in range(0, h - chip_size + 1, stride):
        for x in range(0, w - chip_size + 1, stride):
            chip = data[:, y : y + chip_size, x : x + chip_size]
            tiles.append({
                "data": chip,
                "row": y,
                "col": x,
            })
    return tiles


def extract_window_bounds(
    profile: dict, row: int, col: int, chip_size: int
) -> Tuple[float, float, float, float]:
    transform = profile["transform"]
    left = transform[2] + col * transform[0]
    top = transform[5] + row * transform[4]
    right = left + chip_size * transform[0]
    bottom = top + chip_size * transform[4]
    return (left, bottom, right, top)
