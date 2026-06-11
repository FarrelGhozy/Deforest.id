"""Shared cloud mask utilities — single source of truth for ALL pipeline stages.

Used by:
  - Preprocessing (tiling): filter cloudy chips + save per-pixel cloud mask
  - Label generation: copy cloud mask from chip (not recomputed)
  - Training: ignore_index=255 for cloudy pixels
  - Inference: mask out cloudy predictions
"""

import numpy as np
import rasterio
from pathlib import Path
from rasterio.windows import Window

CLEAR_THRESHOLD: int = 3  # min clear observations to consider pixel "clear"


def cloud_mask_from_clear_count(clear_count: np.ndarray) -> np.ndarray:
    """Convert CLEAR_COUNT band values to binary cloud mask (uint8)."""
    return (clear_count < CLEAR_THRESHOLD).astype(np.uint8)


def read_cloud_mask(
    tif_path: Path,
    row: int,
    col: int,
    chip_size: int = 64,
) -> np.ndarray | None:
    """Read cloud mask for chip window directly from source GeoTIFF.

    Returns (chip_size, chip_size) uint8 array (1=cloud, 0=clear),
    or None if CLEAR_COUNT band is not found.
    """
    with rasterio.open(tif_path) as src:
        desc = src.descriptions
        if not desc or "CLEAR_COUNT" not in desc:
            return None
        band_idx = desc.index("CLEAR_COUNT") + 1
        clear_count = src.read(
            band_idx,
            window=Window.from_slices(
                (row, row + chip_size), (col, col + chip_size)
            ),
        )
    return cloud_mask_from_clear_count(clear_count)


def cloud_fraction(cloud_mask: np.ndarray) -> float:
    """Fraction of cloudy pixels in mask."""
    return float(cloud_mask.sum()) / cloud_mask.size
