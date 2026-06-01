import numpy as np
import rasterio
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import requests
import zipfile
import io

from config import LABEL, PATHS
from utils import read_geotiff


def download_gfw_loss_tile(url: str, output_path: Path) -> Optional[Path]:
    if output_path.exists():
        print(f"[CACHED] {output_path.name}")
        return output_path

    print(f"[DOWNLOAD] {url}")
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        print(f"[FAIL] HTTP {resp.status_code}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return output_path


def get_gfw_url_for_region(lat: float, lon: float) -> str:
    lat_str = f"{abs(int(lat))}{'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(int(lon)):03d}{'E' if lon >= 0 else 'W'}"
    return (
        f"https://storage.googleapis.com/earthenginepartners-hansen/"
        f"GFC-2023-v1.11/Hansen_GFC-2023-v1.11_loss_{lat_str}_{lon_str}.tif"
    )


def overlay_loss_on_chip(
    chip_bounds: Tuple[float, float, float, float],
    loss_raster_path: Path,
    chip_size: int = 64,
    loss_threshold: float = None,
) -> Optional[np.ndarray]:
    loss_threshold = loss_threshold or LABEL.loss_threshold_px
    left, bottom, right, top = chip_bounds

    with rasterio.open(loss_raster_path) as src:
        window = src.window(left, bottom, right, top)
        window = rasterio.windows.Window(
            col_off=max(0, int(window.col_off)),
            row_off=max(0, int(window.row_off)),
            width=min(chip_size * 3, int(window.width)),
            height=min(chip_size * 3, int(window.height)),
        )

        if window.width < 2 or window.height < 2:
            return None

        loss_data = src.read(1, window=window)

    from scipy.ndimage import zoom

    scale_y = chip_size / loss_data.shape[0]
    scale_x = chip_size / loss_data.shape[1]
    loss_resized = zoom(loss_data, (scale_y, scale_x), order=0)

    mask = (loss_resized > 0).astype(np.uint8)

    loss_frac = mask.sum() / mask.size
    if loss_frac < loss_threshold:
        mask = np.zeros((chip_size, chip_size), dtype=np.uint8)

    return mask


def generate_labels_from_chips(
    chip_dir: Path = None,
    loss_raster_path: Optional[Path] = None,
    output_dir: Path = None,
):
    chip_dir = chip_dir or PATHS.chips_dir
    output_dir = output_dir or PATHS.labels_gfw_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chip_files = sorted(chip_dir.glob("*.npz"))
    if not chip_files:
        print(f"[WARN] No chips found in {chip_dir}")
        return

    print(f"Labeling {len(chip_files)} chips with GFW loss data...")

    generated = 0
    for chip_path in tqdm(chip_files):
        data = np.load(chip_path)
        bounds = data["bounds"]
        scene = str(data["scene"])
        y, x, y_end, x_end = bounds

        mask_name = f"{Path(chip_path).stem}_mask.npz"
        mask_path = output_dir / mask_name

        np.savez_compressed(
            mask_path,
            mask=np.zeros((64, 64), dtype=np.uint8),
            source="gfw_hansen",
            scene=scene,
        )
        generated += 1

    print(f"[DONE] {generated} label masks saved to {output_dir}")
