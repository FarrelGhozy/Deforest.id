import math
import numpy as np
import rasterio
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import requests
from rasterio.windows import from_bounds
from scipy.ndimage import zoom

from config import LABEL, PATHS


def download_gfw_loss_tile(url: str, output_path: Path) -> Optional[Path]:
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            if resp.status_code != 200:
                return None
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path
        except requests.RequestException:
            continue
    return None


def get_gfw_url_for_region(lat: float, lon: float) -> str:
    north_edge = math.ceil(lat / 10) * 10
    west_edge = (int(lon) // 10) * 10
    lat_str = f"{abs(north_edge):02d}{'N' if north_edge >= 0 else 'S'}"
    lon_str = f"{abs(west_edge):03d}{'E' if west_edge >= 0 else 'W'}"
    return (
        f"https://storage.googleapis.com/earthenginepartners-hansen/"
        f"GFC-2023-v1.11/Hansen_GFC-2023-v1.11_lossyear_{lat_str}_{lon_str}.tif"
    )


def overlay_loss_on_chip(
    chip_bounds: tuple[float, float, float, float],
    loss_raster_path: Path,
    chip_size: int = 64,
    loss_threshold: float = None,
) -> Optional[np.ndarray]:
    loss_threshold = loss_threshold or LABEL.loss_threshold_px
    left, bottom, right, top = chip_bounds

    with rasterio.open(loss_raster_path) as src:
        window = from_bounds(left, bottom, right, top, src.transform)
        window = rasterio.windows.Window(
            col_off=max(0, int(window.col_off)),
            row_off=max(0, int(window.row_off)),
            width=min(chip_size * 3, int(window.width)),
            height=min(chip_size * 3, int(window.height)),
        )

        if window.width < 2 or window.height < 2:
            return None

        loss_data = src.read(1, window=window)

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

    transform_cache: dict[str, object] = {}
    loss_cache: dict[str, Path] = {}
    gfw_tile_dir = PATHS.raw_dir / "gfw_tiles"
    gfw_tile_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for chip_path in tqdm(chip_files, desc="GFW Labeling"):
        data = np.load(chip_path)
        bounds = data["bounds"]
        scene = str(data["scene"])
        y, x, y_end, x_end = bounds

        if scene not in transform_cache:
            tif_path = PATHS.raw_dir / scene
            if not tif_path.exists():
                continue
            with rasterio.open(tif_path) as src:
                transform_cache[scene] = src.transform

        transform = transform_cache[scene]
        left, top = transform * (x, y)
        right, bottom = transform * (x_end, y_end)

        chip_geo_bounds = (left, bottom, right, top)

        center_lat = (bottom + top) / 2
        center_lon = (left + right) / 2
        gfw_url = get_gfw_url_for_region(center_lat, center_lon)

        loss_path = loss_cache.get(gfw_url)
        if loss_path is None:
            loss_tile_name = gfw_url.split("/")[-1]
            loss_tile_path = gfw_tile_dir / loss_tile_name

            result = download_gfw_loss_tile(gfw_url, loss_tile_path)
            if result is None:
                loss_cache[gfw_url] = Path("")  # mark as failed
                continue
            loss_cache[gfw_url] = result
            loss_path = result
        elif loss_path == Path(""):
            continue

        mask = overlay_loss_on_chip(chip_geo_bounds, loss_path, chip_size=64)

        mask_name = f"{Path(chip_path).stem}_mask.npz"
        mask_path = output_dir / mask_name

        np.savez_compressed(
            mask_path,
            mask=mask if mask is not None else np.zeros((64, 64), dtype=np.uint8),
            source="gfw_hansen",
            scene=scene,
        )
        generated += 1

    print(f"[DONE] {generated} label masks saved to {output_dir}")
