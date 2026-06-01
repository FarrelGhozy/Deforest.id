import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import List, Tuple, Optional
from config import CONFIG


def read_geotiff(path: Path) -> Tuple[np.ndarray, dict, rasterio.profiles.Profile]:
    with rasterio.open(path) as src:
        data = src.read()
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        profile = src.profile
    return data, meta, profile


def cloud_mask_sentinel2(data: np.ndarray, meta: dict,
                         threshold: float = 0.3) -> np.ndarray:
    has_qa = any(b in meta.get("descriptions", []) for b in ["QA60", "MSK_CLDPRB"])
    if has_qa:
        qa_idx = meta["descriptions"].index(
            next(b for b in ["QA60", "MSK_CLDPRB"] if b in meta["descriptions"])
        )
        qa = data[qa_idx]
        cloud = (qa & (1 << 10)) | (qa & (1 << 11))
        return cloud.astype(bool)
    return np.zeros(data.shape[1:], dtype=bool)


def normalize_band(band: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(band[band > 0], (2, 98)) if band.max() > 0 else (0, 1)
    band_clipped = np.clip(band, p2, p98)
    return ((band_clipped - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)


def extract_bands(data: np.ndarray, band_names: List[str],
                  src_band_order: List[str]) -> np.ndarray:
    indices = [src_band_order.index(b) for b in band_names]
    return data[np.array(indices)]


def tile_raster(data: np.ndarray, profile: dict,
                tile_size: int = 512, overlap: int = 64) -> List[dict]:
    _, h, w = data.shape
    stride = tile_size - overlap
    tiles = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile_data = data[:, y_start:y_end, x_start:x_end]

            tile_transform = rasterio.windows.transform(
                Window(x_start, y_start,
                       x_end - x_start, y_end - y_start),
                profile["transform"]
            )

            tiles.append({
                "data": tile_data,
                "bounds": (y_start, y_end, x_start, x_end),
                "transform": tile_transform,
                "width": x_end - x_start,
                "height": y_end - y_start,
            })

    return tiles


def process_scene(scene_path: Path, config) -> List[Path]:
    config.ensure_dirs()
    data, meta, profile = read_geotiff(scene_path)

    src_bands = meta.get("descriptions", [])
    if not src_bands:
        src_bands = [f"B{i}" for i in range(1, data.shape[0] + 1)]

    cloud = cloud_mask_sentinel2(data, meta, config.cloud_threshold)
    cloud_frac = cloud.sum() / cloud.size
    if cloud_frac > config.cloud_threshold:
        print(f"[SKIP] {scene_path.name} — cloud cover {cloud_frac:.1%}")
        return []

    rgb = extract_bands(data, list(config.rgb_bands), src_bands)
    nir = extract_bands(data, [config.nir_band], src_bands)[0]
    red = extract_bands(data, [config.red_band], src_bands)[0]

    ndvi = (nir.astype(np.float32) - red.astype(np.float32)) / (nir + red + 1e-8)

    tiles = tile_raster(data, profile, config.tile_size_px, config.tile_overlap_px)
    saved = []

    for i, tile in enumerate(tiles):
        y_s, y_e, x_s, x_e = tile["bounds"]
        tile_mask = cloud[y_s:y_e, x_s:x_e]
        if tile_mask.sum() / tile_mask.size > config.cloud_threshold * 1.5:
            continue

        tile_rgb = rgb[:, y_s:y_e, x_s:x_e]
        tile_ndvi = ndvi[y_s:y_e, x_s:x_e]

        rgb_norm = np.stack([normalize_band(tile_rgb[j]) for j in range(3)], axis=0)

        scene_stem = scene_path.stem
        tile_name = f"{scene_stem}_tile_{i:04d}_{x_s}_{y_s}"
        out_path = config.tiles_dir / f"{tile_name}.npz"

        np.savez_compressed(
            out_path,
            rgb=rgb_norm,
            ndvi=tile_ndvi.astype(np.float32),
            transform=tile["transform"],
            bounds=(y_s, y_e, x_s, x_e),
            cloud_mask=tile_mask,
            scene=scene_path.name,
        )
        saved.append(out_path)

    return saved
