import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import TILE, PATHS, EXPORT
from utils import (
    read_geotiff,
    get_band_names,
    cloud_mask_sentinel2,
    normalize_band,
    extract_bands,
    compute_ndvi,
    sliding_window_tiles,
)


def process_geotiff(geotiff_path: Path) -> list:
    data, meta, profile = read_geotiff(geotiff_path)
    band_names = get_band_names(geotiff_path)
    _, h, w = data.shape

    if h < TILE.chip_size or w < TILE.chip_size:
        print(f"[SKIP] {geotiff_path.name} — too small ({w}×{h})")
        return []

    cloud = cloud_mask_sentinel2(data, band_names, TILE.cloud_threshold)
    cloud_frac = cloud.sum() / cloud.size
    if cloud_frac > TILE.cloud_threshold:
        print(f"[SKIP] {geotiff_path.name} — cloud cover {cloud_frac:.1%}")
        return []

    rgb = extract_bands(data, list(EXPORT.rgb_bands), band_names)
    nir = extract_bands(data, [EXPORT.nir_band], band_names)[0]
    red = extract_bands(data, [EXPORT.red_band], band_names)[0]
    ndvi = compute_ndvi(nir, red)

    tiles = sliding_window_tiles(data, TILE.chip_size, TILE.stride)
    saved = []

    for tile in tqdm(tiles, desc=f"Tiling {geotiff_path.name}"):
        y, x = tile["row"], tile["col"]
        chip = tile["data"]

        tile_cloud = cloud[y : y + TILE.chip_size, x : x + TILE.chip_size]
        if tile_cloud.sum() / tile_cloud.size > TILE.cloud_threshold * 1.5:
            continue

        tile_ndvi = ndvi[y : y + TILE.chip_size, x : x + TILE.chip_size]
        if tile_ndvi.mean() < TILE.min_ndvi:
            continue

        tile_rgb = rgb[:, y : y + TILE.chip_size, x : x + TILE.chip_size]
        tile_nir = nir[y : y + TILE.chip_size, x : x + TILE.chip_size]
        tile_cloud_mask = tile_cloud.astype(np.uint8)

        rgb_norm = np.stack(
            [normalize_band(tile_rgb[j]) for j in range(3)], axis=0
        )

        scene_stem = geotiff_path.stem
        chip_name = f"{scene_stem}_{y}_{x}"
        out_path = PATHS.chips_dir / f"{chip_name}.npz"

        np.savez_compressed(
            out_path,
            rgb=rgb_norm,
            nir=tile_nir.astype(np.float32),
            ndvi=tile_ndvi.astype(np.float32),
            cloud=tile_cloud_mask,
            bounds=(y, x, y + TILE.chip_size, x + TILE.chip_size),
            scene=geotiff_path.name,
        )
        saved.append(out_path)

    return saved


def process_directory(raw_dir: Path = None):
    raw_dir = raw_dir or PATHS.raw_dir
    PATHS.chips_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(raw_dir.glob("*.tif"))
    if not tif_files:
        print(f"[WARN] No GeoTIFF files found in {raw_dir}")
        return []

    all_chips = []
    for tif_path in tif_files:
        chips = process_geotiff(tif_path)
        all_chips.extend(chips)
        print(f"  → {len(chips)} chips from {tif_path.name}")

    print(f"\n[DONE] Total chips: {len(all_chips)}")
    return all_chips
