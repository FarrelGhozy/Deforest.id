import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Literal
from config import CONFIG


def export_png_pairs(source_dir: Path, output_dir: Path,
                     split: Literal["train", "val"] = "train",
                     val_ratio: float = 0.15) -> dict:
    img_dir = output_dir / split / "img"
    mask_dir = output_dir / split / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(source_dir.glob("*_mask.npz"))
    if not npz_files:
        print(f"[WARN] No mask files in {source_dir}")
        return {"img": 0, "mask": 0}

    n_val = max(1, int(len(npz_files) * val_ratio))
    if split == "val":
        npz_files = npz_files[:n_val]
    else:
        npz_files = npz_files[n_val:]

    count = 0
    for f in npz_files:
        data = np.load(f)
        rgb = data["rgb"]
        mask = data["mask"]

        rgb_img = Image.fromarray(
            np.transpose(rgb.astype(np.uint8), (1, 2, 0))
        )
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        rgb_img.save(img_dir / f"{f.stem.replace('_mask', '')}.png")
        mask_img.save(mask_dir / f"{f.stem.replace('_mask', '')}.png")
        count += 1

    return {"img": count, "mask": count}


def export_geotiff(source_dir: Path, output_dir: Path) -> List[Path]:
    try:
        import rasterio
        from rasterio.crs import CRS
    except ImportError:
        print("[WARN] rasterio not installed, skipping GeoTIFF export")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(source_dir.glob("*_mask.npz"))
    saved = []

    for f in npz_files:
        data = np.load(f)
        mask = (data["mask"] * 255).astype(np.uint8)
        transform = data.get("transform")
        if transform is None or transform.ndim == 0:
            continue

        out_path = output_dir / f"{f.stem.replace('_mask', '')}.tif"
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=mask.dtype,
            crs=CRS.from_epsg(4326),
            transform=transform.item() if hasattr(transform, "item") else transform,
        ) as dst:
            dst.write(mask, 1)

        saved.append(out_path)

    return saved


def export_all(output_dir: Path, config=CONFIG,
               formats: List[str] = None) -> dict:
    if formats is None:
        formats = ["png"]

    result = {}

    for src_name in ["masks_auto", "masks_refined"]:
        src_dir = getattr(config, f"{src_name}_dir")
        if not src_dir.exists():
            continue

        out_subdir = output_dir / src_name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            if fmt == "png":
                for split in ["train", "val"]:
                    r = export_png_pairs(src_dir, out_subdir, split=split)
                    result[f"{src_name}_{split}_png"] = r
            elif fmt == "geotiff":
                r = export_geotiff(src_dir, out_subdir / "geotiff")
                result[f"{src_name}_geotiff"] = len(r)

    return result
