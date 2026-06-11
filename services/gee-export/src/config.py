import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExportConfig:
    gee_project: str = os.getenv("GEE_PROJECT", "deforestprojek")
    gee_credentials: str = os.getenv("GEE_CREDENTIALS", "")

    band_order: tuple = ("B2", "B3", "B4", "B8", "CLEAR_COUNT")
    rgb_bands: tuple = ("B4", "B3", "B2")
    red_band: str = "B4"
    nir_band: str = "B8"
    scale: int = 10
    crs: str = "EPSG:4326"
    max_pixels: int = 1_000_000_000_000

    export_folder: str = "deforest_training"
    export_bucket: str = os.getenv("GEE_EXPORT_BUCKET", "")
    cloud_percent_filter: int = 80
    composite_months: int = 12


@dataclass
class TileConfig:
    chip_size: int = 64
    stride: int = 64
    cloud_threshold: float = 0.3
    min_ndvi: float = 0.0
    output_dir: Path = Path("data/training/unet/chips")


@dataclass
class LabelConfig:
    gfw_url: str = (
        "https://storage.googleapis.com/earthenginepartners-hansen/"
        "GFC-2023-v1.11/Hansen_GFC-2023-v1.11_loss_30N_090E.tif"
    )
    output_label_dir: Path = Path("data/training/unet/labels_gfw")
    loss_threshold_px: float = 0.001


_ROOT = Path(os.getenv("DATA_ROOT", Path(__file__).parents[3]))


@dataclass
class PathConfig:
    raw_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/raw")
    chips_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/chips")
    labels_gfw_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/labels_gfw")
    labels_ndvi_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/labels_ndvi")
    train_img_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/train/images")
    train_mask_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/train/masks")
    val_img_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/val/images")
    val_mask_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/val/masks")
    test_img_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/test/images")
    test_mask_dir: Path = field(default_factory=lambda: _ROOT / "data/training/unet/test/masks")


EXPORT = ExportConfig()
TILE = TileConfig()
LABEL = LabelConfig()
PATHS = PathConfig()
