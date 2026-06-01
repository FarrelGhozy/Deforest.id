import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnnotConfig:
    root: Path = Path("data/annotation")

    raw_dir: Path = field(init=False)
    tiles_dir: Path = field(init=False)
    masks_auto_dir: Path = field(init=False)
    masks_refined_dir: Path = field(init=False)
    export_dir: Path = field(init=False)

    tile_size_px: int = 512
    tile_overlap_px: int = 64

    cloud_threshold: float = 0.3
    ndvi_threshold: float = -0.15
    change_sensitivity: float = 1.5

    bands_order: tuple = ("B2", "B3", "B4", "B8")
    rgb_bands: tuple = ("B4", "B3", "B2")
    nir_band: str = "B8"
    red_band: str = "B4"

    def __post_init__(self):
        self.raw_dir = self.root / "raw"
        self.tiles_dir = self.root / "tiles"
        self.masks_auto_dir = self.root / "masks_auto"
        self.masks_refined_dir = self.root / "masks_refined"
        self.export_dir = self.root / "export"

    def ensure_dirs(self):
        for d in [self.raw_dir, self.tiles_dir, self.masks_auto_dir,
                  self.masks_refined_dir, self.export_dir]:
            d.mkdir(parents=True, exist_ok=True)


CONFIG = AnnotConfig()
