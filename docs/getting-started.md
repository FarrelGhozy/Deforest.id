# Getting Started

## Prerequisites

| Tool | Minimal Versi | Cara Install |
|------|---------------|--------------|
| Python | 3.11 | `sudo apt install python3.11` |
| [uv](https://docs.astral.sh/uv/) | terbaru | `curl -LsSf https://astral.sh/uv/install.sh | sh` |
| Git | — | `sudo apt install git` |

## Clone & Setup

```bash
git clone git@github.com:FarrelGhozy/Deforest.id.git
cd Deforest.id/services/annotation-pipeline
make install
```

## Pipeline Lengkap

### 1. Farrel — Pull Data dari GEE

Export GeoTIFF dari GEE (lihat [Data Format](data-format.md)) dan taruh di:

```
data/annotation/raw/
├── kalimantan_2025_01_15.tif
└── kalimantan_2025_06_20.tif
```

### 2. Preprocess — Tile & Cloud Mask

```bash
make preprocess SCENE=data/annotation/raw/kalimantan_2025_01_15.tif
make preprocess SCENE=data/annotation/raw/kalimantan_2025_06_20.tif
```

### 3. Auto-Annotate — NDVI Change Detection

```bash
make annotate T1=kalimantan_2025_01_15 T2=kalimantan_2025_06_20
```

### 4. Visualize & Refine (Opsional)

```bash
make visualize
```

### 5. Export Training Set

```bash
make export
```

Output siap training di `data/annotation/export/masks_refined/`.

## Cek Status

```bash
make status
```

Output contoh:

```
Raw scenes:     2 files
Tiles:          312 files
Auto masks:     312 files
Refined masks:  89 files
Export:         178 PNGs
```

## Troubleshooting

| Problem | Solusi |
|---------|--------|
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` |
| `module 'numpy' has no attribute ...` | `uv sync` ulang |
| GEE quota exceeded | Turunkan resolusi atau perkecil AOI |
| Cloud cover terlalu tinggi | Filter scene dengan cloud < 20% |
