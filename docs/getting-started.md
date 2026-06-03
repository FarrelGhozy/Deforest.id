# Getting Started

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Git
- GPU recommended untuk training (CUDA)

## Setup

```bash
git clone https://github.com/FarrelGhozy/Deforest.id
cd Deforest.id

# Install Python deps
uv sync
```

---

## Pipeline Lengkap

### 1. Data Acquisition (GEE)

Export Sentinel-2 scenes dari Google Earth Engine. Lihat [Data Format](data-format.md).

```
data/annotation/raw/
  hl_sample_1_2020_01_15.tif
  hl_sample_1_2021_06_01.tif
  ...
```

### 2. Preprocess

```bash
cd services/annotation-pipeline
make preprocess
```

Cloud masking, tiling 512px → 64px, ekstraksi RGB + NIR + NDVI.

### 3. Auto-Annotate

```bash
make annotate
```

NDVI change detection → mask deforestasi awal.

Bisa juga langsung via:

```bash
uv run python scripts/generate_labels.py \
  --chips data/training/unet/chips \
  --baseline-dir data/training/unet/baseline \
  --output data/training/unet/labels_ndvi
```

### 4. Review & Refine

```bash
make visualize  # atau langsung:
uv run streamlit run services/annotation-pipeline/reviewer.py
```

### 5. Split Dataset

```bash
uv run python scripts/split_dataset.py \
  --chips data/training/unet/chips \
  --labels data/training/unet/labels_ndvi \
  --manifest data/training/unet/manifest.json
```

### 6. Training U-Net

```bash
uv run python scripts/train_unet.py \
  --manifest data/training/unet/manifest.json \
  --output models/unet_deforest \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3
```

Training log otomatis tersimpan di `models/unet_deforest/train.log`.
Best model (val IoU tertinggi) di `best.pth`, final model di `final.pth`.

### 7. Inference

```bash
uv run python scripts/infer_unet.py \
  --manifest data/training/unet/manifest.json \
  --model models/unet_deforest/best.pth \
  --output data/training/unet/predictions \
  --batch-size 64
```

Output: `predictions.npy`, `ground_truth.npy`, `metrics.json`.

### 8. Visualisasi

```bash
uv run python scripts/compare_viz.py \
  --manifest data/training/unet/manifest.json \
  --predictions data/training/unet/predictions/predictions.npy \
  --output data/training/unet/comparisons \
  --num-samples 50
```

Menghasilkan 3-panel comparison (RGB / GT / Prediction).

### 9. Review Predictions (Streamlit)

```bash
uv run streamlit run services/annotation-pipeline/review_predictions.py
```

Fitur: navigasi per-sample, IoU histogram, error map (TP/FP/FN), metrics dashboard.

---

## Cek Progress

```bash
make status
```

Atau langsung cek direktori:

| Direktori | Isi |
|-----------|-----|
| `data/annotation/raw/` | GeoTIFF asli dari GEE |
| `data/annotation/tiles/` | Tile .npz 512px |
| `data/annotation/masks_auto/` | Mask auto NDVI |
| `data/annotation/masks_refined/` | Mask hasil review |
| `data/annotation/export/` | Training pairs |
| `data/training/unet/chips/` | Chips 64px |
| `data/training/unet/labels_ndvi/` | Mask final |
| `data/training/unet/predictions/` | Hasil inference |
| `data/training/unet/comparisons/` | Visualisasi |
| `models/unet_deforest/` | Model .pth files |

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| OOM saat training | Turunkan `--batch-size` |
| NaN loss | Cek normalisasi input (NIR /10000) |
| WSL hangs | Set `num_workers=0` |
| Cloud tidak terdeteksi | Sesuaikan threshold di `generate_labels.py` |
