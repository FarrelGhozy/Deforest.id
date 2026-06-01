# Rancangan Training Data — GEE Pull untuk U-Net

> **Tujuan:** Mendapatkan 5.000+ citra satelit 64×64 px dari kawasan Hutan Lindung di Boven Digoel & Merauke (Papua Selatan) untuk training U-Net segmentasi deforestasi, dan komparasi dengan produk existing.

---

## 1. Strategi Umum

```
GEE Export ──► GeoTIFF besar (per scene) ──► Sliding Window ──► 64×64 chips ──► Dataset
                                                                       │
                                                    NDVI change ───────┤──► Mask auto
                                                    GFW loss layer ────┤──► Weak label
                                                    Manual refine ─────┘──► Ground truth
```

**Pendekatan:** Export GeoTIFF dari GEE pada resolusi 10m (Sentinel-2), lalu tile menjadi 64×64 px chips dengan sliding window di Python. Tidak export 1×1 grid dari GEE (boros EECU).

### Kenapa 64×64 px?

| Aspek | Nilai | Catatan |
|-------|-------|---------|
| Ukuran chip | 64×64 px | Standar U-Net input |
| Resolusi lapangan | 640×640 m | Sentinel-2 10m/pixel |
| Cakupan 1 chip | 40,96 ha | 1 grid ~ 2 grid 256m × 256m |
| Cakupan 5.000 chip | 204.800 ha | overlap + penyaringan awan |

### Sumber Data

| Sumber | Produk | Resolusi | Kegunaan |
|--------|--------|----------|----------|
| **Sentinel-2** (utama) | L2A (BOA) | 10m | Input U-Net (RGB + NIR) |
| **Landsat 8/9** (cadangan) | SR Collection 2 | 30m | Fallback jika S2 tertutup awan |
| **GFW / UMD Hansen** | Tree cover loss | 30m | **Weak label** otomatis |
| **KLHK SIGAP** | Kawasan hutan | vektor | Filter area Hutan Lindung |

---

## 2. Area of Interest (AOI)

### Boven Digoel & Merauke — Hutan Lindung

```
Kabupaten      Luas Total   Hutan Lindung (estimasi)    Potensi chips
────────────── ─────────── ─────────────────────────── ──────────────
Boven Digoel   ~18.000 km²         ~4.500 km²                7.000+
Merauke        ~45.000 km²        ~12.000 km²               19.000+
```

**Sumber polygon kawasan:**
- BIG Satupeta — layer Penetapan Kawasan Hutan: `https://kspservices.big.go.id/satupeta/rest/services/PUBLIK/KEHUTANAN/MapServer`
- KLHK SIGAP — layer fungsi kawasan

**Approach:** Tarik polygon Hutan Lindung (fungsitap=100100) dari BIG/KLHK, jadikan mask untuk GEE export.

### Strategi Sampling

```
                    ┌──────────────────┐
                    │   AOI Kawasan     │
                    │   Hutan Lindung   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌────────────┐ ┌────────────┐ ┌────────────┐
      │ Period T1  │ │ Period T2  │ │ Period T3  │
      │ (baseline) │ │ (post-def) │ │ (more data) │
      └────────────┘ └────────────┘ └────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │  Sliding window  │
                    │  → 64×64 chips   │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
            ┌────────────┐   ┌────────────┐
            │ 50% No-deforest │ 50% Deforest │
            │ (Hansen loss=0) │ (Hansen loss>0)│
            └────────────────┘ └──────────────┘
```

**Target distribusi:** 50% negatif (hutan utuh), 50% positif (deforestasi) — untuk menghindari class imbalance.

---

## 3. Pipeline Lengkap

### Fase A: Setup & Authentikasi

```
1. Setup GEE service account
2. Install earthengine-api (tambah ke pyproject.toml)
3. Pull polygon kawasan HL dari BIG Satupeta
4. Simpan polygon sebagai GeoJSON
```

### Fase B: GEE Image Export

Script Python `gee_export.py`:

```python
# Pseudocode — logika utama
for each scene in time_series:
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_hl_polygon)    # Hanya Hutan Lindung
        .filterDate(t1, t2)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))

    composite = collection.median()

    # Bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), QA60
    export = ee.batch.Export.image.toDrive(
        image=composite.select(["B2","B3","B4","B8","QA60"]),
        region=aoi_hl_polygon,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
```

**Constraint GEE:**
- Community tier: ~30.000 EECU/bulan
- 1 scene export (500 km²) ≈ 10.000–15.000 EECU
- Solusi: batch semalaman, atau pakai Contributor tier (unlimited)

### Fase C: Tiling 64×64 Chips

Script `tile_unet.py` — modifikasi dari `preprocess.py`:

```python
def tile_unet(geotiff_path: Path, chip_size=64, stride=64):
    """Sliding window → 64×64 chips, simpan sebagai .npz"""
    data, meta = read_geotiff(geotiff_path)

    for y in range(0, height - chip_size + 1, stride):
        for x in range(0, width - chip_size + 1, stride):
            chip = data[:, y:y+chip_size, x:x+chip_size]

            # Filter cloud (>30% skip)
            if cloud_fraction(chip) > 0.3:
                continue

            # Filter vegetasi (mean NDVI < 0 = bukan hutan)
            if mean_ndvi(chip) < 0:
                continue

            # Simpan
            np.savez_compressed(
                f"{output_dir}/{scene_name}_{y}_{x}.npz",
                rgb=chip[:3],        # B4, B3, B2
                nir=chip[3],         # B8
            )
```

**Target output per scene besar:**

| Scene size | Chips (64×64, stride=64) | Setelah filter awan | Setelah filter NDVI |
|------------|--------------------------|---------------------|---------------------|
| 10.000×10.000 px | ~24.000 | ~16.000 | ~10.000 |

Dari **3–5 scene** besar di Boven Digoel + Merauke sudah dapat 5.000+ chips bersih.

### Fase D: Labeling / Ground Truth

**Opsi 1 — Weak Label dari GFW Hansen (Otomatis, Cepat)**

```
GFW Tree Cover Loss 2020-2024 (30m) → resize ke 10m → overlay 64×64 chip
                                                         │
                                          ┌──────────────┴──────────────┐
                                          ▼                             ▼
                                  forest_loss > 0 px             loss == 0 px
                                  dalam chip                       dalam chip
                                          │                             │
                                          ▼                             ▼
                                  Label: DEFOREST              Label: NO_DEFOREST
```

**Opsi 2 — NDVI Change (Semi-otomatis)**

```
T1 (baseline) ──► NDVI map ──┐
                              ├──► delta NDVI ──► threshold ──► binary mask
T2 (post)     ──► NDVI map ──┘
```

**Opsi 3 — Manual Label via Streamlit**

Pakai existing `visualizer.py` untuk refine mask.

**Rekomendasi:** Opsi 1 untuk initial training, Opsi 2 untuk augmentasi, Opsi 3 untuk validation set.

### Fase E: Struktur Dataset

```
data/training/unet/
├── raw/                  # GeoTIFF hasil GEE export
│   ├── bovendigoel_2020_06.tif     # T1 baseline
│   ├── bovendigoel_2023_06.tif     # T2 deforestasi
│   ├── merauke_2020_06.tif
│   └── merauke_2023_06.tif
│
├── chips/                # 64×64 tiles hasil sliding window
│   ├── bovendigoel_2020_06_100_200.npz
│   ├── bovendigoel_2020_06_100_264.npz
│   └── ...
│
├── labels_gfw/           # Weak label dari GFW Hansen
│   ├── bovendigoel_2020_06_100_200_mask.npz
│   └── ...
│
├── labels_ndvi/          # Label dari NDVI change
│   └── ...
│
├── train/                # Train set (70%)
│   ├── images/           # .npy atau .png 64×64×3
│   └── masks/            # .npy 64×64 binary
│
├── val/                  # Validation set (20%)
│   ├── images/
│   └── masks/
│
└── test/                 # Test set (10%)
    ├── images/
    └── masks/
```

---

## 4. Perbandingan dengan Produk Existing

### Matriks Komparasi

| Aspek | GFW (Hansen) | DETER-B | RADD | **Deforest.id (U-Net)** |
|-------|-------------|---------|------|------------------------|
| **Data** | Landsat 30m | MODIS 250m | Sentinel-1 SAR 10m | **Sentinel-2 10m** |
| **Deteksi** | Perubahan tutupan pohon | Hotspot deforestasi | Perubahan kanopi | **Segmentasi per-grid** |
| **Resolusi** | 30m (≈0,09 ha) | 250m (≈6,25 ha) | 10m (≈1 ha) | **10m (0,01 ha)** |
| **Akurasi** | ~80% global | ~75% Amazon | ~85% | Target: **>90%** |
| **Awan** | Tidak bisa | Tidak bisa | **Bisa (SAR)** | Terbatas |
| **Latency** | Tahunan | ~2 minggu | ~6 hari | **<1 hari** |
| **Biaya** | Gratis | Institusi | Riset | **VPS Rp400rb/bln** |

### Keunggulan Deforest.id

1. **Resolusi 64×64 px dari 10m S2** — mendeteksi deforestasi skala kecil yang lolos dari GFW (30m) dan DETER (250m)
2. **Fine-tuning untuk ekosistem Indonesia** — model global (GFW) akurasinya turun di hutan tropis heterogen
3. **Integrasi kawasan hukum** — tidak ada produk existing yang overlay dengan data One Map Policy
4. **Label dari GFW sebagai weak label** — leverage data existing untuk initial training, lalu fine-tune dengan label manual

---

## 5. Kebutuhan Infrastruktur

| Item | Kebutuhan | Biaya |
|------|-----------|-------|
| Storage dataset | 5.000 chips × ~50 KB = **~250 MB** | ✅ Gratis |
| Storage GeoTIFF | 4–6 file × ~500 MB = **~3 GB** | ✅ Gratis |
| GEE EECU | ~4 export × 15.000 = **60.000 EECU** | ⚠️ Butuh Contributor tier |
| Compute tiling | Laptop/PC lokal | ✅ |
| Compute training | GPU (Colab / lokal RTX) | Colab gratis atau sewa |

---

## 6. Timeline Estimasi

| Fase | Waktu | Deliverable |
|------|-------|-------------|
| **Setup** & polygon HL | 1 hari | GeoJSON kawasan Hutan Lindung |
| **GEE Export** (batch) | ~4 jam (proses GEE) | 4–6 GeoTIFF besar |
| **Tiling** ke 64×64 | 30 menit | ~10.000 chips .npz |
| **Weak labeling** GFW | 30 menit | ~5.000 mask GFW |
| **NDVI change labeling** | 1 jam | ~5.000 mask NDVI |
| **Split** train/val/test | 5 menit | Dataset siap training |
| **U-Net training awal** | ~2 jam (Colab) | Model .h5 pertama |
| **Evaluasi** vs GFW | 1 jam | Matriks komparasi |

---

## 7. Resiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| GEE Community quota habis | Export gagal | Batch bertahap, kompresi, atau upgrade Contributor |
| Tutupan awan >80% (Papua) | Chips terlalu sedikit | Kombinasi multi-temporal, pakai SAR nanti |
| GFW weak label tidak akurat | Model belajar salah | Sampling manual validation set (10%) |
| Hutan Lindung polygon tidak lengkap di BIG | AOI tidak akurat | Fallback ke KLHK SIGAP, atau buffer |
| Overlap antar chips | Data leakage | Pastikan train/val split by scene, not by chip |

---

## 8. Persetujuan

- [ ] **AOI:** Boven Digoel + Merauke — Hutan Lindung (fungsitap=100100)
- [ ] **Ukuran chip:** 64×64 px (10m resolution → 640m×640m lapangan)
- [ ] **Target:** 5.000+ chips (50/50 deforest/no-deforest)
- [ ] **Label awal:** Weak label dari GFW Hansen Tree Cover Loss
- [ ] **Label tambahan:** NDVI change + manual refine lewat Streamlit
- [ ] **Output:** Dataset siap untuk training U-Net segmentasi
