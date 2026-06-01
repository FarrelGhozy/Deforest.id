# Data Format & Struktur Direktori

> **Untuk:** Farrel (GEE Fetcher)
> **Tujuan:** Dokumentasi format output GEE fetcher agar kompatibel dengan annotation pipeline.

---

## Sumber Data

| Parameter | Nilai |
|-----------|-------|
| Platform | Google Earth Engine |
| Satellite | Sentinel-2 (Level-2A) |
| Bands | B2, B3, B4, B8, QA60 |
| Resolusi | 10m (B2–B8) |
| Area | Kalimantan (AOI sesuai batas konsesi GFW) |
| Temporal | Multi-temporal (minimal 2 titik waktu: T1 dan T2) |

## Output Format per Scene

Setiap scene dari GEE harus diexport sebagai **GeoTIFF**:

```
data/annotation/raw/
├── kalimantan_2025_01_15.tif     # T1 — awal
├── kalimantan_2025_06_20.tif     # T2 — akhir
└── ...
```

### GeoTIFF Requirements

| Property | Requirement |
|----------|-------------|
| Format | GeoTIFF, 32-bit float |
| CRS | EPSG:4326 (WGS84) |
| Bands | 5 bands: B2, B3, B4, B8, QA60 |
| NoData | 0 |
| Region | Clip ke AOI |

### Band Order (WAJIB)

| Index | Band | Keterangan |
|-------|------|------------|
| 0 | B2 | Blue — 10m |
| 1 | B3 | Green — 10m |
| 2 | B4 | Red — 10m |
| 3 | B8 | NIR — 10m |
| 4 | QA60 | Cloud mask — 60m |

!!! warning "Urutan Band"
    Urutan bands HARUS sesuai tabel. Annotation pipeline membaca band berdasarkan index, bukan nama.

### Naming Convention

```
{region}_{tahun}_{bulan}_{hari}.tif
```

Contoh: `kalimantan_2025_01_15.tif`, `kalimantan_2025_06_20.tif`

Region wajib **sama** untuk scene yang akan dibandingkan (T1 dan T2).

## Struktur Direktori Lengkap

```
data/
└── annotation/
    ├── raw/                    # ← Farrel taruh GeoTIFF di sini
    ├── tiles/                  # Auto — hasil tiling
    ├── masks_auto/             # Auto — hasil NDVI change
    ├── masks_refined/          # Manual — hasil refine Streamlit
    └── export/                 # Auto — training set
        ├── masks_auto/
        │   ├── train/img + mask/
        │   └── val/img + mask/
        └── masks_refined/
```

## GEE Export Snippet

```python
import ee

def export_scene(aoi, date_start, date_end, description):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )
    image = collection.median()
    bands = ["B2", "B3", "B4", "B8", "QA60"]
    selected = image.select(bands)

    task = ee.batch.Export.image.toDrive(
        image=selected,
        description=description,
        folder="deforest_raw",
        fileNamePrefix=description,
        region=aoi,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    return task
```

## Checklist

- [ ] GeoTIFF sudah di-clip ke AOI
- [ ] Band order: B2, B3, B4, B8, QA60
- [ ] CRS: EPSG:4326
- [ ] NoData: 0
- [ ] Nama file: `{region}_{YYYY}_{MM}_{DD}.tif`
- [ ] Minimal 2 scene region sama
- [ ] Cloud cover < 30%
