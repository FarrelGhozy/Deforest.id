# Deforest.id

**Early Warning System Deforestasi** — deteksi otomatis kerusakan hutan berbasis
citra satelit Sentinel-2 dan deep learning (U-Net).

---

## Hasil Training (v1 — Baseline)

**U-Net** — 50 epochs — RTX 3060 — 5 channel (RGB + NIR + NDVI)

| Metrik | Val | Test |
|--------|-----|------|
| **IoU** | **0.3831** | **0.7256** |
| Dice | — | 0.8410 |
| Precision | — | 0.8340 |
| Recall | — | 0.8480 |

### Sample Predictions

=== "Good Prediction (IoU 0.99)"
    ![](assets/hl_sample_1_deforest_256_2304_iou0.993.png)

=== "Moderate (IoU 0.61)"
    ![](assets/hl_sample_1_deforest_64_1920_iou0.866.png)

=== "Failed (IoU 0.00)"
    ![](assets/hl_sample_2_deforest_768_2752_iou0.000.png)

### Dataset

| Split | Samples | Scenes |
|-------|---------|--------|
| Train | 11,500 | 4 |
| Val   | 2,908  | 1 |
| Test  | 5,695  | 2 |
| **Total** | **20,103** | **7** |

---

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| **Model** | U-Net (PyTorch) |
| **Data** | Sentinel-2 (GEE) |
| **Inference** | Python / ONNX |
| **Backend** | Bun + Elysia.js |
| **Database** | PostgreSQL 16 + PostGIS 3.4 |
| **Frontend** | React 18 + Vite + TypeScript + Leaflet.js |
| **Notifikasi** | Baileys WhatsApp Bot |
| **Infra** | Docker + Proxmox |

---

## Struktur Repositori

```
docs/                          ← MkDocs documentation (GitHub Pages)
services/
  annotation-pipeline/         ← Data preprocessing & labeling
scr/
  training-unet/               ← Training & inference scripts
database/migrations/           ← PostgreSQL + PostGIS schemas
```

---

## Status Pengembangan

- [x] Akuisisi data Sentinel-2 (GEE)
- [x] Preprocessing & cloud masking
- [x] NDVI-based auto-annotation
- [x] Streamlit review & refine
- [x] **U-Net training v1 baseline**
- [x] Inference & visualization
- [ ] Backend API (Elysia.js)
- [ ] Frontend map (React + Leaflet)
- [ ] WhatsApp notification bot
- [ ] Container deployment
