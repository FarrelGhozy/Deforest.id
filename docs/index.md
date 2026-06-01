# Deforest.id

**Early Warning System Deforestasi** — deteksi otomatis kerusakan hutan berbasis citra satelit,
U-Net segmentation, dan notifikasi real-time.

---

## Dokumentasi

| Bagian | Deskripsi |
|--------|-----------|
| [Data Format](data-format.md) | Format output GEE, struktur direktori, naming convention |
| [Annotation Pipeline](annotation-pipeline.md) | Pipeline anotasi NDVI change → refine → export training set |
| [Getting Started](getting-started.md) | Quick start dari clone sampai pipeline jalan |

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| ML Framework | U-Net (TensorFlow/Keras) |
| Data Source | Google Earth Engine (Sentinel-2) |
| Backend | Bun + Elysia.js |
| Database | PostgreSQL 16 + PostGIS 3.4 |
| Frontend | React 18 + Vite + TypeScript + Leaflet.js |
| Notifikasi | Baileys (WhatsApp Bot) |
| Infrastructure | Docker + Proxmox |

## Repository Structure

```
.
├── docs/                          # Dokumentasi (MkDocs → GitHub Pages)
├── services/
│   ├── annotation-pipeline/       # Pipeline anotasi (Python / uv)
│   ├── gee-fetcher/               # GEE data fetcher (Python)
│   ├── ml-inference/              # U-Net training & inference
│   ├── backend-api/               # REST API + WebSocket (Bun/Elysia)
│   ├── frontend/                  # Dashboard (React/Vite)
│   └── wa-bot/                    # WhatsApp bot (Baileys)
├── database/
│   └── migrations/                # SQL migrations
├── mkdocs.yml                     # Konfigurasi MkDocs
├── docker-compose.yml
├── proposal.tex
└── README.md
```

> **Status:** Tahap awal pengembangan — dokumentasi dan pipeline anotasi sedang dibangun.
