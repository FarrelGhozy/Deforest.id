# Deforest.id

> **Sistem Monitoring Kerusakan Hutan Berbasis Grid & Early Warning System (EWS) Terotomatisasi — dengan landasan hukum dari data kawasan hutan resmi pemerintah (KLHK/BIG).**

Deforest.id mengubah pemantauan hutan dari **pasif menjadi proaktif** — secara otomatis menarik citra satelit dari GEE, mendeteksi kerusakan lahan dengan ML, menampilkan peta interaktif dengan overlay kawasan hutan resmi, dan mengirim notifikasi WhatsApp **lengkap dengan dasar hukum** ke petugas lapangan.

## Fitur Utama

| Fitur | Teknologi |
|-------|-----------|
| 🛰 Akuisisi citra satelit otomatis | Google Earth Engine Python API |
| 🧠 Deteksi kerusakan lahan per-grid | U-Net / YOLOv8 → TensorFlow Lite |
| 🗺 Overlay kawasan hutan resmi | KLHK SIGAP + BIG Satupeta (One Map Policy) |
| ⚖️ Legal-aware alert severity | Berdasarkan kombinasi ML confidence + legal status kawasan |
| 🔴🟡🟢 Kode warna grid + kawasan | Leaflet.js + MapLibre GL JS |
| 💬 Notifikasi WA + dasar hukum | Baileys (WhatsApp Web API) |
| 🐳 Distributed container | Docker + Docker Compose |

## Arsitektur

```
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐
  │ GEE      │  │ GIS      │  │ ML       │  │ PostgreSQL       │
  │ Fetcher  │  │ Overlay  │→ │ Inference│ →│ + PostGIS        │
  │(Python)  │  │(KLHK/BIG)│  │(Python)  │  │ ← forest_zones   │
  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘
                                                    │
  ┌──────────┐  ┌──────────┐              ┌─────────┴─────────┐
  │ Frontend │  │ WA Bot   │←─────────────│ Backend API       │
  │(React +  │  │(Baileys) │              │ Bun + Elysia      │
  │ Leaflet) │  │          │              └───────────────────┘
  └──────────┘  └──────────┘
```

### Alur Data

1. **GEE Fetcher** menarik citra Landsat/Sentinel secara periodik
2. **GIS Overlay Engine** menarik data kawasan hutan dari KLHK SIGAP & BIG Satupeta, melakukan spatial join dengan grid cells, mengklasifikasikan setiap grid ke HL/CA/HP/APL
3. **ML Worker** (U-Net) mendeteksi kerusakan per-grid
4. **Legal-aware trigger** menentukan alert severity berdasarkan ML confidence + legal status kawasan
5. **WA Bot** mengirim notifikasi dengan status kawasan, dasar hukum, dan ancaman pidana

## Quick Start

```bash
git clone https://github.com/yourusername/deforest.id.git && cd deforest.id
cp .env.example .env
# Isi: GEE credential, DB password, WA recipient, KLHK/BIG API
docker-compose up -d
```

## Dokumentasi Lengkap

- `docs/planning.md` — Rencana teknis, arsitektur, ERD, roadmap
- `docs/proposal.tex` — Dokumen proposal LaTeX (12 halaman)
- `docs/rancangan-storage.md` — 3 strategi penyimpanan + perbandingan biaya
- `docs/masalah.md` — 15 titik lemah + solusi mutakhir

## Kontak

Tim [Nama Tim] — [email]
