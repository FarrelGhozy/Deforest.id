# Deforest.id 🌲🚨

> **Sistem Monitoring Kerusakan Hutan Berbasis Grid & Early Warning System (EWS) Terotomatisasi**

Deforest.id mengubah pemantauan hutan dari **pasif menjadi proaktif** — secara otomatis menarik citra satelit, mendeteksi kerusakan lahan dengan ML, menampilkan peta interaktif dengan kode warna, dan mengirim notifikasi WhatsApp langsung ke petugas lapangan.

---

## Fitur Utama

| Fitur | Teknologi |
|-------|-----------|
| 🛰 Akuisisi citra satelit otomatis | Google Earth Engine Python API |
| 🧠 Deteksi kerusakan lahan per-grid | YOLOv8 / TensorFlow Lite |
| 🗺 Dashboard peta real-time | Leaflet.js + Mapbox |
| 🔴🟡🟢 Kode warna grid | Merah (severe), Kuning (moderate), Hijau (healthy) |
| 💬 Notifikasi WhatsApp otomatis | Baileys (WhatsApp Web API) |
| 🐳 Distributed container architecture | Docker + Proxmox |

---

## Arsitektur Sistem

```
                        PROXMOX HOST
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐
  │ GEE      │  │ ML       │  │ Backend  │  │ PostgreSQL       │
  │ Fetcher  │ →│ Inference│ →│ Bun/Node │ →│ + PostGIS        │
  │ (Python) │  │ (Python) │  │ (API)    │  │                  │
  └──────────┘  └──────────┘  └────┬─────┘  └──────────────────┘
                                    │
  ┌──────────┐  ┌──────────┐       │        ┌──────────────────┐
  │ Frontend │  │ WA Bot   │       └────────│ Redis Cache      │
  │(React +  │  │(Baileys) │                │                  │
  │ Leaflet) │  │          │                │                  │
  └──────────┘  └──────────┘                └──────────────────┘
```

### Alur Data

1. **GEE Fetcher** menarik citra Landsat/Sentinel secara periodik
2. Citra dibagi ke **grid koordinat** (256m × 256m)
3. **ML Worker** (YOLOv8) mendeteksi anomali per-grid
4. Hasil deteksi disimpan di **PostgreSQL + PostGIS**
5. **Backend API** (Bun + Elysia) menyajikan data via REST & WebSocket
6. **Dashboard** menampilkan peta dengan grid berwarna real-time
7. **WA Bot** mengirim notifikasi jika terdeteksi kerusakan (confidence ≥ 70%)

---

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Data Source | Google Earth Engine |
| Machine Learning | YOLOv8 (Ultralytics) → TensorFlow Lite / ONNX |
| Backend | Bun + Elysia.js |
| Database | PostgreSQL 16 + PostGIS 3.4 |
| Cache & Queue | Redis 7 |
| Frontend | React 18 + Vite + TypeScript |
| Map | Leaflet.js |
| Charts | D3.js |
| WA Bot | Baileys (Node.js) |
| Infrastruktur | Docker + Docker Compose + Proxmox |
| Reverse Proxy | Nginx (Alpine) |

---

## Struktur Proyek

```
deforest.id/
├── docker-compose.yml
├── .env.example
├── services/
│   ├── gee-fetcher/         # Python - GEE API client
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── fetcher.py
│   │       ├── grid_generator.py
│   │       └── image_cropper.py
│   ├── ml-inference/        # Python - YOLOv8 inference
│   │   ├── Dockerfile
│   │   ├── models/
│   │   └── src/
│   │       ├── inference.py
│   │       ├── preprocess.py
│   │       └── postprocess.py
│   ├── backend-api/         # Bun/Elysia - REST + WebSocket
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── routes/
│   │       ├── ws/
│   │       └── db/
│   ├── frontend/            # React + Vite + Leaflet
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── components/
│   │       │   ├── MapView.tsx
│   │       │   ├── GridLayer.tsx
│   │       │   ├── Sidebar.tsx
│   │       │   ├── DetailModal.tsx
│   │       │   └── TimelineChart.tsx
│   │       └── hooks/
│   ├── wa-bot/              # Baileys WhatsApp Bot
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── bot.ts
│   │       └── alert-poller.ts
│   └── nginx/
├── database/
│   ├── migrations/
│   └── seeds/
└── docs/
    ├── planning.md
    ├── proposal.tex
    └── proposal.pdf
```

---

## Database Schema (PostGIS)

| Tabel | Fungsi |
|-------|--------|
| `grid_cells` | Grid koordinat dengan geometry polygon & status warna |
| `satellite_imagery` | Metadata citra satelit dari GEE |
| `detection_logs` | Hasil deteksi ML (confidence, bounding boxes, kategori) |
| `alerts` | Peringatan otomatis (trigger saat confidence ≥ 70%) |
| `notification_logs` | Audit trail pengiriman notifikasi WA |
| `config` | Konfigurasi dinamis (threshold, interval, recipient) |

Trigger otomatis membuat alert saat deteksi **severe/moderate** dengan confidence ≥ 70%.

---

## Resource Container (Proxmox)

| Container | vCPU | RAM | Storage |
|-----------|------|-----|---------|
| GEE Fetcher | 1 | 2 GB | 50 GB |
| ML Inference | 4 | 8 GB | 20 GB |
| Backend API | 1 | 1 GB | 1 GB |
| PostgreSQL + PostGIS | 2 | 4 GB | 100 GB |
| Frontend | 1 | 1 GB | 1 GB |
| WA Bot | 1 | 512 MB | 1 GB |
| Redis | 1 | 1 GB | 5 GB |
| Nginx | 0.5 | 256 MB | 500 MB |
| **Total** | **11.5** | **~18 GB** | **~178 GB** |

---

## Roadmap

| Fase | Hari | Target |
|------|------|--------|
| **F0** Setup | 1–2 | Proxmox, Docker, repo struktur |
| **F1** Data | 3–4 | GEE fetcher, grid generator |
| **F2** ML | 5–7 | YOLOv8 pipeline, alert trigger |
| **F3** API | 5–7 | Backend REST + WebSocket |
| **F4** Bot | 7–8 | WhatsApp notifikasi |
| **F5** UI | 8–10 | Dashboard Leaflet.js |
| **F6** Test | 10–11 | Integration & E2E testing |
| **F7** Demo | 12 | Deploy & presentasi |

---

## Cara Memulai

```bash
# Clone repo
git clone https://github.com/yourusername/deforest.id.git
cd deforest.id

# Copy environment variables
cp .env.example .env
# Isi konfigurasi: GEE credential, DB password, WA recipient, dll.

# Jalankan semua service
docker-compose up -d

# Akses dashboard
open http://localhost:80
```

---

## Kontak

Tim [Nama Tim] — [email]

> *"Dari Pasif Menjadi Proaktif: Mengawal Hutan Indonesia dengan Teknologi"*
# Deforest.id
