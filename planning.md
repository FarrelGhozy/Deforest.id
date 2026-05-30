# Deforest.id — Technical Project Plan

> **Sistem Monitoring Kerusakan Hutan Berbasis Grid & Early Warning System (EWS) Terotomatisasi**
> Hackathon Project — Lead Engineer: [Nama Tim]

---

## 1. Ringkasan Eksekutif & Unique Selling Proposition (USP)

### 1.1 Latar Belakang

Deforestasi di Indonesia merupakan salah satu yang tertinggi di dunia. Sistem monitoring yang ada saat ini bersifat **reaktif dan pasif** — data tersedia tetapi tidak ada mekanisme peringatan dini yang otomatis dan langsung menjangkau pemangku kepentingan di lapangan.

### 1.2 Solusi: Deforest.id

Deforest.id adalah platform **Early Warning System (EWS)** yang mengubah pemantauan hutan dari pasif menjadi **proaktif dan real-time**. Sistem ini secara otomatis:
1. Menarik citra satelit dari **Google Earth Engine**
2. Membagi wilayah hutan ke dalam **grid koordinat** terstandarisasi
3. Mendeteksi anomali/kerusakan lahan menggunakan **YOLOv8 / TensorFlow Lite**
4. Menampilkan hasil dalam **dashboard interaktif berbasis peta** dengan sistem warna (merah/kuning/hijau)
5. Mengirim **notifikasi WhatsApp otomatis** berisi koordinat, confidence score, dan gambar

### 1.3 Unique Selling Proposition (USP) — Untuk Presentasi Juri

| USP | Deskripsi | Dampak |
|-----|-----------|--------|
| **Proaktif vs Reaktif** | Sistem menjemput bola — tidak perlu menunggu laporan manual | Respon 24-48 jam lebih cepat |
| **Grid-Based Precision** | Pembagian area ke dalam grid koordinat memungkinkan deteksi hingga level sub-hektar | Akurasi spasial tinggi |
| **Multi-Platform Alert** | WhatsApp Bot mengirim notifikasi langsung ke HP petugas | Aksesibilitas maksimal tanpa perlu internet mahal |
| **Distributed Architecture** | ML berjalan di container terpisah, tidak membebani backend | Skalabilitas horizontal |
| **Zero Manual Intervention** | Full pipeline otomatis dari akuisisi data hingga notifikasi | Cocok untuk daerah dengan SDM terbatas |

### 1.4 Perbedaan dengan Sistem Existing

| Aspek | Sistem Tradisional | Deforest.id |
|-------|-------------------|-------------|
| Sumber Data | Survey lapangan (manual) | Satelit (otomatis, periodik) |
| Deteksi | Visual interpretasi | Machine Learning (YOLOv8) |
| Pelaporan | Laporan PDF mingguan/bulanan | Dashboard real-time + WA notifikasi |
| Biaya | Tinggi (surveyor, transportasi) | Rendah (hanya infrastruktur cloud) |
| Skalabilitas | Terbatas SDM | Horizontal (Docker + Proxmox) |

---

## 2. Arsitektur Sistem & Alur Data

### 2.1 Arsitektur High-Level

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROXMOX HOST                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Container│  │ Container│  │ Container│  │   Container      │   │
│  │    1     │  │    2     │  │    3     │  │       4          │   │
│  │ GEE Data │  │  ML      │  │ Backend  │  │   PostgreSQL     │   │
│  │ Fetcher  │  │ Inference│  │ Bun/Node │  │   + PostGIS      │   │
│  │ (Python) │  │(Python)  │  │ (API)    │  │                  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────────┘   │
│       │              │             │                               │
│  ┌────┴─────┐  ┌────┴─────┐       │           ┌──────────────────┐│
│  │ Container│  │ Container│       │           │   Container      ││
│  │    5     │  │    6     │       └───────────│       7          ││
│  │ Frontend │  │  WA Bot  │                   │   Redis Cache    ││
│  │(React +  │  │(Baileys) │                   │                  ││
│  │ Leaflet) │  │          │                   │                  ││
│  └──────────┘  └──────────┘                   └──────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Alur Data (Data Flow) — Detail

#### FASE 1: Akuisisi Data Satelit

```
GEE Python API
    │
    ├── Menarik citra Landsat 8/9 atau Sentinel-2
    ├── Periode: Setiap N jam (konfigurabel, default 6 jam)
    ├── Filter: Cloud cover < 20%, area of interest (AOI) polygon
    │
    ▼
Raw Satellite Imagery (GeoTIFF)
    │
    ├── Disimpan sementara di shared Docker volume
    ├── Metadata dicatat ke tabel `satellite_imagery`
    │
    ▼
Grid Generator Module
    │
    ├── Membagi AOI menjadi grid berukuran tetap (contoh: 256m x 256m)
    ├── Setiap grid memiliki: id, centroid (lat/lng), polygon (WKT/GeoJSON)
    ├── Grid disimpan di tabel `grid_cells` (PostGIS geometry)
    │
    ▼
Per-Grid Image Crop
    │
    ├── Setiap grid dipotong (clip) dari citra besar
    ├── Disimpan sebagai: /data/grids/{grid_id}/{timestamp}.png
    ├── Antrian (Queue) dibuat untuk ML Inference
```

#### FASE 2: Machine Learning Inference

```
Queue: /data/grids/{grid_id}/{timestamp}.png
    │
    ▼
ML Container (YOLOv8 / TFLite)
    │
    ├── Load model yang sudah di-train (deforestation detection)
    ├── Input: image tile per grid
    ├── Proses:
    │   ├── Preprocessing (resize ke 640x640, normalize)
    │   ├── Inference (forward pass)
    │   └── Postprocessing (NMS, thresholding)
    │
    ├── Output per grid:
    │   ├── has_damage: boolean
    │   ├── confidence_score: float (0.0 - 1.0)
    │   ├── damage_category: "severe" | "moderate" | "mild" | "none"
    │   ├── bounding_boxes: [{x1, y1, x2, y2, class, confidence}]
    │   └── processed_image: annotated image (dengan bounding box overlay)
    │
    ▼
Results Writer
    │
    ├── Menyimpan hasil ke tabel `detection_logs`
    ├── Menyimpan annotated image ke shared volume
    ├── Jika confidence > threshold (default 0.7):
    │   └── Trigger Alert → masuk ke tabel `alerts`
    │
    ▼
Update Grid Status
    ├── UPDATE grid_cells SET status = 'damaged' / 'healthy' / 'unknown'
    └── UPDATE grid_cells SET last_detection_id = ... 
```

#### FASE 3: Backend API & Database

```
Backend (Bun/Node.js — Elysia/Fastify)
    │
    ├── REST API Endpoints:
    │   ├── GET /api/grids?bbox=...&status=damaged
    │   │   → Return grid cells dalam bounding box (GeoJSON)
    │   ├── GET /api/grids/:id/history
    │   │   → Return historical detection logs untuk grid tertentu
    │   ├── GET /api/stats/summary
    │   │   → Return statistik: total damaged, healthy, trend
    │   ├── GET /api/alerts?since=...
    │   │   → Return alert history
    │   └── POST /api/alerts/acknowledge/:id
    │       → Tandai alert sebagai sudah ditindaklanjuti
    │
    ├── WebSocket (real-time updates):
    │   ├── new_detection → push ke dashboard saat deteksi baru
    │   ├── new_alert → push alert real-time
    │   └── grid_update → update status grid tertentu
    │
    ├── Redis Cache:
    │   ├── Cache hasil query grid (TTL: 5 menit)
    │   ├── Pub/Sub untuk WebSocket broadcasting
    │   └── Queue management untuk ML job
    │
    ▼
PostgreSQL + PostGIS
    ├── Tabel spasial: grid_cells, detection_logs, alerts
    ├── Spatial queries (ST_Within, ST_Intersects, ST_Area)
    └── Time-series logging
```

#### FASE 4: Dashboard Frontend

```
Frontend (React + Vite + Leaflet.js)
    │
    ├── Map Component (Leaflet.js):
    │   ├── Base map: OpenStreetMap / Satellite tile
    │   ├── Overlay: Grid cells dengan warna:
    │   │   ├── 🟢 Hijau: Tidak ada kerusakan (confidence < 0.3)
    │   │   ├── 🟡 Kuning: Kerusakan ringan (0.3 <= confidence < 0.7)
    │   │   └── 🔴 Merah: Kerusakan parah (confidence >= 0.7)
    │   ├── Cluster layer untuk grid dalam jumlah besar
    │   └── Click handler: menampilkan detail grid
    │
    ├── Sidebar / Panel:
    │   ├── Statistik ringkasan (total area, % rusak, trend)
    │   ├── Recent alerts list (real-time dari WebSocket)
    │   ├── Filter: status, tanggal, confidence threshold
    │   └── Search: koordinat atau lokasi
    │
    ├── Detail Modal (saat grid di-click):
    │   ├── Citra asli vs annotated image (side-by-side)
    │   ├── Confidence score
    │   ├── Kategori kerusakan
    │   ├── Timeline chart (confidence over time)
    │   └── Tombol "Kirim Alert" manual
    │
    └── Alert Notification (in-app):
        ├── Toast notification saat deteksi baru
        ├── Sound alert (opsional)
        └── Link ke WhatsApp untuk tindak lanjut
```

#### FASE 5: WhatsApp Notification

```
WA Bot Container (Node.js + Baileys)
    │
    ├── Koneksi: Baileys (WebSocket-based WhatsApp API)
    ├── Autentikasi: QR Code scan (sekali, disimpan di session file)
    │
    ├── Alert Trigger:
    │   └── polling setiap 30 detik ke tabel `alerts` WHERE sent=false
    │
    ├── Format Pesan:
    │   ├── ┌─────────────────────────────────────────┐
    │   │   │ 🚨 *PERINGATAN DINI DEFORESTASI*        │
    │   │   │                                         │
    │   │   │ *Lokasi:*                               │
    │   │   │ Lat: -3.4567, Lng: 114.9876             │
    │   │   │ Grid ID: GRID-2024-A1                   │
    │   │   │                                         │
    │   │   │ *Tingkat Kerusakan:* 🔴 SEVERE          │
    │   │   │ *Confidence:* 94.3%                     │
    │   │   │ *Waktu Deteksi:* 2024-01-15 14:30:22    │
    │   │   │                                         │
    │   │   │ 📍 https://maps.google.com/?q=-3.4567,  │
    │   │   │    114.9876                             │
    │   │   │ 🌐 Lihat Dashboard: https://deforest.id │
    │   │   └─────────────────────────────────────────┘
    │   └── (gambar annotated overlay dilampirkan)
    │
    ├── After Send:
    │   ├── UPDATE alerts SET sent=true, sent_at=NOW()
    │   └── Log status pengiriman
    │
    └── Error Handling:
        ├── Queue ulang jika gagal (max 3 retry)
        ├── Log error ke tabel `notification_logs`
        └── Fallback: SMS gateway (cadangan)
```

### 2.3 Diagram Squence — Skenario End-to-End

```
Timeline:
│
├─ [T=0]     GEE Fetcher menjadwalkan pull citra
├─ [T+2min]  Citra ditarik, grid dipotong-potong
├─ [T+5min]  ML Container memproses 100 grid batch
├─ [T+7min]  Hasil deteksi disimpan di database  
├─ [T+7.5min] Backend push update via WebSocket
├─ [T+8min]  Dashboard menampilkan grid merah baru
├─ [T+8.5min] WA Bot mendeteksi alert baru
├─ [T+9min]  Alert terkirim ke nomor tujuan
└─ [T+∞]     Petugas menerima notifikasi di HP
```

---

## 3. Rincian Tech Stack & Alokasi Resource

### 3.1 Tech Stack Final

| Komponen | Teknologi | Alasan Pemilihan |
|----------|-----------|------------------|
| **Data Source** | Google Earth Engine Python API | Akses gratis ke Landsat/Sentinel, cloud-free composite |
| **ML Framework** | YOLOv8 (Ultralytics) | Ringan, akurasi tinggi, inference cepat di CPU |
| **ML Optimization** | TensorFlow Lite / ONNX | Deployment CPU-friendly, ukuran model kecil (~5-10MB) |
| **Backend API** | Bun + Elysia.js | Runtime cepat (3x Node.js), native TS support, ringan di container |
| **Database** | PostgreSQL 16 + PostGIS 3.4 | Spatial query support, indexing R-Tree, robust |
| **Cache & Queue** | Redis 7 | Pub/Sub untuk WebSocket, queue management, fast cache |
| **Frontend** | React 18 + Vite + TypeScript | Build cepat, HMR, ecosystem matang |
| **Map Library** | Leaflet.js 1.9 | Open-source, tidak perlu API key, plugin ecosystem |
| **Visualization** | D3.js (untuk chart timeline) | Fleksibel untuk data time-series |
| **WA Bot** | Baileys (Node.js) | Library WhatsApp Web, gratis, komunitas aktif |
| **Container Runtime** | Docker + Docker Compose | Orchestrasi sederhana untuk Proxmox |
| **Reverse Proxy** | Nginx (dalam container) | Load balancing, SSL termination, static file serving |
| **Storage** | Docker Volumes (NFS-backed) | Shared storage antar container |

### 3.2 Alokasi Resource Proxmox / Docker

#### Asumsi: Single Proxmox host dengan 16 vCPU, 32GB RAM, 500GB SSD

| Container | CPU | RAM | Storage | Image Size | Priority |
|-----------|-----|-----|---------|------------|----------|
| **1. GEE Fetcher** (Python) | 1 vCPU | 2 GB | 50 GB | ~1.2 GB | Runs periodically |
| **2. ML Inference** (Python) | 4 vCPU | 8 GB | 20 GB | ~3.5 GB (includes model) | **HIGH** — bottleneck utama |
| **3. Backend API** (Bun) | 1 vCPU | 1 GB | 1 GB | ~500 MB | **HIGH** — serving requests |
| **4. PostgreSQL + PostGIS** | 2 vCPU | 4 GB | 100 GB | ~1.5 GB | **HIGH** — I/O intensive |
| **5. Frontend** (React/Vite) | 1 vCPU | 1 GB | 1 GB | ~800 MB (build) | Low — static files |
| **6. WA Bot** (Node.js) | 1 vCPU | 512 MB | 1 GB | ~400 MB | Low — event-based |
| **7. Redis** | 1 vCPU | 1 GB | 5 GB | ~150 MB | Medium — cache + pub/sub |
| **8. Nginx** | 0.5 vCPU | 256 MB | 500 MB | ~50 MB | Low — reverse proxy |
| **Total Alokasi** | **11.5 vCPU** | **17.8 GB** | **~178 GB** | | |

> **Catatan:** Sisa resource (4.5 vCPU, ~14 GB RAM) digunakan untuk overhead Proxmox, monitoring (Prometheus/Grafana opsional), dan buffer scaling.

### 3.3 Strategi Anti-Bottleneck

Masalah utama: **ML Inference memakan CPU/RAM tinggi dan bisa membuat backend lambat.**

Solusi dengan Docker + Proxmox:

```
┌─────────────────────────────────────────────────────────┐
│                    PROXMOX HOST                          │
│                                                         │
│  CPU Cores:    [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]│
│                                                         │
│  Container 2 (ML): pin ke CPU 4-7 (physical cores)     │
│  Container 3 (API): pin ke CPU 8                       │
│  Container 4 (DB):  pin ke CPU 9-10                    │
│  Container 7 (Redis): pin ke CPU 11                    │
│  Sisanya: CPU 1-3, 12-16                               │
│                                                         │
│  RAM: Container 2 (ML) dapat 8GB dedicated              │
│       Container 4 (DB) dapat 4GB dedicated              │
│       Sisanya shared dengan limit                       │
└─────────────────────────────────────────────────────────┘
```

**Strategi tambahan:**
1. **ML Batch Processing** — ML tidak perlu real-time. Proses dalam batch (100 grid sekali jalan).
2. **Redis Queue** — GEE Fetcher push job ke Redis Queue, ML Container consume dari queue.
3. **Async API** — Backend tidak nunggu ML selesai. ML tulis hasil ke DB, WebSocket push ke client.
4. **Rate Limiting** — API dilindungi rate limiter (Redis-based) agar tidak overload.
5. **Connection Pooling** — PostgreSQL max_connections diatur, backend pakai pool (Bun native).

### 3.4 Docker Compose Structure

```yaml
# docker-compose.yml — High Level Structure
services:
  # ── Data Layer ──
  postgis:
    image: postgis/postgis:16-3.4
    volumes: [pgdata:/var/lib/postgresql/data]
    deploy: [resources limits: cpus=2, memory=4G]

  redis:
    image: redis:7-alpine
    volumes: [redisdata:/data]
    deploy: [resources limits: cpus=1, memory=1G]

  # ── Service Layer ──
  gee-fetcher:
    build: ./services/gee-fetcher
    volumes: [griddata:/data]  # shared volume untuk grid images
    depends_on: [redis, postgis]
    deploy: [resources limits: cpus=1, memory=2G]

  ml-inference:
    build: ./services/ml-inference
    volumes: [griddata:/data]
    depends_on: [redis, postgis]
    deploy: [resources limits: cpus=4, memory=8G]

  backend-api:
    build: ./services/backend-api  # Bun + Elysia
    ports: ["3000:3000"]
    depends_on: [redis, postgis]
    deploy: [resources limits: cpus=1, memory=1G]

  wa-bot:
    build: ./services/wa-bot
    volumes: [wasession:/app/session]
    depends_on: [postgis]
    deploy: [resources limits: cpus=1, memory=512M]

  # ── Presentation Layer ──
  frontend:
    build: ./services/frontend  # React + Vite
    deploy: [resources limits: cpus=0.5, memory=512M]

  nginx:
    image: nginx:alpine
    ports: ["80:80", "443:443"]
    depends_on: [backend-api, frontend]
    deploy: [resources limits: cpus=0.5, memory=256M]
```

---

## 4. Desain Skema Database

### 4.1 Entity Relationship Diagram (Text-based)

```
┌──────────────────────┐       ┌──────────────────────────┐
│    grid_cells         │       │    detection_logs         │
│──────────────────────│       │──────────────────────────│
│ PK id UUID           │──1:N──│ PK id UUID               │
│    grid_code VARCHAR  │       │ FK grid_id UUID          │
│    geometry GEOMETRY  │       │    confidence FLOAT       │
│    centroid GEOGRAPHY │       │    damage_category ENUM   │
│    area_ha FLOAT      │       │    bounding_boxes JSONB   │
│    status ENUM        │       │    annotated_path TEXT    │
│    last_detection_id  │       │    raw_image_path TEXT    │
│    created_at         │       │    created_at TIMESTAMPTZ │
│    updated_at         │       │    metadata JSONB         │
└──────────────────────┘       └───────────┬──────────────┘
        │                                   │
        │                                   │ 1:1 (optional)
        │                                   ▼
        │                      ┌──────────────────────────┐
        │                      │    alerts                 │
        │                      │──────────────────────────│
        │                      │ PK id UUID               │
        │                      │ FK detection_id UUID     │
        │                      │    severity ENUM          │
        │                      │    notified BOOLEAN       │
        │                      │    acknowledged BOOLEAN   │
        │                      │    sent_at TIMESTAMPTZ    │
        │                      │    acknowledged_at        │
        │                      │    whatsapp_status TEXT   │
        │                      └──────────────────────────┘
        │
        │                      ┌──────────────────────────┐
        │                      │    satellite_imagery      │
        │                      │──────────────────────────│
        │                      │ PK id UUID               │
        │                      │    source VARCHAR         │
        │                      │    scene_id VARCHAR       │
        │                      │    cloud_cover FLOAT      │
        │                      │    captured_at DATE       │
        │                      │    ingested_at TIMESTAMPTZ│
        │                      │    raw_path TEXT          │
        │                      │    metadata JSONB         │
        │                      └──────────────────────────┘
        │
        │                      ┌──────────────────────────┐
        │                      │    notification_logs      │
        │                      │──────────────────────────│
        │                      │ PK id UUID               │
        │                      │ FK alert_id UUID         │
        │                      │    channel VARCHAR        │
        │                      │    recipient VARCHAR      │
        │                      │    status ENUM            │
        │                      │    error_message TEXT     │
        │                      │    sent_at TIMESTAMPTZ    │
        │                      └──────────────────────────┘
```

### 4.2 DDL Definition

```sql
-- ============================================================
-- ENUM Types
-- ============================================================
CREATE TYPE grid_status AS ENUM ('healthy', 'mild', 'moderate', 'severe', 'unknown');
CREATE TYPE damage_category AS ENUM ('severe', 'moderate', 'mild', 'none');
CREATE TYPE alert_severity AS ENUM ('critical', 'high', 'medium', 'low');
CREATE TYPE notification_status AS ENUM ('pending', 'sent', 'failed', 'retrying');

-- ============================================================
-- Tabel: grid_cells
-- Menyimpan grid koordinat spasial pembagian area hutan
-- ============================================================
CREATE TABLE grid_cells (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grid_code       VARCHAR(32) UNIQUE NOT NULL,         -- e.g., "GRID-2024-A1"
    geometry        GEOMETRY(POLYGON, 4326) NOT NULL,    -- Polygon grid cell di SRID 4326
    centroid        GEOGRAPHY(POINT, 4326) NOT NULL,    -- Titik pusat grid (untuk distance query)
    area_ha         NUMERIC(10, 4),                      -- Luas area dalam hektar
    status          grid_status DEFAULT 'unknown',
    last_detection_id UUID,                              -- FK ke detection_logs (di-set via trigger)
    metadata        JSONB DEFAULT '{}',                  -- Info tambahan (region, zone, dsb)
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    -- Spatial index untuk query bounding box
    CONSTRAINT grid_cells_geometry_valid CHECK (ST_IsValid(geometry))
);

CREATE INDEX idx_grid_cells_geometry_gist ON grid_cells USING GIST (geometry);
CREATE INDEX idx_grid_cells_centroid_gist ON grid_cells USING GIST (centroid);
CREATE INDEX idx_grid_cells_status ON grid_cells (status) WHERE status IN ('severe', 'moderate');
CREATE INDEX idx_grid_cells_grid_code ON grid_cells (grid_code);

-- Trigger: auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_grid_cells_updated_at
    BEFORE UPDATE ON grid_cells
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Tabel: satellite_imagery
-- Metadata citra satelit yang digunakan
-- ============================================================
CREATE TABLE satellite_imagery (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source          VARCHAR(32) NOT NULL,                -- 'landsat-8', 'landsat-9', 'sentinel-2'
    scene_id        VARCHAR(128) NOT NULL,               -- Scene ID dari GEE
    cloud_cover     NUMERIC(5, 2) DEFAULT 0,             -- Persentase awan (0-100)
    captured_at     DATE NOT NULL,                       -- Tanggal akuisisi satelit
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),           -- Waktu di-pull oleh sistem
    raw_path        TEXT,                                -- Path ke file GeoTIFF di storage
    bounds          GEOMETRY(POLYGON, 4326),             -- Bounding box scene
    metadata        JSONB DEFAULT '{}',                  -- Metadata lengkap dari GEE
    checksum        VARCHAR(64)                          -- Untuk deduplikasi
);

CREATE INDEX idx_satellite_imagery_captured_at ON satellite_imagery (captured_at DESC);
CREATE INDEX idx_satellite_imagery_source ON satellite_imagery (source);
CREATE UNIQUE INDEX idx_satellite_imagery_checksum ON satellite_imagery (checksum);
CREATE INDEX idx_satellite_imagery_bounds_gist ON satellite_imagery USING GIST (bounds);

-- ============================================================
-- Tabel: detection_logs
-- Log hasil deteksi ML per grid
-- ============================================================
CREATE TABLE detection_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grid_id         UUID NOT NULL REFERENCES grid_cells(id) ON DELETE CASCADE,
    imagery_id      UUID REFERENCES satellite_imagery(id),
    confidence      NUMERIC(5, 4) NOT NULL,              -- 0.0000 - 1.0000
    damage_category damage_category NOT NULL DEFAULT 'none',
    bounding_boxes  JSONB DEFAULT '[]',                  -- Array of {x1,y1,x2,y2,class,confidence}
    annotated_path  TEXT,                                -- Path ke image dengan bounding box overlay
    raw_image_path  TEXT,                                -- Path ke image crop asli
    inference_time_ms INTEGER,                           -- Waktu inferensi dalam ms
    model_version   VARCHAR(32) DEFAULT 'v1',            -- Versi model yang digunakan
    metadata        JSONB DEFAULT '{}',                  -- Info tambahan preprocessing
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_detection_logs_grid_id ON detection_logs (grid_id);
CREATE INDEX idx_detection_logs_created_at ON detection_logs (created_at DESC);
CREATE INDEX idx_detection_logs_grid_created ON detection_logs (grid_id, created_at DESC);
CREATE INDEX idx_detection_logs_high_confidence ON detection_logs (confidence DESC)
    WHERE confidence >= 0.7;
CREATE INDEX idx_detection_logs_category ON detection_logs (damage_category)
    WHERE damage_category IN ('severe', 'moderate');

-- Partition by month untuk performa jangka panjang
-- (Opsional, implementasi di fase optimasi)

-- ============================================================
-- Tabel: alerts
-- Peringatan yang dikirim ke WhatsApp
-- ============================================================
CREATE TABLE alerts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id    UUID NOT NULL REFERENCES detection_logs(id) ON DELETE CASCADE,
    severity        alert_severity NOT NULL DEFAULT 'medium',
    summary         TEXT,                                -- Ringkasan singkat untuk notifikasi
    notified        BOOLEAN DEFAULT FALSE,
    acknowledged    BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(128),                        -- Nama/user yang acknowledge
    sent_at         TIMESTAMPTZ,
    acknowledged_at TIMESTAMPTZ,
    whatsapp_status notification_status DEFAULT 'pending',
    retry_count     INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_notified ON alerts (notified) WHERE notified = FALSE;
CREATE INDEX idx_alerts_severity ON alerts (severity);
CREATE INDEX idx_alerts_created_at ON alerts (created_at DESC);

-- ============================================================
-- Tabel: notification_logs
-- Log pengiriman notifikasi (audit trail)
-- ============================================================
CREATE TABLE notification_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id        UUID NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    channel         VARCHAR(32) NOT NULL DEFAULT 'whatsapp',  -- 'whatsapp', 'sms', 'email'
    recipient       VARCHAR(128) NOT NULL,                -- Nomor WA / email tujuan
    status          notification_status DEFAULT 'pending',
    error_message   TEXT,
    sent_at         TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_notification_logs_alert_id ON notification_logs (alert_id);
CREATE INDEX idx_notification_logs_status ON notification_logs (status)
    WHERE status IN ('pending', 'failed');

-- ============================================================
-- Tabel: config (key-value untuk konfigurasi dinamis)
-- ============================================================
CREATE TABLE config (
    key             VARCHAR(128) PRIMARY KEY,
    value           JSONB NOT NULL,
    description     TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Seed default configurations
INSERT INTO config (key, value, description) VALUES
    ('detection.threshold', '0.7', 'Minimum confidence untuk mengkategorikan sebagai kerusakan'),
    ('detection.batch_size', '100', 'Jumlah grid per batch ML inference'),
    ('gee.interval_hours', '6', 'Interval pull citra dari GEE (jam)'),
    ('alert.whatsapp.recipients', '["+6281234567890"]', 'Nomor tujuan notifikasi WhatsApp'),
    ('grid.size_meters', '256', 'Ukuran sisi grid dalam meter'),
    ('map.center', '{"lat": -2.5, "lng": 117.0}', 'Default center map (Indonesia)');

-- ============================================================
-- Trigger: Auto-create alert saat deteksi severe/moderate
-- ============================================================
CREATE OR REPLACE FUNCTION auto_create_alert()
RETURNS TRIGGER AS $$
DECLARE
    threshold NUMERIC;
BEGIN
    -- Ambil threshold dari config
    SELECT COALESCE((value->>0)::NUMERIC, 0.7)
    INTO threshold
    FROM config
    WHERE key = 'detection.threshold';

    -- Hanya untuk deteksi dengan confidence di atas threshold
    IF NEW.confidence >= threshold AND NEW.damage_category IN ('severe', 'moderate') THEN
        INSERT INTO alerts (detection_id, severity, summary)
        VALUES (
            NEW.id,
            CASE
                WHEN NEW.damage_category = 'severe' THEN 'critical'
                WHEN NEW.damage_category = 'moderate' THEN 'high'
                ELSE 'medium'
            END,
            FORMAT(
                'Deteksi %s pada grid %s dengan confidence %s%%',
                NEW.damage_category,
                (SELECT grid_code FROM grid_cells WHERE id = NEW.grid_id),
                ROUND(NEW.confidence * 100, 1)
            )
        );

        -- Update status grid
        UPDATE grid_cells
        SET status = NEW.damage_category::text::grid_status,
            last_detection_id = NEW.id,
            updated_at = NOW()
        WHERE id = NEW.grid_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_auto_create_alert
    AFTER INSERT ON detection_logs
    FOR EACH ROW
    WHEN (NEW.confidence >= 0.7)
    EXECUTE FUNCTION auto_create_alert();
```

### 4.3 Query Penting (Contoh)

```sql
-- 1. Grid dalam bounding box untuk map view
SELECT id, grid_code, ST_AsGeoJSON(geometry) as geojson, status,
       last_detection_id
FROM grid_cells
WHERE ST_Intersects(
    geometry,
    ST_MakeEnvelope(114.0, -3.0, 115.0, -2.0, 4326)
);

-- 2. Statistik kerusakan
SELECT
    status,
    COUNT(*) as grid_count,
    SUM(area_ha) as total_area_ha,
    ROUND(SUM(area_ha) * 100.0 / SUM(SUM(area_ha)) OVER(), 2) as percentage
FROM grid_cells
WHERE status != 'unknown'
GROUP BY status;

-- 3. Riwayat deteksi per grid (time series)
SELECT
    dl.created_at,
    dl.confidence,
    dl.damage_category
FROM detection_logs dl
WHERE dl.grid_id = 'some-uuid'
ORDER BY dl.created_at DESC
LIMIT 50;

-- 4. Alert yang belum terkirim (untuk WA Bot polling)
SELECT a.id, a.severity, a.summary,
       dl.confidence, dl.damage_category,
       gc.grid_code,
       ST_X(gc.centroid::geometry) as lat,
       ST_Y(gc.centroid::geometry) as lng,
       dl.annotated_path
FROM alerts a
JOIN detection_logs dl ON a.detection_id = dl.id
JOIN grid_cells gc ON dl.grid_id = gc.id
WHERE a.notified = FALSE
ORDER BY a.created_at ASC
LIMIT 10;

-- 5. Trend harian (untuk chart dashboard)
SELECT
    DATE(dl.created_at) as detection_date,
    COUNT(*) FILTER (WHERE dl.damage_category = 'severe') as severe_count,
    COUNT(*) FILTER (WHERE dl.damage_category = 'moderate') as moderate_count,
    COUNT(*) FILTER (WHERE dl.damage_category = 'mild') as mild_count,
    AVG(dl.confidence) FILTER (WHERE dl.damage_category != 'none') as avg_confidence
FROM detection_logs dl
WHERE dl.created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(dl.created_at)
ORDER BY detection_date DESC;
```

---

## 5. Roadmap Pengembangan (Milestones)

### FASE 0: Setup Infrastruktur (Hari 1-2)

```
┌────────────────────────────────────────────────────────────┐
│ 🛠 FASE 0 — SETUP INFRASTRUKTUR                            │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ [ ] Install Proxmox & konfigurasi VM template               │
│ [ ] Install Docker & Docker Compose di VM                   │
│ [ ] Setup NFS storage untuk shared volume                   │
│ [ ] Setup domain & SSL (Let's Encrypt)                      │
│ [ ] Setup monitoring dasar (htop, docker stats)             │
│ [ ] Buat struktur repository monorepo:                     │
│     ├── /services/gee-fetcher/                              │
│     ├── /services/ml-inference/                             │
│     ├── /services/backend-api/                              │
│     ├── /services/frontend/                                 │
│     ├── /services/wa-bot/                                   │
│     ├── /services/nginx/                                    │
│     ├── /database/ (migrations + seeds)                     │
│     └── docker-compose.yml                                  │
│                                                             │
│ Deliverable: ✅ Semua container bisa start & komunikasi     │
│              ✅ docker-compose up -d running                 │
└────────────────────────────────────────────────────────────┘
```

### FASE 1: Integrasi Google Earth Engine (Hari 3-4)

```
┌────────────────────────────────────────────────────────────┐
│ 🛰 FASE 1 — DATA AKUISISI & GRID GENERATION              │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ [ ] Setup akun GEE & autentikasi (service account)          │
│ [ ] Implementasi fetcher: pull citra berdasarkan AOI polygon│
│ [ ] Implementasi cloud filtering (cloud cover < 20%)        │
│ [ ] Implementasi grid generator (membagi AOI ke grid)       │
│ [ ] Implementasi per-grid image cropping                    │
│ [ ] Simpan metadata ke tabel satellite_imagery              │
│ [ ] Simpan grid ke tabel grid_cells (PostGIS)               │
│ [ ] Integrasi dengan Redis Queue untuk trigger ML           │
│                                                             │
│ Deliverable: ✅ Citra satelit sukses ditarik               │
│              ✅ Grid berhasil dibuat & tersimpan di PostGIS │
│              ✅ Queue trigger ke ML container berfungsi      │
└────────────────────────────────────────────────────────────┘
```

### FASE 2: Machine Learning Pipeline (Hari 5-7)

```
┌────────────────────────────────────────────────────────────┐
│ 🤖 FASE 2 — ML INFERENCE PIPELINE                         │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ [ ] Setup YOLOv8 environment (Ultralytics)                  │
│ [ ] Load pre-trained model (atau fine-tune dengan dataset   │
│     deforestasi Indonesia — jika dataset tersedia)          │
│ [ ] Implementasi preprocessing pipeline:                   │
│     ├── Resize ke 640x640                                   │
│     ├── Normalisasi                                          │
│     └── Augmentasi (jika training)                          │
│ [ ] Implementasi inference worker (consume dari Redis Queue)│
│ [ ] Implementasi post-processing: NMS, thresholding         │
│ [ ] Generate annotated image dengan bounding box overlay    │
│ [ ] Export model ke TensorFlow Lite / ONNX untuk optimasi   │
│ [ ] Simpan hasil ke detection_logs & update grid_cells      │
│ [ ] Trigger alert otomatis via database trigger             │
│                                                             │
│ Deliverable: ✅ ML container bisa infer grid images         │
│              ✅ Hasil deteksi tersimpan di database          │
│              ✅ Alert auto-created untuk deteksi high-conf   │
└────────────────────────────────────────────────────────────┘
```

### FASE 3: Backend API & Database (Hari 5-7 — overlap Fase 2)

```
┌────────────────────────────────────────────────────────────┐
│ ⚙ FASE 3 — BACKEND API & DATABASE                         │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ [ ] Setup Bun + Elysia.js project                          │
│ [ ] Setup PostgreSQL connection pool (Bun native)          │
│ [ ] Setup Redis connection                                  │
│ [ ] Implementasi REST endpoints:                           │
│     ├── GET /api/grids?bbox=...&status=...                  │
│     ├── GET /api/grids/:id                                  │
│     ├── GET /api/grids/:id/history                          │
│     ├── GET /api/stats/summary                              │
│     ├── GET /api/stats/trend?days=30                        │
│     ├── GET /api/alerts?status=pending                      │
│     └── POST /api/alerts/:id/acknowledge                    │
│ [ ] Implementasi WebSocket (Elysia WebSocket plugin):      │
│     ├── new_detection event                                 │
│     ├── new_alert event                                     │
│     └── grid_update event                                   │
│ [ ] Implementasi Redis cache untuk query yang sering        │
│ [ ] API error handling & validation (Zod schema)            │
│ [ ] API documentation (Swagger/OpenAPI via Elysia Swagger)  │
│                                                             │
│ Deliverable: ✅ Semua endpoint REST berfungsi               │
│              ✅ WebSocket real-time push berjalan            │
│              ✅ API ter-dokumentasi di /swagger              │
└────────────────────────────────────────────────────────────┘
```

### FASE 4: WhatsApp Bot (Hari 7-8)

```
┌────────────────────────────────────────────────────────────┐
│ 💬 FASE 4 — WHATSAPP NOTIFICATION BOT                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ [ ] Setup Baileys (WhatsApp Web library)                    │
│ [ ] Implementasi QR authentication & session persistence    │
│ [ ] Implementasi polling ke tabel alerts (notified=false)   │
│ [ ] Format pesan notifikasi (template WA)                  │
│ [ ] Kirim gambar annotated via WA                           │
│ [ ] Implementasi retry logic (max 3x)                       │
│ [ ] Log status pengiriman ke notification_logs              │
│ [ ] Error handling: reconnect otomatis jika disconnect      │
│ [ ] Command handler (optional): /status, /help              │
│                                                             │
│ Deliverable: ✅ Bot bisa kirim notifikasi otomatis          │
│              ✅ Alert terkirim < 1 menit dari deteksi       │
│              ✅ Session persist meski container restart      │
└────────────────────────────────────────────────────────────┘
```

### FASE 5: Dashboard Frontend (Hari 8-10)

```
┌────────────────────────────────────────────────────────────┐
│ 🎨 FASE 5 — DASHBOARD INTERAKTIF                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ [ ] Setup React + Vite + TypeScript project                 │
│ [ ] Setup Leaflet.js dengan OpenStreetMap base layer        │
│ [ ] Implementasi Map Component:                            │
│     ├── Tile layer (OpenStreetMap / Satellite)              │
│     ├── GeoJSON overlay dari grid_cells API                 │
│     ├── Color coding: merah (severe), kuning (moderate),   │
│     │   hijau (healthy), abu-abu (unknown)                  │
│     ├── Grid hover → tooltip (grid_code, status)           │
│     └── Grid click → detail modal                          │
│ [ ] Implementasi Sidebar Panel:                            │
│     ├── Statistik cards (total area, damaged %, healthy %) │
│     ├── Recent alerts list (auto-refresh via WebSocket)     │
│     ├── Filter controls (status, date range, min confidence)│
│     └── Search box (koordinat / grid_code)                 │
│ [ ] Implementasi Detail Modal:                             │
│     ├── Split view: raw image vs annotated image            │
│     ├── Confidence score dengan progress bar               │
│     ├── Timeline chart (D3.js) — confidence over time      │
│     ├── Grid info card (koordinat, area, status history)    │
│     └── Action buttons (Acknowledge, Share, Laporkan)      │
│ [ ] Implementasi WebSocket client (auto-connect, reconnect) │
│ [ ] Implementasi responsive design (mobile-friendly)        │
│ [ ] Dark mode toggle (opsional)                             │
│                                                             │
│ Deliverable: ✅ Dashboard bisa diakses via browser          │
│              ✅ Map menampilkan grid dengan warna benar      │
│              ✅ Real-time update via WebSocket               │
└────────────────────────────────────────────────────────────┘
```

### FASE 6: Testing & Integrasi (Hari 10-11)

```
┌────────────────────────────────────────────────────────────┐
│ 🧪 FASE 6 — TESTING & INTEGRASI                           │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ [ ] Unit Testing:                                          │
│     ├── Backend API (Bun: bun test)                         │
│     ├── ML pipeline (pytest)                                │
│     └── Frontend components (Vitest)                        │
│ [ ] Integration Testing:                                    │
│     ├── GEE → Queue → ML → DB flow                         │
│     ├── DB trigger → Alert → WA Bot flow                   │
│     └── API → WebSocket → Frontend flow                    │
│ [ ] Load Testing (opsional):                               │
│     ├── Simulasi 1000 grid dalam satu batch ML              │
│     └── API response time test (k6/autocannon)              │
│ [ ] End-to-End Flow Test:                                   │
│     ├── Mock GEE data → full pipeline → check output       │
│     └── Verifikasi notifikasi WA sampai                     │
│ [ ] Security checklist:                                     │
│     ├── API rate limiting                                   │
│     ├── CORS configuration                                  │
│     ├── Environment variables for secrets                   │
│     └── No hardcoded credentials                           │
│                                                             │
│ Deliverable: ✅ Semua flow end-to-end berfungsi             │
│              ✅ Test coverage > 70% (critical paths)        │
│              ✅ Siap demo untuk juri                        │
└────────────────────────────────────────────────────────────┘
```

### FASE 7: Deployment & Presentasi (Hari 12)

```
┌────────────────────────────────────────────────────────────┐
│ 🚀 FASE 7 — DEPLOYMENT & PRESENTASI                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ [ ] Final deployment di Proxmox                             │
│ [ ] Seed data demo (simulasi grid dengan berbagai status)   │
│ [ ] Siapkan script demo:                                    │
│     ├── Tampilkan dashboard bersih (semua hijau)            │
│     ├── Jalankan manual ML trigger → grid berubah merah     │
│     ├── Tampilkan notifikasi WA masuk                       │
│     └── Tunjukkan detail grid + history chart               │
│ [ ] Siapkan slide presentasi:                               │
│     ├── Problem statement                                   │
│     ├── Solusi & USP                                        │
│     ├── Arsitektur sistem                                   │
│     ├── Demo langsung (lebih baik dari slide!)              │
│     └── Impact & scalability                                │
│ [ ] Backup & snapshot VM untuk jaga-jaga                    │
│                                                             │
│ Deliverable: ✅ Sistem live dan bisa di-demo               │
│              ✅ Slide presentasi siap                        │
└────────────────────────────────────────────────────────────┘
```

---

## 6. Struktur Direktori Proyek

```
deforest.id/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
│
├── services/
│   ├── gee-fetcher/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   │   ├── main.py              # Entry point
│   │   │   ├── fetcher.py           # GEE API client
│   │   │   ├── grid_generator.py    # Grid division logic
│   │   │   ├── image_cropper.py     # Per-grid cropping
│   │   │   └── models.py            # Pydantic models
│   │   └── config.py
│   │
│   ├── ml-inference/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── models/
│   │   │   └── deforestation_v1.pt  # YOLOv8 weights
│   │   ├── src/
│   │   │   ├── main.py              # Worker entry point
│   │   │   ├── preprocess.py        # Image preprocessing
│   │   │   ├── inference.py         # YOLOv8 inference
│   │   │   ├── postprocess.py       # NMS, thresholding
│   │   │   └── annotator.py         # Bounding box overlay
│   │   └── config.py
│   │
│   ├── backend-api/
│   │   ├── Dockerfile
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── index.ts             # Entry point (Bun/Elysia)
│   │       ├── routes/
│   │       │   ├── grids.ts
│   │       │   ├── stats.ts
│   │       │   └── alerts.ts
│   │       ├── ws/
│   │       │   └── handler.ts       # WebSocket handler
│   │       ├── db/
│   │       │   └── pool.ts          # PostgreSQL connection
│   │       ├── cache/
│   │       │   └── redis.ts         # Redis client
│   │       └── types/
│   │           └── index.ts
│   │
│   ├── frontend/
│   │   ├── Dockerfile
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   ├── tsconfig.json
│   │   ├── index.html
│   │   └── src/
│   │       ├── main.tsx
│   │       ├── App.tsx
│   │       ├── components/
│   │       │   ├── MapView.tsx       # Leaflet map
│   │       │   ├── GridLayer.tsx     # GeoJSON overlay
│   │       │   ├── Sidebar.tsx       # Stats panel
│   │       │   ├── AlertList.tsx     # Recent alerts
│   │       │   ├── DetailModal.tsx   # Grid detail
│   │       │   └── TimelineChart.tsx # D3 chart
│   │       ├── hooks/
│   │       │   ├── useWebSocket.ts
│   │       │   └── useGridData.ts
│   │       ├── api/
│   │       │   └── client.ts
│   │       ├── types/
│   │       │   └── index.ts
│   │       └── styles/
│   │           └── globals.css
│   │
│   ├── wa-bot/
│   │   ├── Dockerfile
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── index.ts             # Entry point
│   │       ├── bot.ts               # Baileys client
│   │       ├── alert-poller.ts      # Polling alerts
│   │       ├── message-formatter.ts # Format pesan
│   │       └── session/             # WA session files
│   │
│   └── nginx/
│       ├── nginx.conf
│       └── Dockerfile
│
├── database/
│   ├── migrations/
│   │   ├── 001_create_enums.sql
│   │   ├── 002_create_grid_cells.sql
│   │   ├── 003_create_satellite_imagery.sql
│   │   ├── 004_create_detection_logs.sql
│   │   ├── 005_create_alerts.sql
│   │   ├── 006_create_notification_logs.sql
│   │   ├── 007_create_config.sql
│   │   └── 008_create_triggers.sql
│   └── seeds/
│       └── seed_demo_data.sql
│
└── docs/
    ├── planning.md
    ├── proposal.tex
    └── api-spec.md
```

---

## 7. Risk Management & Mitigation

| Risiko | Dampak | Probabilitas | Mitigasi |
|--------|--------|--------------|----------|
| GEE API rate limit | Data tidak bisa ditarik | Rendah | Cache data, pull dengan interval, fallback data dummy |
| ML model tidak akurat | False positive/negative | Sedang | Fine-tune dengan dataset lokal, ensemble model, human-in-the-loop |
| WA nomor diblokir | Notifikasi gagal | Rendah | Multi-session, cadangan SMS/email, queue retry |
| Container crash | Service down | Rendah | Docker restart policy: always, healthcheck, resource limits |
| Proxmox host down | Total blackout | Sangat Rendah | Snapshot rutin, backup DB ke cloud, rencana migrasi |
| Dataset deforestasi Indonesia tidak tersedia | Training terbatas | Sedang | Gunakan pre-trained model + few-shot learning, atau rule-based fallback |
| Waktu hackathon terbatas | Tidak selesai | Sedang | Prioritaskan core flow, fitur tambahan sebagai "nice to have" |

---

## 8. Key Metrics — Hackathon Success Criteria

| Metrik | Target | Cara Ukur |
|--------|--------|-----------|
| End-to-end pipeline time | < 15 menit (GEE → Dashboard) | Logging timestamp tiap fase |
| Grid processing speed | > 10 grid/detik per container | ML inference logging |
| Dashboard load time | < 3 detik (first paint) | Lighthouse / manual |
| WebSocket latency | < 500ms (deteksi → dashboard) | Client-side timing |
| Alert delivery time | < 60 detik (deteksi → WA) | Perbandingan created_at vs sent_at |
| System uptime | > 99% selama demo | Docker healthcheck logs |
| Code quality | No any type, lint pass | TypeScript strict, ESLint |

---

## 9. Referensi & Libraries

- **Google Earth Engine Python API**: https://developers.google.com/earth-engine/guides/python_install
- **YOLOv8 (Ultralytics)**: https://docs.ultralytics.com
- **Bun + Elysia.js**: https://elysiajs.com
- **PostgreSQL + PostGIS**: https://postgis.net
- **Leaflet.js**: https://leafletjs.com
- **D3.js**: https://d3js.org
- **Baileys (WA Web)**: https://github.com/WhiskeySockets/Baileys
- **Redis**: https://redis.io
- **Docker Compose**: https://docs.docker.com/compose
- **Proxmox VE**: https://www.proxmox.com

---

> **Dokumen ini adalah cetak biru teknis untuk Deforest.id.**
> Setiap fase dan komponen telah dirancang untuk memaksimalkan impact presentasi
> sambil tetap mempertimbangkan keterbatasan waktu dan resource hackathon.
