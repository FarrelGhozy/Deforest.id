# Rancangan Storage — 3 Pendekatan

> Update data: **1× per hari**
> Grid: 256m × 256m
> Target cakupan: fleksibel (1-25 kabupaten)

---

## Daftar Isi

1. [Pendekatan A: Full Archival (Pencadangan Penuh)](#a-full-archival-pencadangan-penuh)
2. [Pendekatan B: Semi-History (Agregat Harian)](#b-semi-history-agregat-harian)
3. [Pendekatan C: Real-Time Only (Tanpa Riwayat)](#c-real-time-only-tanpa-riwayat)
4. [Perbandingan Langsung](#perbandingan-langsung)
5. [Matriks Keputusan](#matriks-keputusan)

---

## A. Full Archival (Pencadangan Penuh)

### Konsep

> Semua data dari setiap siklus disimpan permanen. Tidak ada yang dihapus.
> Cocok untuk: audit trail, ML re-training, analisis historis mendalam.

### Alur Data Per Siklus Harian

```
06:00 — GEE Fetcher pull citra
07:00 — Grid crop selesai
07:30 — ML inference selesai
08:00 — ALL data written to DB
         │
         ├── detection_logs → INSERT (baris baru, tidak overwrite)
         ├── grid_images   → simpan ke disk (retensi: forever)
         ├── annotated     → simpan ke disk (retensi: forever)
         └── alerts        → INSERT jika severe/moderate
```

### Tabel & Storage

| Tabel | Cara Simpan | Jumlah Baris/tahun (200k grid) | Ukuran/tahun |
|-------|------------|-------------------------------|-------------|
| `detection_logs` | INSERT — 1 baris per grid per siklus | 200k × 365 = **73 juta** | ~8 GB |
| `grid_daily_stats` | INSERT — 1 baris per grid per hari | 200k × 365 = **73 juta** | ~4 GB |
| `grid_cells` | UPDATE status — 1 baris per grid | **200k** statis | ~200 MB |
| `alerts` | INSERT saat severe | ~10-50/hari = **~18k/tahun** | ~50 MB |
| `notification_logs` | INSERT per kirim | ~18k/tahun | ~20 MB |
| `satellite_imagery` | INSERT per siklus | **365 baris** | ~10 MB |
| **Storage images** | Grid PNG (300 KB) + annotated (500 KB) | 200k × 365 × 0.8 GB = **~58 TB** | ❌ **TIDAK MUNGKIN** |
| | *Dengan kompresi & simpan hanya severe annotated* | | ~500 GB/tahun |

**Total storage DB:** ~12 GB/tahun — masih wajar.
**Total storage images:** **~500 GB–58 TB** — TIDAK feasible untuk VPS.

### Catatan Kritis — Full Archival dengan Image

Menyimpan **semua** grid images setiap hari tidak realistis:

| Solusi Image | Storage/tahun | Feasible? |
|-------------|--------------|-----------|
| Simpan semua grid PNG (200k/hari) | ~22 TB | ❌ |
| Simpan semua annotated (200k/hari) | ~36 TB | ❌ |
| Simpan severe annotated only (~50/hari) | ~9 GB | ✅ |
| Simpan sample 1% grid PNG (~2.000/hari) | ~220 GB | ⚠️ Mungkin |
| Simpan severe PNG + annotated (~50/hari) | ~15 GB | ✅ |

**Kesimpulan:** Full archival untuk **metadata DB** feasible (12 GB/tahun). Full archival untuk **images** hanya feasible jika selektif (severe-only + sample).

### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
|-----------|-----------|
| Riwayat lengkap per grid per hari | Storage besar untuk images |
| Audit trail sempurna (EUDR ready) | Backup besar & lama |
| Data training ML melimpah | Query historis bisa lambat tanpa indexing |
| Analisis trend akurat | Biaya VPS naik (butuh volume storage) |

### Rekomendasi Storage

```
DB: VPS internal (PostgreSQL) — 12 GB/tahun ✅
Images:
  ├── Severe annotated → VPS internal — 9 GB/tahun ✅
  ├── Sample 1% grid → Cloud Storage (S3/Wasabi) — ~$3/bln ✅
  └── Full archive → Cold storage (Glacier) — ~$1/bln ✅
```

---

## B. Semi-History (Agregat Harian) — ★ REKOMENDASI

### Konsep

> State terakhir disimpan (overwrite tiap siklus), riwayat disimpan dalam
> bentuk agregat harian. Data mentah dihapus setelah diproses.
> Cocok untuk: keseimbangan coverage, insight, dan biaya.

### Alur Data Per Siklus Harian

```
06:00 — GEE Fetcher pull citra
07:00 — Grid crop selesai
07:30 — ML inference selesai
08:00 — Critical data → UPSERT ke detection_logs (state terakhir)
08:01 — Raw images → DISCARD (kecuali severe)
08:02 — Alert severe → INSERT ke alerts
         │
23:55 — Cron job agregasi harian:
         │
         ├── Hitung AVG confidence, MAX category per grid hari ini
         ├── INSERT ke grid_daily_stats
         └── detection_logs tetap (tidak dihapus, hanya state terakhir)
```

### Tabel & Storage

| Tabel | Cara Simpan | Jumlah Baris/tahun | Ukuran/tahun |
|-------|------------|-------------------|-------------|
| `detection_logs` | **UPSERT** — 1 baris per grid (overwrite) | **200k** (konstan) | ~30 MB |
| `grid_daily_stats` | INSERT — 1 baris per grid per hari | 200k × 365 = **73 juta** | ~4 GB |
| `grid_cells` | UPDATE status — 1 baris per grid | **200k** statis | ~200 MB |
| `alerts` | INSERT saat severe | ~18k/tahun | ~50 MB |
| `notification_logs` | INSERT per kirim | ~18k/tahun | ~20 MB |
| **Storage images** | Severe annotated only | ~50/hari = ~18k/tahun | **~9 GB** |

**Total storage DB:** ~4,3 GB/tahun ✅
**Total storage images:** ~9 GB/tahun ✅
**Total semua:** **<15 GB/tahun** ✅✅✅

### Detail — Cara Kerja UPSERT detection_logs

```sql
-- Tabel: 1 baris per grid, selalu ter-overwrite
CREATE TABLE detection_logs (
    grid_id UUID PRIMARY KEY,           -- ← PK = grid_id, cuma 1 baris per grid
    confidence NUMERIC(5,4) NOT NULL,
    damage_category damage_category NOT NULL,
    bounding_boxes JSONB DEFAULT '[]',
    annotated_path TEXT,
    imagery_id UUID,
    model_version VARCHAR(32),
    updated_at TIMESTAMPTZ DEFAULT NOW()  -- ← waktu deteksi TERAKHIR
);

-- Setiap siklus: UPSERT → timpa baris yang sama
INSERT INTO detection_logs (grid_id, confidence, damage_category, bounding_boxes, annotated_path, imagery_id, model_version, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
ON CONFLICT (grid_id) DO UPDATE SET
    confidence = EXCLUDED.confidence,
    damage_category = EXCLUDED.damage_category,
    bounding_boxes = EXCLUDED.bounding_boxes,
    annotated_path = EXCLUDED.annotated_path,
    imagery_id = EXCLUDED.imagery_id,
    model_version = EXCLUDED.model_version,
    updated_at = NOW();
```

**Tabel ini tetap ~200k baris selamanya.** Tidak peduli sistem berjalan 1 bulan atau 10 tahun.

### Detail — Agregat Harian (grid_daily_stats)

```sql
CREATE TABLE grid_daily_stats (
    id BIGSERIAL PRIMARY KEY,
    grid_id UUID NOT NULL,
    date DATE NOT NULL,
    avg_confidence NUMERIC(5,4),
    max_category damage_category,
    min_confidence NUMERIC(5,4),
    detection_count INTEGER DEFAULT 1,
    UNIQUE(grid_id, date)              -- 1 baris per grid per hari
);

-- Cron job tengah malam:
-- Dari data STATE saat ini, catat sebagai riwayat hari ini
INSERT INTO grid_daily_stats (grid_id, date, avg_confidence, max_category, detection_count)
SELECT grid_id, CURRENT_DATE, confidence, damage_category, 1
FROM detection_logs
ON CONFLICT (grid_id, date) DO UPDATE SET
    avg_confidence = EXCLUDED.avg_confidence,
    max_category = EXCLUDED.max_category,
    detection_count = detection_count + 1;
```

### Query Dashboard (Apa yang Bisa Ditampilkan)

```sql
-- Peta: grid warna → dari detection_logs (state terakhir, real-time)
SELECT grid_id, damage_category, confidence
FROM detection_logs;

-- Trend 30 hari → dari grid_daily_stats
SELECT date, avg_confidence, max_category
FROM grid_daily_stats
WHERE grid_id = 'xxx' AND date >= NOW() - INTERVAL '30 days'
ORDER BY date;

-- Statistik hari ini → dari detection_logs
SELECT damage_category, COUNT(*)
FROM detection_logs
GROUP BY damage_category;

-- Perbandingan minggu ini vs minggu lalu → dari grid_daily_stats
SELECT 'minggu_ini' as periode, AVG(avg_confidence)
FROM grid_daily_stats
WHERE date >= DATE_TRUNC('week', NOW())
UNION ALL
SELECT 'minggu_lalu', AVG(avg_confidence)
FROM grid_daily_stats
WHERE date >= DATE_TRUNC('week', NOW()) - INTERVAL '1 week'
  AND date < DATE_TRUNC('week', NOW());
```

### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
|-----------|-----------|
| Storage kecil (<15 GB/tahun) | Tidak bisa lihat deteksi per jam |
| Coverage luas (10-25 kabupaten) | Data training ML terbatas |
| Trend 30 hari tetap bisa | Tidak ada rollback ke deteksi lama |
| Biaya VPS minimal (Rp 200rb/bln) | Audit trail per siklus tidak ada |
| Tidak perlu cloud storage | |

### Rekomendasi Storage

```
DB: VPS internal — <5 GB/tahun ✅ (PostgreSQL muat)
Images: VPS internal — ~9 GB/tahun ✅
Backup: pg_dump mingguan ke Cloud Storage (~$1-2/bln)
```

---

## C. Real-Time Only (Tanpa Riwayat)

### Konsep

> Hanya state terkini yang disimpan. Begitu siklus baru selesai, data
> siklus sebelumnya hilang. Tidak ada agregat, tidak ada riwayat.
> Cocok untuk: coverage maksimal, early warning murni, biaya minimal.

### Alur Data Per Siklus Harian

```
06:00 — GEE Fetcher pull citra
07:00 — Grid crop selesai
07:30 — ML inference selesai
08:00 — State → UPSERT detection_logs (overwrite)
08:01 — Semua raw images → DISCARD
08:02 — Alert severe → INSERT alerts
         │
         └── Selesai. Tidak ada cron malam. Tidak ada agregat.
```

### Tabel & Storage

| Tabel | Cara Simpan | Jumlah Baris/tahun | Ukuran/tahun |
|-------|------------|-------------------|-------------|
| `detection_logs` | **UPSERT** — 1 baris per grid | **200k** konstan | ~30 MB |
| `grid_cells` | UPDATE status | **200k** konstan | ~200 MB |
| `alerts` | INSERT saat severe | ~18k/tahun | ~50 MB |
| `notification_logs` | INSERT per kirim | ~18k/tahun | ~20 MB |
| `grid_daily_stats` | **TIDAK ADA** | 0 | 0 |
| **Storage images** | **TIDAK ADA** (semua discard) | 0 | **0** |

**Total storage DB:** **~300 MB/tahun** ✅✅✅
**Total storage images:** **0** ✅✅✅
**Total semua:** **<500 MB/tahun** ✅✅✅✅✅

### Tampilan Dashboard (Terbatas)

```sql
-- Peta: grid warna → bisa ✅
SELECT grid_id, damage_category, confidence FROM detection_logs;

-- Statistik hari ini → bisa ✅
SELECT damage_category, COUNT(*) FROM detection_logs GROUP BY damage_category;

-- Trend 30 hari → ❌ TIDAK BISA
-- Perbandingan mingguan → ❌ TIDAK BISA
-- Detail riwayat grid → ❌ TIDAK BISA
-- Chart timeline per grid → ❌ TIDAK BISA
```

**Yang hilang dari dashboard:**
- Chart "confidence over time" per grid — ❌
- "Minggu lalu area ini hijau, sekarang merah" — ❌
- Heatmap perubahan selama 1 bulan — ❌
- Data untuk laporan periodik — ❌

**Yang tetap ada:**
- Peta warna grid real-time — ✅
- Alert masuk (notifikasi WA) — ✅
- Grid detail saat di-click (state terakhir) — ✅
- Jumlah grid merah/kuning/hijau saat ini — ✅

### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
|-----------|-----------|
| Storage sangat kecil (~300 MB/tahun) | ❌ **Tidak ada trend analysis sama sekali** |
| Coverage maksimal (25+ kabupaten) | ❌ Tidak bisa lihat perubahan dari waktu ke waktu |
| Biaya terendah (VPS Rp 200rb) | ❌ Tidak ada data untuk ML re-training |
| Backups hampir tidak perlu | ❌ Juri tanya "gimana trend-nya?" jawab: "tidak ada" |
| Performa DB sangat cepat | ❌ Tidak ada audit trail untuk EUDR |

### Rekomendasi Storage

```
DB: VPS internal — ~300 MB ✅✅ (PostgreSQL, muat puluhan tahun)
Images: Tidak ada ✅
Backup: pg_dump seminggu sekali ~10 detik ✅
```

---

## Perbandingan Langsung

### 1 Kabupaten (200k grid)

| Aspek | A. Full Archival | B. Semi-History | C. Real-Time Only |
|-------|-----------------|----------------|-------------------|
| **Storage DB/tahun** | ~12 GB | ~4,3 GB | **~300 MB** |
| **Storage images/tahun** | ~500 GB (terpaksa selektif) | ~9 GB | **0** |
| **Total storage/tahun** | ~512 GB | **~13 GB** | **~300 MB** |
| **Biaya VPS/bln** | Rp 800rb (+ Cloud Storage) | Rp 200rb | Rp 200rb |
| | | | |
| **Coverage maksimal** | 1-2 kabupaten | **10-25 kabupaten** | **25+ kabupaten** |
| **Trend chart 30 hari** | ✅ Detail per siklus | ✅ Daily aggregate | ❌ Tidak ada |
| **Audit trail EUDR** | ✅ Lengkap | ⚠️ Terbatas | ❌ Tidak ada |
| **Data ML training** | ✅ Melimpah | ⚠️ Sample 1% | ❌ Tidak ada |
| **Peta warna real-time** | ✅ | ✅ | ✅ |
| **Alert WA** | ✅ | ✅ | ✅ |
| **Grid detail (state skr)** | ✅ | ✅ | ✅ |
| | | | |
| **Update data** | 1×/hari | 1×/hari | 1×/hari |
| **Riwayat yg disimpan** | Semua | Rata-rata harian | Tidak ada |
| **Bisa lihat "kemarin"** | ✅ Bisa (detail) | ✅ Bisa (agregat) | ❌ Tidak bisa |

### 10 Kabupaten (2 juta grid)

| Aspek | A. Full Archival | B. Semi-History | C. Real-Time Only |
|-------|-----------------|----------------|-------------------|
| **Storage DB/tahun** | ~120 GB | ~43 GB | **~3 GB** |
| **Storage images/tahun** | ~5 TB ❌ | ~90 GB | **0** |
| **VPS termurah** | ❌ Tidak muat | ✅ 200 GB cukup | ✅ 200 GB cukup |
| **Coverage** | ❌ Tidak feasible | ✅ **10-25 kabupaten** | ✅ **25+ kabupaten** |

---

## Matriks Keputusan

### Pilih A — Full Archival Jika:

- ✅ Proyek ditujukan untuk **penelitian/akademik** (butuh data historis lengkap)
- ✅ Target **1-2 kabupaten** saja
- ✅ Ada budget untuk **cloud storage** (S3/Glacier)
- ✅ Butuh **audit trail tingkat EUDR** (setiap deteksi tercatat)
- ✅ Ada rencana **ML re-training** rutin dengan data sendiri

### Pilih B — Semi-History Jika: ★

- ✅ **Ini yang paling recommended untuk hackathon/kompetisi**
- ✅ Mau coverage **10-25 kabupaten** dengan biaya minimal
- ✅ Trend chart 30 hari **sudah cukup** untuk presentasi ke juri
- ✅ Ingin keseimbangan antara **insight** dan **biaya**
- ✅ VPS Rp 200rb/bulan adalah target budget

### Pilih C — Real-Time Only Jika:

- ✅ Target coverage **sangat luas** (25+ kabupaten)
- ✅ Fokus utama adalah **early warning** (bukan analisis)
- ✅ Tidak masalah **tidak bisa lihat perubahan historis**
- ✅ Budget sangat terbatas
- ⚠️ **Tapi:** perlu siap menjawab pertanyaan juri tentang ketiadaan trend

### Rekomendasi Akhir

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│   HACKATHON / KOMPETISI →  Pilih B (Semi-History)           │
│                                                               │
│   Alasan:                                                     │
│   1. Coverage luas → impressif untuk demo                    │
│   2. Trend 30 hari → cukup jawab pertanyaan juri             │
│   3. Storage kecil → VPS murah, no cloud dependency          │
│   4. Tidak perlu backup complex                              │
│   5. Fokus开发 di pipeline, bukan di infrastructure          │
│                                                               │
│   Kalau waktu sisa → upgrade ke A (Full Archival)            │
│   untuk tambah tabel image archive.                          │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Lampiran: Estimasi Biaya VPS per Pendekatan

### A. Full Archival

| Item | Harga | Catatan |
|------|-------|---------|
| VPS 8 vCPU, 16 GB RAM, 400 GB NVMe | ~Rp 800rb/bln | Storage besar untuk images |
| Cloud Storage (S3/Wasabi) | ~$10/bln (~Rp 160rb) | Archive images & backup |
| **Total** | **~Rp 960rb/bln** | |

### B. Semi-History

| Item | Harga | Catatan |
|------|-------|---------|
| VPS 4 vCPU, 8 GB RAM, 200 GB NVMe | ~Rp 400rb/bln | Lebih dari cukup |
| Cloud Storage (opsional backup) | ~$2/bln (~Rp 32rb) | Weekly pg_dump |
| **Total** | **~Rp 432rb/bln** | |

### C. Real-Time Only

| Item | Harga | Catatan |
|------|-------|---------|
| VPS 4 vCPU, 8 GB RAM, 100 GB NVMe | ~Rp 350rb/bln | Storage hampir tidak terpakai |
| **Total** | **~Rp 350rb/bln** | |

---

> Dokumen ini adalah lampiran dari `planning.md` dan `masalah.md`.
> Tujuan: membantu memilih strategi penyimpanan sebelum coding dimulai.
