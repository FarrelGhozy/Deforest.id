# Deforest.id — Analisis Titik Lemah & Solusi Mutakhir

> Dokumen ini berisi identifikasi, analisis, dan solusi atas seluruh kelemahan
> potensial proyek Deforest.id. Disusun berdasarkan riset terkini (2025–2026).
> **Tujuan:** Antisipasi pertanyaan juri, mitigasi risiko teknis, dan memperkuat
> argumentasi proyek.

---

## Daftar Isi

1. [Akurasi & Keandalan ML](#1-akurasi--keandalan-ml)
2. [Dataset Training](#2-dataset-training)
3. [Keterbatasan Google Earth Engine](#3-keterbatasan-google-earth-engine)
4. [Tutupan Awan di Hutan Tropis](#4-tutupan-awan-di-hutan-tropis)
5. [WhatsApp Bot & Baileys](#5-whatsapp-bot--baileys)
6. [False Positive & False Negative](#6-false-positive--false-negative)
7. [Skalabilitas Grid](#7-skalabilitas-grid)
8. [Deployment & Infrastruktur](#8-deployment--infrastruktur)
9. [Keamanan & Privasi Data](#9-keamanan--privasi-data)
10. [Validasi Pengguna & Produk](#10-validasi-pengguna--produk)
11. [Regulasi & Kepatuhan](#11-regulasi--kepatuhan)
12. [Performa WebSocket & Real-time](#12-performa-websocket--real-time)
13. [Biaya Operasional Jangka Panjang](#13-biaya-operasional-jangka-panjang)
14. [Kompetitor & Alternatif Existing](#14-kompetitor--alternatif-existing)
15. [Ketergantungan pada Pihak Ketiga](#15-ketergantungan-pada-pihak-ketiga)

---

## 1. Akurasi & Keandalan ML

### 1.1 Masalah

**YOLOv8 untuk deteksi deforestasi memiliki akurasi yang sangat terbatas.**

Berdasarkan riset terbaru (Nature Scientific Reports, 2025) yang menguji YOLOv8
dengan LangChain agent untuk deteksi deforestasi:

| Metrik | Nilai | Arti |
|--------|-------|------|
| mAP@0.5 | **0.071** | Hanya 7.1% deteksi tepat |
| Recall maksimal | **0.24** | 76% area deforestasi terlewat |
| Precision | ~**0.10** | 90% alarm adalah false positive |

Riset lain mengonfirmasi:
- YOLOv8 kesulitan mendeteksi **objek kecil** (pohon tumbang awal, stump)
- Pada grid cell dengan banyak objek bertumpuk, YOLOv8 rawan **salah deteksi**
- Validation loss **naik di epoch akhir** → indikasi **overfitting** karena dataset
  kecil dan tidak seimbang
- Framework seperti U-Net + ResNet50 mencapai **99.91% accuracy** untuk
  segmentasi deforestasi — jauh di atas YOLOv8

### 1.2 Solusi

#### a. Ganti Arsitektur: YOLOv8-StarNet-CGA atau SCS-YOLOv8
Riset 2026 (Forests Journal) mengembangkan modifikasi YOLOv8 yang mengganti C2f
module dengan **StarNet** dan menambahkan **Coordinate-Guidance Attention (CGA)**:

| Metrik | YOLOv8 | YOLOv8-StarNet-CGA | Peningkatan |
|--------|--------|-------------------|-------------|
| Precision | baseline | +**8.6%** | Deteksi lebih tepat |
| Recall | baseline | +**13%** | Lebih sedikit terlewat |
| mAP50 | baseline | +**11.7%** | Akurasi keseluruhan naik |
| mAP50-95 | baseline | +**14.8%** | Konsistensi di berbagai IoU |

**Implementasi:** Ganti backbone C2f dengan StarNet (4-stage, star operation-based)
dan tambahkan CGA module untuk dynamic feature focusing.

#### b. Gunakan Resolusi Input Lebih Tinggi (960×960)
Riset VHRTrees (Frontiers, 2025) membuktikan:
- YOLOv8m dengan resolusi **960×960**, optimizer SGD, batch size 16, 50 epoch
  → **F1-score 0.932, mAP@0.5 0.934**
- Dibanding 640×640, peningkatan signifikan di deteksi objek kecil

#### c. Ganti Loss Function: SIoU
SIoU (SCYLLA-Intersection over Union) mempertimbangkan **shape, orientation,
dan scale** — tidak hanya overlap area seperti CIoU. Untuk objek deforestasi
yang irregular bentuknya (api, tebangan selektif), SIoU memberikan konvergensi
lebih cepat dan akurasi bounding box lebih tinggi.

**Catatan:** Untuk objek statis/seperti pohon sakit, SIoU bisa over-constraint.
Gunakan **scene-adaptive weighting**: kurangi weight angle/shape cost untuk
target reguler, fokus ke position & overlap.

#### d. Pertimbangkan U-Net + ResNet50 Sebagai Alternatif
Untuk segmentasi (bukan detection), **U-Net + ResNet50 backbone** mencapai
99.91% accuracy di dataset deforestasi Brazil. Jika kebutuhan proyek adalah
**menandai area terdampak per-grid** (bukan mendeteksi objek individual),
pendekatan segmentasi lebih cocok daripada object detection.

### 1.3 Rekomendasi

> **Untuk MVP:** Gunakan YOLOv8 dengan resolusi 960×960 + SGD optimizer +
> data augmentation (horizontal flip, rotation ±15°, mosaic).
>
> **Untuk akurasi tinggi:** Modifikasi backbone ke StarNet + CGA, atau
> ganti ke U-Net + ResNet50 untuk segmentasi.
>
> **Untuk produksi:** Ensemble YOLOv8 (deteksi) + U-Net (segmentasi) +
> NDVI threshold sebagai validasi silang.

---

## 2. Dataset Training

### 2.1 Masalah

**Dataset deforestasi Indonesia yang terannotasi sangat terbatas.**

- Labeled dataset yang ada mayoritas untuk Amazon Brazil, bukan hutan tropis
  Indonesia (yang didominasi gambut, mangrove, hutan kerangas)
- YOLOv8 butuh ribuan annotated images per kelas. Untuk deforestasi Indonesia,
  dataset publik hampir tidak ada
- Domain shift: model yang dilatih di Brazil Amazon **tidak bisa langsung**
  dipakai di Indonesia karena perbedaan spektral, tekstur, dan pola deforestasi
- Crowdsourcing (Global Forest Watch) menghasilkan ~43.000 gambar untuk Amazon,
  tapi belum ada padanannya untuk Indonesia

### 2.2 Solusi

#### a. Transfer Learning + Fine-tuning dengan Data Sintetis
- Mulai dengan pre-trained YOLOv8 COCO weights
- Fine-tune dengan dataset publik yang paling mendekati:
  - **Tropical deforestation:** Amazon dataset (Bragagnolo et al., 2021)
  - **Forest/non-forest segmentation:** VHRTrees (26.000 tree boundaries)
  - **Global forest change:** Hansen/GFW dataset (30m resolution)
- Augmentasi data sintetis menggunakan **Conditional GAN** untuk
  menghasilkan variasi hutan Indonesia

#### b. Few-Shot Learning dengan Foundation Models
Pendekatan terbaru (Edge AI Alliance, 2025): foundation models seperti
**Prithvi** (IBM-NASA geospatial FM) atau **Clay Foundation Model** bisa
melakukan zero-shot/few-shot classification untuk deforestasi tanpa perlu
ribuan gambar. Cukup 10-50 sample per kelas.

**Keuntungan:**
- Tidak perlu dataset besar
- Generalisasi lebih baik ke domain baru (Indonesia)
- Bisa bedakan tipe deforestasi

#### c. Semi-Supervised + Active Learning
- ML memberikan prediksi awal (meskipun noisy)
- Human-in-the-loop mengoreksi sample yang paling tidak pasti
- Sample terkoreksi dimasukkan ke training set iteratif
- Dalam 3-5 siklus, akurasi naik drastis dengan effort labeling minimal

#### d. Crowdsourcing via Platform
Global Forest Watch membuktikan crowdsourcing bekerja:
- 5.500+ kontributor dari 96 negara
- 43.108 gambar terklasifikasi dalam 6 bulan
- ResNet18 mencapai **>90% akurasi** dari data crowd

**Implementasi:** Tambahkan fitur "Verifikasi" di dashboard → petugas/NGO
bisa konfirmasi apakah deteksi benar atau false positive → hasilnya jadi
training data tambahan. Ini juga jadi argumen USP: "setiap notifikasi
meningkatkan akurasi sistem."

### 2.3 Rekomendasi

> **Untuk MVP:** Pre-trained YOLOv8 (COCO) + threshold NDVI sebagai backup.
> Akurasi mungkin rendah tapi pipeline tetap jalan.
>
> **Untuk lomba:** Tunjukkan *plan* pengumpulan dataset — kerjasama dengan
> KLHK/Balai KSDA, kontribusi crowd via dashboard verifikasi.
>
> **Untuk produksi:** Foundation model (Prithvi/Clay) + active learning loop.

---

## 3. Keterbatasan Google Earth Engine

### 3.1 Masalah

**GEE memberlakukan kuota yang ketat (per April 2026).**

Mulai 27 April 2026, semua proyek nonkomersial GEE memiliki kuota bulanan:

| Tier | Kuota EECU-jam | Syarat |
|------|---------------|--------|
| **Community** | 150 EECU-jam (540.000 EECU-detik) | Semua proyek nonkomersial |
| **Contributor** | 1.000 EECU-jam (3.600.000 EECU-detik) | Wajib billing account (tidak ditagih) |
| **Partner** | 100.000 EECU-jam (360.000.000 EECU-detik) | Aplikasi khusus, untuk NGO/riset |

**Dampak untuk Deforest.id:**
- Setiap crop grid 256m × 256m dari citra Sentinel-2 memakan ~0.5 EECU
- Dengan Community Tier (540.000 EECU-detik/bulan):
  - Per grid: ~0.5 EECU → ~18.000 EECU-detik (estimasi)
  - Maksimal: **~30 grid per siklus** (jika siklus 6 jam)
  - Area coverage: **~2 km² per siklus** — sangat kecil
- Dengan Contributor Tier (3.600.000 EECU-detik/bulan):
  - **~200 grid per siklus** → ~13 km² per siklus
- Jika kuota habis, GEE masuk *restricted mode* (performance turun drastis)

**Masalah tambahan:**
- **Max concurrent batch tasks:** rata-rata hanya 2 tasks
- **Max asset storage:** 250 GB per proyek
- **Max assets:** 10.000
- **Memory limit:** tiap request bisa gagal dengan "User memory limit exceeded"

### 3.2 Solusi

#### a. Pakai Contributor Tier (gratis dengan billing account)
- Cukup link billing account ke GCP (kartu kredit)
- GEE tidak akan menagih selama tetap nonkomersial
- Kuota naik dari 150 → 1.000 EECU-jam/bulan

#### b. Pakai Partner Tier untuk Skala Besar
Jika proyek dampak lingkungan, ajukan Partner Tier:
- 100.000 EECU-jam/bulan
- Syarat: proyek terkait climate mitigation, adaptasi, atau perlindungan
- Approval butuh beberapa minggu — **urus sejak awal proyek**

#### c. Strategi Optimasi Kuota
- Simpan (cache) grid images setelah diproses — jangan re-pull setiap siklus
- Gunakan **GEE Asset** untuk menyimpan grid tetap (tidak perlu re-compute)
- Proses hanya grid yang *berubah* — bandingkan dengan deteksi sebelumnya,
  skip grid yang statusnya masih sama
- Batch export: export citra GeoTIFF utuh sekali, potong grid di local

#### d. Alternatif Data Source
Jangan gantungkan 100% ke GEE. Tambahkan alternatif:

| Platform | Kuota/Harga | Kelebihan |
|----------|-------------|-----------|
| **Microsoft Planetary Computer** | Gratis, kuota besar | STAC API, Sentinel-2/Landsat ready |
| **Copernicus Data Space** | Gratis, unlimited download | Data Sentinel langsung dari ESA |
| **AWS Open Data Registry** | Gratis (biaya egress saja) | Sentinel-2, Landsat di S3 |
| **USGS EarthExplorer** | Gratis | Landsat archive terbesar |

**Implementasi:** Buat abstraction layer — `SatelliteDataSource` interface.
GEE sebagai primary, Planetary Computer sebagai fallback.

#### e. Pre-Fetch & Cache Strategy
- Tarik data GEE sekali sehari (bukan per 6 jam)
- Simpan di shared volume sebagai GeoTIFF
- Grid cropping dilakukan **lokal**, bukan di GEE — ini menghemat EECU drastis
- GEE hanya dipakai untuk: (1) pull citra mentah, (2) cloud masking

### 3.3 Rekomendasi

> **Untuk MVP:** Pull citra GEE sekali sehari → simpan lokal → crop grid
> lokal → infer lokal. GEE hanya sebagai "source" bukan "processor."
>
> **Untuk jangka panjang:** Ajukan Partner Tier + Planetary Computer
> sebagai secondary source.

---

## 4. Tutupan Awan di Hutan Tropis

### 4.1 Masalah

**Hutan tropis Indonesia ditutupi awan hampir sepanjang tahun.**

Riset (Nature Scientific Data, 2023) menganalisis 5 tahun data Landsat +
Sentinel-2 di seluruh tropis:

| Metrik | Nilai |
|--------|-------|
| Rata-rata tutupan awan tahunan (Kalimantan) | **>80%** |
| Maksimum hari berturut-turut tanpa citra jelas | **>120 hari** (Oceania) |
| Rata-rata waktu tunggu citra bebas awan | **~100 hari** (Asia deforestasi front) |
| Bulan kritis untuk optik (Indonesia) | **November–Maret** (musim hujan) |

**Dampak:** Jika hanya mengandalkan citra optik (Landsat/Sentinel-2), sistem
bisa **buta total selama 3-4 bulan** di musim hujan. Ini ironis karena
deforestasi ilegal sering terjadi justru saat musim hujan (aktivitas
tersembunyi oleh awan).

### 4.2 Solusi

#### a. Integrasi SAR (Synthetic Aperture Radar) — Sentinel-1
Radar **menembus awan** — tidak terpengaruh cuaca. Sentinel-1 (C-band)
gratis dengan resolusi 10m, revisit 6-12 hari.

**Bukti dari sistem DETER-R (Brazil, operational sejak 2022):**
- False positive rate: **<0.2%** (sangat rendah)
- 33.234 alerts tambahan yang tidak terdeteksi sistem optik dalam 1 tahun
- Tambahan area: **105.238 ha** (5% dari total deteksi)
- Pada musim hujan, kontribusi naik ke **8.1%**

**Cara integrasi:**
- Pipeline paralel: optical (Sentinel-2) + radar (Sentinel-1)
- Jika optical tidak tersedia (awan), fallback ke radar
- Radar juga mendeteksi **degradasi halus** yang optical sering lewatkan
  (setelah hujan, vegetasi cepat pulih — optical lihat "hijau", radar
  lihat "struktur berubah")

#### b. Data Fusion: Optical + SAR
State-of-the-art: gabungkan optical (spectral) + SAR (structural) sebagai
multi-channel input ke ML model.

**Contoh implementasi:**
- Input 6-channel: R, G, B, NIR (dari Sentinel-2), VV, VH (dari Sentinel-1)
- Model seperti **U-Net** atau **DeepLabV3+** bisa menerima multi-channel
- Akurasi deteksi meningkat 5-15% dibanding single-source

#### c. Komposit Temporal
Jangan bergantung pada satu citra. Gunakan **median composite** dari 30 hari
terakhir — piksel yang tertutup awan digantikan oleh piksel bebas awan dari
tanggal lain dalam periode yang sama.

**GEE implementasi:**
```javascript
var composite = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterDate('2024-01-01', '2024-01-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();
```

#### d. Prioritaskan Sentinel-2 (bukan Landsat)
Sentinel-2 memiliki **revisit 5 hari** (vs Landsat 16 hari) dan **resolusi
10m** (vs Landsat 30m). Dengan 3-4 satelit (2A + 2B), probabilitas dapat
citra bebas awan meningkat drastis.

### 4.3 Rekomendasi

> **Untuk MVP:** Optical-only (Sentinel-2) dengan komposit temporal 30 hari.
> Sadari keterbatasan dan sampaikan di presentasi.
>
> **Untuk lomba (nilai tambah besar):** Demonstrasikan prototype integrasi
> SAR — cukup tunjukkan bahwa *arsitektur mendukung* data fusion, meskipun
> pipeline optical-nya yang jalan penuh.
>
> **Untuk produksi:** Wajib integrasi Sentinel-1 SAR. Tanpa SAR, sistem
> tidak bisa klaim "real-time monitoring" di Indonesia.

---

## 5. WhatsApp Bot & Baileys

### 5.1 Masalah

**WhatsApp secara agresif memblokir akun yang menggunakan Baileys.**

Berdasarkan issue tracker Baileys (GitHub, 2025–2026):

| Masalah | Frekuensi | Dampak |
|---------|-----------|--------|
| **405 Connection Failure** | Sering (Feb 2026) | Tidak bisa pairing device baru |
| **Account restricted (463)** | Sering | Nomor tidak bisa kirim pesan |
| **Permanent ban** | Kadang | Nomor hilang total |
| **Temporary ban** | Sering | Lockout 12-72 jam |

**Penyebab utama ban:**
1. **Server IP** — Meta mendeteksi login dari datacenter IP → flag sebagai bot
2. **Pola pengiriman robotik** — interval tetap, volume tinggi dalam waktu singkat
3. **Warmup** — Nomor baru langsung kirim banyak pesan → red flag
4. **Platform mismatch** — Sejak Feb 2026, WA reject `UserAgent.Platform.WEB`
5. **Multi-device session** — Baileys terdeteksi sebagai unofficial client

### 5.2 Solusi

#### a. Gunakan Wrapper Anti-Ban: `baileys-antiban`
Library open-source (MIT) yang wrapping Baileys dengan:

```typescript
import { wrapSocket } from 'baileys-antiban';

const sock = makeWASocket({ ... });
const safeSock = wrapSocket(sock, {
  warmupDays: 7,        // Gradual ramp-up
  maxPerMinute: 8,      // Max 8 messages/minute
  maxPerHour: 200,      // Max 200 messages/hour
  jitterStdDev: 1500,   // Gaussian delay: 1.5s std dev
  healthCheckInterval: 60000  // Check session health
});
```

**Fitur:**
- Gaussian-distributed delays (bukan fixed interval)
- Warmup period 7 hari — volume naik gradual
- Auto-pause saat risk tinggi
- Health monitoring — deteksi "connected but dead"

#### b. Gunakan MACOS Platform (Workaround 405 Error)
Sejak Feb 2026, WA block `Platform.WEB`. Solusi: ganti ke `Platform.MACOS`:

```typescript
// Baileys v7.0.0-rc9+
const sock = makeWASocket({
  auth: state,
  // Platform.MACOS otomatis dipakai di versi terbaru
});
```

#### c. Multi-Session & Load Balancing
Gunakan **WaSP Protocol** (session management library):

```typescript
import { SessionManager } from 'wasp-protocol';

const manager = new SessionManager({
  sessions: 3,  // 3 nomor berbeda
  failover: true,
  rateLimit: {
    perSession: { maxPerMinute: 8 },
    global: { maxPerMinute: 20 }
  }
});
```

- 3+ nomor WA berbeda untuk failover
- Jika 1 nomor kena ban, otomatis pindah ke nomor lain
- Distribusi beban pesan merata

#### d. Cadangan: SMS Gateway & Email
Ketergantungan hanya pada WA berbahaya. Siapkan cadangan:

| Channel | Library | Biaya | Keandalan |
|---------|---------|-------|-----------|
| WhatsApp (Baileys) | baileys-antiban | Gratis | Rendah (risk ban) |
| SMS | Twilio / Nexmo | ~$0.05/SMS | Tinggi |
| Email | SendGrid / SMTP | Gratis 100/hr | Tinggi |
| Telegram Bot | node-telegram-bot-api | Gratis | Tinggi |

**Implementasi:** Buat `NotificationChannel` abstraction — WA sebagai
*primary*, SMS/Email/Telegram sebagai *fallback*.

#### e. Hindari Baileys — Gunakan API Resmi (Jika Budget Ada)
Untuk produksi serius:

| Layanan | Biaya | Catatan |
|---------|-------|---------|
| **WhatsApp Business API** | ~$0.005/pesan | Resmi, tidak kena ban, perlu approval |
| **Twilio for WhatsApp** | ~$0.005/pesan + $0.15/hari | Sandbox atau production approval |
| **WATI / ChatAPI** | $49-299/bulan | Managed service, resmi |

**Untuk lomba:** Baileys cukup. Tapi siapkan jawaban:
> *"Untuk MVP kami gunakan Baileys. Untuk produksi, kami akan migrasi ke
> WhatsApp Business API untuk reliability dan compliance."*

### 5.3 Rekomendasi

> **Untuk MVP:** Baileys + `baileys-antiban` + MACOS platform + session
> persistence. Test dengan nomor cadangan.
>
> **Untuk demo:** Rekam video notifikasi WA, jangan demo langsung real-time
> WA (resiko kena ban pas presentasi).
>
> **Untuk produksi:** Migrasi ke WhatsApp Business API atau Twilio.

---

## 6. False Positive & False Negative

### 6.1 Masalah

**Sistem deteksi otomatis selalu menghasilkan false positive dan false negative.**

Definisi untuk konteks Deforest.id:

| Jenis Error | Contoh | Dampak |
|-------------|--------|--------|
| **False Positive** | Sawit terdeteksi sebagai deforestasi | Alarm palsu, petugas capek, kredibilitas turun |
| **False Negative** | Tebangan liar terlewat | Kerusakan hutan tidak tertangani |

**Sumber false positive pada deforestasi:**
- Lahan pertanian/pertambangan yang sudah ada sebelumnya (bukan deforestasi baru)
- Perubahan musiman (daun gugur, kebakaran alami)
- Bayangan awan (salah deteksi sebagai perubahan)
- Awan tipis yang tidak ter-filter — citra jadi lebih gelap

**Sumber false negative:**
- Deforestasi skala kecil (tebang pilih, <0.5 ha) — di bawah resolusi
- Deforestasi tersembunyi di bawah kanopi
- Re-vegetasi cepat — area sudah hijau kembali saat citra berikutnya

### 6.2 Solusi

#### a. Multi-Temporal Analysis (Change Detection)
Jangan deteksi dari single image — bandingkan time series:

```python
# Konsep: deteksi perubahan drastis di time series NDVI
# Bukan deteksi "ini hutan atau bukan" dari 1 gambar
ndvi_t0 = calculate_ndvi(image_t0)  # baseline (1 bulan lalu)
ndvi_t1 = calculate_ndvi(image_t1)  # sekarang
delta_ndvi = ndvi_t0 - ndvi_t1

# Deforestasi = penurunan NDVI signifikan + persisten
if delta_ndvi > threshold and persist > 14_days:
    alert()
```

**Keuntungan:** Mengeliminasi false positive dari perubahan musiman.

#### b. Multi-Model Voting (Ensemble)
Tiga model berbeda memberikan prediksi, majority vote menentukan hasil:

```
Grid Image
    │
    ├── YOLOv8 (deteksi) ──┐
    ├── U-Net (segmentasi) ─┤── Voting → Final Decision
    └── NDVI (threshold) ──┘
```

- Jika 2 dari 3 setuju → hasil dipakai
- Jika hanya 1 yang positif → anggap noise
- Confidence final = rata-rata confidence dari 3 model

#### c. Human-in-the-Loop untuk High-Confidence Alert
**Prinsip:** Sistem auto-send WA untuk alert dengan confidence > 90%.
Untuk confidence 70-90%, masuk queue review manusia di dashboard.
Untuk confidence < 70%, hanya dicatat di log.

```
Confidence
    │
    ├── > 90%   → Auto Kirim WA + Dashboard (full otomatis)
    ├── 70-90%  → Dashboard warning, WA dikirim setelah review
    └── < 70%   → Log only (tidak ada notifikasi)
```

#### d. Contextual Validation Layer
Kurangi false positive dengan data kontekstual:

| Data Konteks | Sumber | Fungsi |
|-------------|--------|--------|
| Peta konsesi/izin | KLHK | Jangan trigger alert di area konsesi legal |
| Tata guna lahan resmi | BIG/RTRW | Filter area non-hutan (sawah, tambang eksisting) |
| Fire hotspot | NASA FIRMS | Jika ada hotspot, kemungkinan kebakaran (bukan tebangan) |
| Awan (cloud mask) | Sentinel-2 QA | Pastikan bukan bayangan awan |

**Implementasi:** Spatial join antara grid cell dan layer konteks.
Simpan sebagai `grid_cells.metadata.context` (JSONB).

#### e. Confidence Calibration
YOLOv8 secara default overconfident. Kalibrasi dengan **temperature scaling**
atau **Platt scaling** sehingga confidence mencerminkan probabilitas aktual.

**Idenya:** Model sering output confidence 0.95 padahal salah.
Kalibrasi: confidence 0.95 berarti benar 95% dari waktu.

```python
# Platt scaling: logistic regression di validation set
from sklearn.linear_model import LogisticRegression
calibrator = LogisticRegression()
calibrator.fit(model_raw_scores, is_correct)
calibrated_confidence = calibrator.predict_proba(raw_scores)[:, 1]
```

### 6.3 Rekomendasi

> **Untuk MVP:** Single model (YOLOv8) + NDVI threshold sebagai backup.
> Tapi pastikan di presentasi jelaskan *rencana* ensemble.
>
> **Untuk produksi:** Ensemble 3 model + human-in-the-loop + contextual
> validation + confidence calibration.

---

## 7. Skalabilitas Grid

### 7.1 Masalah

**Jumlah grid membengkak eksponensial seiring luas area.**

| Luas Area | Ukuran Grid | Jumlah Grid | Waktu Infer (1 detik/grid) |
|-----------|-------------|-------------|---------------------------|
| 1 km² | 256m × 256m | ~16 | 16 detik |
| 100 km² | 256m × 256m | ~1.525 | 25 menit |
| 1.000 km² (kabupaten) | 256m × 256m | ~15.250 | 4.2 jam |
| 50.000 km² (provinsi) | 256m × 256m | ~762.500 | ~8.8 hari |

Pada skala provinsi, ML inference **8.8 hari non-stop** — tidak feasible
untuk sistem peringatan dini.

**Masalah tambahan:**
- Storage grid images: 50.000 km² × ~500 KB/grid = **~370 GB per siklus**
- Database rows: 762.500 grid × riwayat deteksi → miliaran baris dalam 1 tahun
- Query spatial di PostGIS untuk 762.500 polygon bisa lambat tanpa optimasi

### 7.2 Solusi

#### a. Adaptive Grid — Bukan Seragam
Jangan bagi semua area dengan grid seragam. Gunakan **hierarchical grid**:

```
Level 0: Hutan utuh = grid besar (1 km × 1 km) — diproses jarang
Level 1: Buffer zone = grid sedang (500m × 500m)
Level 2: Area berisiko tinggi = grid kecil (100m × 100m) — prioritas
```

**Implementasi:**
- Mulai dengan grid besar untuk identifikasi area berubah
- Jika area berubah terdeteksi, *subdivide* menjadi grid lebih kecil
- Jika area stabil (3 siklus tanpa perubahan), *merge* jadi grid besar lagi

Teknik ini bisa mengurangi jumlah grid hingga **90%** tanpa kehilangan presisi.

#### b. Prioritized Processing — Antrean Prioritas
Tidak semua grid perlu diproses setiap siklus:

| Prioritas | Grid | Frekuensi Proses |
|-----------|------|------------------|
| **High** | Grid dengan deteksi severe/moderate sebelumnya | Setiap siklus (6 jam) |
| **Medium** | Grid di buffer zone/area konflik | Setiap hari |
| **Low** | Grid hutan utuh yang stabil | Setiap minggu |
| **Background** | Grid yang belum pernah diproses | Satu kali, lalu turun prioritas |

#### c. Batch Processing & GPU Acceleration
ML inference per grid lambat di CPU. Dengan **GPU**:

| Hardware | Waktu/grid | Waktu/15.000 grid |
|----------|-----------|-------------------|
| CPU-only (4 vCPU) | ~1 detik | ~4.2 jam |
| NVIDIA T4 (cloud) | ~15 ms | ~3.7 menit |
| NVIDIA A10G | ~8 ms | ~2 menit |
| TensorRT optimized | ~4 ms | ~1 menit |

Solusi: batch inference — kumpulkan 100 grid, infer sekaligus sebagai batch.

```python
# Batch inference — jauh lebih cepat daripada per-grid
results = model(batch_images)  # shape: [B, 640, 640, 3] → B=100
```

#### d. Database Optimasi untuk Grid Besar
- **Partitioning:** `detection_logs` partition per bulan
- **Spatial indexing:** GIST index wajib di `grid_cells.geometry`
- **Materialized view:** Untuk query statistik yang sering (dashboard)
- **Grid clustering:** Kelompokkan grid secara geografis untuk reduce scan

#### e. Edge Processing — Inference di Lokasi
Untuk skala besar, jangan tarik semua data ke server. Sebaliknya, deploy
ML di edge (VPS regional) yang lebih dekat ke sumber data (GEE).

### 7.3 Rekomendasi

> **Untuk MVP:** Fokus ke area kecil (<100 km²) — cukup untuk demo.
> Adaptive grid + prioritas processing.
>
> **Untuk lomba:** Jangan klaim skala provinsi. Fokus ke skala kabupaten/
> taman nasional. "Kami mulai dari area prioritas, scalable secara
> horizontal dengan tambahan compute node."
>
> **Untuk produksi:** GPU batch inference + partitioning + edge processing.

---

## 8. Deployment & Infrastruktur

### 8.1 Masalah

**7 container terpisah = 7 titik kegagalan potensial.**

| Container | Risiko | Efek Gagal |
|-----------|--------|------------|
| GEE Fetcher | Gagal pull data | Data tidak update |
| ML Inference | OOM/crash | Tidak ada deteksi baru |
| Backend API | Error/restart | Dashboard down |
| PostgreSQL | Corrupt/full | Semua data hilang |
| Frontend | Build error | UI tidak muncul |
| WA Bot | Ban/disconnect | Notifikasi tidak terkirim |
| Redis | Data loss | Cache kosong, queue hilang |

**Masalah lain:**
- Docker daemon bisa crash
- Network bridge antar container bisa disconnect
- Disk full karena grid images menumpuk
- VPS provider bisa downtime

### 8.2 Solusi

#### a. Healthcheck & Auto-Restart untuk Semua Container
```yaml
services:
  backend-api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

#### b. Graceful Degradation — Jangan Single Point of Failure
Setiap container harus bisa *gagal secara independen* tanpa meruntuhkan
sistem:

```
GEE Fetcher mati → Grid terakhir tetap terlihat di dashboard (stale data)
ML Inference mati → Dashboard masih tampilkan deteksi terakhir
Backend API restart → WebSocket reconnect otomatis, requests diqueue
PostgreSQL mati → Redis cache masih serve data untuk beberapa menit
WA Bot mati → Alert diqueue, terkirim saat bot kembali online
```

**Implementasi kunci:** Backend API harus bisa serve data dari Redis cache
meskipun PostgreSQL down. Gunakan **cache-aside pattern**:

```typescript
async function getGrids(bbox: string) {
  // 1. Coba Redis cache dulu
  const cached = await redis.get(`grids:${bbox}`);
  if (cached) return JSON.parse(cached);

  // 2. Fallback ke PostgreSQL
  const data = await db.query('SELECT...');
  
  // 3. Set cache dengan TTL 5 menit
  await redis.set(`grids:${bbox}`, JSON.stringify(data), 'EX', 300);
  return data;
}
```

#### c. Disk Management — Jangan Sampai Penuh
Grid images menumpuk cepat. Implementasi **retensi otomatis**:

```
Grid images:
  └── Retensi: 7 hari → hapus yang lebih dari 7 hari
  └── Kecuali annotated images dengan deteksi severe → simpan permanent

DB logs:
  └── detection_logs: retensi 90 hari → partition drop
  └── notification_logs: retensi 30 hari
```

**Cron job di host:**
```bash
# Hapus grid images > 7 hari
find /data/grids/*.png -mtime +7 -delete
```

#### d. Monitoring & Alerting Infrastruktur
Minimal:

```yaml
services:
  cadvisor:  # Container metrics
    image: gcr.io/cadvisor/cadvisor:latest
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
```

**Yang dimonitor:**
- CPU/RAM usage per container (alert jika >80%)
- Disk usage (alert jika >85%)
- Container restarts (alert dalam 1 jam >3 restart)
- PostgreSQL connection count
- Redis memory usage
- API response time (alert jika >2 detik)

#### e. Backup Strategy
- **PostgreSQL:** `pg_dump` setiap 6 jam ke cloud storage
- **Redis:** RDB snapshot setiap jam
- **Docker volumes:** rsync ke secondary disk setiap hari
- **docker-compose.yml:** version-controlled di git

### 8.3 Rekomendasi

> **Untuk MVP:** healthcheck + restart always + backup harian.
>
> **Untuk lomba:** Siapkan *demo script versi offline* — semuanya running
> lokal, tidak butuh internet. Jangan gantungkan pada VPS.
>
> **Untuk produksi:** Prometheus + Grafana + PagerDuty/Telegram alert.

---

## 9. Keamanan & Privasi Data

### 9.1 Masalah

**Data kerusakan hutan bisa sensitif secara politis dan ekonomis.**

| Risiko | Dampak |
|--------|--------|
| Data deforestasi ilegal bocor | Pelaku bisa menghilangkan bukti |
| Koordinat presisi publik | Oknum bisa menghindari patroli |
| API tanpa autentikasi | Abuse, scraping, DDoS |
| Credential GEE/DB di .env | Bocor ke publik via git |
| WA session file | Bisa dipakai untuk spam/scam |

### 9.2 Solusi

#### a. API Authentication (Minimum JWT)
```typescript
// Elysia middleware
app.use(jwt({
  secret: process.env.JWT_SECRET,
  exp: '24h'
}));
```

**Endpoint tiers:**
| Endpoint | Auth | Rate Limit |
|----------|------|------------|
| GET /api/grids?bbox=... | Public | 100 req/min |
| GET /api/grids/:id/history | JWT required | 30 req/min |
| GET /api/alerts | JWT + Admin | 10 req/min |
| POST /api/alerts/acknowledge | JWT + role check | 5 req/min |
| POST /api/admin/* | JWT + Admin | 2 req/min |

#### b. Environment Variable Management
Jangan simpan secrets di .env file yang tercommit:

```bash
# ✅ Aman: Docker secrets
docker secret create pg_password -

# ✅ Aman: VPS environment variables (via docker compose)
services:
  backend-api:
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password

# ❌ TIDAK AMAN:
# environment:
#   - DB_PASSWORD=supersecret123
```

**Wajib:**
- `.env` di `.gitignore`
- `.env.example` tanpa nilai real
- Ganti semua password sebelum deploy ke VPS

#### c. Data Access Control
- Grid location: publik (ini penting untuk transparansi)
- Alert details: petugas terautentikasi
- Raw imagery: petugas terautentikasi
- User management: admin only

#### d. Secure Session WA
File session WA sangat sensitif — bisa dipakai untuk login sebagai akun
tersebut. Proteksi:

```yaml
services:
  wa-bot:
    volumes:
      - wasession:/app/session
    # Session file hanya bisa diakses container wa-bot
```

Dan enkripsi session file:
```typescript
// Simpan session terenkripsi
const encrypted = CryptoJS.AES.encrypt(
  JSON.stringify(session),
  process.env.SESSION_ENCRYPTION_KEY
).toString();
```

#### e. API Security Hardening
```yaml
services:
  nginx:
    config:
      # Rate limiting
      limit_req_zone $binary_remote_addr zone=api:10m rate=30r/s;
      # CORS ketat
      add_header Access-Control-Allow-Origin "https://deforest.id";
      # Security headers
      add_header X-Content-Type-Options "nosniff";
      add_header X-Frame-Options "DENY";
      add_header Strict-Transport-Security "max-age=31536000";
```

### 9.3 Rekomendasi

> **Untuk MVP:** JWT auth + rate limiting + environment variables.
>
> **Untuk lomba:** Cukup demo tanpa auth untuk kemudahan akses, tapi
> jelaskan *rencana* security di presentasi.
>
> **Untuk produksi:** Full auth + role-based access + audit log + session
> encryption.

---

## 10. Validasi Pengguna & Produk

### 10.1 Masalah

**Proyek ini berisiko menjadi "solusi cari masalah."**

Pertanyaan juri yang paling sulit:
> *"Siapa yang sudah kamu interview? Apa yang mereka butuhkan?"*
> *"Kenapa mereka butuh ini? Udah ada Global Forest Watch gratis."*
> *"Petugas lapangan di pelosok bisa akses ini? Pulsa internet?"*

**Risiko:** Proyek dibangun berdasarkan asumsi tim, bukan kebutuhan nyata
pengguna — ini adalah **kelemahan #1 yang paling sering dihukum juri**.

### 10.2 Solusi

#### a. Lakukan Minimal User Research (Bahkan 3-5 Interview)
Sebelum presentasi, lakukan setidaknya:

| Siapa | Pertanyaan | Target Insight |
|-------|------------|----------------|
| Petugas Balai KSDA/TN | "Sistem monitoring apa yg dipakai sekarang? Apa frustasinya?" | Validasi need |
| NGO lingkungan (WALHI, dll) | "Seberapa cepat mereka butuh data?" | Validasi urgency |
| Dinas Kehutanan provinsi | "Anggaran untuk monitoring berapa? Proses anggaran gimana?" | Validasi bisnis |

**Output:** Kutipan langsung dari pengguna yang bisa dimasukkan ke slide.

#### b. Borderline: Global Forest Watch vs Deforest.id
Jangan claim "ini lebih baik dari GFW." Itu tidak kredibel. Sebaliknya:

> *"Global Forest Watch adalah platform monitoring global yang sangat baik
> dengan update bulanan. Deforest.id berbeda: kami adalah **early warning
> system** yang fokus pada **notifikasi proaktif real-time** ke petugas
> melalui WhatsApp. GFW adalah **atlas**, kami adalah **sistem saraf**.
> Bukan kompetitor, tapi complementary layer."*

#### c. Offline Mode untuk Daerah Tanpa Internet
Petugas lapangan sering di area tanpa signal. Solusi:

- Dashboard: **Progressive Web App (PWA)** — bisa diakses offline
- Data grid: download area tertentu sebelum berangkat
- WA: teks tetap terkirim lewat 2G/Edge (gambar dikompres <200KB)
- **SMS fallback** untuk daerah blind spot total

#### d. Bahasa Daerah untuk Notifikasi
Petugas lapangan mungkin tidak fasih Bahasa Inggris. Siapkan template
notifikasi multi-bahasa:

```
[Nama Hutan] — PERINGATAN DEFORESTASI
Grid: GRID-2024-A1
Status: SAKIT PARAH (Severe)
Keyakinan: 94%
Lokasi: -3.4567, 114.9876
Waktu: 15 Jan 2024 14:30
Klik: https://deforest.id/grid/GRID-2024-A1
```

Bisa pakai Bahasa Indonesia, Jawa, Dayak, dll tergantung lokasi.

### 10.3 Rekomendasi

> **Untuk MVP:** Minimal 3 wawancara pengguna potensial. Dokumenkan dan
> masukkan ke presentasi — "berdasarkan wawancara dengan 3 petugas KSDA..."
>
> **Untuk lomba:** Ini salah satu diferensiator terbesar. Tim yang bisa
> tunjukkan *user validation* langsung unggul dari tim yang cuma punya
> ide teknis.

---

## 11. Regulasi & Kepatuhan

### 11.1 Masalah

**Data satelit dan deteksi deforestasi memiliki implikasi hukum.**

- Di Indonesia, data citra satelit resolusi tinggi diatur oleh **UU No. 4
  Tahun 2011 tentang Informasi Geospasial** dan **PP No. 45 Tahun 2021**
- Data tentang deforestasi ilegal bisa menjadi **barang bukti hukum** —
  ada konsekuensi jika data tidak akurat atau tidak bisa dipertanggungjawabkan
- Kerja sama dengan instansi pemerintah (KLHK) memerlukan **izin dan MoU**
- **Lisensi:** Sentinel-2 (CC BY-SA 4.0), Landsat (USGS free and open),
  GEE (tier-based)

### 11.2 Solusi

#### a. Manfaatkan EUDR sebagai Justifikasi (Nilai Tambah Besar)

**EU Deforestation Regulation (EUDR)** mulai berlaku penuh 30 Desember 2025
untuk perusahaan besar, Juni 2026 untuk UKM.

**Persyaratan EUDR:**
1. Produk (karet, sawit, kopi, kakao, kedelai, kayu, sapi) harus **bebas deforestasi** setelah 31 Desember 2020
2. Perusahaan wajib melakukan **due diligence** — buktikan asal produk
3. Geolokasi **setiap plot lahan** harus tercatat

**Ini adalah peluang pasar raksasa untuk Deforest.id:**
- Perusahaan sawit di Indonesia butuh sistem monitoring untuk EUDR compliance
- Regulasi mewajibkan penggunaan citra satelit untuk verifikasi
- Deforest.id bisa jadi tool *due diligence* yang murah

#### b. Jaga chain of custody data
Untuk potensi penggunaan hukum:
- Setiap deteksi: simpan **hash** dari citra asli (bukti integritas)
- Metadata lengkap: timestamp, source image ID, model version, parameter
- Audit trail: siapa akses apa, kapan
- Pastikan data immutable (tidak bisa diubah setelah ditulis)

#### c. Lisensi & Atribusi
**Wajib mencantumkan:**
- Data satelit: "Contains modified Copernicus Sentinel data [tahun]"
- GEE: "Powered by Google Earth Engine"
- OpenStreetMap (dasar peta): "© OpenStreetMap contributors"

### 11.3 Rekomendasi

> **Untuk MVP:** Fokus ke EUDR compliance angle sebagai justifikasi pasar.
> Ini lebih kuat secara bisnis daripada "menyelamatkan hutan."
>
> **Untuk presentasi:** Frame proyek sebagai **"EUDR compliance tool untuk
> industri sawit/karet Indonesia"** — ada market jelas, ada regulasi yang
> mendorong, ada kebutuhan nyata.

---

## 12. Performa WebSocket & Real-time

### 12.1 Masalah

**WebSocket untuk 762.500 grid update real-time tidak trivial.**

- Jika semua grid diupdate setiap 6 jam, berarti ~35 grid/detik harus
  di-push ke dashboard — ini masih manageable
- Masalah muncul saat **burst**: ML selesai batch, tiba-tiba 1.000 grid
  update dalam 10 detik → flood WebSocket
- Koneksi WebSocket terputus di jaringan tidak stabil (umum di Indonesia)
- Browser render ribuan grid polygon bisa **lag/freeze**

### 12.2 Solusi

#### a. Throttle & Batch WebSocket Updates
Jangan push per-grid. Push batch setiap 5 detik atau per 100 grid:

```typescript
// Backend: kumpulkan update, flush periodik
const updateBuffer: GridUpdate[] = [];
setInterval(() => {
  if (updateBuffer.length > 0) {
    ws.emit('grid_updates', updateBuffer.splice(0));  // batch
  }
}, 5000);  // setiap 5 detik
```

#### b. Client-Side Rendering Optimasi
Ribuan grid polygon di Leaflet bisa lambat. Solusi:

1. **Canvas rendering** (bukan SVG):
   ```typescript
   L.geoJSON(data, {
     renderer: L.canvas(),  // lebih cepat untuk banyak polygon
     ...
   });
   ```

2. **Grid clustering** — zoom rendah: grup grid, zoom tinggi: detail
3. **Viewport culling** — hanya render grid yang terlihat di layar
4. **Tile-based rendering** — bagi peta jadi tile, render per tile

#### c. WebSocket Reconnection Logic
```typescript
function useWebSocket() {
  let ws = new WebSocket('wss://api.deforest.id/ws');
  let reconnectAttempts = 0;

  ws.onclose = () => {
    const delay = Math.min(1000 * 2 ** reconnectAttempts, 30000);
    setTimeout(() => {
      reconnectAttempts++;
      ws = new WebSocket('wss://api.deforest.id/ws');
    }, delay);
  };

  ws.onopen = () => {
    reconnectAttempts = 0;  // reset setelah sukses
  };
}
```

#### d. HTTP Fallback untuk WebSocket
Jika WebSocket gagal, fallback ke polling HTTP:

```typescript
// Deteksi koneksi
if ('WebSocket' in window) {
  useWebSocket();
} else {
  // Polling setiap 10 detik
  setInterval(() => {
    fetch('/api/grids/updated-since?since=' + lastUpdate);
  }, 10000);
}
```

### 12.3 Rekomendasi

> **Untuk MVP:** Canvas rendering + batch WebSocket + viewport culling.
> Fokus ke performa dengan 5.000-10.000 grid (cukup untuk demo skala
> kabupaten).

---

## 13. Biaya Operasional Jangka Panjang

### 13.1 Masalah

**Biaya bisa membengkak seiring skala.**

| Komponen | Skala Kecil (MVP) | Skala Provinsi |
|----------|-------------------|----------------|
| VPS | Rp 200rb/bln | Rp 3-5jt/bln |
| GPU Cloud | - | $0.5-1/jam (T4) |
| Storage | 50 GB (gratis) | 1-5 TB |
| GEE Quota | Gratis (Community) | Butuh paid plan |
| WA API | Gratis (Baileys) | $0.005/pesan |
| Domain | Rp 150rb/thn | Rp 150rb/thn |
| **Total** | **~Rp 350rb/bln** | **~Rp 10jt/bln** |

### 13.2 Solusi

#### a. Biaya Tetap Rendah dengan Arsitektur Cloud-Native

**Skenario MVP:**
| Item | Solusi | Biaya |
|------|--------|-------|
| VPS 4 vCPU, 8GB RAM, 160GB | DigitalOcean / Vultr | ~$24/bln (Rp 400rb) |
| GPU (opsional) | RunPod serverless | ~$0.20/jam (jika dipakai) |
| Database | PostgreSQL in VPS | Sudah termasuk VPS |
| Storage Grid | DigitalOcean Volume 100GB | ~$10/bln |
| CI/CD | GitHub Actions (gratis) | $0 |
| Monitoring | Uptime Kuma (self-host) | $0 |
| Domain | Niagahoster | Rp 150rb/thn |

**Total MVP: ~Rp 500rb-750rb/bulan**

#### b. Funding & Revenue Model (Jangka Panjang)

| Sumber | Cara | Potensi |
|--------|------|---------|
| **Hibah riset** | KLHK, World Bank, UNDP | Rp 200-500jt/proyek |
| **SaaS B2B** | Perusahaan sawit/karet untuk EUDR | $10-50rb/tahun/korporasi |
| **Subscription** | NGO/universitas | Rp 50-200jt/tahun |
| **Data API** | Akses data dan alert | Rp 5-20jt/bulan |
| **Hackathon prize** | Jika menang | Rp 20-200jt |

### 13.3 Rekomendasi

> **Untuk lomba:** Fokus ke *hackathon prize* dan *hibah riset* sebagai
> funding source. Jangan klaim "kami akan mandiri secara bisnis" tanpa
> bukti. Tapi tunjukkan bahwa *potensi pasarnya ada* (EUDR).
>
> **Untuk produksi:** Model bisnis utama: **B2B SaaS untuk EUDR compliance**
> pada perusahaan sawit/karet/kopi di Indonesia.

---

## 14. Kompetitor & Alternatif Existing

### 14.1 Masalah

**Banyak sistem serupa sudah ada dan gratis.**

| Platform | Fitur | Kelemahan | Harga |
|----------|-------|-----------|-------|
| **Global Forest Watch** | Peta interaktif, alerts, data historis | Update bulanan, tidak ada WA, dashboard komplex | Gratis |
| **GFW Pro** | API, data mentah, analisis custom | Mahal, learning curve tinggi | $10-50rb/tahun |
| **DETER/INPE** (Brazil) | NRT alerts, area spesifik | Hanya Brazil | Gratis |
| **RADD** (Wageningen) | Radar-based alerts, NRT | Tidak ada dashboard WA | Gratis |
| **SIGAP** (KLHK) | Monitoring nasional Indonesia | Akses terbatas, tidak publik | N/A |
| **Planet Labs** | VHR imagery (<1m) | Berbayar, mahal | $100-500/tahun (edu) |

### 14.2 Solusi — Strategi Positioning

**Jangan competing head-to-head. Posisikan Deforest.id sebagai layer tambahan:**

```
           ┌──────────────────────────────┐
           │       DEFOREST.ID            │
           │  Grid-based EWS + WA Bot     │
           │  (+Notifikasi Proaktif)      │
           ├──────────────────────────────┤
           │                              │
           │   Global Forest Watch        │
           │   RADD / DETER / SIGAP       │
           │   (Sumber Data & Alerts)     │
           │                              │
           └──────────────────────────────┘
```

**Nilai tambah Deforest.id yang tidak dimiliki kompetitor:**
1. ✅ **Grid-based precision** — tidak ada yang melakukan grid division
    secara real-time dengan object detection per-grid
2. ✅ **WA notifikasi** — GFW cuma kirim email. Tidak ada yang kirim WA
    otomatis dengan gambar annotated dan koordinat
3. ✅ **Dashboard warna** — GFW dan lainnya peta statis. Deforest.id
    grid real-time yang berubah warna otomatis
4. ✅ **Grid time-series** — riwayat confidence per grid yang bisa
    di-trace (tidak ada di sistem lain)
5. ✅ **Terintegrasi end-to-end** — pengguna tidak perlu pindah platform

### 14.3 Rekomendasi

> **Untuk lomba:** Jujur tentang eksistensi GFW. Jangan bilang "tidak ada
> sistem monitoring deforestasi." Tapi bilang "tidak ada yang melakukan
> peringatan dini proaktif via WA dengan grid-based detection."
>
> **Narrative untuk juri:** *"Global Forest Watch memberi Anda peta.
> Deforest.id memberi Anda alarm di saku. Ini perbedaan antara tahu ada
> kebakaran dan bangun karena alarm asap berbunyi."*

---

## 15. Ketergantungan pada Pihak Ketiga

### 15.1 Masalah

**Proyek ini bergantung pada 6+ layanan eksternal.**

| Layanan | Risiko | Jika Mati |
|---------|--------|-----------|
| Google Earth Engine | API change, quota habis, deprecation | Tidak bisa tarik data satelit |
| Copernicus (Sentinel) | Satelit bermasalah, kebijakan berubah | Tidak ada citra baru |
| WhatsApp/Meta | Baileys diblokir total, API berubah | WA Bot mati |
| Leaflet.js/OSM | OSM tile server down | Peta dasar tidak tampil |
| VPS Provider | Downtime, maintenance | Semua service down |
| GitHub Packages | Untuk image registry | Docker build gagal |
| npm/PyPI | Library vulnerability, removal | Dependency broken |

### 15.2 Solusi

#### a. Setiap Dependency Harus Punya Fallback
```yaml
Data Source:
  Primary: Google Earth Engine
  Fallback: Microsoft Planetary Computer
  Fallback 2: Copernicus Data Space

Map Tiles:
  Primary: OpenStreetMap (tile provider)
  Fallback: CartoDB basemap
  Fallback 2: Stamen Terrain

WA Notifications:
  Primary: Baileys
  Fallback: Twilio SMS
  Fallback 2: Email (SendGrid)

ML Runtime:
  Primary: ONNX Runtime (CPU)
  Fallback: TensorFlow Lite
  Fallback 2: OpenCV DNN module
```

#### b. Dependency Pinning — Hindari "Works on My Machine"
```dockerfile
# Buruk — version tidak tetap
FROM python:3.11
RUN pip install ultralytics

# Baik — pinned exact versions
FROM python:3.11-slim@sha256:abc123...
RUN pip install ultralytics==8.3.0
```

Semua dependency pinning:
- Python: `requirements.txt` dengan hash
- Node.js: `package-lock.json` (commit ke git)
- Docker images: gunakan digest (`image@sha256:...`)

#### c. Offline-First — Jangan Gantungkan ke API Eksternal Saat Demo
Untuk demo lomba, semua harus jalan **tanpa internet**:

1. Pre-load grid images sample (simpan di volume lokal)
2. Seed database dengan dummy detections (berbagai status & warna)
3. Gunakan tile map offline (download tile cache)
4. WA bot: kirim ke nomor demo (atau skip, pakai recording video)

#### d. Self-Healing Infrastructure
```yaml
services:
  wa-bot:
    # Restart otomatis jika crash
    restart: unless-stopped
    # Healthcheck
    healthcheck:
      test: ["CMD", "node", "healthcheck.js"]
      interval: 30s
      retries: 5
      start_period: 10s
```

### 15.3 Rekomendasi

> **Untuk MVP:** Jangan khawatir berlebihan tentang dependency. Cukup
> miliki 1 fallback per dependency kritis (data source & WA).
>
> **Untuk demo:** Wajib offline-capable. Jangan gantungkan pada API
> eksternal saat presentasi.
>
> **Untuk produksi:** Setiap dependency kritis punya minimal 1 fallback.

---

## Ringkasan: Prioritas Mitigasi

Berdasarkan tingkat urgensi dan dampak:

| Prioritas | Masalah | Tindakan | Deadline |
|-----------|---------|----------|----------|
| 🔴 **Critical** | ML akurasi rendah (mAP 0.07) | Ganti backbone StarNet + CGA, atau ganti ke segmentasi | Fase ML |
| 🔴 **Critical** | GEE quota baru (Apr 2026) | Ajukan Contributor/Partner Tier, optimasi EECU | Fase Data |
| 🔴 **Critical** | Wa Bot kena ban | baileys-antiban + MACOS platform + session backup | Fase Bot |
| 🟡 **High** | Cloud cover Indonesia | Integrasi Sentinel-1 SAR, komposit temporal | Fase Data |
| 🟡 **High** | Dataset deforestasi Indonesia | Transfer learning + active learning loop | Fase ML |
| 🟡 **High** | User validation | Minimal 3 wawancara sebelum final | Fase Presentasi |
| 🟡 **High** | Demo tanpa internet | Offline build + seed data | Fase Demo |
| 🟢 **Medium** | False positive | NDVI + ensemble voting | Fase ML |
| 🟢 **Medium** | Skalabilitas grid | Adaptive grid + prioritas processing | Fase Data |
| 🟢 **Medium** | Security | JWT + environment vars | Fase API |
| 🟢 **Medium** | EUDR compliance angle | Siapkan argumentasi bisnis | Fase Presentasi |

---

## Referensi

1. *Real-time deforestation anomaly detection using YOLO and LangChain agents*
   — Nature Scientific Reports, 2025
2. *Multi-Scenario Recognition and Detection Model Based on Improved YOLOv8*
   — Forests Journal (MDPI), 2026
3. *VHRTrees: a new benchmark dataset for tree detection in satellite imagery*
   — Frontiers in Forests and Global Change, 2025
4. *Earth Engine Noncommercial Tiers* — Google Developers, April 2026
5. *Spatial and Temporal Availability of Cloud-free Optical Observations in the
   Tropics* — Nature Scientific Data, 2023
6. *DETER-R: Near-Real Time Tropical Forest Disturbance Warning System Based on
   Sentinel-1* — Remote Sensing (MDPI), 2022
7. *Baileys v7.0.0 Release Notes* — WhiskeySockets (GitHub), 2026
8. *Wide-Area Near-Real-Time Monitoring of Tropical Forest Degradation Using
   Sentinel-1* — Remote Sensing (MDPI), 2020
9. *Deforestation detection using deep learning-based semantic segmentation:
   a systematic review* — Frontiers, 2024
10. *EU Regulation on Deforestation-free Products (EUDR)* — European Commission, 2023/2025/2026
11. *Earth observation as enabler for implementing the EUDR* — npj Climate Action, 2025
12. *Constructing a knowledge base from remote sensing indicators for
    deforestation assessment* — Applied Intelligence (Springer), 2025
13. *Crowd-Driven Deep Learning Tracks Amazon Deforestation* — Remote Sensing (MDPI), 2023
14. *Assessing the Performance of Deep Learning Networks for Real-Time
    Deforestation Segmentation* — SEMISH, 2025
15. *Google Earth Engine Pricing* — Google Cloud, 2026
