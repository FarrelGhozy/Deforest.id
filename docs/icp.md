# Idea Concept Paper (ICP)

Kami telah menyusun **Idea Concept Paper** setebal 3 halaman untuk submission lomba, mencakup:

- **Problem Statement**: gap resolusi monitoring (6,25 ha REDD+ vs 2 ha ambang ekologis)
- **Proposed Solution**: pipeline U-Net + pseudo-labeling NDVI temporal + kerangka peringatan dini 3 level (Waspada/Siaga/Kritis) dengan amplifikasi untuk hutan lindung
- **Initial Results**: IoU 0,73, Dice 0,84 pada 5.695 sampel test
- **Roadmap**: SAR integration, field validation, web platform, policy linkage

## Dokumen

| File | Deskripsi |
|------|-----------|
| `paper/icp.tex` | Source LaTeX (3 halaman, 10pt twocolumn, English) |
| `paper/icp.pdf` | PDF siap submit (65 KB) |
| `paper/icp-refs.bib` | Bibliografi (5 referensi) |

## Struktur Paper

1. **Abstract** — Ringkasan 1 paragraf
2. **Problem Statement** — Data deforestasi Indonesia, gap monitoring, urgensi
3. **Proposed Solution**
   - Core Technical Innovation (pseudo-labeling via temporal NDVI differencing)
   - Early Warning Framework (tabel threshold + amplifikasi hutan lindung)
   - Cloud Detection Without QA60 (heuristik RGB-NDVI)
4. **Methodology** — 6-tahap pipeline dari akuisisi GEE hingga inferensi
5. **Initial Results** — Metrik test set + perbandingan dengan NDVI threshold baseline
6. **Impact and Scalability** — Zero annotation cost, commodity hardware, policy alignment
7. **Roadmap** — 4 langkah pengembangan ke depan

## Kebaruan ICP

- **Pseudo-labeling otomatis**: training labels dari temporal NDVI differencing, tanpa anotasi manual
- **Cloud heuristic**: solusi atas QA60 yang tidak berfungsi di data GEE
- **Kerangka ambang terintegrasi**: menyatukan tiga pilar (FAO 0,5 ha, ekologis 2 ha, nasional 6,25 ha)
- **Amplifikasi hutan lindung**: peringatan otomatis dinaikkan 1 level untuk kawasan lindung
