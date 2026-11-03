# Eksperimen & Evaluasi Model Klasifikasi Sentimen Ulasan Pelanggan E-Commerce

Repositori ini berisi laporan eksperimen, perbandingan performa, analisis *trade-off*, dan rekomendasi teknis (*technical approach*) untuk fitur klasifikasi sentimen otomatis pada ulasan produk *e-commerce*. Proyek ini membandingkan dua paradigma AI:
1. **Model Klasik Machine Learning**: TF-IDF Vectorizer + Logistic Regression (Scikit-Learn).
2. **LLM API (Zero-Shot)**: Google Gemini API (`gemini-3.5-flash-lite`) via `google-genai` SDK.

---

## 1. Problem Statement & Dataset

### Problem Statement
Sebuah tim produk *e-commerce* ingin mengotomatiskan proses kurasi dan pelabelan sentimen ulasan pelanggan (*review sentiment*) menjadi **`positif`** atau **`negatif`**. Tujuannya adalah untuk:
- Membantu calon pembeli menyaring ulasan produk secara cepat.
- Memberikan sinyal kepuasan pelanggan kepada *merchant* dan tim operasional *quality control*.
- Mengurangi kebutuhan peninjauan ulasan secara manual.

### Deskripsi Dataset
Dataset yang digunakan adalah `customer_reviews_sentiment.csv` dengan karakteristik:
- **Total Data**: 200 baris ulasan pelanggan di berbagai kategori produk e-commerce (pakaian, elektronik, aksesoris, perabotan rumah tangga).
- **Struktur Kolom**:
  - `review_id`: Identifikasi unik ulasan.
  - `product_name`: Nama produk yang diulas.
  - `review_text`: Isi teks ulasan dari pelanggan.
  - `sentiment`: Label sentimen biner (`positif` atau `negatif`).
- **Data Splitting**: 
  - Dibagi menggunakan `train_test_split` dengan rasio **80% data latih (160 baris)** dan **20% data uji (40 baris)**.
  - Dilakukan secara terstratifikasi (`stratify=y`) untuk menjaga proporsi seimbang antara kelas positif (22 sampel uji) dan negatif (18 sampel uji) pada kedua pendekatan agar evaluasi adil (*fair evaluation*).

---

## 2. Ringkasan Eksperimen Kedua Pendekatan

### Pendekatan 1: Model Klasik Machine Learning (Scikit-Learn)
- **Ekstraksi Fitur**: Menggunakan `TfidfVectorizer` untuk mengubah teks ulasan menjadi representasi bobot numerik (*Term Frequency - Inverse Document Frequency*).
- **Algoritma**: `LogisticRegression()` untuk pembelajaran klasifikasi biner.
- **Workflow**:
  1. *Fit* vektorizer dan latih model pada `X_train` (160 data).
  2. Transformasi `X_test` dan lakukan inferensi label `y_pred_classic`.
  3. Evaluasi metrik performa terhadap `y_test`.

### Pendekatan 2: LLM API (Google Gemini)
- **Model**: `gemini-3.5-flash-lite` menggunakan library resmi terbaru `google-genai`.
- **Desain Prompt (Zero-Shot Prompting)**:
  ```text
  Klasifikasikan sentimen ulasan berikut sebagai "positif" atau "negatif".
  Jawab hanya dengan satu kata: positif atau negatif.

  Ulasan: "{review_text}"
  Sentimen:
  ```
- **Konfigurasi & Resiliensi API**:
  - `temperature = 0.2` dan `max_output_tokens = 10` untuk respons yang konsisten dan hemat kuota.
  - `AutomaticFunctionCallingConfig(disable=True)` untuk menghindari peringatan AFC.
  - **Penanganan Rate Limit & Retry**: Menerapkan deteksi `ClientError 429` (batas 15 RPM pada Free Tier) dengan jeda otomatis ~26–28 detik agar kuota ter-reset, serta *exponential backoff retry* untuk error server transien (500, 502, 503).
  - **Pacing**: Jeda 1.0 detik antar-permintaan dan normalisasi string output menjadi `'positif'` atau `'negatif'`.

---

## 3. Tabel Hasil Evaluasi dan Perbandingan

Kedua pendekatan dievaluasi menggunakan *ground truth* data uji (`y_test`, $N = 40$).

### 3.1 Ringkasan Metrik Utama

| Pendekatan | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Klasik (Logistic Regression)** | **1.0000 (100%)** | **1.0000** | **1.0000** | **1.0000** |
| **LLM API (Gemini 3.5 Flash Lite)** | **1.0000 (100%)** | **1.0000** | **1.0000** | **1.0000** |

### 3.2 Detail Metrik per Kelas Sentimen

| Label Kelas | Metrik | Model Klasik (Scikit-Learn) | LLM API (Gemini) |
| :--- | :--- | :---: | :---: |
| **Negatif** (18 ulasan) | Precision | 1.00 | 1.00 |
| | Recall | 1.00 | 1.00 |
| | F1-Score | 1.00 | 1.00 |
| **Positif** (22 ulasan) | Precision | 1.00 | 1.00 |
| | Recall | 1.00 | 1.00 |
| | F1-Score | 1.00 | 1.00 |

### 3.3 Confusion Matrix

```
Model Klasik:                    LLM API (Gemini):
        Pred: Neg   Pred: Pos            Pred: Neg   Pred: Pos
Act: Neg   18          0          Act: Neg   18          0
Act: Pos    0         22          Act: Pos    0         22
```

> **Catatan Analisis**: Kedua model mencapai skor sempurna (100%) pada dataset uji ini karena kata-kata kunci sentimen pada dataset terpolarisasi sangat jelas (*misal: "lambat", "rusak", "beda jauh" vs "puas", "awet", "cepat"*).

---

## 4. Analisis Trade-off dan Limitation

Meskipun akurasi kuantitatif kedua model setara pada dataset pengujian, karakteristik operasional di sistem produksi nyata memiliki *trade-off* mendasar:

| Dimensi Evaluasi | Model Klasik (TF-IDF + Logistic Regression) | LLM API (Google Gemini) |
| :--- | :--- | :--- |
| **Latensi Inferensi** | **Ultra Cepat (< 10 ms)**. Diproses langsung di CPU lokal. | **Sedang - Lambat (500 – 2000 ms)** per ulasan via network call. |
| **Throughput & Skalabilitas** | **Sangat Tinggi**. Mampu melayani ribuan RPS tanpa antrean eksternal. | **Terbatas Kuota**. Tunduk pada kuota API (Free Tier: 15 RPM). |
| **Biaya Operasional (OpEx)** | **Mendekati Rp 0**. Tidak ada biaya per-panggilan API; cukup CPU server biasa. | **Berbayar per-token**. Biaya terakumulasi seiring naiknya volume ulasan. |
| **Kebutuhan Data Latih** | **Wajib ada data latih teranotasi** (*annotated dataset*). | **Tidak perlu data latih** (*Zero-shot / Few-shot learning*). |
| **Pemahaman Bahasa & Konteks** | **Kaku**. Hanya mencocokkan kata kunci; rentan gagal pada sarkasme, slang, dan typo. | **Sangat Luas**. Memahami konteks semantik, kalimat campuran (*mixed review*), dan sarkasme. |
| **Ketergantungan Eksternal** | **Nol**. Berjalan *on-premise* / *private cloud* secara mandiri. | **Tinggi**. Bergantung pada ketersediaan (*uptime* & *network*) pihak ketiga. |

---

## 5. Rekomendasi Technical Approach

### Rekomendasi Utama: *Tiered / Hybrid Architecture*

Untuk aplikasi *production-grade* e-commerce berskala menengah hingga besar, rekomendasi teknis terbaik adalah menerapkan **Arsitektur Bertingkat (*Hybrid Tiered Approach*)**:

```
                       [ Ulasan Baru Masuk ]
                                 │
                                 ▼
             ┌───────────────────────────────────────┐
             │  TIER 1: Model Klasik (Scikit-Learn)  │
             │  - Latensi < 10 ms, Biaya Rp 0       │
             └───────────────────┬───────────────────┘
                                 │
                   Skor Probabilitas Keyakinan (p)
                                 │
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
       [ Yakin / High Margin ]             [ Ragu / Ambigu ]
       (p >= 0.85 atau p <= 0.15)         (0.15 < p < 0.85)
               │                                   │
               ▼                                   ▼
    Output Sentimen Langsung              ┌─────────────────────────────────┐
    (Menangani ~85-90% ulasan)            │ TIER 2: LLM API (Gemini Flash)  │
                                          │ - Cek semantik & konteks mendalam│
                                          └────────────────┬────────────────┘
                                                           │
                                                           ▼
                                                Output Sentimen Final
                                                (Menangani ~10-15% kasus sulit)
```

#### Alasan Strategis Pemilihan Hybrid:
1. **Efisiensi Biaya & Latensi**: Sekitar 85–90% ulasan e-commerce menggunakan kalimat sederhana dan langsung selesai di Tier 1 secara instan tanpa biaya API.
2. **Kualitas Tinggi pada Kasus Rumit**: Ulasan ambigu, sarkas, atau panjang yang membuat model klasik ragu diteruskan ke LLM Tier 2 untuk analisis konteks mendalam.
3. **Resiliensi Tinggi**: Jika API eksternal mengalami kendala jaringan atau *rate limit*, sistem tetap berjalan 100% menggunakan model Tier 1 (*graceful degradation*).

#### Panduan Jika Memilih Solusi Tunggal:
- **Gunakan Model Klasik jika**: Prioritas utama adalah biaya nol, latensi ultra-rendah (misal: feedback instan saat pembeli mengetik), dan telah memiliki data historis berlabel.
- **Gunakan LLM API jika**: Prioritas adalah *time-to-market* instan (fase MVP), belum ada data latih teranotasi, atau ulasan didominasi variasi bahasa non-formal yang sangat dinamis.

---

## 6. Panduan Menjalankan Notebook / Script

### 6.1 Prasyarat
- Python 3.10 atau lebih baru (direkomendasikan Python 3.12).
- Google Gemini API Key (dapat diperoleh gratis di [Google AI Studio](https://aistudio.google.com/)).

### 6.2 Langkah Instalasi

1. **Clone repository & buat virtual environment**:
   ```bash
   git clone <url-repository-anda>
   cd assignment-model-experiment-Arkhananta-Satrya-Ariadilla
   python -m venv .venv
   ```

2. **Aktivasi virtual environment**:
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependensi yang diperlukan**:
   ```bash
   pip install pandas scikit-learn google-genai python-dotenv
   ```

4. **Konfigurasi Environment Variable**:
   Buat file `.env` di direktori *root* atau set environment variable:
   ```env
   GEMINI_API_KEY=AIzaSyD...your_api_key_here...
   ```

### 6.3 Menjalankan Notebook
Buka dan jalankan file notebook di VS Code atau Jupyter Lab:
```bash
jupyter notebook notebook/experiment_notebook.ipynb
```
Jalankan setiap *cell* secara berurutan dari Section 1 hingga Section 6. Hasil prediksi dan tabel metrik evaluasi akan langsung ter-render.
