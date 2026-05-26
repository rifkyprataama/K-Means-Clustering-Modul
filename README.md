# Laporan Praktikum Pembelajaran Mesin: K-Means Clustering

**Nama:** Rifky Daffa Pratama  
**Program Studi:** Teknik Informatika  
**Universitas:** Universitas Islam Negeri Sunan Gunung Djati Bandung  

Repositori ini dibuat untuk memenuhi tugas mata kuliah Pembelajaran Mesin, yang mencakup implementasi Alur K-Means Clustering pada dua skenario dataset yang berbeda: tugas utama (Modul 1) dan tugas perluasan (Latihan Mandiri).

---

## 1. Modul 1: Klusterisasi GPS Trajectories
* **Direktori Folder:** `/modul`
* **Script Program:** `Kmeans_Clustering.py`
* **Dataset:** `go_track_tracks.csv` (Sumber: UCI Machine Learning Repository)
* **Karakteristik Fitur:** Mengelompokkan data perjalanan berbasis Android menggunakan parameter `distance` (jarak) dan `speed` (kecepatan) ke dalam 3 jenis kelompok fungsional.
* **Penyesuaian Sintaks:** Berdasarkan kaidah *Unsupervised Learning*, pelaporan berbasis *supervised metrics* (*Accuracy*, *F1-Score*) dilewati karena tiadanya label kelas historis baku (`y_test`) pada berkas data dasar.

## 2. Latihan Mandiri: Segmentasi Pelanggan Mall
* **Direktori Folder:** `/latihan`
* **Script Program:** `Latihan_KMeans.py`
* **Dataset:** `Mall_Customers.csv` (Sumber: Kaggle Data Public)
* **Karakteristik Fitur:** Menerapkan arsitektur logika pemrosesan yang identik dengan Modul 1 untuk memetakan kelompok konsumen ritel. Data dibersihkan dari fitur non-kategoris non-numerik, lalu dikelompokkan menjadi 5 segmentasi pasar strategis berdasarkan fitur `Annual Income` (Pendapatan Tahunan) dan `Spending Score` (Skor Pengeluaran).

---

## Alur Metodologi Pengerjaan (Kedua Berkas)
1. **Data Acquisition:** Memuat berkas ekstensi `.csv` menggunakan struktur data *DataFrame* Pandas.
2. **Data Slicing & Dropping:** Mengeliminasi fitur identitas/indeks yang tidak memiliki korelasi matematis terhadap jarak antartitik objek cluster.
3. **Exploratory Data Analysis (EDA):** Mengamati sebaran awal distribusi spasial data menggunakan diagram pencar grafik *Scatter Plot*.
4. **Feature Scaling:** Mentransformasi orientasi nilai matriks data menuju skala seragam [0, 1] melalui kalkulasi *Min-Max Scaler*.
5. **Algoritma KMeans:** Menginisialisasi parameter centroid awal, melakukan fitting matriks, dan mengekstraksi nilai *labels_* untuk disatukan kembali ke tabel utama.
6. **Centroid Mapping:** Menampilkan koordinat titik pusat geometris massa (*Centroid*) di atas representasi spasial kelompok data akhir.

## Persyaratan Dependensi Eksekusi
Guna menjalankan kedua skrip `.py` di atas melalui *Interactive Window* VS Code, pastikan pustaka berikut telah tersedia di sistem operasi Anda:
```bash
pip install pandas numpy matplotlib scikit-learn