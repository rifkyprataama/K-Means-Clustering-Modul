# Modul 1: K-Means Clustering GPS Trajectories

**Nama:** Rifky Daffa Pratama  
**Program Studi:** Teknik Informatika  
**Universitas:** Universitas Islam Negeri Sunan Gunung Djati Bandung  

## Deskripsi Pengerjaan
Repositori ini berisi implementasi algoritma K-Means Clustering menggunakan dataset GPS Trajectories (dari aplikasi Android Go!Track) yang diambil dari UCI Machine Learning Repository. Proyek ini bertujuan untuk menyelesaikan Modul 1 dengan mengelompokkan data perjalanan berdasarkan kemiripannya.

## Dokumentasi & Penyesuaian Modul
Kode pada repositori ini (`Kmeans_Clustering.py`) ditulis dengan merujuk langsung pada instruksi Modul 1. Namun, terdapat penyesuaian teknis untuk menghindari *error* dari instruksi modul asli:
- **Penghapusan Evaluasi Klasifikasi:** Bagian perhitungan metrik `accuracy_score`, `classification_report`, dan `f1_score` tidak diimplementasikan. Metrik klasifikasi tersebut membutuhkan target label (`y_test`), yang mana tidak berlaku pada algoritma *Unsupervised Learning* seperti K-Means.

## Cara Menjalankan Program
1. Pastikan library yang dibutuhkan telah terinstal (`pip install pandas numpy matplotlib scikit-learn`).
2. Pastikan file dataset `go_track_tracks.csv` berada di dalam satu folder yang sama dengan *script*.
3. Jalankan *script* menggunakan VS Code Interactive Window atau terminal: `python Kmeans_Clustering.py`.