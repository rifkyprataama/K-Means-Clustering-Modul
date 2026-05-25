#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Membaca data dataset go_track_tracks.csv
baca = pd.read_csv("go_track_tracks.csv")

# Melihat 5 data teratas
print("--- 5 Data Teratas ---")
print(baca.head())

# Melihat informasi tentang dataset
print("\n--- Informasi Dataset ---")
baca.info()

# Mengurangi kolom yang tidak dipergunakan dalam penentuan clustering (misalnya 'linha')
if 'linha' in baca.columns:
    baca = baca.drop(["linha"], axis=1)

print("\n--- Data Setelah Kolom 'linha' Dihapus ---")
print(baca.head())

# Menentukan variabel yang akan diklusterkan (mengambil kolom indeks 1 dan 2)
baca_x = baca.iloc[:, 1:3]
print("\n--- Variabel yang Diklusterkan ---")
print(baca_x.head())

# Memvisualkan persebaran data awal (Berdasarkan distance dan speed)
plt.figure(figsize=(8,6))
plt.scatter(baca.distance, baca.speed, s=10, c="red", marker="o", alpha=0.5)
plt.title("Persebaran Data Awal (Distance vs Speed)")
plt.xlabel("Distance")
plt.ylabel("Speed")
plt.show()

# Mengubah variabel data yang berbentuk data frame menjadi array
x_array = np.array(baca_x)
print("\n--- Data Array (5 Baris Pertama) ---")
print(x_array[:5])

# Melakukan normalisasi data menggunakan Min-Max Scaling
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
print("\n--- Data Scaled (5 Baris Pertama) ---")
print(x_scaled[:5])

# Membuat model KMeans dengan 3 kluster
kmeans = KMeans(n_clusters=3, random_state=42)

# Training model sesuai instruksi modul
kmeans.fit(baca)

# Menambahkan label kluster ke dalam dataframe
baca["kluster"] = kmeans.labels_

# Plot hasil clustering
plt.figure(figsize=(8,6))

# Visualisasi sebaran kluster berdasarkan 'id_android' dan 'speed' sesuai instruksi modul
plt.scatter(
    baca['id_android'],
    baca['speed'],
    c=baca['kluster'],
    cmap='viridis'
)

# Menambahkan titik centroid
plt.scatter(
    kmeans.cluster_centers_[:, 1], 
    kmeans.cluster_centers_[:, 2], 
    s=200,
    c='red',
    marker='X',
    label='Centroid'
)

# Menambahkan label sesuai modul
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
#%%