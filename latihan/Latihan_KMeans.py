#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 1. Membaca data dataset Mall_Customers.csv
baca = pd.read_csv("Mall_Customers.csv")

# Melihat 5 data teratas
print("--- 5 Data Teratas ---")
print(baca.head())

# Melihat informasi tentang dataset
print("\n--- Informasi Dataset ---")
baca.info()

# Mengurangi kolom yang tidak dipergunakan dalam penentuan clustering 
# (Menghapus CustomerID, Gender, dan Age agar menyisakan kolom numerik target)
if 'CustomerID' in baca.columns:
    baca = baca.drop(["CustomerID", "Gender", "Age"], axis=1)

print("\n--- Data Setelah Pembersihan Kolom ---")
print(baca.head())

# Menentukan variabel yang akan diklusterkan (Annual Income dan Spending Score)
baca_x = baca.iloc[:, 0:2]
print("\n--- Variabel yang Diklusterkan ---")
print(baca_x.head())

# Memvisualkan persebaran data awal
plt.figure(figsize=(8,6))
plt.scatter(baca['Annual Income (k$)'], baca['Spending Score (1-100)'], s=30, c="blue", marker="o", alpha=0.5)
plt.title("Persebaran Data Awal (Income vs Spending Score)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
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

# Membuat model KMeans (Menggunakan 5 kluster karena karakteristik data Mall sangat rapi terbagi 5)
kmeans = KMeans(n_clusters=5, random_state=42)

# Training model sesuai alur praktikum
kmeans.fit(baca_x)

baca["kluster"] = kmeans.labels_

# Plot hasil clustering
plt.figure(figsize=(8,6))

# Visualisasi sebaran kluster
plt.scatter(
    baca['Annual Income (k$)'],
    baca['Spending Score (1-100)'],
    c=baca['kluster'],
    cmap='viridis',
    s=50
)

# Menambahkan titik centroid hasil training
plt.scatter(
    kmeans.cluster_centers_[:, 0], # Centroid untuk Sumbu X (Annual Income)
    kmeans.cluster_centers_[:, 1], # Centroid untuk Sumbu Y (Spending Score)
    s=200,
    c='red',
    marker='X',
    label='Centroid'
)

# Menambahkan label kelengkapan grafik
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering: Segmentasi Pelanggan Mall')
plt.legend()
plt.show()
#%%