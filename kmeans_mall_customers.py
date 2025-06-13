# K-Means Clustering untuk Customer Segmentation

# 1. Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. Load Dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv'
df = pd.read_csv(url)

# 3. EDA Singkat
print(df.head())
print(df.describe())
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()

# 4. Preprocessing dan Feature Selection
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Menentukan Jumlah Cluster Optimal (Elbow Method)
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(k_range, wcss, marker='o')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS')
plt.title('Metode Elbow untuk Menentukan k')
plt.show()

# 6. KMeans Clustering dengan k=5
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# 7. Visualisasi Cluster
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set1')
plt.title('Segmentasi Pelanggan berdasarkan K-Means')
plt.show()

# 8. Evaluasi Cluster
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil_score:.3f}")
