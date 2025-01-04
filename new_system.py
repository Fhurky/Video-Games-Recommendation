import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
import os
import joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "16"  # Mantıksal çekirdek sayısını belirtin


# Veri yükleme
data = pd.read_csv("datasets/new_games_steam.csv")

# Eksik 'name' verilerini kaldır
data = data.dropna(subset=['name'])

# 'name' sütununu string'e çevir
data['name'] = data['name'].astype(str)

# Model için veriyi hazırlama
X = data.drop(["name"], axis=1)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN ile benzer oyunları bulma
knn = NearestNeighbors(n_neighbors=15, metric='cosine')
knn.fit(X_scaled)

game_index = 37  # Örnek oyun indeksi
game_index -=2
distances, indices = knn.kneighbors([X_scaled[game_index]])

print(f"Benzer oyunlar {data['name'].iloc[game_index]} için (KNN):")
for i in indices[0]:
    print("-> " + data['name'].iloc[i])

# KNN modelini kaydetme
joblib.dump(knn, "knn_model.pkl")
print("KNN modeli 'knn_model.pkl' olarak kaydedildi.")

# K-Means ile kümeleme
kmeans = KMeans(n_clusters=4000, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# Veriye K-Means küme etiketlerini ekleme
data['kmeans_cluster'] = kmeans_clusters

# Hedef oyunun K-Means kümesini bulma
target_kmeans_cluster = data.loc[game_index, 'kmeans_cluster']

similar_games_kmeans = data[data['kmeans_cluster'] == target_kmeans_cluster]
print(f"Benzer oyunlar {data['name'].iloc[game_index]} için (K-Means):")
for name in similar_games_kmeans['name']:
    print("-> " + name)

# K-Means modelini kaydetme
joblib.dump(kmeans, "kmeans_model.pkl")
print("K-Means modeli 'kmeans_model.pkl' olarak kaydedildi.")


