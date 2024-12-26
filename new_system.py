import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans

# Veri yükleme
data = pd.read_csv("datasets/new_games_steam.csv")

# Eksik 'name' verilerini kaldır
data = data.dropna(subset=['name'])

# 'name' sütununu string'e çevir
data['name'] = data['name'].astype(str)

# Gerekli sütunları kaldırma
columns_to_drop = [
    "Grid-Based Movement", "Logic", "Pirates", "Tennis", "Faith", "Time Attack", "Motorbike",
    "Hockey", "Tanks", "Nonlinear", "Kickstarter", "Building", "Sailing", "Solitaire",
    "Intentionally Awkward Controls", "Psychological", "Dating Sim", "6DOF", "Top-Down Shooter",
    "Web Publishing", "RTS", "Space Sim", "Capitalism", "Music-Based Procedural Generation",
    "Hack and Slash", "Skating", "Memes", "Dynamic Narration", "Text-Based", "Clicker", "Noir",
    "Cold War", "Martial Arts", "Lovecraftian", "Mystery Dungeon", "Hardware", "Nature",
    "Lemmings", "Psychedelic", "Sniper", "Tutorial", "4 Player Local", "Mars", "Match 3",
    "Dark Humor", "Audio Production", "Gun Customization", "Western", "Swordplay", "Real-Time",
    "ATV", "Dinosaurs", "Ninja", "Tactical", "Alternate History", "Transhumanism", "Dog",
    "Bikes", "Football"
]
data = data.drop(columns_to_drop, axis=1)

# Model için veriyi hazırlama
X = data.drop(["name"], axis=1)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN ile benzer oyunları bulma
knn = NearestNeighbors(n_neighbors=15, metric='cosine')
knn.fit(X_scaled)

game_index = 2911  # Örnek oyun indeksi
distances, indices = knn.kneighbors([X_scaled[game_index]])

print(f"Benzer oyunlar {data['name'].iloc[game_index]} için (KNN):")
for i in indices[0]:
    print("-> " + data['name'].iloc[i])

# DBSCAN ile kümeleme
dbscan = DBSCAN(eps=1.5, min_samples=5, metric='cosine')
db_clusters = dbscan.fit_predict(X_scaled)

# Veriye DBSCAN küme etiketlerini ekleme
data['db_cluster'] = db_clusters

# Hedef oyunun DBSCAN kümesini bulma
target_db_cluster = data.loc[game_index, 'db_cluster']

if target_db_cluster == -1:
    print(f"{data['name'].iloc[game_index]} oyununa benzer bir küme bulunamadı (DBSCAN).")
else:
    similar_games_db = data[data['db_cluster'] == target_db_cluster]
    print(f"Benzer oyunlar {data['name'].iloc[game_index]} için (DBSCAN):")
    for name in similar_games_db['name']:
        print("-> " + name)

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
