import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

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

# Modelleri yükleme
knn = joblib.load("knn_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
gmm = joblib.load("gmm_model.pkl")  # GMM modeli yüklendi

# Örnek oyun seçimi
game_index = 37  # Örnek oyun indeksi
game_index -= 2  # İndeks düzeltmesi

# KNN ile benzer oyunları bulma
distances, indices = knn.kneighbors([X_scaled[game_index]])
print(f"Benzer oyunlar {data['name'].iloc[game_index]} için (KNN):")
for i in indices[0]:
    print("-> " + data['name'].iloc[i])

# K-Means ile benzer oyunları bulma
data['kmeans_cluster'] = kmeans.predict(X_scaled)
target_kmeans_cluster = data.loc[game_index, 'kmeans_cluster']
similar_games_kmeans = data[data['kmeans_cluster'] == target_kmeans_cluster]
print(f"Benzer oyunlar {data['name'].iloc[game_index]} için (K-Means):")
for name in similar_games_kmeans['name'][:10]:  # İlk 10 oyun
    print("-> " + name)

# GMM ile benzer oyunları bulma
data['gmm_cluster'] = gmm.predict(X_scaled)
target_gmm_cluster = data.loc[game_index, 'gmm_cluster']
similar_games_gmm = data[data['gmm_cluster'] == target_gmm_cluster]
print(f"Benzer oyunlar {data['name'].iloc[game_index]} için (GMM):")
for name in similar_games_gmm['name'][:10]:  # İlk 10 oyun
    print("-> " + name)
