import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os
import joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Fiziksel çekirdek sayısını belirtin

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

# Gaussian Mixture Model ile kümeleme
gmm = GaussianMixture(n_components=200, random_state=42)  # Küme sayısını ihtiyacınıza göre belirleyin
gmm_clusters = gmm.fit_predict(X_scaled)

# Veriye GMM küme etiketlerini ekleme
data['gmm_cluster'] = gmm_clusters

# Hedef oyun
game_index = 37  # Örnek oyun indeksi
target_gmm_cluster = data.loc[game_index, 'gmm_cluster']

# Hedef kümede yer alan oyunlar
similar_games_gmm = data[data['gmm_cluster'] == target_gmm_cluster]

# 10 adet oyun seçme (örneğin, rastgele)
similar_games_sample = similar_games_gmm.sample(10, random_state=42)

# Önerilen oyunları yazdırma
print(f"Benzer oyunlar {data['name'].iloc[game_index]} için (GMM, 10 adet):")
for name in similar_games_sample['name']:
    print("-> " + name)

# Modeli kaydet
joblib.dump(gmm, "gmm_model.pkl")
print("Model başarıyla kaydedildi.")