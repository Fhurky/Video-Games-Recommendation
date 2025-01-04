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

# Model için veriyi hazırlama
X = data.drop(["name"], axis=1)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gaussian Mixture Model ile kümeleme
gmm = GaussianMixture(n_components=250, random_state=42)  # Küme sayısını ihtiyacınıza göre belirleyin
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