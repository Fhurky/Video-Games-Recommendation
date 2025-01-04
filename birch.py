import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import joblib

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

# Birch modelini oluştur ve eğit
birch = Birch(n_clusters=10)
birch.fit(X_scaled)

# Modeli kaydet
joblib.dump(birch, "birch_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model başarıyla eğitildi ve kaydedildi.")