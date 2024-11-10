import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv("games_steam.csv")

scaler = StandardScaler()

X = data.drop(["name"], axis = 1)

knn = NearestNeighbors(n_neighbors=15, metric='cosine')
knn.fit(X)

game_index = 110
print(X.iloc[game_index])
distances, indices = knn.kneighbors([X.iloc[game_index]])

print(f"Benzer oyunlar {data['name'].iloc[game_index]} için:")
for i in indices[0]:
    print("-> "+data['name'].iloc[i])

