import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv("datasets/games_genres.csv", low_memory=False)
data_platfomrs = pd.read_csv(("games_platforms.csv"))

data = data.loc[:, ~data.columns.str.contains('^Unnamed', na=False)]
data = data.drop(["Genres"], axis = 1)

scaler = StandardScaler()

mean_rating = data["Rating"].mean()
data["Rating"] = data["Rating"].fillna(mean_rating)
scaler.fit(data[["Rating"]])
data["Rating"] = scaler.transform(data[["Rating"]])

data["Wishlist"] = data["Wishlist"].apply(lambda x: float(str(x).replace('K', '')) * 1000 if 'K' in str(x) else float(x))
mean_rating = data["Wishlist"].mean()
data["Wishlist"] = data["Wishlist"].fillna(mean_rating)
scaler.fit(data[["Wishlist"]])
data["Wishlist"] = scaler.transform(data[["Wishlist"]])

data["Lists"] = data["Lists"].apply(lambda x: float(str(x).replace('K', '')) * 1000 if 'K' in str(x) else float(x))
mean_rating = data["Lists"].mean()
data["Lists"] = data["Lists"].fillna(mean_rating)
scaler.fit(data[["Lists"]])
data["Lists"] = scaler.transform(data[["Lists"]])

data["Reviews"] = data["Reviews"].apply(lambda x: float(str(x).replace('K', '')) * 1000 if 'K' in str(x) else float(x))
mean_rating = data["Reviews"].mean()
data["Reviews"] = data["Reviews"].fillna(mean_rating)
scaler.fit(data[["Reviews"]])
data["Reviews"] = scaler.transform(data[["Reviews"]])

data["Plays"] = data["Plays"].apply(lambda x: float(str(x).replace('K', '')) * 1000 if 'K' in str(x) else float(x))
mean_rating = data["Plays"].mean()
data["Plays"] = data["Plays"].fillna(mean_rating)
scaler.fit(data[["Plays"]])
data["Plays"] = scaler.transform(data[["Plays"]])

X = data[['Rating', 'Wishlist', 'Lists','Quiz/Trivia', 'Strategy', 'Fighting', 'Adventure',
       'Turn Based Strategy', 'Simulator', 'Point-and-Click', 'Shooter',
       'Brawler', 'Platform', 'Real Time Strategy', 'Sport', 'Pinball',
       'Indie', 'Tactical', 'Music', 'Arcade', 'Puzzle', 'MOBA',
       'Visual Novel', 'Card & Board Game', 'Racing']]

knn = NearestNeighbors(n_neighbors=15, metric='cosine')
knn.fit(X)
game_index = 115
distances, indices = knn.kneighbors([X.iloc[game_index]])

print(f"Benzer oyunlar {data['Title'].iloc[game_index]} için:")
for i in indices[0]:
    print("-> "+data['Title'].iloc[i])
