import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import StandardScaler

# Download latest version
path = kagglehub.dataset_download("trolukovich/steam-games-complete-dataset")
path = path + "/steam_games.csv"
data = pd.read_csv(path)

# Delete unnecesary columns
data = data.drop(["url", 'types', 'recommended_requirements', 'desc_snippet', 'recent_reviews', 'all_reviews', 'developer', 'minimum_requirements'], axis = 1)
data = data.drop(['mature_content', 'game_description', 'languages', 'game_details', 'publisher', 'discount_price', 'release_date'], axis = 1)

# Convert dolars column
def convert_to_dollars(value):
    if isinstance(value, str):
        if value == 'Free to Play':
            return 0.0  # Free to Play'i 0 dolar olarak kabul ediyoruz
        elif value.startswith('$'):
            return float(value.replace('$', ''))  # Dolar işaretini kaldır ve floata çevir
        else:
            try:
                return float(value)  # Diğer sayıları floata çevir
            except ValueError:
                return np.nan
    return value

data['original_price'] = data['original_price'].apply(convert_to_dollars)

average_price = data['original_price'].mean()
data['original_price'] = data['original_price'].fillna(average_price)

data['achievements'] = data['achievements'].apply(lambda x: 1 if pd.notnull(x) and pd.to_numeric(x, errors='coerce') is not None else 0)

popular_tags = []
for i in range(len(data)):
    tags = data.iloc[i]["popular_tags"]
    if isinstance(tags, str): 
        tags = tags.split(",")
        popular_tags.extend(tags)

# Get the tags in a list
popular_tags = list(set(popular_tags))

data_genres = pd.DataFrame(columns=popular_tags)
data = pd.concat([data.reset_index(drop=True), data_genres], axis=1)

# Fill the new columns
for genre in popular_tags:
    data[genre] = data['popular_tags'].apply(lambda tags: 1 if isinstance(tags, str) and genre in tags else 0)

data = data.drop(columns=['popular_tags'])
data = data.drop(["genre"], axis = 1)

scaler = StandardScaler()
scaler.fit(data[["original_price"]])
data["original_price"] = scaler.transform(data[["original_price"]])

data.to_csv('new_games_steam.csv', index=False)

