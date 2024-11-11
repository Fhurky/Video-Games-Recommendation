import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv("datasets/new_games_steam.csv")

scaler = StandardScaler()

data = data.drop([
    "Grid-Based Movement", "Logic", "Pirates", "Tennis", "Faith", "Time Attack", "Motorbike", 
    "Hockey", "Tanks", "Nonlinear", "Kickstarter", "Building", "Sailing", "Solitaire"
    , "Intentionally Awkward Controls", "Psychological", 
    "Dating Sim", "6DOF", "Top-Down Shooter", "Web Publishing", "RTS", "Space Sim", 
    "Capitalism", "Music-Based Procedural Generation", "Hack and Slash", "Skating", 
    "Memes", "Dynamic Narration", "Text-Based", "Clicker", "Noir", "Cold War", 
    "Martial Arts", "Lovecraftian", "Mystery Dungeon", "Hardware", 
    "Nature", "Lemmings", "Psychedelic", "Sniper", "Tutorial", "4 Player Local" 
    , "Mars", "Match 3", "Dark Humor", "Audio Production", "Gun Customization",
    "Western", "Swordplay", "Real-Time", "ATV", "Dinosaurs", "Ninja", "6DOF", "Tactical",
     "Alternate History", "Transhumanism", "Dog", "Bikes", "Football"
], axis=1)

X = data.drop(["name"], axis = 1)

knn = NearestNeighbors(n_neighbors=15, metric='cosine')
knn.fit(X)

game_index = 1
distances, indices = knn.kneighbors([X.iloc[game_index]])

print(f"Benzer oyunlar {data['name'].iloc[game_index]} için:")
for i in indices[0]:
    print("-> "+data['name'].iloc[i])

