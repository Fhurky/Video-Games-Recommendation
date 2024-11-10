import pandas as pd

data = pd.read_csv("games_genres.csv")
data = data.loc[:, ~data.columns.str.contains('^Unnamed', na=False)]
data = data.drop(["Genres"], axis = 1)

print(data.columns)