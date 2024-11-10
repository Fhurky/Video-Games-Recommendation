import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("matheusfonsecachaves/popular-video-games")
path = path + "/backloggd_games.csv"
data = pd.read_csv(path)
data = data.drop(["Summary","Release_Date","Developers","Backlogs","Playing","Platforms"], axis=1)
# Drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

Genres = []
for i in range(len(data)):
    Genres.extend(data.iloc[i]["Genres"].replace("[", "").replace("]", "").replace("'", "").split(", "))

Genres = list(set(Genres))
# DataFrame oluşturuluyor
data_genres = pd.DataFrame(columns=Genres)
# concat() ile birleştirme
df_concat = pd.concat([data, data_genres], ignore_index=True)

# Her satırın genres değerini alıp ilgili sütunları 1 veya 0 yapma
for index, row in df_concat.iterrows():
    for category in Genres:
        if category in row['Genres']:
            df_concat.at[index, category] = 1
        else:
            df_concat.at[index, category] = 0


df_concat.to_csv('games_genres.csv', index=False)