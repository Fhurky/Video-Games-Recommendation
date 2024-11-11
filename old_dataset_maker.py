import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("matheusfonsecachaves/popular-video-games")
path = path + "/backloggd_games.csv"
data = pd.read_csv(path)
data = data.drop(["Summary","Wishlist","Lists","Reviews","Release_Date","Developers","Backlogs","Playing", "Plays"], axis=1)
# Drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

Platforms = []
for i in range(len(data)):
    Platforms.extend(data.iloc[i]["Platforms"].replace("[", "").replace("]", "").replace("'", "").split(", "))

Platforms = list(set(Platforms))
# DataFrame oluşturuluyor
data_Platforms = pd.DataFrame(columns=Platforms)
# concat() ile birleştirme
df_concat = pd.concat([data, data_Platforms], ignore_index=True)

# Her satırın genres değerini alıp ilgili sütunları 1 veya 0 yapma
for index, row in df_concat.iterrows():
    for category in Platforms:
        if category in row['Platforms']:
            df_concat.at[index, category] = 1
        else:
            df_concat.at[index, category] = 0


df_concat.to_csv('games_platforms.csv', index=False)