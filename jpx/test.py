import pandas as pd


# data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
data_2 = pd.read_csv("kaggle_data\\JPX\\stock_list.csv")
print(data_2.loc[1000])
print(list(data_2))
print(data_2.info())

