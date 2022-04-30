import pandas as pd

# data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
# print(list(data))
# print(data['AdjustmentFactor'].describe())
# # print(len(data[data['SupervisionFlag']==True]))

data = pd.read_csv("kaggle_data\\JPX\\train_files\\options.csv")
print(data)
print(list(data))