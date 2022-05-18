import pandas as pd
import numpy as np

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')

train_data_price = train_data_price[['Date', 'SecuritiesCode', 'Open','Close']]
train_data_price['test'] = train_data_price.groupby(['SecuritiesCode'])['Close'].shift(3)

print(train_data_price)