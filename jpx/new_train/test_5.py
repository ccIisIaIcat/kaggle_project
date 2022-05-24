import pandas as pd
import numpy as np

new_data = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
new_data = new_data[['Date','Open','Close','SecuritiesCode','AdjustmentFactor','SupervisionFlag']]
print(list(new_data))

print(new_data['SupervisionFlag'].mean())
# print(new_data[(new_data['SecuritiesCode'] == 6861) & (new_data['Date']<='2017-01-20')])




    