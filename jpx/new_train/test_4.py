import pandas as pd

stock_info = pd.read_csv('kaggle_data/JPX/stock_list.csv')
stock_info = stock_info[['SecuritiesCode','17SectorCode']]
print(stock_info['17SectorCode'].dtypes)