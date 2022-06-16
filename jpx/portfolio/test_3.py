import pandas as pd

my_data = pd.read_csv('kaggle_data/JPX/tool_data/rank_info.csv')
my_data = my_data[['Date','SecuritiesCode','lag_rank_30']]
my_data['Date'] = my_data.groupby(['SecuritiesCode'])['Date'].shift(-1)
my_data.to_csv('kaggle_data/JPX/tool_data/rank_info_3.csv')