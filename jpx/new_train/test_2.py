import pandas as pd
import numpy as np

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')


train_data_price = train_data_price[['Date','SecuritiesCode','Open','Close','Target']]

lag_list_ = [1,2,3,5]

def get_lag_info(train_data,lag_list):
    for lag_length in lag_list:
        train_data[f'lag_{lag_length}_info'] = train_data.groupby(['SecuritiesCode'])['Open'].shift(lag_length-1)
        train_data[f'lag_{lag_length}_info'] = (train_data['Close']-train_data[f'lag_{lag_length}_info'])/train_data[f'lag_{lag_length}_info']
    return train_data


train_data_price = get_lag_info(train_data_price,lag_list_)


new_list = train_data_price[train_data_price['SecuritiesCode'] == 1301]
print(new_list)

        
