import pandas as pd
import numpy as np

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')


train_data_price = train_data_price[['Date','SecuritiesCode','Open','Close','Target']]

lag_list_ = [1,2,3,5]

def get_lag_info(train_data,lag_list):
    for lag_length in lag_list:
        train_data[f'lag_{lag_length}_info'] = train_data.groupby(['SecuritiesCode'])['Close'].shift(lag_length)
        train_data[f'lag_{lag_length}_info'] = (train_data['Close']-train_data[f'lag_{lag_length}_info'])/train_data[f'lag_{lag_length}_info']
    train_data['target_ne'] = train_data.groupby(['SecuritiesCode'])['Target'].shift(-1)
    return train_data


train_data_price = get_lag_info(train_data_price,lag_list_)
train_data_price = train_data_price.drop(columns=['Open','Close'])


new_list = train_data_price[train_data_price['SecuritiesCode'] == 1301]
print(new_list)

        
