import pandas as pd
import numpy as np

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')

print(list(train_data_price))

train_data_price['log_change'] = np.log(train_data_price['Close'])-np.log(train_data_price['Open'])
train_data_price = train_data_price[['Date','SecuritiesCode','log_change','Target']]

lag_list_ = [1,2,3,4]

def get_lag_info(train_data,lag_list):
    for lag_length in lag_list:
        print(lag_length)
        train_data[f'lag_{lag_length}_lochg'] = train_data.groupby(['SecuritiesCode'])['log_change'].transform(lambda x: x)
        for i in range(lag_length-1):
            print("i:",i)
            train_data[f'lag_{lag_length}_lochg'] = train_data.groupby(['SecuritiesCode'])[f'lag_{lag_length}_lochg'].transform(lambda x: x+x.shift(i+1))
    return train_data

train_data_price = get_lag_info(train_data_price,lag_list_)


new_list = train_data_price[train_data_price['SecuritiesCode'] == 1301]
print(new_list)

        
