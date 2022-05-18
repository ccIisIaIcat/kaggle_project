import pandas as pd
import numpy as np

LAG_LIST = [1,2,3,5] #lag信息序列
K_FOLDS = 7 #测试个数

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
train_data_price = train_data_price[['Date','SecuritiesCode','Open','Close','Target']]



date_list = train_data_price.sort_values(by='Date',ascending=True)['Date'].unique()
print(date_list)
print(len(date_list))

def get_lag_info(train_data,lag_list):
    for lag_length in lag_list:
        train_data[f'lag_{lag_length}_info'] = train_data.groupby(['SecuritiesCode'])['Close'].shift(lag_length)
        train_data[f'lag_{lag_length}_info'] = (train_data['Close']-train_data[f'lag_{lag_length}_info'])/train_data[f'lag_{lag_length}_info']
    train_data['target_ne'] = train_data.groupby(['SecuritiesCode'])['Target'].shift(-1)
    train_data = train_data.drop(columns=['Open','Close'])
    train_data = train_data.dropna(axis=0,how='any')
    return train_data


# train_data_price = get_lag_info(train_data_price,lag_list_)

def get_date_divide(date_list,k_folds):

    return 0

# new_list = train_data_price[train_data_price['SecuritiesCode'] == 1301]
# print(new_list)

        
