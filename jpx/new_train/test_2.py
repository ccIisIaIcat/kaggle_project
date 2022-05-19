import pandas as pd
import numpy as np

LAG_LIST = [1,2,3,5,7,10,15] #lag信息序列
K_FOLDS = 7 #测试个数
OVERLAP_RATIO = 0.3 #cv测试集覆盖比率
TEST_DATA_RATIO = 0.3 #每组中测试集所占比例

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
train_data_price = train_data_price[['Date','SecuritiesCode','Open','Close','Target']]

def get_date_divide(date_list,k_folds, overlap_ratio,test_data_ratio):
    # (k*0.3+0.7)*l = L => l = L/(k*0.3+0.7)
    single_length = int(len(date_list)/(k_folds*overlap_ratio+1-overlap_ratio))
    gap_size = int(single_length*overlap_ratio)
    date_set = []
    for i in range(k_folds):
        start_date = date_list[i*gap_size]
        if i*gap_size+single_length < len(date_list):
            end_date = date_list[i*gap_size+single_length]
        else :
            end_date = date_list[len(date_list)-1]
        mid_date = date_list[i*gap_size+int(single_length*(1-test_data_ratio))]
        date_set.append([start_date,mid_date,end_date])
    return date_set

def get_lag_info(train_data,lag_list):
    for lag_length in lag_list:
        train_data[f'lag_{lag_length}_info'] = train_data.groupby(['SecuritiesCode'])['Close'].shift(lag_length)
        train_data[f'lag_{lag_length}_info'] = (train_data['Close']-train_data[f'lag_{lag_length}_info'])/train_data[f'lag_{lag_length}_info']
    train_data['target_ne'] = train_data.groupby(['SecuritiesCode'])['Target'].shift(-1)
    train_data = train_data.drop(columns=['Open','Close'])
    train_data = train_data.dropna(axis=0,how='any')
    return train_data


train_data_price = get_lag_info(train_data_price,LAG_LIST)
date_list = train_data_price.sort_values(by='Date',ascending=True)['Date'].unique()
print(list(train_data_price))
date_list = get_date_divide(date_list,K_FOLDS,OVERLAP_RATIO,TEST_DATA_RATIO)

for time_set in date_list:
    train_data = train_data_price[(train_data_price['Date']>time_set[0]) & (train_data_price['Date']<time_set[1])]
    test_data = train_data_price[(train_data_price['Date']>time_set[1]) & (train_data_price['Date']<time_set[2])]
    x_train = train_data.drop(columns=['Date','SecuritiesCode','Target'])
    y_train = train_data['Target']
    x_test = test_data.drop(columns=['Date','SecuritiesCode','Target'])
    y_test = test_data['Target']
    print(train_data)
    print(time_set)



# new_list = train_data_price[train_data_price['SecuritiesCode'] == 1301]
# print(new_list)

        
