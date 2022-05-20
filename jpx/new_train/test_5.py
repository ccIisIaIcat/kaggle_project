import pandas as pd
import numpy as np

new_data = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
new_data = new_data[['Date','SecuritiesCode','AdjustmentFactor']]
print(list(new_data))

def get_adjustment(list_,lag_length):
    list_ = np.array(list_)
    base_point = 1
    answer_list = []
    for i in range(lag_length):
        base_point = base_point*list_[i]
        answer_list.append(base_point)
    for i in range(len(list_)-lag_length):
        base_point = base_point*list_[i+lag_length]/list_[i]
        answer_list.append(base_point)
    return answer_list

new_data['adj'] = new_data.groupby(['SecuritiesCode'])['AdjustmentFactor'].transform(lambda ls_:get_adjustment(ls_,100))

print(new_data[(new_data['SecuritiesCode'] == 6861) & (new_data['Date']<='2017-01-20')])




    