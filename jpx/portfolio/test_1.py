from distutils.log import info
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 投资组合权重
toprank_weight_ratio = 2
portfolio_size = 200
weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
# weights = np.insert(weights,0,0)
# print(weights)
lbl = preprocessing.LabelEncoder()

test_list = [1,1,1,1,1.2]
test_list_2 = [1,1,1,1,1.2]
test_list = np.array(test_list)
test_list_2 = np.array(test_list_2)

def cal_sharp_ratio(list_):
    return list_.mean()/list_.std()

def get_best_part(original_list,original_weight,new_weight,matrix,signal_list,positive_nagetive):
    max_sharp = -99999
    if positive_nagetive == 1:
        for i in range(len(matrix)):
            if signal_list[i] == 0:
                sharp_now = cal_sharp_ratio((original_list*original_weight+matrix[i]*new_weight)/(original_weight+new_weight))
                if sharp_now > max_sharp:
                    max_sharp = sharp_now
                    answer_id = i
    else:
        for i in range(len(matrix)):
            if signal_list[i] == 0:
                sharp_now = cal_sharp_ratio((original_list*original_weight-matrix[i]*new_weight)/(original_weight+new_weight))
                if sharp_now > max_sharp:
                    max_sharp = sharp_now
                    answer_id = i
    print(max_sharp)
    return answer_id
      
def get_the_best_portfolio(info_matrix):
    up_portfolio = []
    down_portfolio = []
    signal_list = np.zeros(len(info_matrix))
    id_list_up = np.zeros(200)
    id_list_down = np.zeros(200)
    temp_list = np.zeros(len(info_matrix[0]))
    weight_now = 0
    weight_now_down = 0
    for i in range(400):
        if i%2 == 0:
            id_now = i // 2
            max_id = get_best_part(temp_list,weight_now,weights[id_now],matrix=info_matrix,signal_list=signal_list,positive_nagetive=1)
            temp_list = (temp_list*weight_now+info_matrix[max_id]*weights[id_now])/(weight_now+weights[id_now])
            signal_list[max_id] = 1
            weight_now += weights[id_now]
            id_list_up[id_now] = max_id
            up_portfolio.append(info_matrix[max_id])
        else:
            id_now = i // 2
            max_id = get_best_part(temp_list,weight_now,weights[id_now],matrix=info_matrix,signal_list=signal_list,positive_nagetive=-1)
            temp_list = (temp_list*weight_now-info_matrix[max_id]*weights[id_now])/(weight_now+weights[id_now])
            signal_list[max_id] = -1
            weight_now_down += weights[id_now]
            id_list_down[id_now] = max_id
            down_portfolio.append(info_matrix[max_id])
    return id_list_up,id_list_down

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
train_data_price = train_data_price[['Date','SecuritiesCode','Target']]
security_list = train_data_price['SecuritiesCode'].unique()
date_list = train_data_price['Date'].unique()
train_data_price['Signal'] = train_data_price.groupby(['SecuritiesCode'])['Target'].transform('count')
train_data_price = train_data_price[train_data_price['Signal'] == 1202]
train_data_price = train_data_price[train_data_price['Date']>'2010-12-01']

test_matrix = []
new_info = pd.DataFrame(train_data_price.groupby(['SecuritiesCode'])['Target'])
new_info = list(new_info[1])
for obj in new_info:
    test_matrix.append(np.array(obj.values))
    print(len(np.array(obj.values)))
print(get_the_best_portfolio(test_matrix))



    
    





# train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
# train_data_price = train_data_price[['Date','SecuritiesCode','Open','Close','Target','AdjustmentFactor']]
# stock_info = pd.read_csv('kaggle_data/JPX/stock_list.csv')
# stock_info = stock_info[['SecuritiesCode','17SectorCode']]
# stock_info['17SectorCode'] = lbl.fit_transform(stock_info['17SectorCode'].astype(str))
# stock_info['17SectorCode'] = lbl.fit_transform(stock_info['17SectorCode'].astype(int))
# train_data_price = pd.merge(train_data_price,stock_info,how='left',on='SecuritiesCode')

