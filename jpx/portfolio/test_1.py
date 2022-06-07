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

def combine_sharp_ratio(list_1,list_2,weight_1,weight_2):
    return cal_sharp_ratio((list_1*weight_1+list_2*weight_2)/(weight_1+weight_2))

def combine_sharp_ratio_2(list_1,list_2,weight_1,weight_2):
    return cal_sharp_ratio((list_1*weight_1-list_2*weight_2)/(weight_1+weight_2))

def get_best_part(original_list,original_weight,new_weight,matrix,signal_list):
    max_sharp = -99999
    for i in range(len(matrix)):
        if signal_list[i] == 0:
            sharp_now = combine_sharp_ratio(original_list,matrix[i],original_weight,new_weight)
            if sharp_now > max_sharp:
                max_sharp = sharp_now
                answer_id = i
    return answer_id

def get_best_part_2(original_list,original_weight,new_weight,matrix,signal_list):
    max_sharp = -99999
    for i in range(len(matrix)):
        if signal_list[i] == 0:
            sharp_now = combine_sharp_ratio_2(original_list,matrix[i],original_weight,new_weight)
            if sharp_now > max_sharp:
                max_sharp = sharp_now
                answer_id = i
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
            max_id = get_best_part(temp_list,weight_now,weights[id_now],matrix=info_matrix,signal_list=signal_list)
            signal_list[max_id] = 1
            weight_now += weights[id_now]
            id_list_up[id_now] = max_id
            up_portfolio.append(info_matrix[max_id])
        else:
            id_now = i // 2
            max_id = get_best_part_2(temp_list,weight_now,weights[id_now],matrix=info_matrix,signal_list=signal_list)
            signal_list[max_id] = -1
            weight_now_down += weights[id_now]
            id_list_down[i] = max_id
            down_portfolio.append(info_matrix[max_id])
    
    





# train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
# train_data_price = train_data_price[['Date','SecuritiesCode','Open','Close','Target','AdjustmentFactor']]
# stock_info = pd.read_csv('kaggle_data/JPX/stock_list.csv')
# stock_info = stock_info[['SecuritiesCode','17SectorCode']]
# stock_info['17SectorCode'] = lbl.fit_transform(stock_info['17SectorCode'].astype(str))
# stock_info['17SectorCode'] = lbl.fit_transform(stock_info['17SectorCode'].astype(int))
# train_data_price = pd.merge(train_data_price,stock_info,how='left',on='SecuritiesCode')

