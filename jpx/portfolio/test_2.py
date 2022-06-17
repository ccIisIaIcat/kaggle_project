#用于计算排名指标并进行保存
#步骤：1、获得清洗后的数据，提取日期序列，构造计算对象；2、根据构造的日期对象计算排名数据，并保存在csv中
#可以计算排名用于构造预测指标，也可以计算排名用于样本指标
#先计算作为样本指标的排序集合

from itertools import count
from typing import final
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import preprocessing

PORTFOLIO_SIZE = 30 #指定排名组合计算的是多少天前的组合

# 投资组合权重
toprank_weight_ratio = 2
portfolio_size = 200
weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
lbl = preprocessing.LabelEncoder()

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
train_data_price = train_data_price[['Date','SecuritiesCode','Target']]
train_data_price['Signal'] = train_data_price.groupby(['SecuritiesCode'])['Target'].transform('count')
train_data_price = train_data_price[train_data_price['Signal'] == 1202]
date_list = train_data_price['Date'].unique()

date_obj_list = []

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
    return answer_id
      
def get_the_best_portfolio(info_matrix,scode_list):
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
    up_list = []
    down_list = []
    for num_1 in id_list_up:
        up_list.append(scode_list[int(num_1)])
    for num_2 in id_list_down:
        down_list.append(scode_list[int(num_2)])
    return up_list,down_list

for i in range(len(date_list)-PORTFOLIO_SIZE-2):
    obj_now = [date_list[i],date_list[i+PORTFOLIO_SIZE]]
    date_obj_list.append([obj_now,date_list[i+PORTFOLIO_SIZE+2]])

up_num_list = []
down_num_list = []
for i in range(200):
    up_num_list.append(200-i)
    down_num_list.append(i-200)

train_data_price = train_data_price[['Date','SecuritiesCode','Target']]
n_sum = len(date_obj_list)
counter = 0
final_df = pd.DataFrame(columns=['SecuritiesCode',f'lag_rank_{PORTFOLIO_SIZE}','Date'])
print(date_obj_list)

for date_obj in date_obj_list:
    counter += 1
    print(date_obj,"percent:",counter/n_sum,"   counter:",counter)
    new_data = train_data_price[(train_data_price['Date']>=date_obj[0][0]) | (train_data_price['Date']<=date_obj[0][1])]
    test_matrix = []
    new_info = pd.DataFrame(new_data.groupby(['SecuritiesCode'])['Target'])
    scode_list = list(new_info[0].values)
    new_info = list(new_info[1])
    for obj in new_info:
        test_matrix.append(np.array(obj.values))
    a_list,b_list = get_the_best_portfolio(test_matrix,scode_list)
    temp_pd = pd.DataFrame()
    temp_pd['SecuritiesCode'] = (a_list+b_list)
    temp_pd[f'lag_rank_{PORTFOLIO_SIZE}'] = (up_num_list+down_num_list)
    temp_pd['Date'] = date_obj[1]
    final_df = pd.concat([final_df,temp_pd],ignore_index=True)
    if counter % 50 == 0:
        final_df.to_csv('kaggle_data/JPX/tool_data/rank_info_5.csv')
    
