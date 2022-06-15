#用于计算排名指标并进行保存
#步骤：1、获得清洗后的数据，提取日期序列，构造计算对象；2、根据构造的日期对象计算排名数据，并保存在csv中
#可以计算排名用于构造预测指标，也可以计算排名用于样本指标
#先计算作为样本指标的排序集合

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import preprocessing

PORTFOLIO_SIZE = 30 #指定排名组合计算的是多少天前的组合

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
train_data_price = train_data_price[['Date','SecuritiesCode','Target']]
train_data_price['Signal'] = train_data_price.groupby(['SecuritiesCode'])['Target'].transform('count')
train_data_price = train_data_price[train_data_price['Signal'] == 1202]
date_list = train_data_price['Date'].unique()

date_obj_list = []

for i in range(len(date_list)-PORTFOLIO_SIZE-1):
    obj_now = [date_list[i],date_list[i+PORTFOLIO_SIZE]]
    date_obj_list.append([obj_now,date_list[i+PORTFOLIO_SIZE+1]])

print(date_obj_list)
