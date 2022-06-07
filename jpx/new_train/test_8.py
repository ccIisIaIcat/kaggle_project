import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import preprocessing

LAG_LIST = [1,3,7,30] #lag信息序列
K_FOLDS = 7 #测试集个数
OVERLAP_RATIO = 0.05 #cv测试集覆盖比率
TEST_DATA_RATIO = 0.2 #每组中测试集所占比例
TEST_PERIOD = 15 #选择用之前多少天的投资组合评分作为target
seed0 = 8586 #随机种子
#提升树的参数设定
params = {
    'early_stopping_rounds': 100,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': 5,
    'verbose': -1,
    'max_bin':600,
    'min_data_in_leaf':50,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'subsample_freq': 1,
    'feature_fraction': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 2,
    'seed':seed0,
    'feature_fraction_seed': seed0,
    'bagging_fraction_seed': seed0,
    'drop_seed': seed0,
    'data_random_seed': seed0,
    'extra_trees': True,
    'extra_seed': seed0,
    'zero_as_missing': True,
    "first_metric_only": True
    }

lbl = preprocessing.LabelEncoder()

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
train_data_price = train_data_price[['Date','SecuritiesCode','Open','Close','Target','AdjustmentFactor']]
stock_info = pd.read_csv('kaggle_data/JPX/stock_list.csv')
stock_info = stock_info[['SecuritiesCode','17SectorCode']]
stock_info['17SectorCode'] = lbl.fit_transform(stock_info['17SectorCode'].astype(str))
stock_info['17SectorCode'] = lbl.fit_transform(stock_info['17SectorCode'].astype(int))
train_data_price = pd.merge(train_data_price,stock_info,how='left',on='SecuritiesCode')