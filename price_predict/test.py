#学习使用lgbm的自定义函数
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
seed0 = 8586

train_data = pd.read_csv('kaggle_data/price_predict/TrainData.csv')

#提升树的参数设定
params = {
    'early_stopping_rounds': 50,
    'objective': 'regression',
    'metric': 'rmse',
#     'metric': 'None',
    'boosting_type': 'gbdt',
    'max_depth': 5,
    'verbose': -1,
    'max_bin':600,
    'min_data_in_leaf':50,
    'learning_rate': 0.03,
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


x_data = train_data[['BEDROOMS', 'BATHROOMS', 'GARAGE', 'LAND_AREA', 'FLOOR_AREA',  'CBD_DIST', 'NEAREST_STN_DIST',  'LATITUDE', 'LONGITUDE', 'NEAREST_SCH_DIST']]
y_data = train_data[['PRICE']]
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.3,random_state = 42)

train_dataset = lgb.Dataset(x_train,y_train)
val_dataset = lgb.Dataset(x_test,y_test,reference=train_dataset)

def judge_function(a,train_data):
    b = train_data.get_label()
    a = np.ravel(a)
    b = np.ravel(b)
    answer = abs((b-a)/b)
    answer = answer.mean()
    return 'diff',answer, True

def loglikelood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess

model = lgb.train(params = params,
                          train_set = train_dataset, 
                          valid_sets=[train_dataset, val_dataset],
                          valid_names=['tr', 'vl'],
                          num_boost_round = 5000,
                          verbose_eval = 100,   
                          feval = judge_function,
                         )

new_test_val = pd.read_csv('kaggle_data/price_predict/TestData.csv')
new_test_val = new_test_val[['BEDROOMS', 'BATHROOMS', 'GARAGE', 'LAND_AREA', 'FLOOR_AREA',  'CBD_DIST', 'NEAREST_STN_DIST',  'LATITUDE', 'LONGITUDE', 'NEAREST_SCH_DIST']]
new_test_judge = pd.read_csv('kaggle_data/price_predict/result.csv')
new_test_judge = new_test_judge[['PRICE']]
answer = model.predict(new_test_val)
new_test_judge['compare'] = answer
new_test_judge['diff'] = abs((new_test_judge['PRICE']-new_test_judge['compare'])/new_test_judge['PRICE'])
print(new_test_judge['diff'].mean())


