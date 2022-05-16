import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import time
import datetime
import pickle
import gc
from tqdm import tqdm


n_fold = 7
seed0 = 8586
use_supple_for_train = True
not_use_overlap_to_train = False

TRAIN_CSV = 'kaggle_data/g_research/train.csv'
SUPPLE_TRAIN_CSV = 'kaggle_data/g_research/supplemental_train.csv'
ASSET_DETAILS_CSV = 'kaggle_data/g_research/asset_details.csv'

#设置展示窗口大小
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 350)

lags = [60,300,900] #暂时也不知道干嘛的

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

#之前也经常见到的用于减少内存使用的函数
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# 读取资产信息并按照资产ID排序
df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")

#读取所需的数据（注意到博主只用到了‘close’列的数据，应该是使用‘close’数据构造回滚特征
df_train = pd.read_csv(TRAIN_CSV, usecols=['timestamp','Asset_ID', 'Close', 'Target'])

if use_supple_for_train:    
    df_supple = pd.read_csv(SUPPLE_TRAIN_CSV, usecols=['timestamp','Asset_ID', 'Close', 'Target'])
#     display(df_supple)
    df_train = pd.concat([df_train, df_supple])
    del df_supple
df_train = reduce_mem_usage(df_train)
# df_train
