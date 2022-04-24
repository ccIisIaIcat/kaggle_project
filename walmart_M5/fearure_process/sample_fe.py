#一些基本的导入
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
from math import ceil
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

#编写一些查看内存使用情况的函数
#查看当前进程所占用内存
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

#对大小的格式化输出
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


#对dataframe进行压缩处理
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# 新的连接方式防止数据类型丢失
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on] #选取作为连接键的df1中的列
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')#根据这些列左连接df2
    new_columns = [col for col in list(merged_gf) if col not in merge_on]#去除新连接表中不是连接键的列
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)#把这些新的键的列和df1合并
    return df1

#声明一些全局变量
TARGET = 'sales'         # 主要的目标值
END_TRAIN = 1913         # 训练集最后一天的id
MAIN_INDEX = ['id','d']  # 确定数据的连接键

#加载数据
print('Load Main Data')

#各项数据加载
train_df = pd.read_csv('E:\\kaggle_project\\kaggle_data\\walmart\\sales_train_validation.csv')
prices_df = pd.read_csv('E:\\kaggle_project\\kaggle_data\\walmart\\sell_prices.csv')
calendar_df = pd.read_csv('E:\\kaggle_project\\kaggle_data\\walmart\\calendar.csv')
