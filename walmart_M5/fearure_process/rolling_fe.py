#一些对回滚特征变量的构造
#基本包的导入
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

import time

warnings.filterwarnings('ignore')

#查看内存的工具函数
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

#一些常量定义
TARGET = 'sales'         # Our main target
END_TRAIN = 1913         # Last day in train set
MAIN_INDEX = ['id','d']  # We can identify item by these columns

#导入数据包
train_df = pd.read_csv('kaggle_data\\walmart\\sales_train_validation.csv')

#这里只读取加州的数据
train_df = train_df[train_df['state_id']=='CA']

#查看数据大小
print('Shape', train_df.shape)

#一个常规的三维列构造
index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
train_df = pd.melt(train_df, 
                  id_vars = index_columns, 
                  var_name = 'd', 
                  value_name = TARGET)
