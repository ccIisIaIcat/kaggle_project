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

# 查看数据大小
# print('Shape', train_df.shape)

#一个常规的三维列构造
index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
train_df = pd.melt(train_df, 
                  id_vars = index_columns, 
                  var_name = 'd', 
                  value_name = TARGET)

#为了构造回滚列时计算方便，这里只取需要的三列
temp_df = train_df[['id','d',TARGET]]

#获取前八天的回滚数据
LAG_DAYS = [col for col in range(1,8)]
temp_df = train_df[['id','d',TARGET]]

temp_df = train_df[['id','d',TARGET]]

start_time = time.time()
for i in range(1,8):
    print('Shifting:', i)
    temp_df['lag_'+str(i)] = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(i))
    
print('%0.2f min: Time for loops' % ((time.time() - start_time) / 60))

start_time = time.time()
temp_df = temp_df.assign(**{'{}_lag_{}'.format(col, l): temp_df.groupby(['id'])[col].transform(lambda x: x.shift(l))for l in LAG_DAYS for col in [TARGET]})
print('%0.2f min: Time for bulk shift' % ((time.time() - start_time) / 60))

#均值的回滚数据
temp_df = train_df[['id','d','sales']]
start_time = time.time()
for i in [14,30,60]:
    print('Rolling period:', i)
    temp_df['rolling_mean_'+str(i)] = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).mean())
    temp_df['rolling_std_'+str(i)]  = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).std())
print('%0.2f min: Time for loop' % ((time.time() - start_time) / 60))

print(temp_df.head())
temp_df = temp_df.iloc[:,3:]
print(temp_df.head())
print(list(temp_df))

from scipy import sparse 
temp_matrix = sparse.csr_matrix(temp_df)
print(temp_matrix)

temp_matrix_restored = pd.DataFrame(temp_matrix.todense())
restored_cols = ['roll_' + str(i) for i in list(temp_matrix_restored)]
temp_matrix_restored.columns = restored_cols

del temp_df, train_df, temp_matrix, temp_matrix_restored

grid_df = pd.read_pickle('kaggle_data\\walmart\\tool_data\\grid_part_1.pkl')

grid_df = grid_df[['id','d','sales']]
SHIFT_DAY = 28

start_time = time.time()
print('Create lags')

LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+15)]
grid_df = grid_df.assign(**{
        '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
        for l in LAG_DAYS
        for col in [TARGET]
    })

# Minify lag columns
for col in list(grid_df):
    if 'lag' in col:
        grid_df[col] = grid_df[col].astype(np.float16)

print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

# Rollings
# with 28 day shift
start_time = time.time()
print('Create rolling aggs')

for i in [7,14,30,60,180]:
    print('Rolling period:', i)
    grid_df['rolling_mean_'+str(i)] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)
    grid_df['rolling_std_'+str(i)]  = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)

# Rollings
# with sliding shift
for d_shift in [1,7,14]: 
    print('Shifting period:', d_shift)
    for d_window in [7,14,30,60]:
        col_name = 'rolling_mean_tmp_'+str(d_shift)+'_'+str(d_window)
        grid_df[col_name] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)
    
    
print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

print('Save lags and rollings')
grid_df.to_pickle('kaggle_data\\walmart\\tool_data\\lags_df_'+str(SHIFT_DAY)+'.pkl')

print(grid_df.info())