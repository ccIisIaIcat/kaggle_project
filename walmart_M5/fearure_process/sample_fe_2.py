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
# train_df = pd.read_csv('E:\\kaggle_project\\kaggle_data\\walmart\\sales_train_validation.csv')
prices_df = pd.read_csv('E:\\kaggle_project\\kaggle_data\\walmart\\sell_prices.csv')
calendar_df = pd.read_csv('E:\\kaggle_project\\kaggle_data\\walmart\\calendar.csv')

print('Prices')

#获取一些价格的统计数据
prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

#对统计的价格做正则化处理
prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']

#评估同一商店同一商品价格的波动（不同价格的个数）
#和同一商店同一价格的不同商品个数
prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

#作者意图做一些回测信息，用到了calender里的一些属性
calendar_prices = calendar_df[['wm_yr_wk','month','year']]#获取三列信息
calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])#去重
prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')#把月份和年的信息追加到prices_data后方
del calendar_prices

#之后就可以根据标签获得一些价格的动量信息
prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

#把不用的列删除
del prices_df['month'], prices_df['year']

#把新的信息追加在grid_data后面并保存
grid_df = pd.read_pickle('E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_1.pkl')

original_columns = list(grid_df)
grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
keep_columns = [col for col in list(grid_df) if col not in original_columns]
grid_df = grid_df[MAIN_INDEX+keep_columns]
grid_df = reduce_mem_usage(grid_df)

#保存为grid_part_2
grid_df.to_pickle('E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_2.pkl')
print('Size:', grid_df.shape)