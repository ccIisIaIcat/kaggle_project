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

#把训练集矩阵转为id-日期-销量值三列
#其中id索引列为'id','item_id','dept_id','cat_id','store_id','state_id'
print('Create Grid')
index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
grid_df = pd.melt(train_df, 
                  id_vars = index_columns, 
                  var_name = 'd', 
                  value_name = TARGET)

#相比原数据集获得了更多条数据
print('Train rows:', len(train_df), len(grid_df))

#为了后续方便预测，创建预测表格
add_grid = pd.DataFrame()
for i in range(1,29):
    temp_df = train_df[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['d'] = 'd_'+ str(END_TRAIN+i)
    temp_df[TARGET] = np.nan
    add_grid = pd.concat([add_grid,temp_df])

#把新生成的表格追加到grid后面
grid_df = pd.concat([grid_df,add_grid])
grid_df = grid_df.reset_index(drop=True)

#把使用过的文件从内存中删除
del temp_df, add_grid

#train_df也可以删除
del train_df

#查看grid的内存使用情况
print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

#把一些string类型转为category类型，可以节省一部分内存且不会造成有用信息的损失
for col in index_columns:
    grid_df[col] = grid_df[col].astype('category')

#再次查看内存使用
print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

#下面的操作是为了删除销量为0的列，其中这些列的wm_yr_wk列为零
#把某一商品出现的第一次沃尔玛周作为新的一列
release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id','item_id','release']

#把该价格合并在grid_df表中
grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
del release_df

#下面我们去删除一些零值列，这项操作需要wm_yr_wk信息
#所以把该列合并到grid中
grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])

#把零值列删除
grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
grid_df = grid_df.reset_index(drop=True)

print(">>>>>>>>>>>>>>..")
#查看一下grid的内存占用
print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

#把新的release列和最小值做差，并保存为int16，进一步节约内存
grid_df['release'] = grid_df['release'] - grid_df['release'].min()
grid_df['release'] = grid_df['release'].astype(np.int16)

#查看一下新的grid的内存占用
print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

#把新的信息存起来方便后续使用
print('Save Part 1')
grid_df.to_pickle('kaggle_data\\walmart\\tool_data\\grid_part_1.pkl')

print('Size:', grid_df.shape)
