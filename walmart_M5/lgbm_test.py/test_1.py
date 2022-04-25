#基本包的导入
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool       #多线程包导入
import lightgbm as lgb #训练模型导入

warnings.filterwarnings('ignore')

#计算机参数
N_CORES = psutil.cpu_count()
#路径参数
BASE = "E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_1.pkl"
PRICE = "E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_2.pkl"
CALENDAR = "E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_3.pkl"

#随机种子生成函数
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

#多线程函数
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

#数据读取
def get_data_by_store(store):
    
    # 读取并连接基本特征
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    df = df[df['store_id']==store] #只保留和商店id相关的特征
