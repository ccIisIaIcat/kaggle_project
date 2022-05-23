import numpy as np
import pandas as pd
import math
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

#初始化随机种子
def seed_everything(seed=2021):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

SEED=2021
seed_everything(SEED)

train = pd.read_csv("kaggle_data/JPX/train_files/stock_prices.csv",parse_dates=["Date"])
train=train.drop(columns=['RowId','ExpectedDividend','AdjustmentFactor','SupervisionFlag']).dropna().reset_index(drop=True)

# 一些常见的工具函数
def calc_spread_return_per_day(df, portfolio_size=200, toprank_weight_ratio=2):
    assert df['Rank'].min() == 0
    assert df['Rank'].max() == len(df['Rank']) - 1
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
    purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
    short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
    return purchase - short

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size=200, toprank_weight_ratio=2):
    buf = df.groupby('Date').apply(calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio#, buf

def add_rank(df):
    df["Rank"] = df.groupby("Date")["Target"].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

def fill_nan_inf(df):
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df

# 用于计算剔除了规定list之内的股票之后
def check_score(df,preds,Securities_filter=[]):
    tmp_preds=df[['Date','SecuritiesCode']].copy()
    tmp_preds['Target']=preds
    
    #Rank Filter. Calculate median for this date and assign this value to the list of Securities to filter.
    tmp_preds['target_mean']=tmp_preds.groupby("Date")["Target"].transform('median')
    # 对于处在securities_list之中的股票，把他们的target改为当日股票target的中位数
    tmp_preds.loc[tmp_preds['SecuritiesCode'].isin(Securities_filter),'Target']=tmp_preds['target_mean']
    
    tmp_preds = add_rank(tmp_preds)
    df['Rank']=tmp_preds['Rank']
    score=round(calc_spread_return_sharpe(df, portfolio_size= 200, toprank_weight_ratio= 2),5)
    score_mean=round(df.groupby('Date').apply(calc_spread_return_per_day, 200, 2).mean(),5)
    score_std=round(df.groupby('Date').apply(calc_spread_return_per_day, 200, 2).std(),5)
    print(f'Competition_Score:{score}, rank_score_mean:{score_mean}, rank_score_std:{score_std}')