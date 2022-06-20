import pandas as pd
import numpy as np

def add_rank(df):
    df["Rank"] = df.groupby("Date")['lag_rank_30'].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

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
    return sharpe_ratio, buf

data = pd.read_csv('kaggle_data/JPX/tool_data/rank_info_3.csv')
data = data[data['Date']=='2017-02-21']
data = data[['SecuritiesCode','lag_rank_30']]
test_info = pd.read_csv('kaggle_data/JPX/supplemental_files/stock_prices.csv')
test_info = test_info[['Date','SecuritiesCode','Target']]
test_info = pd.merge(test_info,data,on=['SecuritiesCode'],how='left')
test_info['lag_rank_30'] = test_info['lag_rank_30'].fillna(0)
print(test_info)
add_rank(test_info)
print(calc_spread_return_sharpe(test_info)[0])

#没有用。。。放弃挣扎了