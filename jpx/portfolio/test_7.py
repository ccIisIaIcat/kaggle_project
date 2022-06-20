#用于检验旧的投资组合到底有没有用
import pandas as pd
import numpy as np

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

def add_rank(df):
    df["Rank"] = df.groupby("Date")["lag_rank_7"].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

info_data = pd.read_csv('kaggle_data/JPX/tool_data/test_rank_lag_7_7.csv')
info_data = info_data[['SecuritiesCode','lag_rank_7','Date']]
train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
train_data_price = train_data_price[['Date','SecuritiesCode','Target']]
train_data_price = pd.merge(info_data,train_data_price,on=['Date','SecuritiesCode'],how='left')
add_rank(train_data_price)
print(calc_spread_return_sharpe(train_data_price)[0])