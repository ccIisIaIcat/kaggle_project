import pandas as pd
import numpy as np
from warnings import filterwarnings
from random import randint
filterwarnings("ignore")

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
    df["Rank"] = df.groupby("Date")["Target"].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

df = pd.read_csv('kaggle_data\\JPX\\supplemental_files\\stock_prices.csv', parse_dates=["Date"])
df = df[['Date','SecuritiesCode','Target']]
data_day_1 = df[(df['Date'] == '2021-12-06')].reset_index(drop=True)
# data_day_1 = add_rank(df)
# print(calc_spread_return_sharpe(data_day_1))

def change_rank(df,value):
    n = len(df)-1
    temp_value = calc_spread_return_per_day(df)
    while(temp_value>value):
        a1 = randint(0,n)
        a2 = randint(0,n)
        df[df['Rank']==a1]['Rank'] = a2
        df[df['Rank']==a2]['Rank'] = a1
        temp_value = calc_spread_return_per_day(df)
        print(temp_value)
    return df

change_rank




