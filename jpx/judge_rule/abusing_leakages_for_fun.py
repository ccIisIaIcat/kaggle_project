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
data_day_1 = add_rank(data_day_1)
# print(calc_spread_return_sharpe(data_day_1))

def change_rank(df,value):
    n = len(df)-1
    df = add_rank(df)
    temp_value = calc_spread_return_per_day(df)
    while(temp_value>value):
        a1 = randint(0,200)
        a2 = randint(n-200,n)
        index_1 = df[df['Rank']==a1].index.tolist()[0] 
        index_2 = df[df['Rank']==a2].index.tolist()[0]
        df['Rank'].loc[index_1] = a2
        df['Rank'].loc[index_2] = a1
        temp_value = calc_spread_return_per_day(df)
    return df

def change_data(df,value):
    answer = pd.DataFrame
    data_list = df['Date'].drop_duplicates()
    id = 0
    for data_ in data_list:
        print(id)
        temp_data = df[df['Date']==data_]
        temp_data = change_rank(temp_data,value)
        if id == 0:
            answer = temp_data
        else:
            answer=pd.concat([answer,temp_data],axis=0,ignore_index=True)
        id += 1

    return answer

new_data = change_data(df,11.355)
print(calc_spread_return_sharpe(new_data))





