import pandas as pd
import numpy as np
from warnings import filterwarnings
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

df = pd.read_csv('kaggle_data\\JPX\\supplemental_files\\stock_prices.csv', parse_dates=["Date"])
df = df[['Date','SecuritiesCode','Target']]
data_day_1 = df[df['Date'] == '2021-12-06']
data_day_2 = df[df['Date'] == '2021-12-07']
data_day_3 = df[df['Date'] == '2021-12-08']
print(data_day_1,data_day_2,data_day_3)



