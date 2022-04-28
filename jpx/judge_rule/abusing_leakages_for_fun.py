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

def add_rank(df):
    df["Rank"] = df.groupby("Date")["Target"].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

def adjuster(ff, step=1, offset=95, cap=11.4):
    org_score = calc_spread_return_per_day(ff)
    if cap >= org_score: return ff.Rank.values
    for i in range(0, 2000, step):
        f, l = ff.index[i], ff.index[i+offset]
        ff.loc[f, "Rank"], ff.loc[l, "Rank"] = ff.loc[l, "Rank"], ff.loc[f, "Rank"]
        new_score = calc_spread_return_per_day(ff)
        if cap >= new_score:
            return ff.Rank.values
df = pd.read_csv('kaggle_data\\JPX\\example_test_files\\stock_prices.csv', parse_dates=["Date"])
df = add_rank(df)
df = df.sort_values(["Date", "Rank"])

for date in df.Date.unique():
    df.loc[df.Date==date, "Rank"] = adjuster(df[df.Date==date])

_, buf = calc_spread_return_sharpe(df)
print(buf.mean(), buf.std(), buf.mean() / buf.std(), buf.min(), buf.max())