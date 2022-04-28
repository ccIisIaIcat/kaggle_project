from hashlib import new
import score_judge
import pandas as pd


# sample_submission = pd.read_csv("kaggle_data\\JPX\\example_test_files\\tool_sample.csv")
#生成一个随机样本表，来观察计算结果
new_data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
new_data = new_data[['Date','SecuritiesCode','Target']]
new_data['Rank'] = new_data.groupby(["Date"])['SecuritiesCode'].rank(ascending=True,method='first').astype(int)-1



print(score_judge.calc_spread_return_sharpe(new_data))







