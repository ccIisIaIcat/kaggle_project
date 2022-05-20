import pandas as pd

new_data = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
new_data = new_data.drop(columns=['RowId','High','Low','Volume'])
print(list(new_data))
print(new_data['AdjustmentFactor'])
print(len(new_data[new_data['AdjustmentFactor']!= 1]))

print(new_data[(new_data['SecuritiesCode']== 6861) & (new_data['Date'] <= '2017-01-20')])