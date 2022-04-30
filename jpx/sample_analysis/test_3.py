#用一些朴素特征和模型（树模型）对未来结果进行预测
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
gap = int(len(data)/100)
id_list = np.linspace(start = 0, stop = len(data)-gap*2, num = 400)
date_list = []
for i in id_list:
    date_list.append([data.loc[int(i)]['Date'],data.loc[int(i)+gap]['Date']])

data['incress_ratio'] = (data['Close']-data['Open'])/data['Open']
data['highest_incress'] = (data['High']-data['Open'])/data['Open']
data['lowest_reduce'] = (data['Open']-data['Low'])/data['Open']
data['change_ration'] = (data['High']-data['Low'])/data['Low']
data = data[['RowId', 'Date', 'SecuritiesCode','Target', 'incress_ratio', 'highest_incress', 'lowest_reduce','change_ration']]
X = data[[ 'incress_ratio', 'highest_incress', 'lowest_reduce','change_ration','Target']]
y = data[[ 'Target']]
print(list(data))
X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.4, random_state=0)

