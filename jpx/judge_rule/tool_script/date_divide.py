import pandas as pd
import numpy as np

data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
gap = int(len(data)/100)
id_list = np.linspace(start = 0, stop = len(data)-gap*2, num = 400)
date_list = []
for i in id_list:
    date_list.append([data.loc[int(i)]['Date'],data.loc[int(i)+gap]['Date']])
    print(data.loc[int(i)]['Date'],data.loc[int(i)+gap]['Date'])

# #可以在这里获得新的data集()
# for i in range(len(date_list)):
#     new_data = data[(data['Date']>date_list[i][0]) & (data['Date']<date_list[i][1])]

