#单纯基于股价信息做一些预测
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_ratio(data):
    r1 = data['Target'].corr(data['incress_ratio'])
    r2 = data['Target'].corr(data['highest_incress'])
    r3 = data['Target'].corr(data['lowest_reduce'])
    r4 = data['Target'].corr(data['change_ration'])
    return [r1,r2,r3,r4]

data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
data['incress_ratio'] = (data['Close']-data['Open'])/data['Open']
data['highest_incress'] = (data['High']-data['Open'])/data['Open']
data['lowest_reduce'] = (data['Open']-data['Low'])/data['Open']
data['change_ration'] = (data['High']-data['Low'])/data['Low']
data = data[['RowId', 'Date', 'SecuritiesCode','Target', 'incress_ratio', 'highest_incress', 'lowest_reduce','change_ration']]
gap = int(len(data)/100)
id_list = np.linspace(start = 0, stop = len(data)-gap*2, num = 400)
date_list = []
for i in id_list:
    date_list.append([data.loc[int(i)]['Date'],data.loc[int(i)+gap]['Date']])
r1_list = []
r2_list = []
r3_list = []
r4_list = []

for i in range(len(date_list)):
    new_data = data[(data['Date']>date_list[i][0]) & (data['Date']<date_list[i][1])]
    r_list = get_ratio(new_data)
    print(i)
    r1_list.append(r_list[0])
    r2_list.append(r_list[1])
    r3_list.append(r_list[2])
    r4_list.append(r_list[3])


# for i in id_list:
#     end_position += gap
#     new_data = data[(data.index.to_list()[0]>int(i)) & (data.index.to_list()[0]<end_position)]
#     print(new_data)
#     r_list = get_ratio(new_data)
#     r1_list.append(r_list[0])
#     r2_list.append(r_list[1])
#     r3_list.append(r_list[2])
#     r4_list.append(r_list[3])



tool_list = np.linspace(start = 0, stop = len(r1_list)-1, num = len(r1_list))
plt.plot(tool_list,r1_list)
plt.plot(tool_list,r2_list)
plt.plot(tool_list,r3_list)
plt.plot(tool_list,r4_list)
plt.show()


