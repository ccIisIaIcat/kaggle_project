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
id_list = np.linspace(start = 0, stop = len(data), num = 100)
end_position = int(len(data)/100)

r1_list = []
r2_list = []
r3_list = []
r4_list = []

for i in id_list:
    end_position += gap
    print(int(i),end_position)
    new_data = data[(int(data['RowId'])>int(i)) & (int(data['RowId'])<end_position)]
    r_list = get_ratio(new_data)
    r1_list.append(r_list[0])
    r2_list.append(r_list[1])
    r3_list.append(r_list[2])
    r4_list.append(r_list[3])

tool_list = np.linspace(start = 0, stop = 99, num = 100)
plt.plot(tool_list,r1_list)
plt.plot(tool_list,r2_list)
plt.plot(tool_list,r3_list)
plt.plot(tool_list,r4_list)
plt.show()


