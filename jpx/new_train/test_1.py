import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def cal_p(lis_,k,m1,m2,level):
    sum = 0
    sub = 0
    for i in range(len(lis_)):
        j = i
        while(j<len(lis_)  and lis_[j]>m1 and lis_[j]<m2 and (j-i)<=k-1):
            j += 1
        if j-i == k:
            sum += 1
            if (j < len(lis_) and lis_[j] > level):
                sub += 1
    print(k,m1,m2,sub,sum)
    if sum < 5:
        return -1
    else:
        return sub/sum


price_data = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
print(list(price_data))

price_data['change'] = (price_data['Close']-price_data['Open'])/price_data['Open']
price_data = price_data[['Date', 'SecuritiesCode','change']]
SecuritiesCode = price_data['SecuritiesCode'].unique()

new_list = price_data[price_data['SecuritiesCode'] == 1381]
new_list = new_list['change'].reset_index(drop=True)
k_list = [1,2,3,4,5,6]
LEVEL = 0

def get_matrix(s_list):
    max_num = s_list.max()
    min_num = s_list.min()
    gap = (max_num-min_num)/13
    m_list = []
    for i in range(12):
        m_list.append([min_num+i*gap,min_num+i*gap+gap])
    answer = []
    for k_ in k_list:
        answer.append([])
    for k_ in range(len(k_list)):
        for m_ in m_list:
            answer[k_].append(cal_p(new_list,k_list[k_],m_[0],m_[1],LEVEL))
    return answer

ax = sns.heatmap(get_matrix(new_list))

plt.show()

