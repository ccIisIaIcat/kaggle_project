from cmath import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import v
#统一几个指标：1、收益率；2、相关系数矩阵；
#计算最优夏普率：方法一：根基收益率排序；
#方法二：（先做一些统计学实验）


train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
date_list_2 = pd.DataFrame()
security_list = pd.DataFrame()
date_list = train_data_price['Date'].unique()
date_list_2['Date'] = train_data_price['Date'].unique()
security_list['SecuritiesCode'] = train_data_price['SecuritiesCode'].unique()
date_list_2['tool'] = 1
security_list['tool'] = 1
new_data_frame = pd.merge(date_list_2,security_list,on='tool',how='left')
train_data_price = pd.merge(new_data_frame,train_data_price,on=['SecuritiesCode','Date'],how='left')
train_data_price = train_data_price[['Date','SecuritiesCode','Target']]

train_data_price = train_data_price[['Date','SecuritiesCode','Target']]
train_data_price = train_data_price[(train_data_price['Date']>='2020-06-01') & (train_data_price['Date']<='2020-09-01')]


test_matrix = []
train_data_price['Target'] = train_data_price.groupby(['SecuritiesCode'])['Target'].apply(lambda x: x.fillna(x.mean()))
train_data_price.dropna(axis=0,inplace=True)
new_info = pd.DataFrame(train_data_price.groupby(['SecuritiesCode'])['Target'])
scode_list = list(new_info[0].values)
new_info = list(new_info[1])
for obj in new_info:
    test_matrix.append(np.array(obj.values))
print(test_matrix[3])


#定义一个'和相关系数'
def combine_ratio(list_1,list_2):
    temp_list = list_1+list_2
    temp_list = temp_list-temp_list.mean()
    scale = np.inner(temp_list,temp_list)
    if scale != 0:
        temp_list = temp_list/scale
    return ((list_1+list_2)).std()

def combine_ratio_matrix(goal_matrix):
    answer_matrix = []
    for i in range(len(goal_matrix)):
        print(i/len(goal_matrix))
        temp_list = []
        for j in range(len(goal_matrix)):
            temp_list.append(combine_ratio(goal_matrix[i],goal_matrix[j]))
        answer_matrix.append(temp_list)
    return answer_matrix

def combine_ratio_weight(list_1,list_2,weight_1,weight_2):
    return ((list_1*weight_1+list_2*weight_2)/(weight_1+weight_2)).std()

# 下面我们来构造投资组合
# 基本思路：1、选择收益率最高的点，以该点为基准，寻找（收益率-和相关系数）最大的点一起放入uppackage
# 2、执行完1之后，找收益率最低的点，以该点为基准，寻找（-收益率-和相关系数）最大的点一起放入downpackage
# 3、反复执行1、2直到 uppackage 和 downpackage 的数目都达到200
# 一点改进：在和相关系数前加一个系数k，考察k取多少的时候，统计意义上一对儿组合的sharp ratio最大
# 可以先进行第二步工作
# 先看看分布的大体量纲(根据计算结果k取100左右比较合适，但有可能是由收益中的极端值引起的)



def get_best_friend(value_list,sum_corr_list,k,self_id):
    max = -9999999
    max_id = -1
    for i in range(len(value_list)):
        temp_value = value_list[i]-sum_corr_list[i]*k
        if temp_value>max and i!=self_id:
            max = temp_value
            max_id = i
    return max_id

def get_best_friend(value_list,sum_corr_list,k,self_id):
    max = -9999999
    max_id = -1
    for i in range(len(value_list)):
        temp_value = value_list[i]-sum_corr_list[i]*k
        if temp_value>max and i!=self_id:
            max = temp_value
            max_id = i
    return max_id

def get_worst_friend(value_list,sum_corr_list,k,self_id):
    max = -9999999
    max_id = -1
    for i in range(len(value_list)):
        temp_value = -value_list[i]-sum_corr_list[i]*k
        if temp_value>max and i!=self_id:
            max = temp_value
            max_id = i
    return max_id


# combine_matrix = combine_ratio_matrix(test_matrix)
# earning_list = np.sum(test_matrix,axis=1)

# id_this = 0
# for id_this in [0,1,2,3]:
#     k_list = []
#     for j in range(600):
#         k_list.append(70+j*0.01)
#     new_sharp_set = []
#     tool_set = []
#     i = 0
#     for k_ in k_list:
#         i+=1
#         tool_set.append(i)
#         temp_friend_id = get_best_friend(earning_list,combine_matrix[id_this],k_,0)
#         print(temp_friend_id)
#         print(earning_list[temp_friend_id],combine_matrix[id_this][temp_friend_id])
#         new_list = (test_matrix[temp_friend_id]+test_matrix[id_this])/2
#         new_sharp_set.append(new_list.mean()/new_list.std())
#     print(earning_list)
#     print(combine_matrix[id_this])
#     plt.plot(tool_set,new_sharp_set)
#     plt.show()
# print(combine_matrix[3])




    


