from turtle import up
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
new_info = pd.DataFrame(train_data_price.groupby(['SecuritiesCode'])['Target'])
new_info.dropna(axis=1,inplace=True)
scode_list = list(new_info[0].values)
new_info = list(new_info[1])
for obj in new_info:
    test_matrix.append(np.array(obj.values))

def add_rank(df):
    df["Rank"] = df.groupby("Date")['lag_rank'].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

def add_rank_2(df):
    df["Rank"] = df.groupby("Date")['Target'].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

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

def get_best_friend(value_list,sum_corr_list,k,self_id,signal_list):
    max = -9999999
    max_id = -1
    for i in range(len(value_list)):
        temp_value = value_list[i]-sum_corr_list[i]*k
        if temp_value>max and i!=self_id and signal_list[i]==0:
            max = temp_value
            max_id = i
    return max_id

def get_worst_friend(value_list,sum_corr_list,k,self_id,signal_list):
    max = -9999999
    max_id = -1
    for i in range(len(value_list)):
        temp_value = -value_list[i]-sum_corr_list[i]*k
        if temp_value>max and i!=self_id and signal_list[i]==0:
            max = temp_value
            max_id = i
    return max_id

def get_best_id(value_list,signal_list):
    max = -99999
    best_id = -1
    for i in range(len(value_list)):
        if value_list[i]>max and signal_list[i]==0:
            max = value_list[i]
            best_id = i
    return best_id

def get_worst_id(value_list,signal_list):
    min = 99999
    worst_id = -1
    for i in range(len(value_list)):
        if value_list[i]<min and signal_list[i]==0:
            min = value_list[i]
            worst_id = i
    return worst_id


def get_the_best_portfolio(value_list,combine_matrix,scode_list,portfolio_size,k_value):
    signal_list = np.zeros(len(value_list))
    id_list_up = np.zeros(portfolio_size)
    id_list_down = np.zeros(portfolio_size)
    for i in range(portfolio_size*2):
        print(i)
        if i%2 == 0:
            id_now = i // 2
            best_id = get_best_id(value_list=value_list,signal_list=signal_list)
            signal_list[best_id] = 1
            max_id = get_best_friend(value_list=value_list,sum_corr_list=combine_matrix[best_id],k=k_value,self_id=best_id,signal_list=signal_list)
            signal_list[max_id] = 1
            id_list_up[id_now] = max_id
        else:
            id_now = i // 2
            worst_id = get_worst_id(value_list=value_list,signal_list=signal_list)
            signal_list[worst_id] = -1
            max_id = get_worst_friend(value_list=value_list,sum_corr_list=combine_matrix[worst_id],k=k_value,self_id=worst_id,signal_list=signal_list)
            signal_list[max_id] = -1
            id_list_down[id_now] = max_id
    up_list = []
    down_list = []
    for num_1 in id_list_up:
        up_list.append(scode_list[int(num_1)])
    for num_2 in id_list_down:
        down_list.append(scode_list[int(num_2)])
    return up_list,down_list

combine_matrix = combine_ratio_matrix(test_matrix)
earning_list = np.sum(test_matrix,axis=1)

up_num_list = []
down_num_list = []

for i in range(200):
    up_num_list.append(200-i)
    down_num_list.append(i-200)


train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
tool_set = []
sharp_set = []
value_set = []
i = 0
for k_ in [0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4]:
    tool_set.append(i)
    i += 1
    train_data_price = train_data_price[['Date','SecuritiesCode','Target']]
    a_list,b_list = get_the_best_portfolio(value_list=earning_list,combine_matrix=combine_matrix,scode_list=scode_list,portfolio_size=200,k_value=k_)
    temp_pd = pd.DataFrame()
    temp_pd['SecuritiesCode'] = (a_list+b_list)
    temp_pd['lag_rank'] = (up_num_list+down_num_list)
    train_data_price = pd.merge(train_data_price,temp_pd,on=['SecuritiesCode'],how='left')
    train_data_price['lag_rank'] = train_data_price['lag_rank'].fillna(0)
    print(train_data_price)
    print(list(train_data_price))
    add_rank(train_data_price)
    sha_ = calc_spread_return_sharpe(train_data_price)[0]
    sharp_set.append(sha_)
    print(sha_)
    add_rank_2(train_data_price)
    sha_ = calc_spread_return_sharpe(train_data_price)[0]

plt.plot(tool_set,sharp_set)
plt.show()



    


