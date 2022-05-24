import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 投资组合权重
toprank_weight_ratio = 2
portfolio_size = 200
weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
print(weights)

test_list = [1,1,1,1,1.2]
test_list_2 = [1,1,1,1,1.2]
test_list = np.array(test_list)
test_list_2 = np.array(test_list_2)

def cal_sharp_ratio(list_):
    return list_.mean()/list_.std()

def combine_sharp_ratio(list_1,list_2):
    return cal_sharp_ratio(list_1+list_2)


def get_top_list(list_now,matrix,ratio_a,ratio_b):
    max_sharp = -100
    max_id = -1
    for i in range(len(matrix)):
        sharp_now = combine_sharp_ratio(ratio_a*list_now,ratio_b*matrix[i])
        if sharp_now > max_sharp:
            max_sharp = sharp_now
            max_id = i
    return max_id,max_sharp