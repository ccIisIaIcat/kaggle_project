import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt


LAG_LIST = [1,2,3,4,8,16,25,32,48,56,64,78,90] #lag信息序列
K_FOLDS = 7 #测试集个数
OVERLAP_RATIO = 0.3 #cv测试集覆盖比率
TEST_DATA_RATIO = 0.2 #每组中测试集所占比例
seed0 = 8586 #随机种子
#提升树的参数设定
params = {
    'early_stopping_rounds': 100,
    'objective': 'regression',
    'metric': 'rmse',
#     'metric': 'None',
    'boosting_type': 'gbdt',
    'max_depth': 5,
    'verbose': -1,
    'max_bin':600,
    'min_data_in_leaf':50,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'subsample_freq': 1,
    'feature_fraction': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 2,
    'seed':seed0,
    'feature_fraction_seed': seed0,
    'bagging_fraction_seed': seed0,
    'drop_seed': seed0,
    'data_random_seed': seed0,
    'extra_trees': True,
    'extra_seed': seed0,
    'zero_as_missing': True,
    "first_metric_only": True
    }

train_data_price = pd.read_csv('kaggle_data/JPX/train_files/stock_prices.csv')
train_data_price = train_data_price[['Date','SecuritiesCode','Open','Close','Target']]

def get_date_divide(date_list,k_folds, overlap_ratio,test_data_ratio):
    # (k*0.3+0.7)*l = L => l = L/(k*0.3+0.7)
    single_length = int(len(date_list)/(k_folds*overlap_ratio+1-overlap_ratio))
    gap_size = int(single_length*overlap_ratio)
    date_set = []
    for i in range(k_folds):
        start_date = date_list[i*gap_size]
        if i*gap_size+single_length < len(date_list):
            end_date = date_list[i*gap_size+single_length]
        else :
            end_date = date_list[len(date_list)-1]
        mid_date = date_list[i*gap_size+int(single_length*(1-test_data_ratio))]
        date_set.append([start_date,mid_date,end_date])
    return date_set

def get_lag_info(train_data,lag_list):
    train_data['target_ne'] = train_data.groupby(['SecuritiesCode'])['Target'].shift(-1)
    for lag_length in lag_list:
        train_data[f'lag_{lag_length}_info'] = train_data.groupby(['SecuritiesCode'])['Close'].shift(lag_length)
        train_data[f'lag_{lag_length}_info'] = (train_data['Close']-train_data[f'lag_{lag_length}_info'])/train_data[f'lag_{lag_length}_info']
    train_data = train_data.drop(columns=['Open','Close'])
    train_data = train_data.dropna(axis=0,how='any')
    return train_data

def add_rank(df):
    df["Rank"] = df.groupby("Date")["predict_tartget"].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

def add_random_rank(df):
    df["Rank"] = df.groupby("Date")['SecuritiesCode'].rank(ascending=False, method="first") - 1 
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


train_data_price = get_lag_info(train_data_price,LAG_LIST)
date_list = train_data_price.sort_values(by='Date',ascending=True)['Date'].unique()
print(list(train_data_price))
date_list = get_date_divide(date_list,K_FOLDS,OVERLAP_RATIO,TEST_DATA_RATIO)

ratio_set = []
random_ratio_set = []
features_importance = []
for time_set in date_list:
    print(time_set)
    train_data = train_data_price[(train_data_price['Date']>time_set[0]) & (train_data_price['Date']<time_set[1])]
    test_data = train_data_price[(train_data_price['Date']>time_set[1]) & (train_data_price['Date']<time_set[2])]
    x_train = train_data.drop(columns=['Date','SecuritiesCode','Target'])
    y_train = train_data['Target']
    x_test = test_data.drop(columns=['Date','SecuritiesCode','Target'])
    y_test = test_data['Target']
    train_dataset = lgb.Dataset(x_train,y_train)
    val_dataset = lgb.Dataset(x_test,y_test,reference=train_dataset)
    model = lgb.train(params = params,
                          train_set = train_dataset, 
                          valid_sets=[train_dataset, val_dataset],
                          valid_names=['tr', 'vl'],
                          num_boost_round = 5000,
                          verbose_eval = 100,   
                         )
    features_importance.append(model.feature_importance())
    test_data['predict_tartget'] = model.predict(x_test)
    add_rank(test_data)
    test_data = test_data[['Date','SecuritiesCode','Target','Rank']]
    ratio_set.append(calc_spread_return_sharpe(test_data)[0])
    add_random_rank(test_data)
    random_ratio_set.append(calc_spread_return_sharpe(test_data)[0])
    print(">>>>>>>>>>>>>>")

tool_set = []
name_list = []
for i in range(K_FOLDS):
    tool_set.append(i+1)
    name_list.append(f'num_{i+1}')

tool_set_2 = []
for i in range(len(LAG_LIST)+1):
    tool_set_2.append(i+1)


print(features_importance)

print(list(train_data))

def compare_importance(list_):
    sum_ = list_.sum()
    list_ = list_/sum_
    return list_


for i in range(K_FOLDS):
    plt.plot(tool_set_2,compare_importance(features_importance[i]) )
plt.legend(name_list)
plt.show()
plt.plot(tool_set,ratio_set)
plt.plot(tool_set,random_ratio_set)
plt.legend(['predict','Random'])
plt.show()


# new_list = train_data_price[train_data_price['SecuritiesCode'] == 1301]
# print(new_list)

        
