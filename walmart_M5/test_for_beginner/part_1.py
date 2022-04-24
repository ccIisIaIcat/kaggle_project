import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime, timedelta

print("lll")

#读取所需的文件
cal = pd.read_csv("E:\\kaggle_project\\kaggle_data\\walmart\\calendar.csv")
ss = pd.read_csv("E:\\kaggle_project\\kaggle_data\\walmart\\sample_submission.csv")
stv = pd.read_csv("E:\\kaggle_project\\kaggle_data\\walmart\\sales_train_validation.csv")


#读取表最前面的id信息和最近28天的数据
last_28 = stv.iloc[:, np.r_[0,-28:0]]

#信息矩阵转为三列列表，id作为主列，变量名为d，值列为demand
last = last_28.melt('id', var_name='d', value_name='demand')

#左连接cal列表，连接键d
last = last.merge(cal,on="d")

#获得每个产品在不同周几的产品销量的平均值
by_weekday = last.groupby(["id","wday"])["demand"].mean()

#生成一个提交样本的副本
sub = ss.copy()

#把副本的列名改为新的日期名
sub.columns = ['id'] + ['d_' + str(1914+x) for x in range(28)]

#只选出validation列（valiation为方便预测做出的已有数据的预测集
sub = sub.loc[sub.id.str.contains('validation')]

#把sub转为三维列,只取id,d,wday,并根据wday和id左连接求出的demand
sub = sub.melt('id', var_name='d', value_name='demand')
sub = sub.merge(cal)[['id', 'd', 'wday']]
df = sub.join(by_weekday, on=['id', 'wday'])

#取需要的三列转化为矩阵形式
df = df.pivot(index='id', columns='d', values='demand')
#重置索引列
df.reset_index(inplace=True)
print(df)

#重置id顺序保证和提交格式的顺序一致
submission = ss[['id']].copy()
submission = submission.merge(df)

#把表翻倍以符合提交格式
submission = pd.concat([submission, submission], axis=0)

#保证和提交样表的id相同
submission['id'] = ss.id.values

#重命名列
submission.columns = ['id'] + ['F' + str(i) for i in range(1,29)]

#保存信息
submission.to_csv('E:\\kaggle_project\\kaggle_data\\walmart\\answer\\submission.csv', index=False)



