# 基于xgboost实现预测
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt 

#1、数据获取
titanic = pd.read_csv("kaggle_data/titanic/train.csv")

#数据预览
print(titanic.describe())

#2、数据基本处理
#2-1、确定特征值，目标值
x = titanic[["Pclass","Age","Sex","SibSp","Parch","Fare"]]
y = titanic[["Survived"]]

#2-2、处理缺失值
x["Age"].fillna(x["Age"].mean(),inplace=True)

#2-3、划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)

#3、特征工程（对于训练集中的字典数据进行热编码处理）

transfer = DictVectorizer(sparse=False)

x_train = transfer.fit_transform(x_train.to_dict(orient="record"))
x_test = transfer.fit_transform(x_test.to_dict(orient="record"))


#4、xgboost模型训练和模型评估
xg = XGBClassifier()
xg.fit(x_train,y_train)
    #查看训练结果
# print(xg.score(x_test,y_test))
    #进行优化
#针对max_depth进行调优
depth_range = range(10)
score = []
for i in depth_range:
    xg = XGBClassifier(eta=0.3,gamma=0,max_depth=i)
    xg.fit(x_train,y_train)
    score.append(xg.score(x_test,y_test))

#可视化
plt.plot(depth_range,score)
plt.show()




