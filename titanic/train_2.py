# 基于random_forest实现预测
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
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
print(x["Age"])

#2-3、划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)

#3、特征工程（对于训练集中的字典数据进行热编码处理）

transfer = DictVectorizer(sparse=False)

x_train = transfer.fit_transform(x_train.to_dict(orient="record"))
x_test = transfer.fit_transform(x_test.to_dict(orient="record"))

#4、模型预测
clf = RandomForestClassifier()

#5、构造参数集方便参数的交叉验证
parameters = {
    'n_estimators':[4,6,9],
    'max_features':['log2','sqrt','auto'],
    'criterion':['gini','entropy'],
    'max_depth':[2,3,5,10],
    'min_samples_split':[2,3,5],
    'min_samples_leaf':[1,5,8]
}

acc_scorer = make_scorer(accuracy_score)
grid_obj = GridSearchCV(clf,parameters,scoring=acc_scorer)
grid_obj = grid_obj.fit(x_train,y_train)

#获得最佳模型参数
clf = grid_obj.best_estimator_
print(grid_obj.best_estimator_)
# RandomForestClassifier(criterion='entropy', max_depth=5, max_features='log2',
#                        min_samples_split=3, n_estimators=9)

#参数预测
print(clf.score(x_test,y_test))
#0.8340807174887892


