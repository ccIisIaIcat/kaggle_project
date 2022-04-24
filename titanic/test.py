#为了熟悉函数进行的一些小测试
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt 

test = [{"sex":"famle","score":20.3},{"sex":"male","score":55},{"sex":"male","score":35}]

transfer = DictVectorizer(sparse=False)
titanic = pd.read_csv("kaggle_data/titanic/train.csv")

data = transfer.fit_transform(test)
print(data)

