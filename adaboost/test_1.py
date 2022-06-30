from lzma import FORMAT_ALONE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# 创建随机数种子
rng = np.random.RandomState(111)
# 训练集X为300个0到10之间的随机数
X = np.linspace(0, 10, 300)[:, np.newaxis]
# 定义训练集X的目标变量
y = np.sin(1*X).ravel() + np.sin(2*X).ravel() + np.sin(3* X).ravel()+np.cos(3*X).ravel() +rng.normal(0, 0.3, X.shape[0])

plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='k', label='data', s=10, zorder=1, edgecolors=(0, 0, 0))
plt.xlabel("X")
plt.ylabel("y", rotation=0)
plt.show()

# 定义不同迭代次数的AdaBoost回归器模型
adbr_1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=1, random_state=123)
adbr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=10, random_state=123)
adbr_3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=123)

# 拟合上述三个模型
adbr_1.fit(X, y)
adbr_2.fit(X, y)
adbr_3.fit(X, y)

# 读取各个模型的最大迭代次数
adbr_1_n_estimators = adbr_1.get_params(True)['n_estimators']
adbr_2_n_estimators = adbr_2.get_params(True)['n_estimators']
adbr_3_n_estimators = adbr_3.get_params(True)['n_estimators']

# 预测
y_1 = adbr_1.predict(X)
y_2 = adbr_2.predict(X)
y_3 = adbr_3.predict(X)
# 画出各个模型的回归拟合效果
plt.figure(figsize=(10, 6))
# 画出训练数据集（用黑色表示）
plt.scatter(X, y, c="k", s=10, label="Training Samples")
# 画出adbr_1模型（最大迭代次数为1)的拟合效果（用红色表示）
plt.plot(X, y_1, c="r", label="n_estimators=%d" % adbr_1_n_estimators, linewidth=1)
# 画出adbr_2模型（最大迭代次数为10)的拟合效果（用绿色表示）
plt.plot(X, y_2, c="g", label="n_estimators=%d" % adbr_2_n_estimators, linewidth=1)
# 画出adbr_3模型（最大迭代次数为100)的拟合效果（用蓝色表示）
plt.plot(X, y_3, c="b", label="n_estimators=%d" % adbr_3_n_estimators, linewidth=1)

plt.xlabel("data")
plt.ylabel("target")
plt.title("AdaBoost_Regressor Comparison with different n_estimators when max_depth=3")
plt.legend()
plt.show()

# 拟合不同基学习器深度的回归模型
adbr_4 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=123)
adbr_5 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=100, random_state=123)
adbr_6 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=100, random_state=123)

# 拟合上述3个模型
adbr_4.fit(X, y)
adbr_5.fit(X, y)
adbr_6.fit(X, y)

# 预测
y_4 = adbr_4.predict(X)
y_5 = adbr_5.predict(X)
y_6 = adbr_6.predict(X)

# 画出各个模型的回归拟合效果
plt.figure(figsize=(10, 6))
# 画出训练数据集（用黑色表示）
plt.scatter(X, y, c="k", s=10, label="Training Samples")
# 画出adbr_4模型（基学习器深度为3)的拟合效果（用红色表示）
plt.plot(X, y_4, c="r", label="max_depth=4" , linewidth=1)
# 画出adbr_5模型（基学习器深度为4)的拟合效果（用绿色表示）
plt.plot(X, y_5, c="g", label="max_depth=5" , linewidth=1)
# 画出adbr_6模型（基学习器深度为5)的拟合效果（用蓝色表示）
plt.plot(X, y_6, c="b", label="max_depth=6" , linewidth=1)

plt.xlabel("data")
plt.ylabel("target")
plt.title("AdaBoost_Regressor Comparison with different max_depth when n_estimators=100")
plt.legend()
plt.show()

