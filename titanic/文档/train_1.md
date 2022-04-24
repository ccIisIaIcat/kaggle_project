train_1:
    基于xgboost提升树方法，并尝试了不同树高下的预测准确率和结果绘制
    使用到的sklearn和pandas函数有：

        1、数据读取[pd.read_csv()]

        2、从dataframe中读取特定列data[["a1","a2"]]

        3、缺失值处理data["a1"].fillna(data["a1"].mean(),inplace=True),其中，平均数mean也可以用众数mode、中位数median代替，inplace是指是否用新数据代替旧数据，如果是false的话则会生成新的副本

        4、训练集和测试集的划分 train_test_split（from sklearn.model_selection import train_test_split）
            参数：test_size:测试集所占的分数比例，默认0.3
                 random_state:随机种子，默认0
        
        5、特征处理，热编码，使用（from sklearn.feature_extraction import DictVectorizer）
            方法：先声明一个transfer对象，然后调用对象的fit_transfer(x)进行调用，其中x是一个字典列表
                fit后调用vec.get_feature_names()函数可以查看每一列的参数

            注意：当DictVectorizer的sparse参数为true时，返回的是稀疏矩阵中的非零值，为false时，返回一个二维矩阵
                当一列有多个字典参数时，会自动生成多列，每行用0，1表示
        
        6、dataframe的to_dict方法：以每一行的列名为key，具体值为value，返回的具体形式由orient决定
            orient的值有，"dict","record","list"等，可查看：https://blog.csdn.net/m0_37804518/article/details/78444110
        
        7、数据预览
            dataframe的describe()方法可以返回数据的一些基本信息
        
        8、xgboost实例化参数
            暂时值考虑三个参数：eta=0.3(学习率),gamma=0(过拟合程度),max_depth=i(每次学习的树高)

            训练方法：fit(x_train,y_train)

            结果检查方法：score(x_test,y_test)
        
        
