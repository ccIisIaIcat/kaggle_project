train_2:
    基于随机森林方法实现预测

    1、y的dataframe转置(y_test.values.ravel()

    2、GridSearchCv,交叉参数检验（适用于小样本），以随机森林为例
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
        clf = grid_obj.best_estimator_