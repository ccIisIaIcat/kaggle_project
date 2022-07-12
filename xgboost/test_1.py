import pandas as pd
import numpy as np

d=pd.DataFrame()
d['feature']=[0,1,2,np.nan]
y=pd.DataFrame([0,1,2,3])

import xgboost as xgb
clf=xgb.XGBRegressor(n_estimators=1)
clf.fit(d,y)
xgb.plot_tree(clf)