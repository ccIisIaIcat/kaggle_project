#新增的一些优化
import pandas as pd
import numpy as np

grid_df = pd.read_pickle('E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_1.pkl')
grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)
del grid_df['wm_yr_wk']
grid_df.to_pickle('E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_1.pkl')

del grid_df