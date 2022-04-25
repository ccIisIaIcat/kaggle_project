#一个简单的回顾，看看处理所得的数据
import pandas as pd

data_1 = pd.read_pickle("E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_1.pkl")
data_2 = pd.read_pickle("E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_2.pkl")
data_3 = pd.read_pickle("E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_3.pkl")
data_4 = pd.read_pickle("E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\lags_df_28.pkl")
print(data_4.head())
print(list(data_4))
print(data_4[data_4["id"]=="HOBBIES_1_008_CA_1_validation"])

