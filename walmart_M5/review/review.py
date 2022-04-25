#一个简单的回顾，看看处理所得的数据
import pandas as pd

data_1 = pd.read_pickle("E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_1.pkl")
data_2 = pd.read_pickle("E:\\kaggle_project\\kaggle_data\\walmart\\tool_data\\grid_part_2.pkl")
print(data_2.head())

