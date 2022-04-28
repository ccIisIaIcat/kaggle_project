#编写一个根据提交的submission和已有的历史数据计算得分的脚本

TOTAL_STOCK = 2000 #总体的股票个数
FIRST_DATE = '2017-01-04' #测试集第一天的数据
END_DATE = '2021-12-03' #测试最后一天的数据

#为方便测试，每次计算需要提交预测开始日期和结束日期
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class score_judge():
    start_test_date = ''
    end_test_date = ''
    judge_data = pd.DataFrame
    submission_dataframe = pd.DataFrame
    data_series = pd.DataFrame
    def __init__(self,start_date,end_date,submission) -> None:
        self.start_test_date = pd.to_datetime(start_date,format = '%Y-%m-%d')
        self.end_test_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
        self.submission_dataframe = submission
        self.get_dataset()
        self.process_submission_set()
    def get_dataset(self):
        self.judge_data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
        self.judge_data = self.judge_data[['Date','SecuritiesCode','Target']]
        self.judge_data["Date"] = pd.to_datetime(self.judge_data["Date"],format='%Y-%m-%d') 
        self.judge_data = self.judge_data[(self.judge_data["Date"]>=self.start_test_date) & (self.judge_data["Date"]<=self.end_test_date)]
    def process_submission_set(self):
        self.submission_dataframe["Date"] = pd.to_datetime(self.submission_dataframe["Date"],format='%Y-%m-%d')
        self.submission_dataframe = pd.merge(self.submission_dataframe,self.judge_data,how='left',on=['Date','SecuritiesCode'])
        data_frame_up = self.submission_dataframe[self.submission_dataframe["Rank"]<200]
        data_frame_down = self.submission_dataframe[self.submission_dataframe["Rank"]>=1800]
        data_frame_up['tag'] = data_frame_up['Target']*(2-data_frame_up['Rank']/200)
        data_frame_down['tag'] = data_frame_down['Target']*(2-(1999-data_frame_down['Rank'])/200)
        self.data_series = data_frame_up.groupby(['Date'])['tag'].sum()-data_frame_down.groupby(['Date'])['tag'].sum()
    def score(self):
        return self.data_series.mean()/self.data_series.std()


sample_submission = pd.read_csv("kaggle_data\\JPX\\example_test_files\\tool_sample.csv")
s = score_judge('2020-12-04','2020-12-09',sample_submission)
print(s.score())


