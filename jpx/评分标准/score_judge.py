#编写一个根据提交的submission和已有的历史数据计算得分的脚本

TOTAL_STOCK = 2000 #总体的股票个数

FIRST_DATE = '2017-01-04' #测试集第一天的数据
END_DATE = '2021-12-03' #测试最后一天的数据

#为方便测试，每次计算需要提交预测开始日期和结束日期
import pandas as pd

class score_judge():
    start_test_date = ''
    end_test_date = ''
    judge_data = pd.DataFrame
    submission_dataframe = pd.DataFrame
    def __init__(self,start_date,end_date,submission) -> None:
        self.start_test_date = pd.to_datetime(start_date,format = '%Y-%m-%d')
        self.end_test_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
        self.submission_dataframe = submission
        self.get_dataset()
    def test_function(self):
        print(self.start_test_date,self.end_test_date)
        print(self.judge_data.head())
    def get_dataset(self):
        self.judge_data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
        self.judge_data = self.judge_data[['Date','SecuritiesCode','Target']]
        self.judge_data["Date"] = pd.to_datetime(self.judge_data["Date"],format='%Y-%m-%d') 
        self.judge_data = self.judge_data[(self.judge_data["Date"]>=self.start_test_date) & (self.judge_data["Date"]<=self.end_test_date)]
    def process_submission_set(self):
        self.submission_dataframe["Date"] = pd.to_datetime(self.submission_dataframe["Date"],format='%Y-%m-%d')
        self.submission_dataframe =  self.submission_dataframe[(self.submission_dataframe["Rank"]<200) |  (self.submission_dataframe["Rank"]>=1800)]

sub = pd.read_csv("kaggle_data\\JPX\\example_test_files\\sample_submission.csv")
# s = score_judge('2002-4-26','2022-4-27',sub)
# print(pd.to_datetime(s.start_test_date,format = '%Y-%m-%d')>pd.to_datetime(s.end_test_date,format = '%Y-%m-%d'))
# s.test_function()

def process_submission_set(submission_dataframe):
        submission_dataframe["Date"] = pd.to_datetime(submission_dataframe["Date"],format='%Y-%m-%d')
        submission_dataframe =  submission_dataframe[(submission_dataframe["Rank"]<200) | (submission_dataframe["Rank"]>=1800)]
        print(submission_dataframe.groupby('Date')['Rank'].describe())

process_submission_set(sub)
