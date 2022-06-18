import time
import pandas as pd


temp_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(temp_time)
new_dataframe = pd.DataFrame()
new_dataframe['time_record'] = [temp_time]
while True:
    time.sleep(1)
    temp_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    new_dataframe.loc[len(new_dataframe.index)] = [temp_time]
    new_dataframe.to_csv('temp_time.csv')