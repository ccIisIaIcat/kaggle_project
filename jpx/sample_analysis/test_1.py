import matplotlib.pyplot as plt
import mpl_finance as mpf
import pandas as pd

fig = plt.figure(figsize=(8, 6), dpi=100, facecolor="white")
graph_KAV = fig.add_subplot(1, 1, 1)

data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
data = data[data['SecuritiesCode']==1333]
data = data[['Date','Open','Close','High','Low']]
mpf.candlestick2_ochl(graph_KAV, data['Open'], data['Close'], data['High'], data['Low'], width=0.5,colorup='r', colordown='g')  # 绘制K线走势

data = pd.read_csv("kaggle_data\\JPX\\train_files\\stock_prices.csv")
data = data[data['SecuritiesCode']==1332]
data = data[['Date','Open','Close','High','Low']]
mpf.candlestick2_ochl(graph_KAV, data['Open'], data['Close'], data['High'], data['Low'], width=0.5,colorup='r', colordown='g')  # 绘制K线走势

plt.show()

