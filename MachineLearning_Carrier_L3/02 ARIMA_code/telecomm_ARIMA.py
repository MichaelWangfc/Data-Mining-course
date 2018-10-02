#!-*- coding:utf-8 -*-
import random
import time
import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import matplotlib.pyplot as plt
start = '2016-12-31'
end = time.strftime('%Y-%m-%d')
datestart = datetime.datetime.strptime(start, '%Y-%m-%d')
dateend = datetime.datetime.strptime(end, '%Y-%m-%d')
dayincomedict = {}
value = 9000
while datestart < dateend:
    datestart += datetime.timedelta(days=1)
    dayDate = datestart.strftime('%Y-%m-%d')
    income = random.randint(value, 12110)
    dayincomedict[dayDate] = income

dayincomelist = sorted(dayincomedict.items(), key=lambda x: x[0], reverse=False)
print len(dayincomelist)
dayindex = []
dayincome = []
for item in dayincomelist:
    dayindex.append(item[0])
    dayincome.append(item[1])
print dayindex
print dayincome
from pandas.core.frame import DataFrame
import pandas as pd
daydict = {
    "date": dayindex,
    "income": dayincome
}
df = DataFrame(daydict)
# print df.head()
df = df.set_index('date')
df.index = pd.to_datetime(df.index)
ts = df['income']
# print ts.head().index
# def draw_ts(timeseries):
#     timeseries.plot()
#     plt.title('date & income')
#     plt.ylabel("income(w)")
#     plt.show()
# draw_ts(ts)

from statsmodels.tsa.stattools import adfuller
#判断时序数据稳定性
# def test_stationarity(timeseries):
#     # 这里以一年为一个窗口，每一个时间t的值由它前面12个月（包括自己）的均值代替，标准差同理。
#     rolmean = pd.rolling_mean(timeseries, window=12)
#     rolstd = pd.rolling_std(timeseries, window=12)
#     # plot rolling statistics:
#     fig = plt.figure()
#     fig.add_subplot()
#     orig = plt.plot(timeseries, color='blue', label='Original')
#     mean = plt.plot(rolmean, color='red', label='rolling mean')
#     std = plt.plot(rolstd, color='black', label='Rolling standard deviation')
#     plt.legend(loc='best')
#     plt.title('Rolling Mean & Standard Deviation')
#     plt.show(block=False)
#     # Dickey-Fuller test:
#     print 'Results of Dickey-Fuller Test:'
#     dftest = adfuller(timeseries, autolag='AIC')
#     # dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
#     for key, value in dftest[4].items():
#         dfoutput['Critical value (%s)' % key] = value
#
#     print dfoutput
#
#
# # ts = data['#Passengers']
# test_stationarity(ts)
# #由于原数据值域范围比较大，为了缩小值域，同时保留其他信息，常用的方法是对数化，取log。
ts_log = np.log(ts)
#Moving Average--移动平均
# moving_avg = pd.rolling_mean(ts_log,12)
# plt.plot(ts_log ,color = 'blue')
# plt.plot(moving_avg, color='red')
# plt.show()
#然后作差：
# ts_log_moving_avg_diff = ts_log-moving_avg
# ts_log_moving_avg_diff.dropna(inplace = True)
# test_stationarity(ts_log_moving_avg_diff)
# halflife的值决定了衰减因子alpha：  alpha = 1 - exp(log(0.5) / halflife)
# expweighted_avg = pd.ewma(ts_log,halflife=12)
# ts_log_ewma_diff = ts_log - expweighted_avg
# test_stationarity(ts_log_ewma_diff)
#
#Differencing--差分
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
# test_stationarity(ts_log_diff)
#
# #3.Decomposing-分解
# # 分解(decomposing) 可以用来把时序数据中的趋势和周期性数据都分离出来:
# from statsmodels.tsa.seasonal import seasonal_decompose
# def decompose(timeseries):
#     # 返回包含三个部分 trend（趋势部分） ， seasonal（季节性部分） 和residual (残留部分)
#     decomposition = seasonal_decompose(timeseries)
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid
#     plt.subplot(411)
#     plt.plot(ts_log, label='Original')
#     plt.legend(loc='best')
#     plt.subplot(412)
#     plt.plot(trend, label='Trend')
#     plt.legend(loc='best')
#     plt.subplot(413)
#     plt.plot(seasonal, label='Seasonality')
#     plt.legend(loc='best')
#     plt.subplot(414)
#     plt.plot(residual, label='Residuals')
#     plt.legend(loc='best')
#     plt.tight_layout()
#     return trend, seasonal, residual
# #
# # 消除了trend 和seasonal之后，只对residual部分作为想要的时序数据进行处理
# trend , seasonal, residual = decompose(ts_log)
# residual.dropna(inplace=True)
# test_stationarity(residual)
#
#ACF and PACF plots:
# from statsmodels.tsa.stattools import acf, pacf
# lag_acf = acf(ts_log_diff, nlags=20)
# lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
# #Plot ACF:
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')
#
# #Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
# plt.show()
#
from statsmodels.tsa.arima_model import ARIMA
# # model = ARIMA(ts_log, order=(1, 1, 0))
# # results_ARIMA = model.fit(disp=-1)
# # plt.plot(ts_log_diff)
# # plt.plot(results_AR.fittedvalues, color='red')
# # plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
# # plt.show()

model = ARIMA(ts_log, order=(1, 1, 0))
results_ARIMA = model.fit(disp=-1)
reslist= model.predict(dayindex,"2018-07-22","2018-08-22")
print reslist
# # plt.plot(ts_log_diff)
# # plt.plot(results_MA.fittedvalues, color='red')
# # plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
#
model = ARIMA(ts_log, order=(1, 1, 1))
# results_ARIMA = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
# plt.show()
#
#ARIMA拟合的其实是一阶差分ts_log_diff，predictions_ARIMA_diff[i]是第i个月与i-1个月的ts_log的差值。
#由于差分化有一阶滞后，所以第一个月的数据是空的，
# predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
# print predictions_ARIMA_diff.head()
#累加现有的diff，得到每个值与第一个月的差分（同log底的情况下）。
#即predictions_ARIMA_diff_cumsum[i] 是第i个月与第1个月的ts_log的差值。
# predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#先ts_log_diff => ts_log=>ts_log => ts
#先以ts_log的第一个值作为基数，复制给所有值，然后每个时刻的值累加与第一个月对应的差值(这样就解决了，第一个月diff数据为空的问题了)
#然后得到了predictions_ARIMA_log => predictions_ARIMA
# predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
# predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
# predictions_ARIMA = np.exp(predictions_ARIMA_log)
# plt.figure()
# plt.plot(ts)
# plt.plot(predictions_ARIMA)
# plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
# plt.show()
