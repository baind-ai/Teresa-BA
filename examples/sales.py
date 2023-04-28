import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv('D:/BArbeit/Teresa-BA/examples/trend.csv', parse_dates=['login_date'], index_col=['login_date'])
df = pd.read_csv('D:/BArbeit/Teresa-BA/examples/trend.csv', sep=";", usecols=["sales"])
print(f"Total samples: {len(df)}")
print(df.head())

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df.sales)

f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title('1st Order Differencing')
ax1.plot(df.sales.diff())

ax2 = f.add_subplot(122)
plot_acf(df.sales.diff().dropna(), ax = ax2)
plt.show()

f2 = plt.figure()
ax12 = f2.add_subplot(121)
ax12.set_title('2st Order Differencing')
ax12.plot(df.sales.diff().diff())

ax22 = f2.add_subplot(122)
plot_acf(df.sales.diff().diff().dropna(), ax = ax22)
plt.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(df.sales.dropna())
print('p-value: ', result[1])

result = adfuller(df.sales.diff().dropna())
print('p-value1: ', result[1])

result = adfuller(df.sales.diff().diff().dropna())
print('p-value2: ', result[1])

from statsmodels.graphics.tsaplots import plot_pacf
f3 = plt.figure()
ax3 = f3.add_subplot(121)
ax3.set_title('1st Order Differencing')
ax3.plot(df.sales.diff())

ax4 = f3.add_subplot(122)
plot_pacf(df.sales.diff().dropna(), ax = ax4, lags=1)
plt.show()

f4 = plt.figure()
ax5 = f4.add_subplot(121)
ax5.set_title('2st Order Differencing')
ax5.plot(df.sales.diff().diff())

ax6 = f4.add_subplot(122)
plot_pacf(df.sales.diff().diff().dropna(), ax = ax6,lags=1)
plt.show()

from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(df.sales, order=(1,1,1)).fit()
prediction = arima_model.predict(len(df.sales), len(df.sales + 6))
print(arima_model.summary())
