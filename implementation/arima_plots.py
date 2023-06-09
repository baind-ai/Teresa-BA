import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./baind.csv', sep=",", names=["sensor", "temp", "time_stamp"],index_col=None)
print(f"Total samples: {len(df)}")
print(df.head())

sensor_name = df['sensor'].values[0]
plt.plot(df["time_stamp"],df["temp"])
plt.savefig("plot_data.pdf")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

dta = df

#dta.index = pd.date_range(start=15000, end=17000, freq='T')
arima_model = ARIMA(df['temp'], order=(1,1,1)).fit()
fig, ax = plt.subplots()
ax = dta.loc['20000':].plot(ax=ax)
plot_predict(arima_model, 14000, 20000, ax=ax)
ax.set_ylim([-6, 30])
#plt.show()
plt.savefig("plot_library.pdf")

import arima_impl

predictions = arima_impl.predict_arima(arima_model, 1, 1, 1, 100)
plt.plot(predictions)
plt.savefig("plot_own_impl.pdf")


