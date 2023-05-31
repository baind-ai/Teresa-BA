import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./baind.csv', sep=",", names=["sensor", "temp", "time_stamp"])
print(f"Total samples: {len(df)}")
print(df.head())

sensor_name = df['sensor'].values[0]
plt.plot(df["time_stamp"],df["temp"])
plt.savefig("plot_data.pdf")