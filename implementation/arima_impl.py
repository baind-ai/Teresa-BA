import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Laden der Zeitreihe-Daten
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    return data

# Differenzieren der Zeitreihe
def difference(data, d):
    diff = []
    for i in range(d, len(data)):
        value = data[i] - data[i-d]
        diff.append(value)
    return np.array(diff)

# Implementierung von Autocorrelation Function (ACF) und Partial Autocorrelation Function (PACF)
# mithilfe der Parameter p und q
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def arima_parameters(data, p, q):
    # Plotten der Autokorrelationsfunktion
    plot_acf(data, lags=q)
    
    # Plotten der Partialautokorrelationsfunktion
    plot_pacf(data, lags=p)

    np.linalg.lstsq()

    #yt = c + φ1 * yt-1 + φ2 * yt-2 + ... + φp * yt-p + et
    #yt = μ + et + θ1 * et-1 + θ2 * et-2 + ... + θq * et-q
    return p, q

# Erstellen und Anpassen des ARIMA-Modells
def arima_model(data, p, d, q):
    # Implementierung von ARIMA-Modell
    # mit p, d und q als Modellparameter
    # Anpassung des Modells an die Daten
    
    return model

# Vorhersage der nächsten n Schritte
def arima_forecast(model, n):
    # Implementierung der Vorhersage der nächsten n Schritte
    # mit dem ARIMA-Modell
    
    return forecast



#2:
import pandas as pd

# Lade die Zeitreihe in ein pandas DataFrame
df = pd.read_csv('D:/BArbeit/Teresa-BA/examples/trend.csv', sep=";", usecols=["sales"])

# Überprüfe, ob die Zeitreihe stationär ist
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df['Wert'])
print(f'ADF Statistik: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')
print(f'Kritische Werte: {adf_test[4]}')

# Führe Differenzierung durch, falls notwendig, um die Zeitreihe stationär zu machen
if adf_test[1] > 0.05:
    df_diff = df.diff().dropna()
else:
    df_diff = df

from statsmodels.tsa.arima.model import ARIMA
# Trainiere ARMA-Modell auf der differenzierten oder ursprünglichen Zeitreihe
model = ARIMA(df_diff, order=(1,0,1))
model_fit = model.fit()
