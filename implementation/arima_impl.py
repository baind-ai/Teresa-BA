import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Laden der Zeitreihe-Daten
def load_data():
    #data = np.loadtxt(file_path, delimiter=",")
    values = []
    df = pd.read_csv('D:/BArbeit/Teresa-BA/examples/trend.csv', usecols=['sales'])
    values = df['sales']
    return values

# Differenzieren der Zeitreihe
def difference(data, d):
    diff = []
    for i in range(d, len(data)):
        value = data[i] - data[i-d]
        diff.append(value)
    return np.array(diff)

"""
# Implementierung von Autocorrelation Function (ACF) und Partial Autocorrelation Function (PACF)
# mithilfe der Parameter p und q
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def arima_parameters(data, p, q):
    # Plotten der Autokorrelationsfunktion
    plot_acf(data, lags=q)
    
    # Plotten der Partialautokorrelationsfunktion
    plot_pacf(data, lags=p)

    given_values = data
    predicted_values = []
    predicted_values = np.copy(given_values)
    c = 1

    expected_value = 0
    for i in range(0, len(data)):
        value = data[i]
        percentage = 1/len(data)
        multiplied = value*percentage
        expected_value= expected_value + multiplied


    #yt = c + φ1 * yt-1 + φ2 * yt-2 + ... + φp * yt-p + et
    #yt = μ + et + θ1 * et-1 + θ2 * et-2 + ... + θq * et-q


    #Yt​=(1+ϕ1​)Yt−1​−ϕ1​Yt−2​+εt​+θ1​εt−1​
    
    #(1−ϕ1​B)(1−B)Yt​=(1+θ1​B)εt​ Hierbei ist B der sogenannte Rückverschiebungs- oder Lag-Operator, der definiert ist als BYt​=Yt−1​, ϕ1​ ist der AR-Koeffizient, θ1​ ist der MA-Koeffizient und εt​ ist der Fehlerterm.
    #Die obige Gleichung kann auch umgeschrieben werden als:
    #Yt​−ϕ1​Yt−1​−Yt−1​+ϕ1​Yt−2​=εt​+θ1​εt−1​
    

    """
    # Autoregression (AR)
def ar_model(time_series, p):
    X = []
    y = []
    for i in range(p, len(time_series)):
        X.append(time_series[i-p:i])
        y.append(time_series[i])
    X = np.array(X)
    y = np.array(y)
    coefficients = np.linalg.pinv(X) @ y
    predictions = X @ coefficients
    residuals = y - predictions
    return coefficients, residuals

# Moving Average (MA)
def ma_model(residuals, q):
    coefficients = ar_model(residuals, q)[0]
    return coefficients

# ARIMA
def arima_model(time_series, p, d, q):
    # Integration
    integrated_time_series = difference(time_series, d)
    
    # Autoregression
    ar_coefficients, residuals = ar_model(integrated_time_series, p)
    
    # Moving Average
    ma_coefficients = ma_model(residuals, q)
    
    return ar_coefficients, ma_coefficients


def predict_arima(time_series, p, d, q, n_steps):
    # Fit ARIMA model
    ar_coefficients, ma_coefficients = arima_model(time_series, p, d, q)
    
    # Make predictions
    predictions = []
    for _ in range(n_steps):
        # Predict next value
        ar_term = np.dot(ar_coefficients, time_series[-p:])
        ma_term = np.dot(ma_coefficients, predictions[-q:])
        prediction = ar_term + ma_term
        predictions.append(prediction)
        
        # Update time series
        time_series = np.append(time_series, prediction)
    
    return predictions

# Make predictions
n_steps = 10
time_series = load_data()
predictions = predict_arima(time_series, p, d, q, n_steps)

# Plot predictions
plt.plot(predictions)
plt.show()

        
   # In dieser Funktion werden die Zeitreihe und die Verzögerungsordnung (p) als Eingabe verwendet. Die Funktion erstellt dann eine Designmatrix X und einen Zielvektor y aus der Zeitreihe. Die Designmatrix X enthält verzögerte Beobachtungen der Zeitreihe und der Zielvektor y enthält die entsprechenden aktuellen Beobachtungen.

#Die Funktion verwendet dann die Pseudo-Inverse von X und Matrixmultiplikation, um die Koeffizienten des Autoregressionsmodells zu berechnen. Diese Koeffizienten werden dann zurückgegeben.

"""   
        # ARIMA with constant term
def arima_model_with_constant(time_series, p, d, q):
    # Integration
    integrated_time_series = integration(time_series, d)
    
    # Add constant term
    X = np.ones((len(integrated_time_series), 1))
    integrated_time_series = np.hstack((X, integrated_time_series.reshape(-1, 1)))
    
    # Autoregression
    ar_coefficients = ar_model(integrated_time_series, p)
    
    # Moving Average
    residuals = integrated_time_series[p:, 1] - integrated_time_series[p:].dot(ar_coefficients)
    ma_coefficients = ma_model(residuals, q)
    
    return ar_coefficients[0], ar_coefficients[1:], ma_coefficients
Kopieren
In dieser Implementierung wird die arima_model_with_constant-Funktion verwendet, um ein ARIMA-Modell mit einem konstanten Term anzupassen. Die Funktion nimmt die Zeitreihe und die Parameter des ARIMA-Modells (p, d und q) als Eingabe.

Die Funktion integriert zunächst die Zeitreihe und fügt dann einen konstanten Term hinzu. Dieser konstante Term wird durch eine Spalte von Einsen repräsentiert.

Die Funktion verwendet dann die ar_model-Funktion, um die Koeffizienten des Autoregressionsmodells zu berechnen. Der erste Koeffizient des Autoregressionsmodells repräsentiert den konstanten Term und kann verwendet werden, um den Erwartungswert der Zeitreihe zu schätzen.
        
        
        

    for i in range(0,20):
        
        error_term_AR = 
        error_term_MA = 

        total_sum_lags = 0

        for j in range(0,p):
            quantifier = np.linalg.lstsq()
            lag_value = predicted_values[len(data) + i - j - 1]
            part_AR =  quantifier*lag_value
            total_sum_lags = total_sum_lags + part_AR
        

        total_sum_errors = 0
        for k in range(0,q):
            quantifier = np.linalg.lstsq()
            error_value = given_values[len(data) + i - k - 1]
            part_MA = quantifier*error_value
            total_sum_errors = total_sum_errors + part_MA

        predicted_values[len(data) + i] = c + total_sum_lags + error_term_AR  +  expected_value + total_sum_errors + error_term_MA

    
    return predicted_values

# Erstellen und Anpassen des ARIMA-Modells
def arima_model(data, p, d, q):
    values = arima_parameters(difference(data,d), p, q)
    plt.plot(values)
    return True

arima_model(load_data(), 1,1,1)



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
"""