import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import load_data

data_raw_list = load_data.load_alcohol()[2]

data_first = data_raw_list[1]
df = load_data.prepare_data_alcohol(data_first)

fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(16, 12))

for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i + 1]]
    ax.plot(data, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=3)
    ax.set_title(df.columns[i + 1])
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()

df = df.drop(columns=['finish', 'drink_predict', 'restless_sleep', 'difficulty_sleep', 'hours_sleep'])

nobs = 4
df_train, df_test = df[0:-nobs], df[-nobs:]

df_diff = df_train.diff().dropna()

model = VAR(df_diff)
x = model.select_order(maxlags=5)
print(x.summary())
model_fitted = model.fit(5)

lag_order = model_fitted.k_ar
forecast_input = df_diff.values[-lag_order:]

fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_1d')

for col in df_train.columns:
    df_forecast[str(col) + '_forecast'] = df_train[col].iloc[-1] + df_forecast[str(col) + '_1d'].cumsum()

df_forecast = df_forecast.loc[:, ['craving_forecast', 'want_drink_forecast']]
print(df_forecast.head())
print(df_test['craving'].head())


def forecast_accuracy(forecast, actual):
    print('MAPE:', np.mean(np.abs(forecast - actual) / np.abs(actual)))
    print('ME:', np.mean(forecast - actual))
    print('MAE:', np.mean(np.abs(forecast - actual)))
    print('MPE:', np.mean((forecast - actual) / actual))
    print('RMSE:', np.mean((forecast - actual) ** 2) ** .5)
    print('CORR:', np.corrcoef(forecast, actual)[0, 1])


print('Craving Performance:')
forecast_accuracy(df_forecast['craving_forecast'].values, df_test['craving'])
print('Want to drink Performance:')
forecast_accuracy(df_forecast['want_drink_forecast'].values, df_test['want_drink'])
