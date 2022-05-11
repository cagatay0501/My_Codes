# %% Import Library

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# %% Import Data
df = pd.read_csv("multiple_linear_regression_dataset.csv", sep=";")

# %% Multiple Linear Regression

# Aslında burada model oluşturuyoruz.

x = df.iloc[:, [0, 2]].values  # 2 değeride aldı burada 2 side etki ettiği için
y = df.maas.values.reshape(-1, 1)


# %% mlr = Multiple Linear Regression

mlr = LinearRegression()
mlr.fit(x, y)  # Benim x ve y mi kullanarak bana bir line fit et.

print("b0 = ", mlr.intercept_)

print("b1, b2 = ", mlr.coef_)

# %% Predict

mlr.predict(np.array([[10, 35], [5, 35]]))
