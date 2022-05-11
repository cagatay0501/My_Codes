from sklearn.ensemble import RandomForestRegressor
# Random forest ensemble üyesi
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %% R^2 score for random forest
df = pd.read_csv("random+forest+regression+dataset.csv", sep=";",
                 header=None)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

# %%
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x, y)
y_head = rf.predict(x)

# %% R^2

print("My R^2 score: ", r2_score(y, y_head))
# İlk sıraya gerçek değerler 2. sıraya predict edilen değer.


# %% R^2 score for Linear Regression.


df = pd.read_csv("linear_regression_dataset.csv", sep=";")


plt.scatter(df.deneyim, df.maas, color="blue")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.show()
# Veri lineer regressiona uygun.

lr = LinearRegression()

x = df.deneyim.values.reshape(-1, 1)  # Burada array formatına çevrdik.
y = df.maas.values.reshape(-1, 1)  # Burada array formatına çevirdik.

lr.fit(x, y)


# %% Visualize Line

y_head = lr.predict(x)

plt.plot(x, y_head, color="red")

lr.predict([[100]])


# %% R^2

print("R^2 score: ", r2_score(y, y_head))
