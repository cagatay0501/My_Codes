from sklearn.ensemble import RandomForestRegressor
# Random forest ensemble uyesi

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random+forest+regression+dataset.csv", sep=";",
                 header=None)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

# %%
rf = RandomForestRegressor(n_estimators=100, random_state=42)
"""
n_estimators = number of tree, (Icinde kullanacagim tree sayisi)

random_state = Basic anlatacak olursak, bu bir ID, bizim kullandigimiz
algoritma datayi sub dataya getirirken random bir sekilde boler. Bu durumda
her defasinda farkli bir predict degeri aliriz. Bunu onlemek icin
random_state belirleriz ve her defasinda bizim tanimladigimiz araliktan bolerek
bize belirli bir degeri dondurur. (döndürür.)
"""
rf.fit(x, y)

print("7.8 seviyesinde fiyatin ne kadar olduğu: ", rf.predict([[7.8]]))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
y_head = rf.predict(x_)

# Visualize
plt.scatter(x, y, color="red")
plt.plot(x_, y_head, color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
