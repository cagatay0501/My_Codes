from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% Import Data

df = pd.read_csv("decision+tree+regression+dataset.csv",
                 sep=";", header=None)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

# %%  Decision Tree Regression
tr = DecisionTreeRegressor()   # random sate = 0

tr.fit(x, y)

tr.predict([[5.5]])  # Çift parantez kullanmayu unutma


y_head = tr.predict(x)

# %% Visualize
plt.scatter(x, y, color="red")
plt.plot(x, y_head, color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()

"""
tr = tree regression
dtr = Decsion tree regression

Görselde bu değerlerin decision treede mean değerlerini sonuç vermesini
bekleriz. Fakat bizim sonucumu bu aşamada düz bir çizgi halinde olacaktır.
Böyle durumlarda splitleri göremeyiz.Bizim değerlerimizin değişen bir price
şeklinde gitmesinin nedeni bizim istenen değerleri vermiş olmamız.
(Predict ettirmemiz). DTR'dasplitleri görebilmek için daha büyük bir aralıkta
predict yaptırmalıyız.
"""

# %%  decision tree regression
tree_reg = DecisionTreeRegressor()   # random sate = 0
tree_reg.fit(x, y)


tree_reg.predict([[5.5]])
x_ = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
y_head = tree_reg.predict(x_)
# %% visualize
plt.scatter(x, y, color="red")
plt.plot(x_, y_head, color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
