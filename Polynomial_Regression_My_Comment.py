# %% Import Library
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# %% Import Data
df = pd.read_csv("polynomial+regression.csv", sep=";")

# %% Data check

y = df.araba_max_hiz.values.reshape(-1, 1)
x = df.araba_fiyat.values.reshape(-1, 1)

# %% Data Plot

plt.scatter(x, y)
plt.ylabel("Arabalarin Max Hizlari")
plt.xlabel("Arabalarin Fiyatlari")
plt.show()

# Linear Regression = b0 + b1*x
# Multiple LR = b0 + b1*x1 + b2*x2

lr = LinearRegression()
lr.fit(x, y)

# %% Predict (Linear)
y_head = lr.predict(x)
plt.plot(x, y_head, color="red", label="linear")
plt.show

print("10 milyon tllik araba hizi = ", lr.predict([[10000]]))
"""
Yukarıdaki dogru olamaz (871.66401826 km hız) hata sebebi linear kullanılması.
Polylomial Linear Regression kullanmamız gerekiyor.
"""
# %% Polynomial Regression


# Polinomial Regression = b0 + b1*x1 + b2*x^2 + b3*x^3 + ... +bn*x^n


# %% Fit


pf = PolynomialFeatures(degree=4)  # 2 denendi , 8 denendi optimumu 4 gibi

"""
Şimdi eğer kodumuzu:

x_polynomial = pf.fit(x)
şeklinde yazarsak bu ne anlama gelir.

- Bu benim modelimi oluştur demektir yani PolynomialFeatures(degree=2)
kullanır ama bunu bana vermez.
Ama ben polynomial features'u oluşturduktan sonra bunu elde etmek istiyorum.

Bu sebeple:
x_polynomial = pf.fit_transform(x)
demem gerekiyor. Peki bu ne anlama geliyor:

Buradaki PolynomialFeatures(degree=2) u kullan benim x imi 2. dereceden
polynomial feature'a çevir. Yani basitçe uygula ve çevir ve bunu
x_polynomial'a eşitle.

"""

x_polynomial = pf.fit_transform(x)
lr.fit(x_polynomial, y)


# %% Predict (Polynomial)

y_head2 = lr.predict(x_polynomial)

plt.plot(x, y_head2, color="purple", label="poly")
plt.legend()
plt.show()
" Degree ile oynayarak dataya en iyi oturan modeli bulabiliriz."
