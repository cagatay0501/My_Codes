# sklearn ayı zamanda datasetleri de içinde bulunduran bir kütüphane
# Data setimizi sklearnden indirelim bu kez.

# %% Import Library and DataSet

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns=feature_names)
df["sinif"] = y

x = data

# %% PCA

# from sklearn.decomposition import PCA burada kullanıldı.

"""
Biz bunu neden yapıyoruz? Şimdi bizim datamızda 4 adet feature var. Bu datayı
renkleri kullanarak görselleştirebilirz. Bizim amacımız burada pca kullanarak
bu 4 boyutlu datayı 2 boyuta çekmek ve böyle görselleştirmek.

whiten=True yazarak aslında datayı normalize ediyor.
"""

pca = PCA(n_components=2, whiten=True)  # whitten = normalize
pca.fit(x)
x_pca = pca.transform(x)

"""
Burada pca.fit(x) yaparak x'e pca uygulayacak modeli oluşturduk.
x_pca = pca.transform(x)
yaparak bunu x_pca ya eşitledik. Şuan elimizde 2 boyutlu x_pca var.

Peki bunun hangisi principle component hangisi second component?

pca.explained_variance_ratio_ bize bunu söyleyecek.

varianc ratio:  [0.92461872 0.05306648] çıktısını aldığımızda
% 92 olan benim principle %5 olan ise second component değerimi verir.
"""

print("varianc ratio: ", pca.explained_variance_ratio_)

"""
Peki amacımız neydi?
- Variance korumak, 4 ten 2 ye düşürüyorum datamı ama  datamın bana sağladığı
bilgileri korumak istiyorum. Peki ne kadarını koruduk?

Bunu yaparkende yukarıda hesapladığımız.
"varianc ratio: ", pca.explained_variance_ratio_ yani

pca.explained_variance_ratio_ değerinin çıktılarını toplamamız gerekiyor.
"""

print("sum: ", sum(pca.explained_variance_ratio_))

"""
Sonuç 0.977685206318795 yani datamdan %2.232 lik bir bilgi kaybım var.

Şimdi bu datamızın nasıl göründüğüne bakalım.
"""

# %% 2D
df["p1"] = x_pca[:, 0]
df["p2"] = x_pca[:, 1]

color = ["red", "green", "blue"]

for each in range(3):
    plt.scatter(df.p1[df.sinif == each], df.p2[df.sinif == each],
                color=color[each], label=iris.target_names[each])
plt.xlabel("p1")
plt.ylabel("p2")
plt.legend()
plt.show()
