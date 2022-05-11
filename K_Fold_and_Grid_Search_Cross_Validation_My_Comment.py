
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

# %% Normalization

x = (x - np.min(x))/(np.max(x) - np.min(x))

# %% Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=42)

# %% KNN MODEL

knn = KNeighborsClassifier(n_neighbors=3)  # K = n_neighbors

# %% K - Fold Cross Validation K == 10

accuracies = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10)

# Estimator cv yaparken kullanacağı algoritma.
# cv bölme sayısı

print("Avarage accuracy: ", np.mean(accuracies))
print("Avarage accuracy: ", np.std(accuracies))  # Tutarlılık.

# %%
knn.fit(x_train, y_train)
print("test accuracy: ", knn.score(x_test, y_test))


# %% Grid Search Cross Varidation

grid = {"n_neighbors": np.arange(1, 50)}
knn = KNeighborsClassifier()


knn_cv = GridSearchCV(knn, grid, cv=10)
"""
knn_cv =,
 GridSearchCV(knn, grid, cv = 10)
 
 GridSearchCV içine:
     1 tane kullanacağı algoritmayı alır.
     Daha sonra gridleri alır
     Ve kaça böleceğimiz  cv alır.

burada grid içerisindeki n neigbors yerine başka şeylerde olabilirdi.
Mesela logistic Regression için orada, ridge ve lasso gibi regulazition
teknikleri olabilirdi. Onlar olabilirdi. Veya treeler için ne kadar derinlite
olacağı olabilirdi bir treenin.
"""
knn_cv.fit(x, y)

# %% Print Hyperparameter KNN algoritmasındaki K değeri

print("tuned hyperparameter K: ", knn_cv.best_params_)

print("tuned parametreye göre en iyi accuracy (Best Score): ",
      knn_cv.best_score_)

# %% Grid Search Cross Varidation with Logistic Regression


# Öncelikle şunu unutmayalım  logistic regression bir binary classifier dır.

x = x[:100, :]
y = y[:100]

param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
# L1 == lasso / L2 = ridge parametrelerini ifade eder.

log_reg = LogisticRegression()

log_reg_cv = GridSearchCV(log_reg, param_grid, cv=10)

log_reg_cv.fit(x, y)

print("tuned hyperparameter K: ", log_reg_cv.best_params_)
print("tuned parametreye göre en iyi accuracy (Best Score): ",
      log_reg_cv.best_score_)
