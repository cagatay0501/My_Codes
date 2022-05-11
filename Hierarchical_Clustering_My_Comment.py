# %% Impport Library

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram  # Dendogram için
from sklearn.cluster import AgglomerativeClustering  # Hierarchical Cluster
# için import ettik.


# %% Create a Data

# Class 1
x1 = np.random.normal(25, 5, 100)
y1 = np.random.normal(25, 5, 100)

# Class 2
x2 = np.random.normal(55, 5, 100)
y2 = np.random.normal(60, 5, 100)

# Class 3
x3 = np.random.normal(55, 5, 100)
y3 = np.random.normal(15, 5, 100)


x = np.concatenate((x1, x2, x3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

# Şimdi bir ditionary de bunları birleştirelim.
dictionary = {"x": x, "y": y}

data = pd.DataFrame(dictionary)

# %% Dendogram

# Hierarchical cluster yapmadan önceki ilk step dendogram çizmek.

# from scipy.cluster.hierarchy import linkage,dendrogram
# import edilmelidir.

merg = linkage(data, method="ward")  # scipy'ın hierarchical algoritmasıdır.

dendrogram(merg, leaf_rotation=90)
plt.xlabel("Data_Point")
plt.ylabel("Euclidean_Distance")
plt.show()

# Sonucuna göre 3 adet cluster sayısı ideal.


# %% Hierarchical Clustering Algortihm

"""
AgglomerativeClustering : Her bir data pointten tek tek işlemlerle 1 clustera
ulaşma işlemine denir.
"""

hierarchical_cluster = AgglomerativeClustering(n_clusters=3,
                                               affinity="euclidean",
                                               linkage="ward")
cluster = hierarchical_cluster.fit_predict(data)

data["label"] = cluster

# %% Data Visualization

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color="red")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color="green")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color="blue")
plt.show()
