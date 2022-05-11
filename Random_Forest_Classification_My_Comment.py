from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %% Data Import
data = pd.read_csv("data_logistic_regression.csv")

# %% Data Editing
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean, M.texture_mean,
            color="red", label="kotu", alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean,
            color="green", label="iyi", alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# %% Normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42)

# %% RandomForestClassifier Algorithm

rfc = RandomForestClassifier(n_estimators=100, random_state=1)

rfc.fit(x_train, y_train)

print("Random forest algorithm result: ", rfc.score(x_test, y_test))

y_pred = rfc.predict(x_test)
y_true = y_test

# %% Congfusion Matrix
# from sklearn.metrics import confusion_matrix burada import edildi
cm = confusion_matrix(y_true, y_pred)

# %% Cm Visualization

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt="0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
