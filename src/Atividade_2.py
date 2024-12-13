# %%
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# %%
df
# %%
df.describe().transpose()

# %%
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

ax.set_title("First three PCA dimensions")
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()
# %%
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target)

print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.35, random_state = 42)

print(f'Training data (x) {x_train.shape}')
print(f'Training data (y) {y_train.shape}')
print(f'Testing data (x) {x_test.shape}')
print(f'Testing data (y) {y_test.shape}')
# %%
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
classifier.fit(x_train, y_train)
# %%
y_pred = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)
result = pd.DataFrame(y_test.values, columns=['Real'])
result['Predict'] = y_pred
result['Probability'] = np.round(y_prob[:,1],2)
result
# %%
cm = confusion_matrix(y_test, y_pred)
print(cm)
# %%
precision = metrics.precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)
recall = metrics.recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)