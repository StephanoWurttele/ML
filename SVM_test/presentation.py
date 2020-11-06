import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("TkAgg")

cancer = datasets.load_breast_cancer()
fig = plt.figure()
x = cancer.data
y = cancer.target
r = list(x for x in range(1,570) if ((x % 40 == 0) or x == 569))
#r = [100,200,300,400]
print(r)
x_range = 10
mini = 2
x_vals = []
print(len(x))
kernels = ["linear", "rbf", "sigmoid"]
for i in kernels[:2]:
  print(i)
  temp = []
  for j in r:
    x_train, x_test, y_train, y_test = sklearn.model_selection.\
        train_test_split(x, y, test_size=0.2)
    clf = svm.SVC(kernel=i, gamma='scale')
    clf.fit(x_train[:j], y_train[:j])
    y_pred = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    if acc < mini:
      mini = acc
    temp.append(acc)
  x_temp = np.array(temp).reshape((-1,1))
  y_temp = np.array(r)
  regr = LinearRegression()
  regr.fit(x_temp, y_temp)
  x_new = np.linspace(mini, 1, 200)
  y_new = regr.predict(x_new[:, np.newaxis])
  plt.plot(x_new, y_new, label=i)
  x_vals.clear()
   
mini = 2
for k in r:
  clf = KNeighborsClassifier(n_neighbors=10)
  clf.fit(x_train[:k], y_train[:k])
  y_pred = clf.predict(x_test)
  acc = metrics.accuracy_score(y_test, y_pred)
  if acc < mini:
    mini = acc
  x_vals.append(acc)
print(x_vals)
x_temp = np.array(x_vals).reshape((-1, 1))
y_temp = np.array(r)
regr = LinearRegression()
regr.fit(x_temp, y_temp)
x_new = np.linspace(mini, 1, 200)
y_new = regr.predict(x_new[:, np.newaxis])
plt.plot(x_new, y_new, label="KNN")
fig.suptitle('Model comparison', fontsize=20)
plt.xlabel('Number of tests', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
plt.legend()
fig.savefig('figura_comparacion_outlier.jpg')
