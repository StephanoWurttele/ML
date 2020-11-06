import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target
print(x[0])
#skimage.io.imshow(x[0])

x_range = range(20)
results = [[], [], [], []]



kernels = ["linear", "rbf", "sigmoid"]
for j in x_range:
  x_train, x_test, y_train, y_test = sklearn.model_selection.\
    train_test_split(x, y, test_size=0.2)
  for i in kernels:
    clf = svm.SVC(kernel=i, gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    results[kernels.index(i)].append(acc)

  clf = KNeighborsClassifier(n_neighbors=10)
  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)
  acc = metrics.accuracy_score(y_test, y_pred)
  results[3].append(acc)

for x in range(len(results) - 2):
  plt.plot(x_range, results[x], label = kernels[x])
plt.plot(x_range, results[3], label = "KNN")
plt.xlabel = "Iteration"
plt.ylabel = "Precision"
plt.legend()
plt.show()

x = np.array(x)
y = np.array(y)
regr = LinearRegression()
regr.fit(x, y)
x_new = np.linspace(1, 2, 200)
y_new = regr.predict(x_new[:, np.newaxis])
plt.plot(x_new, y_new)
plt.savefig('figura.jpg')
 
