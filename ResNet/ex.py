import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = [[1.0], [2.0], [3.0]]
y = [1.0, 4.0, 9.0]
x = np.array(x)
y = np.array(y)
regr = LinearRegression()
regr.fit(x, y)
x_new = np.linspace(1, 2, 200)
y_new = regr.predict(x_new[:, np.newaxis])
plt.plot(x_new, y_new)
plt.savefig('figura.jpg')
