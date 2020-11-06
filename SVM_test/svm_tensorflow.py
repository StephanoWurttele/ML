from sklearn import datasets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
iris = datasets.load_iris()

x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

for y in iris.target:
  print(y)

print("---------------------------------------")
for x in iris.data:
  print(x)

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

batch_size = 100
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape = [None,1], dtype=float32)
A = tf.Variable(tf.random_normal(shape=[2,1]))
B = tf.Variable(tf.random_normal(shape=[1,1]))

model_output = tf.subtract(tf.matmul(x_data, A), b)

12_norm = tf.reduce_sum(tf.square(A))
alpha = tf.constant([0,1])
classification_term = tf.reduce_mean(tf.maximum(0, tf.subtract(1.,tf.multiply(model_output, y_target
