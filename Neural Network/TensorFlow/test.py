import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def BoundaryFunction(domain, Ns):
  '''
  Computing the boundary function G(x)
  t0 = starting time
  tN = end time
  Nt = number of points in time
  Ns = size of data set
  '''

  x_train = np.random.rand(Ns, 1)
  y_train = np.zeros((Ns, 1))
  #y_train[for k in x_train[k] < 0.1:] = 0.1

  model = tf.keras.Sequential([
  # Adds a densely-connected layer with 64 units to the model:
  layers.Dense(10, activation='sigmoid', input_shape=(1,)),
  # Add another:
  layers.Dense(10, activation='sigmoid'),
  # Add a softmax layer with 10 output units:
  layers.Dense(1, activation='sigmoid')])

  model.compile(optimizer=tf.train.GradientDescentOptimizer(10),
              loss='mse')

  model.fit(x_train, y_train, epochs=20)

  #x_test = np.random.rand(Ns, 1)
  #y_test = x_test - np.ones((Ns, 1)) * domain[0]
  #model.evaluate(x_test, y_test)

  return model

def DistanceFunction(domain, Ns):
  '''
  Computing distance function
  Ns = size of data set
  '''

  x_train = np.random.rand(Ns, 1)
  y_train = x_train
  #y_train = 1/2. + 1/2. * np.sin(2*np.pi*x_train) #- np.ones((Ns, 1)) * float(domain[0])

  model = tf.keras.Sequential([
  # Adds a densely-connected layer with 64 units to the model:
  layers.Dense(10, activation='sigmoid', input_shape=(1,)),
  # Add another:
  layers.Dense(10, activation='sigmoid'),
  # Add another:
  layers.Dense(10, activation='sigmoid'),
  # Add a softmax layer with 10 output units:
  layers.Dense(1, activation='sigmoid')])

  model.compile(optimizer=tf.train.GradientDescentOptimizer(5),
              loss='mse')

  model.fit(x_train, y_train, epochs=15)

  #x_test = np.random.rand(Ns, 1)
  #y_test = x_test - np.ones((Ns, 1)) * domain[0]
  #model.evaluate(x_test, y_test)

  return model

domain = [0.0, 1.0]
Ns = 10000
# Compute the distance function D(x)
distANN = DistanceFunction(domain, Ns)
#bndryANN = BoundaryFunction(domain, Ns)

x = np.linspace(0, 1, 101)
print(x)
y = distANN.predict(x)

plt.plot(x, y, 'b-o')
plt.show()
