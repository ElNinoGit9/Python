import numpy as np
import random
from Data_object import DataClass
import matplotlib.pyplot as plt

class NeuralNetworkClass:
    def __init__(self, sizes):

        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []

        for k in range(0, self.numLayers - 1):
            self.biases.append(np.random.normal(0, 1, self.sizes[k + 1]))
            self.weights.append(np.random.normal(0, 1, [self.sizes[k + 1], self.sizes[k]]))

    def Feedforward(self, a):
        a = np.squeeze(np.array(a))
        for k in range(0, self.numLayers - 1):

                b = self.biases[k]
                w = self.weights[k]
                a = self.sigmoid(np.dot(w,a) + b)

        return a

    def sigmoid(self, z):

         return 1.0/(1.0 + np.exp(-z))

    def sigmoid_prime(self, z):

         return self.sigmoid(z) * (1 - self.sigmoid(z))

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        ''' Stochastic Gradient Descent'''

        n = np.shape(training_data[0])[1]

        for k in range(0, epochs):

                train = np.matrix(training_data[0])
                train_res = np.matrix(training_data[1])
                shuffled = np.random.permutation(n)
                shuffled = [int(i) for i in shuffled]

                for l in range(0, n, mini_batch_size):

                        train_tmp     = [np.array(training_data[0][:, k]) for k in shuffled[l:l + mini_batch_size]]
                        train_res_tmp = [np.array(training_data[1][:, k]) for k in shuffled[l:l + mini_batch_size]]
                        train_tmp     = np.stack(train_tmp, axis=1)
                        train_res_tmp = np.stack(train_res_tmp, axis=1)
                        train_tmp     = np.matrix(train_tmp)
                        train_res_tmp = np.matrix(train_res_tmp)

                        mini_batch = []

                        mini_batch.append(train_tmp)
                        mini_batch.append(train_res_tmp)

                        self.update_mini_batch(mini_batch, eta)

                if test_data:
                        res = self.evaluate(test_data)
                        print('epoch ', k)
                        print('accuracy = ', res)

    def update_mini_batch(self, mini_batch, eta):

            nabla_b = []
            nabla_w = []

            for k in range(0, self.numLayers - 1):
                    nabla_b.append(np.zeros((np.shape(self.biases[k]))))
                    nabla_w.append(np.zeros((np.shape(self.weights[k]))))

            mini_batch_size = np.shape(mini_batch)
            for n in range(0, mini_batch_size[0]):
                    x = mini_batch[0][:, n]
                    y = mini_batch[1][:, n]
                    [delta_nabla_b, delta_nabla_w] = self.backprop(x, y)
                    for l in range(0, self.numLayers - 1):

                            nabla_b[l] = nabla_b[l] + delta_nabla_b[l]
                            nabla_w[l] = nabla_w[l] + delta_nabla_w[l]

            for m in range(0, self.numLayers - 1):

                    self.biases[m]  = self.biases[m]  - (eta/len(mini_batch)) * nabla_b[m]
                    self.weights[m] = self.weights[m] - (eta/len(mini_batch)) * nabla_w[m]

    def backprop(self, x, y):

            nabla_b = []
            nabla_w = []
            for k in range(0, self.numLayers - 1):
                    nabla_b.append(np.zeros((np.shape(self.biases[k]))))
                    nabla_w.append(np.zeros((np.shape(self.weights[k]))))

            act = np.squeeze(np.array(x))
            acts = []
            acts.append(act)
            zs = []

            for k in range(0, self.numLayers - 1):
                    b = self.biases[k]
                    w = self.weights[k]
                    act = np.squeeze(np.asarray(act))
                    z = np.dot(w, act) + b
                    zs.append(z)
                    act = self.sigmoid(z)
                    acts.append(act)


            delta       = np.multiply(self.cost_derivative(acts[-1], np.squeeze(np.array(y))), self.sigmoid_prime(zs[-1]))
            nabla_b[-1] = delta
            nabla_w[-1] = np.outer(delta, np.transpose(acts[-2]))

            for l in range(0, self.numLayers - 2):
                    z = zs[-l - 2]
                    sp = self.sigmoid_prime(z)
                    test = np.dot(np.transpose(self.weights[-l - 1]), delta)
                    delta = np.multiply(np.dot(np.transpose(self.weights[-l - 1]), delta), sp)
                    nabla_b[-l - 2] = delta
                    nabla_w[-l - 2] = np.outer(delta, np.transpose(acts[-l -3]))

            return nabla_b, nabla_w

    def evaluate(self, test_data):

            test_size = np.shape(test_data[0])
            test_results = np.zeros(test_size[1])

            x = np.transpose(test_data[0])
            y = np.transpose(test_data[1])
            out = np.zeros(test_size[1])

            for k in range(0, test_size[1]):

                    m = np.argmax(self.Feedforward(np.squeeze(x[k, :])))
                    #m2 = np.argmax(np.squeeze(y[k, :]))
                    m2 = np.argmax(np.squeeze(y[k, :]))
                    test_results[k] = (m - m2)
                    out[k] = (m == m2)

            return sum(out)/test_size[1]

    def cost_derivative(self, output_acts, y):

            return output_acts - y
