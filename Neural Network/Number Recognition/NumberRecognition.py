import numpy as np
import random
from Data_object import DataClass
from NeuralNetwork_Object import NeuralNetworkClass
import matplotlib.pyplot as plt


def plot_number(self, number):

    Z = np.reshape(number, (16, 16))
    plt.imshow(Z)
    plt.show()

path = 'C:/Users/Markus/Documents/Python Scripts/Neural Network/Number Recognition/'
pathRefSet = path + 'refSet.csv'
pathRefAns = path + 'refAns.csv'
pathTestSet = path + 'testSet.csv'
pathTestAns = path + 'testAns.csv'

dat = DataClass(pathRefSet, pathRefAns, pathTestSet, pathTestAns)
data_RefSet, data_RefAns, data_TestSet, data_TestAns = dat.readData()

training_data = []
data_RefSet = np.matrix(data_RefSet)
data_RefAns = np.matrix(data_RefAns)
data_TestSet = np.matrix(data_TestSet)
data_TestAns = np.matrix(data_TestAns)

data_RefSet = np.transpose(data_RefSet)
data_RefAns = np.transpose(data_RefAns)
data_TestSet = np.transpose(data_TestSet)
data_TestAns = np.transpose(data_TestAns)

training_data.append(data_RefSet[:, :1700])
training_data.append(data_RefAns)

test_data = []
test_data.append(data_TestSet)
test_data.append(data_TestAns)

NN = NeuralNetworkClass([256, 10, 10, 10, 10])

NN.SGD(training_data, 100, 5, 1, None)

res = NN.evaluate(test_data)

print('Result: ', res)
