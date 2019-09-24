import numpy as np
import random
from Data_object import DataClass
from NeuralNetwork_Object import NeuralNetworkClass

def data_divide(data, back_time, front_time, test_frame):
    ''' Divides data into Reference and Test sets'''

    refSet = data[:back_time]
    refAns = data[back_time:back_time + front_time]

    for k in range(back_time + 1, len(data) - test_frame - front_time + 1):
        refSet = np.vstack((refSet, data[k - back_time:k]))
        refAns = np.vstack((refAns, data[k:k + front_time]))

    testSet = data[-test_frame:-test_frame + back_time]
    testAns = data[-test_frame + back_time:-test_frame + back_time + front_time]

    for l in range(back_time + 1, test_frame - front_time):

        testSet = np.vstack((testSet, data[-test_frame + l - back_time:-test_frame + l]))
        testAns = np.vstack((testAns, data[-test_frame + l:-test_frame + l + front_time]))

    testSet = np.vstack((testSet, data[-front_time - back_time + 1:-front_time + 1]))
    testAns = np.vstack((testAns, data[-front_time:]))

    return np.transpose(refSet), np.transpose(refAns), np.transpose(testSet), np.transpose(testAns)

#data = [i/100. for i in range(0, 100)]

path = 'C:/Users/Markus/Documents/Python Scripts/Neural Network/Invest AI/'

pathData = path + 'OMXS308618.csv'

dat = DataClass(pathData)
dat_data = dat.readData()

print(dat_data)

'''
data_RefSet, data_RefAns, data_TestSet, data_TestAns = data_divide(data, 10, 5, 20)

print(data_RefSet)
print(data_RefAns)
print(data_TestSet)
print(data_TestAns)

training_data = []
#data_RefSet = np.matrix(data_RefSet)
#data_RefAns = np.matrix(data_RefAns)
#data_TestSet = np.matrix(data_TestSet)
#data_TestAns = np.matrix(data_TestAns)

#data_RefSet = np.transpose(data_RefSet)
#data_RefAns = np.transpose(data_RefAns)
#data_TestSet = np.transpose(data_TestSet)
#data_TestAns = np.transpose(data_TestAns)

training_data.append(data_RefSet)
training_data.append(data_RefAns)

test_data = []
test_data.append(data_TestSet)
test_data.append(data_TestAns)

NN = NeuralNetworkClass([10, 10, 10, 10, 5])

NN.SGD(training_data, 100, 10, 1, None)

res = NN.evaluate(test_data)

print('Result: ', res)
'''
