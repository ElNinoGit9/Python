import numpy as np
import csv
class DataClass:
    def __init__(self, refSet, refAns, testSet, testAns):

        self.refSetPath = refSet
        self.refAnsPath = refAns
        self.testSetPath = testSet
        self.testAnsPath = testAns

    def readData(self):

        file1 = open(self.refSetPath, newline='')
        reader1 = csv.reader(file1)

        head = next(reader1)
        dataRefSet = [float(i) for i in head]
        for row in reader1:
                dataRefSet = np.vstack((dataRefSet, [float(i) for i in row]))


        file2 = open(self.refAnsPath, newline='')
        reader2 = csv.reader(file2)

        head = next(reader2)
        dataRefAns = [float(i) for i in head]
        for row in reader2:
                dataRefAns = np.vstack((dataRefAns, [float(i) for i in row]))

        file3 = open(self.testSetPath, newline='')
        reader3 = csv.reader(file3)

        head = next(reader3)
        dataTestSet = [float(i) for i in head]
        for row in reader3:
                dataTestSet = np.vstack((dataTestSet, [float(i) for i in row]))

        file4 = open(self.testAnsPath, newline='')
        reader4 = csv.reader(file4)

        head = next(reader4)
        dataTestAns = [float(i) for i in head]
        for row in reader4:
                dataTestAns = np.vstack((dataTestAns, [float(i) for i in row]))

        return dataRefSet, dataRefAns, dataTestSet, dataTestAns
