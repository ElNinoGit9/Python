import numpy as np
import csv
class DataClass:
    def __init__(self, data_path):

        self.dataPath = data_path

    def readData(self):

        file1 = open(self.dataPath, newline='')
        reader1 = csv.reader(file1)

        head = next(reader1)
        head2 = next(reader1)
        dataSet = []
        for row in reader1:
                # row = [Date, Open, High, Low, Close, Volume, Adj. Close]
                tmp = row[1].split(';')[1]
                #print(tmp)
                tmp2 = float(tmp)
                dataSet.append(tmp2)
        return dataSet
