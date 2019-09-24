class NumberAlgorithmClass:
    def __init__(self, num, method):
        import numpy as np
        self.number = num
        self.method = method

    def GetSequence(self):

        if self.method is 'LuhnAlgorithm':
            from LuhnAlgorithm import LuhnAlgorithm
            self.check = LuhnAlgorithm(self.number)

Num = NumberAlgorithmClass(7905125121, 'LuhnAlgorithm')
Num.GetSequence()

print(Num.check)
