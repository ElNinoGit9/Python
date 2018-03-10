class NumberSequenceClass:
    def __init__(self, N, method):
        import numpy as np
        self.N = N
        self.method = method

    def GetSequence(self):

        if self.method is 'Fibonacci':
            from Fibonacci import Fibonacci
            self.seq = Fibonacci(self.N)
        elif self.method is 'Farey':
            from Farey import Farey
            self.seq = Farey(self.N)

Num = NumberSequenceClass(10, 'Farey')
Num.GetSequence()

print Num.seq
