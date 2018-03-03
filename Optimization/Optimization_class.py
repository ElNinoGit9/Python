class OptimizationClass:
    def __init__(self, interval, tol, N, method, func):
        import numpy as np
        self.minv = interval[0]
        self.maxv = interval[1]
        self.tol = tol
        self.N = N
        self.method = method
        self.func = f
        self.funcp = fp

    def optimize(self):

        if self.method is 'GoldenSearch':
            from GoldenSearch import GoldenSearch
            [self.min, self.xmin] = GoldenSearch([self.minv, self.maxv], self.func, self.tol)
        elif self.method is 'Fibonacci':
            from Fibonacci import Fibonacci
            [self.min, self.xmin] = Fibonacci([self.minv, self.maxv], self.func, self.tol)
        elif self.method is 'QuadraticInterpolation':
            from QuadraticInterpolation import QuadraticInterpolation
            [self.min, self.xmin] = QuadraticInterpolation((self.minv + self.maxv)/2., self.func, self.tol)
        elif self.method is 'NelderMead':
            from NelderMead import NelderMead
            [self.min, self.xmin] = NelderMead(self.func, self.tol, self.N)
        elif self.method is 'SteepestDescent':
            from SteepestDescent import SteepestDescent
            [self.min, self.xmin] = SteepestDescent([self.minv, self.maxv], self.func, self.funcp, self.tol, self.N)

import numpy as np
'''def f (x, y): return (x-2)*(x-2) + (y + 1)*(y + 1)'''
def f (x): return (x-2)**4 - 1
def fp(x): return 4*(x-2)**3

Opt = OptimizationClass([0, 2.1], 1e-2, 20, 'GoldenSearch', f)
Opt.optimize()

print 'minimum =', Opt.min, 'at x =', Opt.xmin
