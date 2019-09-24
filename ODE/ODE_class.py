class ODEClass:
    def __init__(self, minv, maxv, N, method, ft, f0):
        import numpy as np
        self.minv = minv
        self.maxv = maxv
        self.N = N
        self.method = method
        self.funct = ft
        self.f0 = f0

    def Solve(self):

        if self.method is 'EulerForward':
            from EulerForward import EulerForward
            [self.t, self.y] = EulerForward([self.minv, self.maxv], self.f0, self.funct, self.N)
        if self.method is 'EulerBackward':
            from EulerBackward import EulerBackward
            [self.t, self.y] = EulerBackward([self.minv, self.maxv], self.f0, self.funct, self.N)
        if self.method is 'CrankNicolson':
            from CrankNicolson import CrankNicolson
            [self.t, self.y] = CrankNicolson([self.minv, self.maxv], self.f0, self.funct, self.N)
        elif self.method is 'Heun':
            from Heun import Heun
            [self.t, self.y] = Heun([self.minv, self.maxv], self.f0, self.funct, self.N)
        elif self.method is 'RungeKutta4':
            from RungeKutta4 import RungeKutta4
            [self.t, self.y] = RungeKutta4([self.minv, self.maxv], self.f0, self.funct, self.N)

def ft(t, y): return y + t

Ode = ODEClass(0, 1, 500, 'EulerForward', ft, 0)
Ode.Solve()
print(Ode.t, Ode.y)
