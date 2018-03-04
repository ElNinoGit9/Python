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

        if self.method is 'Euler':
            from Euler import Euler
            [self.t, self.y] = Euler([self.minv, self.maxv], self.f0, self.funct, self.N)
        elif self.method is 'Heun':
            from Heun import Heun
            [self.t, self.y] = Heun([self.minv, self.maxv], self.f0, self.funct, self.N)
        elif self.method is 'RungeKutta4':
            from RungeKutta4 import RungeKutta4
            [self.t, self.y] = RungeKutta4([self.minv, self.maxv], self.f0, self.funct, self.N)

def ft(t, y): return y + t

Ode = ODEClass(0, 1, 500, 'MidpointMethod', ft, 0)
Ode.Solve()
print Ode.t, Ode.y
