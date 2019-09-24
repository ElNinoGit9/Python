class RootFindingClass:
    def __init__(self, minv, maxv, N, method, f, fp, tol):
        import numpy as np
        self.minv = float(minv)
        self.maxv = float(maxv)
        self.N = N
        self.method = method
        self.f = f
        self.fp = fp
        self.tol = tol

    def Solve(self):

        if self.method is 'Bisection':
            from Bisection import Bisection
            self.x_n = Bisection([self.minv, self.maxv], self.N, self.f)
        elif self.method is 'RegulaFalsi':
            from RegulaFalsi import RegulaFalsi
            self.x_n = RegulaFalsi([self.minv, self.maxv], self.N, self.f)
        elif self.method is 'Secant':
            from Secant import Secant
            self.x_n = Secant([self.minv, self.maxv], self.N, self.f)
        elif self.method is 'NewtonRaphson':
            from NewtonRaphson import NewtonRaphson
            self.x_n = NewtonRaphson([self.minv, self.maxv], self.N, self.f, self.fp)
        elif self.method is 'Brent':
            from Brent import Brent
            self.x_n = Brent([self.minv, self.maxv], self.N, self.f, self.tol)

#def f (x): return x**3 - 27
#def fp(x): return 3*x**2
#
#Eqn = RootFindingClass(1, 10, 10, 'Brent', f, fp, 1e-4)
#Eqn.Solve()
#
#print Eqn.x_n
