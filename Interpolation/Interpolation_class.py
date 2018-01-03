class InterpolationClass:
    def __init__(self, N, method, f, fp, tol):
        import numpy as np
        self.N = N
        self.method = method
        self.f = f

    def Solve(self):

        if self.method is 'Lagrange':
            self.Lagrange()
        elif self.method is 'Newton':
            self.Newton()
        elif self.method is 'Chebyshev':
            self.Chebyshev()

    def Lagrange(self):
        import numpy as np
        print 'Lagrange'

    def Newton(self):
        import numpy as np
        print 'Newton'

    def Chebyshev(self):
        import numpy as np
        print 'Chebyshev'

def f (x): return x**3 - 27

Int = InterpolationClass('Lagrange', f)
Int.Solve()
print Int.x_n
