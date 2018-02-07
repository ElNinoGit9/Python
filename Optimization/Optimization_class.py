class OptimizationClass:
    def __init__(self, minv, maxv, tol, method, func):
        import numpy as np
        self.minv = minv
        self.maxv = maxv
        self.tol = tol
        self.method = method
        self.func = f

    def optimize(self):

        if self.method is 'GoldenSearch':
            self.GoldenSearch()
        elif self.method is 'Fibonacci':
            self.Fibonacci()
        elif self.method is 'QuadraticInterpolation':
            self.QuadraticInterpolation()
        elif self.method is 'NelderMead':
            self.NelderMead()
        elif self.method is 'SteepestDescent':
            self.SteepestDescent()
        elif self.method is 'Newton':
            self.Newton()

    def GoldenSearch(self):
        import numpy as np
        print 'GoldenSearch'

        a   = self.minv
        b   = self.maxv
        tol = self.tol
        r1  = (np.sqrt(5) - 1)/2.
        r2  = r1*r1
        h   = a - b
        ya  = self.func(a)
        yb  = self.func(b)
        c   = a + r2*h
        d   = a + r1*h
        yc  = self.func(c)
        yd  = self.func(d)

        A = []
        B = []
        C = []
        D = []
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)

        k = 0

        while (np.abs(yb - ya) > tol) | (h > tol):
            k = k + 1

            if (yc < yd):
                b = d
                yb = yd
                d = c
                yd = yc
                h = b - a
                c = a + r2*h
                yc = self.func(c)
            else:
                a = c
                ya = yc
                c = d
                yc = yd
                h = b - a
                d = a + r1*h
                yd = self.func(d)

            A.append(a)
            B.append(b)
            C.append(c)
            D.append(d)

            dp = np.abs(b - a)
            dy = np.abs(yb - ya)
            p = a
            yp = ya

            if (yb < ya):
                p = b
                yp = yb

            G = [np.transpose(A), np.transpose(B), np.transpose(C), np.transpose(A)]
            S = [p, yp]
            E = [dp, dy]

        self.min  = S[1]
        self.xmin = S[0]

    def Fibonacci(self):
        import numpy as np
        print 'Fibonacci'

    def QuadraticInterpolation(self):
        import numpy as np
        print 'QuadraticInterpolation'

    def NelderMead(self):
        import numpy as np
        print 'NelderMead'

    def SteepestDescent(self):
        import numpy as np
        print 'SteepestDescent'

    def Newton(self):
        import numpy as np
        print 'NewtonMethod'

import numpy as np
def f (x): return np.sin(2*np.pi*x)

Opt = OptimizationClass(0, .9, 1e-8, 'GoldenSearch', f)
Opt.optimize()
print 'minimum =', Opt.min, 'at x =', Opt.xmin
