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
        h   = b-a
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

        def fib(x):

            if x < 3:
                return 1
            else:
                return fib(x - 1) + fib(x - 2)


        a = self.minv
        b = self.maxv
        e = 0
        i = 1
        F = 1

        while F <= (b-a)/self.tol:
            F = fib(i)
            i = i + 1

        n = i - 1
        A = np.zeros((1, n-2))
        B = np.zeros((1, n-2))
        A[0,0] = a
        B[0,0] = b
        c = A[0,0] + (float(fib(n-2))/fib(n)) * (B[0,0] - A[0,0])
        d = A[0,0] + (float(fib(n-1))/fib(n)) * (B[0,0] - A[0,0])
        k = 0

        while k < n - 3:

            if self.func(c) > self.func(d):
                A[0, k + 1] = c
                B[0, k + 1] = B[0, k]
                c = d
                d = A[0, k+1] + (float(fib(n - k - 1))/fib(n - k)) * (B[0, k+1] - A[0, k+1])

            else:
                A[0, k+1] = A[0, k]
                B[0, k+1] = d
                d = c
                c = A[0, k+1] + (float(fib(n-k-2))/fib(n-k)) * (B[0, k+1] - A[0, k+1])

            k = k + 1

        if self.func(c) > self.func(d):
            A[0, n-3] = c
            B[0, n-3] = B[0, n-4]
            c = d
            d = A[0, n-3] + (0.5 + e) * (B[0, n-3] - A[0, n-3])
        else:
            A[0, n-3] = A[0, n-4]
            B[0, n-3] = d
            d = c
            c = A[0, n-3] + (0.5 - e) * (B[0, n-3] - A[0, n-3])

        if self.func(c) > self.func(d):
            a = c
            b = B[0, n-3]
        else:
            a = A[0, n-3]
            b = d

        self.xmin = (a+b)/2
        self.min = self.func((a+b)/2)

    def QuadraticInterpolation(self):
        import numpy as np
        print 'QuadraticInterpolation'

        p0 = self.minv
        maxj = 20
        maxk = 30
        big = 1e6
        err = 1
        k = 1
        P = []
        P.append(p0)
        cond = 0
        h = 1

        if (np.abs(p0) > 1e4):

            h = float(np.abs(p0))/1e4

        while (k < maxk) & (err > self.tol) & (cond != 5):

            f1 = (self.func(p0 + 1e-5) - self.func(p0 - 1e-5)) / float(2e-5)

            if (f1 > 0):
                h = -np.abs(h)

            p1 = p0 + h
            p2 = p0 + 2*h
            pmin = p0 + 0
            y0 = self.func(p0)
            y1 = self.func(p1)
            y2 = self.func(p2)
            ymin = y0 + 0
            cond = 0
            j = 0

            while (j < maxj) & (np.abs(h) > self.tol) & (cond == 0):

                if (y0 <= y1):
                    p2 = p1 + 0
                    y2 = y1 + 0
                    h = h/2.
                    p1 = p0 + h
                    y1 = self.func(p1)
                else:
                    if (y2 < y1):
                        p1 = p2 + 0
                        y1 = y2 + 0
                        h = 2*h
                        p2 = p0 + 2*h
                        y2 = self.func(p2)
                    else:
                        cond = -1

                j = j + 1

                if (np.abs(h) > big) | (np.abs(p0) > big):
                    cond = 5

            if cond == 5:
                pmin = p1 + 0
                ymin = self.func(p1)
            else:
                d = 4 * y1 - 2 * y0 - 2 * y2

                if (d < 0):
                    hmin = h * (4 * y1 - 3 * y0 - y2)/float(d)
                else:
                    hmin = h/3.
                    cond = 4

                pmin = p0 + hmin
                ymin = self.func(pmin)
                h = np.abs(h)
                h0 = np.abs(hmin)
                h1 = np.abs(hmin - h)
                h2 = np.abs(hmin - 2*h)

                if (h0 < h):
                    h = h0 + 0
                if (h1 < h):
                    h = h1 + 0
                if (h2 < h):
                    h = h2 + 0
                if (h == 0):
                    h = hmin + 0
                if (h < self.tol):
                    cond = 1
                if (np.abs(h) > big) | (np.abs(pmin) > big):
                    cond = 5


                e0 = np.abs(y0 - ymin)
                e1 = np.abs(y1 - ymin)
                e2 = np.abs(y2 - ymin)

                if (e0 != 0) & (e0 < err):
                    err = e0 + 0
                if (e1 != 0) & (e1 < err):
                    err = e1 + 0
                if (e2 != 0) & (e2 < err):
                    err = e2 + 0
                if (e0 != 0) & (e1 == err) & (e2 == 0):
                    error = 0
                if err < self.tol:
                    cond = 2

                p0 = pmin + 0
                k = k + 1
                P.append(p0)

            if (cond == 2) & (h < self.tol):
                cond = 3

        p = p0 + 0
        dp = h + 0
        dy = err + 0
        Opt.xmin = p + 0
        Opt.min = self.func(p)

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
def f (x): return np.sin(np.pi*x)

Opt = OptimizationClass(0, .9, 1e-6, 'QuadraticInterpolation', f)
Opt.optimize()
print 'minimum =', Opt.min, 'at x =', Opt.xmin
