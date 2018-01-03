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
            self.Bisection()
        elif self.method is 'RegulaFalsi':
            self.RegulaFalsi()
        elif self.method is 'Secant':
            self.Secant()
        elif self.method is 'NewtonRaphson':
            self.NewtonRaphson()
        elif self.method is 'Brent':
            self.Brent()

    def Bisection(self):
        import numpy as np
        print 'Bisection'

        upper = self.maxv
        lower = self.minv
        X = []

        for n in range(1, self.N):

            middle = (upper + lower)/2

            if self.f(middle)*self.f(upper) > 0:
                upper = middle
            else:
                lower = middle

            X.append(middle)

        self.x_n = X

    def RegulaFalsi(self):
        import numpy as np
        print 'Regula Falsi'

        upper = self.maxv
        lower = self.minv
        X = []

        for n in range(1, self.N):

            middle = (lower*self.f(upper) - upper*self.f(lower))/(self.f(upper) - self.f(lower))

            if self.f(middle)*self.f(upper) > 0:
                upper = middle
            else:
                lower = middle

            X.append(middle)

        self.x_n = X

    def Secant(self):
        import numpy as np
        print 'Secant'

        upper = self.maxv
        lower = self.minv
        X = []

        for n in range(1, self.N):

            middle = (lower*self.f(upper) - upper*self.f(lower))/(self.f(upper) - self.f(lower))

            lower = upper
            upper = middle
            X.append(middle)

        self.x_n = X

    def NewtonRaphson(self):
        import numpy as np
        print 'Newton Rhapson'

        xp = (self.maxv + self.minv)/2
        X = []

        for n in range(1, self.N):

            xp = xp - self.f(xp)/self.fp(xp)
            X.append(xp)

        self.x_n = X

    def Brent(self):
        import numpy as np
        print 'Brent'

        a = self.maxv
        b = self.minv
        X = []
        c = a
        d = 0
        m = True

        for n in range(1, self.N):

            if (self.f(c) != self.f(a)) and (self.f(c) != self.f(b)):
                s = a*f(b)*f(c)/((f(a) - f(b))*(f(a) - f(c))) + b*f(a)*f(c)/((f(b) - f(a))*(f(b) - f(c))) + c*f(a)*f(b)/((f(c) - f(a))*(f(c) - f(b)))
            else:
                s = (b*self.f(a) - a*self.f(b))/(self.f(a) - self.f(b))

            if (((s < (3.*a + b)/4.) and (s > b)) \
            or (m and (np.abs(s-b) >= np.abs(b-c)/2.)) \
            or ((not m) and (np.abs(s-b) >= np.abs(c-d)/2.)) \
            or (m and (np.abs(b-c) < self.tol)) \
            or ((not m) and (np.abs(c-d) < self.tol))):
                s = (a+b)/2.
                m = True
            else:
                m = False

            d = c
            c = b
            if f(a)*f(s) < 0:
                b = s
            else:
                a = s

            if np.abs(f(a)) < np.abs(f(b)):
                tmp = a
                a = b
                b = tmp

            X.append(s)

        self.x_n = X

def f (x): return x**3 - 27
def fp(x): return 3*x**2

Eqn = RootFindingClass(1, 10, 10, 'RegulaFalsi', f, fp, 1e-4)
Eqn.Solve()
print Eqn.x_n
