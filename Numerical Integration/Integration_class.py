class IntegrationClass:
    def __init__(self, minv, maxv, N, method, func):
        import numpy as np
        self.minv = minv
        self.maxv = maxv
        self.N = N
        self.method = method
        self.func = f
        self.dx = (maxv-minv)/float(N)

    def integrate(self):

        if self.method is 'Midpoint':
            self.MidpointRule()
        elif self.method is 'Trapeziodal':
            self.TrapeziodalRule()
        elif self.method is 'Simpson':
            self.SimpsonsRule()
        elif self.method is 'Boole':
            self.BoolesRule()
        elif self.method is 'Legendre':
            self.GaussLegendre()
        elif self.method is 'Lobatto':
            self.GaussLobatto()
        elif self.method is 'Romberg':
            self.Romberg()

    def MidpointRule(self):
        import numpy as np
        print 'Midpoint rule'

        gridMid = np.linspace(self.minv + self.dx/2., self.maxv - self.dx/2., self.N, endpoint=True)

        self.I = self.dx * (np.sum(self.func(gridMid[0:])))

    def TrapeziodalRule(self):
        import numpy as np
        print 'Trapeziodal rule'

        self.grid = np.linspace(self.minv, self.maxv, self.N + 1, endpoint=True)

        self.I = self.dx * (np.sum(self.func(self.grid[1:-1])) + self.func(self.grid[0])/2. + self.func(self.grid[-1])/2.)

    def SimpsonsRule(self):
        import numpy as np
        print 'Simpsons rule'

        self.grid = np.linspace(self.minv, self.maxv, self.N + 1, endpoint=True)

        self.I = 2.*self.dx/6. * (self.func(self.grid[0]) + 4*np.sum(self.func(self.grid[1:-1:2])) + 2*np.sum(self.func(self.grid[2:-1:2])) + self.func(self.grid[-1]))

    def BoolesRule(self):
        import numpy as np
        print 'Booles rule'

        self.grid = np.linspace(self.minv, self.maxv, self.N + 1, endpoint=True)

        self.I = 2.*self.dx/45. * (7*self.func(self.grid[0]) + 32*np.sum(self.func(self.grid[1:-1:4])) \
        + 12*np.sum(self.func(self.grid[2:-1:4])) + 32*np.sum(self.func(self.grid[3:-1:4])) \
        + 14*np.sum(self.func(self.grid[4:-1:4])) + 7*self.func(self.grid[-1]))

    def GaussLegendre(self):
        import numpy as np
        print 'Gauss-Legendre'

        if self.N == 1:
            p = np.array([0])
            w = np.array([2])
        elif self.N == 2:
            p = np.array([0.57735, -.57735])
            w = np.array([1, 1])
        elif self.N == 3:
            p = np.array([0, 0.774597, -0.774597])
            w = np.array([0.888889, 0.555556, 0.555556])
        elif self.N == 4:
            p = np.array([0.339981, -0.339981, 0.861136, -0.861136])
            w = np.array([0.652145, 0.652145, 0.347855, 0.347855])
        elif self.N == 5:
            p = np.array([0, 0.538469, -0.538469, 0.90618, -0.90618])
            w = np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927])
        else:
            print 'number of points undefined'

        a = (self.maxv - self.minv)/2.
        b = (self.maxv + self.minv)/2.
        self.I = a * np.dot(self.func(a*p + b), w)

    def GaussLobatto(self):
        import numpy as np
        print 'Gauss-Lobatto'

        if self.N == 3:
            p = np.array([0, 1, -1])
            w = np.array([4/3., 1/3., 1/3.])
        elif self.N == 4:
            p = np.array([np.sqrt(1/5.), -np.sqrt(1/5.), 1, -1])
            w = np.array([5/6., 5/6., 1/6., 1/6.])
        elif self.N == 5:
            p = np.array([0, np.sqrt(3./7.), -np.sqrt(3./7.), 1, -1])
            w = np.array([32/45., 49/90., 49/90., 1/10., 1/10.])
        elif self.N == 6:
            p = np.array([np.sqrt(1/3. - 2*np.sqrt(7)/21.), -np.sqrt(1/3. - 2*np.sqrt(7)/21.), np.sqrt(1/3. + 2*np.sqrt(7)/21.), -np.sqrt(1/3. + 2*np.sqrt(7)/21.), 1, -1])
            w = np.array([(14 + np.sqrt(7))/30., (14 + np.sqrt(7))/30., (14 - np.sqrt(7))/30., (14 - np.sqrt(7))/30., 1/15., 1/15.])
        elif self.N == 7:
            p = np.array([0, np.sqrt(5/11. - 2/11.*np.sqrt(5/3.)), -np.sqrt(5/11. - 2/11.*np.sqrt(5/3.)), np.sqrt(5/11. + 2/11.*np.sqrt(5/3.)), -np.sqrt(5/11. + 2/11.*np.sqrt(5/3.)), 1, -1])
            w = np.array([256/525., (124 + 7*np.sqrt(15))/350., (124 + 7*np.sqrt(15))/350., (124 - 7*np.sqrt(15))/350., (124 - 7*np.sqrt(15))/350., 1/21., 1/21.])
        else:
            print 'number of points undefined'

        a = (self.maxv - self.minv)/2.
        b = (self.maxv + self.minv)/2.
        self.I = a * np.dot(self.func(a*p + b), w)

    def Romberg(self):
        import numpy as np
        import math
        print 'Romberg integration'

        tol = 0.001
        M = 1
        h = self.maxv - self.minv
        err = 1
        J = -1
        R = np.zeros((4,4))
        R[0, 0] = h * (self.func(self.maxv) + self.func(self.minv)) / 2.

        while ((err > tol) & (J < 2)) | (J < 2):
            J = J + 1
            h = h/2.
            s = 0

            for p in range(1, M+1):

                x = self.minv + h*(2*p - 1)
                s = s + self.func(x)

            R[J+1, 0] = R[J, 0]/2. + h*s
            M = 2*M

            for K in range(0, J+1):
                R[J+1, K+1] = R[J+1, K] + (R[J+1, K] - R[J, K])/(math.pow(4.0, K+1) - 1)

            err = np.abs(R[J, J] - R[J+1, J+1])

        self.I = R[J+1, J+1]



def f (x): return x*x*x*x

Int = IntegrationClass(0, 1, 10, 'Romberg', f)
Int.integrate()
print Int.I
