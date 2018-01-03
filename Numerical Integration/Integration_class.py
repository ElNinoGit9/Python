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

def f (x): return x*x*x*x

Int = IntegrationClass(0, 1, 12, 'Boole', f)
Int.integrate()
print Int.I
