class gridClass:
    'Common base class for all employees'
    def __init__(self, minv, maxv, N):
        import numpy as np
        self.Nt = N
        self.dt = (maxv - minv) / float(N)
        self.t  = np.linspace(minv, maxv, N, endpoint=True)

    def plot_grid(self):
        import matplotlib.pyplot as plt
        import numpy as np
        print "plotting grid"

class problemClass:
    def __init__(self, a):
        import numpy as np

        self.mu = 1
        self.sigma = 1

    def createMatrices(self, grid_obj):
        import numpy as np

    def createData(self, grid_obj):

        def gE(x, t): return self.u_a (x, t)
        def gED(x, t): return self.uX_a(x, t)
        def gW(x, t): return self.u_a (x, t)
        def gWD(x, t): return self.uX_a(x, t)
        def f(x, t): return self.u_a(x, t)

class SchemeClass:
    def __init__(self):
        self.a = 1

    def Solve(self, problem, stoch, grid):

        self.EulerMaruyama(problem, stoch, grid)
        self.Milstein(problem, stoch, grid)
        self.RungeKutta(problem, stoch, grid)

    def EulerMaruyama(self, problem, stoch, grid):
        import numpy as np

        ys = np.zeros((grid.Nt, 1))

        def afunc(x, t): return np.ones((R, 1))
        def b_xfunc(x, t): return np.ones((1, 1))
        def bfunc(x, t): return np.ones((1, 1))
        def f(t): return np.ones((1, 1))

        Z       = stoch.grid
        a       = afunc
        b       = bfunc
        R       = stoch.R
        dt      = stoch.R * grid.dt * np.ones((1, stoch.N_sim))
        X       = np.zeros((stoch.N_sim, grid.Nt/R))
        X[:, 1] = f(0)

        for k in range(0, grid.Nt/R - 1):

            tp = stoch.grid[:, R*k]
            Xp = X[:, k]
            Zp = np.cumsum(Z[:, R*k:R*k + 1], axis = 1)

            X[:, k + 1] = X[:, k] + a(Xp, tp) * dt + b(Xp, tp) * Zp[:, -1]

        self.solution = np.mean(X, axis = 0)

    def Milstein(self, problem, stoch, grid):
        import numpy as np

        ys = np.zeros((grid.Nt, 1))

        def afunc(x, t): return 1
        def b_xfunc(x, t): return 1
        def bfunc(x, t): return 1
        def f(t): return 1

        Z      = stoch.grid
        a      = afunc
        b      = bfunc
        b_x    = b_xfunc
        R      = stoch.R
        dt     = stoch.R * grid.dt * np.ones((1, stoch.N_sim))
        X      = np.zeros((stoch.N_sim, grid.Nt/R))
        X[:,1] = f(0)

        for k in range(0, grid.Nt/R - 1):

            tp = stoch.grid[:, R*k]
            Xp = X[:, k]
            Zp = np.cumsum(Z[:, R*k:R*k + 1], axis = 1)
            X[:, k + 1] = X[:, k] + a(Xp, tp) * dt + b(Xp, tp) * Zp[:, -1] + 1/2 * b_x(Xp, tp) * (pow(Zp[:, -1],2) - dt)

    def RungeKutta(self, problem, stoch, grid):
        import numpy as np

        ys = np.zeros((grid.Nt, 1))

        def afunc(x, t): return 1
        def b_xfunc(x, t): return 1
        def bfunc(x, t): return 1
        def f(t): return 1

        Z       = stoch.grid
        a       = afunc
        b       = bfunc
        b_x     = b_xfunc
        R       = stoch.R
        dt      = stoch.R * grid.dt * np.ones((1, stoch.N_sim))
        X       = np.zeros((stoch.N_sim, grid.Nt/R))
        X[:, 1] = f(0)

        for k in range(0, grid.Nt/R - 1):

            tp = stoch.grid[:, R*k]
            Xp = X[:, k]
            Zp = np.cumsum(Z[:, R*k:R*k + 1], axis = 1)

            Yp = Xp + a(Xp, tp) * dt + b(Xp, tp) * np.sqrt(dt)
            X[:, k + 1] = X[:, k] + a(Xp, tp) * dt + b(Xp, tp) * Zp[:, -1] + 1/2 * (b(Yp, tp) - b(Xp, tp)) * (pow(Zp[:, -1], 2) - dt) * 1./np.sqrt(dt)

    def plotSolution(self, grid):
        import matplotlib.pyplot as plt

        plt.plot(grid.t, self.solution)
        plt.show()

class stochasticClass:
    def __init__(self, N_sim, R, grid):
        import numpy as np
        self.N_sim = N_sim
        self.grid = np.sqrt(grid.dt) * np.random.normal(0, 1, self.N_sim * grid.Nt)
        self.grid = np.reshape(self.grid, (self.N_sim, grid.Nt))
        self.R = R

g = gridClass(0, 1, 1000)
g.plot_grid()

p = problemClass(1)
p.createMatrices(g)
p.createData(g)

st = stochasticClass(100, 1, g)

s = SchemeClass()
s.Solve(p, st, g)

s.plotSolution(g)
