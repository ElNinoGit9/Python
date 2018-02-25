class ODEClass:
    def __init__(self, minv, maxv, N, method, func, f0):
        import numpy as np
        self.minv = minv
        self.maxv = maxv
        self.N = N
        self.method = method
        self.func = f
        self.f0 = f0

    def Solve(self):

        if self.method is 'Euler':
            self.Euler()
        elif self.method is 'Heun':
            self.Heun()
        elif self.method is 'Taylor':
            self.Taylor()
        elif self.method is 'RungeKutta4':
            self.RungeKutta4()
        elif self.method is 'RungeKuttaFehlberg':
            self.RungeKuttaFehlberg()
        elif self.method is 'ABMM':
            self.ABMM()
        elif self.method is 'MilneSimpson':
            self.MilneSimpson()
        elif self.method is 'Hamming':
            self.Hamming()

    def Euler(self):
        import numpy as np
        print 'Euler'

        h = (self.maxv - self.minv)/float(self.N)
        y = np.zeros((self.N + 1, 1))
        y = y[:,0]
        t = np.linspace(self.minv, self.maxv, self.N + 1, endpoint = True)
        y[0] = self.f0

        for j in range(0, self.N):
            y[j+1] = y[j] + h*self.func(t[j], y[j])

        self.Solution = [np.transpose(t), np.transpose(y)]


    def Heun(self):
        import numpy as np
        print 'Heun'

    def Taylor(self):
        import numpy as np
        print 'Taylor'

    def RungeKutta4(self):
        import numpy as np
        print 'RungeKutta4'

    def RungeKuttaFehlberg(self):
        import numpy as np
        print 'RungaKuttaFehlberg'

    def ABMM(self):
        import numpy as np
        print 'ABMM'

    def MilneSimpson(self):
        import numpy as np
        print 'MilneSimpson'

    def Hamming(self):
        import numpy as np
        print 'Hamming'

def f (t, y): return y + t

Ode = ODEClass(0, 1, 5, 'Euler', f, 0)
Ode.Solve()
print Ode.Solution[1]
