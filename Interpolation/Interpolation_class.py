class InterpolationClass:
    def __init__(self, x0, y0, xn, method):
        import numpy as np
        self.x0 = x0
        self.y0 = y0
        self.xn = xn
        self.method = method

    def Solve(self):

        if self.method is 'Constant':
            from Constant import Constant
            self.yn = Constant(self.x0, self.y0, self.xn)
        elif self.method is 'Linear':
            from Linear import Linear
            self.yn = Linear(self.x0, self.y0, self.xn)
        elif self.method is 'Polynomial':
            from Polynomial import Polynomial
            self.yn = Polynomial(self.x0, self.y0, self.xn)
        elif self.method is 'Splines':
            from Splines import Splines
            self.yn = Splines(self.x0, self.y0, self.xn)

import matplotlib.pyplot as plt
import numpy as np
from math import *

def f(x): return np.sin(pi*x)
x0 = np.linspace(-4,4,20, endpoint = True)
y0 = f(x0)
xn = np.linspace(-4,4,100, endpoint = True)

Int = InterpolationClass(x0, y0, xn, 'Polynomial')
Int.Solve()

plt.plot(Int.xn, Int.yn)
plt.show()

plt.plot(Int.xn, f(Int.xn))
plt.show()

plt.plot(Int.xn, f(Int.xn), Int.xn, Int.yn)
plt.show()
