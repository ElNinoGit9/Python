def Milstein(X, a, b, b_x, dt, tp, Zp):
    import numpy as np

    Xn = X + a(X, tp) * dt + b(X, tp) * Zp + 1/2 * b_x(X, tp) * (pow(Zp, 2) - dt)

    return Xn
