def EulerMaruyama(X, a, b, b_x, dt, tp, Zp):
    import numpy as np

    Xn = X + a(X, tp) * dt + b(X, tp) * Zp

    return Xn
