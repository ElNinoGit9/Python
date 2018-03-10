def Secant(interval, N, f):
    import numpy as np
    print 'Secant'

    upper = interval[1]
    lower = interval[0]
    X = []

    for n in range(1, N):

        middle = (lower * f(upper) - upper * f(lower))/(f(upper) - f(lower))

        lower = upper
        upper = middle
        X.append(middle)

    return X
