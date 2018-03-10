def NewtonRaphson(interval, N, f, fp):
    import numpy as np
    print 'Newton Rhapson'

    xp = (interval[1] + interval[0])/2
    X = []

    for n in range(1, N):

        xp = xp - f(xp)/fp(xp)
        X.append(xp)

    return X
