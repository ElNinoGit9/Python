def ForwardTimeCentralSpace(u, c):
    import numpy as np

    un = np.zeros(len(u))

    un[1:-1] = (1 - 2*c) * u[1:-1] + c*u[:-2] + c * u[2:]

    return un[1:-1]
