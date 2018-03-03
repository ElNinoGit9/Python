def MidpointRule(interval, f, N):
    import numpy as np
    print 'Midpoint rule'

    h = (interval[1] - interval[0])/float(N)

    grid = np.linspace(interval[0] + h/2., interval[1] - h/2., N, endpoint=True)

    I = h * (np.sum(f(grid[0:])))

    return I
