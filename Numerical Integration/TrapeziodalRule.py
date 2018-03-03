def TrapeziodalRule(interval, f, N):
    import numpy as np
    print 'Trapeziodal rule'

    h = (interval[1] - interval[0])/float(N)

    grid = np.linspace(interval[0], interval[1], N + 1, endpoint=True)

    I = h * (np.sum(f(grid[1:-1])) + f(grid[0])/2. + f(grid[-1])/2.)

    return I
