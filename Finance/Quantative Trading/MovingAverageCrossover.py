def MovingAverageCrossover(signals, a, b, n):
    import numpy as np

    short = np.sum(signals['Adj Close'][n-a:n])/float(a)
    long = np.sum(signals['Adj Close'][n-b:n])/float(b)

    if long < short:
        return 1
    else:
        return -1
