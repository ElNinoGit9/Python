def MovingAverageCrossover(stockPrices, a, b, n, amounts):
    import numpy as np

    short = np.sum(stockPrices[n-a:n])/a
    long = np.sum(stockPrices[n-b:n])/b
    bestInd = np.argmax([long, short])

    if amounts[bestInd] == 1: # If the best the current setting
        return 0, 0 # Do nothing
    elif bestInd == 1: # If not, and we want to go long
        return 1, 0 # Buy
    else: # Else, go short
        return 0, 1 # Sell
