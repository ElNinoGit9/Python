class StrategyClass:
    def __init__(self, method, stock, startDate, endDate, a, b):
        import pandas as pd
        import pandas_datareader as pdr
        import datetime
        import quandl
        import numpy as np
        import fix_yahoo_finance as yf

        self.initStocks(stock, startDate, endDate)

        if method is 'MovingAverageCrossover':
            from MovingAverageCrossover import MovingAverageCrossover
            self.method = MovingAverageCrossover
            self.methodName = 'Moving Average Crossover'
            self.a = a
            self.b = b

    def initStocks(self, stock, startDate, endDate):
        import pandas as pd
        import pandas_datareader as pdr
        import datetime
        import quandl
        import numpy as np
        import fix_yahoo_finance as yf

        def get(tickers, startdate, enddate):
          def data(ticker):
            return (yf.download(ticker, start=startdate, end=enddate))
          datas = map (data, tickers)
          return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

        tickers = [stock]
        self.signals = get(tickers, datetime.datetime(startDate[0], startDate[1], startDate[2]), datetime.datetime(endDate[0], endDate[1], endDate[2]))

    def Run(self):
        print('Run strategy')
        import pandas as pd
        import pandas_datareader as pdr
        import datetime
        import quandl
        import numpy as np
        import fix_yahoo_finance as yf

        solver = self.method

        self.money = 100
        self.share = 0

        nDays = len(self.signals)
        self.value = np.zeros(nDays - self.b + 1)
        self.comp = np.zeros(nDays - self.b + 1)

        self.value[0] = 100
        self.comp[0] = 100

        self.BuySell(self.signals, 1, self.b)

        for n in range(self.b, nDays):

            indBuySell = solver(self.signals, self.a, self.b, n)
            self.BuySell(self.signals, indBuySell, n)

            self.value[n - self.b + 1] = self.money + self.share * self.signals['Adj Close'][n]
            self.comp[n - self.b + 1] = self.comp[0] * self.signals['Adj Close'][n]/self.signals['Adj Close'][self.b]

    def BuySell(self, signals, BuySellInd, n):

        if BuySellInd > 0:
            price = signals['Adj Close'][n]
            self.share = self.share + BuySellInd * self.money/price
            self.money = (1 - BuySellInd) * self.money
        elif BuySellInd < 0:
            price = signals['Adj Close'][n]
            self.money = self.money - BuySellInd * self.share * price
            self.share = (1 + BuySellInd) * self.share


    def Plot(self):
        import matplotlib.pyplot as plt
        # Initialize the plot figure
        fig = plt.figure()

        plt.xlabel('t (days)')
        plt.ylabel('value')
        plt.plot(self.value)
        plt.plot(self.comp)
        plt.legend(['Moving Average', 'All in'], loc=1, frameon=False)
        plt.show()

g = StrategyClass('MovingAverageCrossover', 'AAPL', [2012, 10, 1], [2018, 1, 1], 50, 100)

g.Run()
g.Plot()
