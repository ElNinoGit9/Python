import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class DataClass:

    def __init__(self, stock, startDate, endDate):
        import pandas as pd
        import pandas_datareader as pdr
        import datetime
        import quandl
        import fix_yahoo_finance as yf

        def get(tickers, startdate, enddate):
          def data(ticker):
            return (yf.download(ticker, start=startdate, end=enddate))
          datas = map (data, tickers)
          return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

        tickers = [stock]
        self.data = get(tickers, datetime.datetime(startDate[0], startDate[1], startDate[2]), datetime.datetime(endDate[0], endDate[1], endDate[2]))
        self.stockPrices = [n for n in self.data['Adj Close']]

class StrategyClass:
    def __init__(self, method, a, b):

        if method is 'MovingAverageCrossover':
            from MovingAverageCrossover import MovingAverageCrossover
            self.method = MovingAverageCrossover
            self.methodName = 'Moving Average Crossover'
            self.a = a
            self.b = b

    def Run(self, data_obj):

        solver = self.method

        self.money = 100
        self.share = 0

        nDays = len(data_obj.stockPrices)

        self.value = np.zeros(nDays - self.b + 1)
        self.comp  = np.zeros(nDays - self.b + 1)

        self.value[0] = 100

        self.startDate = 200
        amounts = [1, 0] # Start with 100% money
        buyAmounts, sellAmounts = solver(data_obj.stockPrices, self.a, self.b, self.b, amounts)

        self.Buy(buyAmounts, data_obj.stockPrices[self.startDate])
        self.Sell(sellAmounts, data_obj.stockPrices[self.startDate])

        for n in range(self.b, nDays):

            totalNetWorth = self.share * data_obj.stockPrices[n] + self.money
            amounts = [self.money / totalNetWorth, self.share * data_obj.stockPrices[n] / totalNetWorth]

            buyAmounts, sellAmounts = solver(data_obj.stockPrices, self.a, self.b, n, amounts)

            self.Buy(buyAmounts, data_obj.stockPrices[n])
            self.Sell(sellAmounts, data_obj.stockPrices[n])

            self.value[n - self.b + 1] = self.money + self.share * data_obj.stockPrices[n]


    def Buy(self, amount, price):

            self.share = self.share + amount * self.money/price
            self.money = (1 - amount) * self.money

    def Sell(self, amount, price):

            self.money = self.money + amount * self.share * price
            self.share = (1 - amount) * self.share

            return [amount, 1 - amount]

    def Plot(self):
        import matplotlib.pyplot as plt
        # Initialize the plot figure
        self.comp  = self.value[0] * data_obj.stockPrices / data_obj.stockPrices[self.startDate]
        fig = plt.figure()

        plt.xlabel('t (days)')
        plt.ylabel('value')
        plt.plot(self.value)
        plt.plot(self.comp)
        plt.legend(['Moving Average', 'All in'], loc=1, frameon=False)
        plt.show()

kmin = 1
kmax = 150
kstep = 2
lmin = 1
lmax = 150
lstep = 2
returns = np.zeros((len(range(kmin, kmax, kstep)), len(range(lmin, lmax, lstep))))

dat = DataClass('AAPL', [2012, 10, 1], [2018, 1, 1])
for k in range(kmin, kmax, kstep):
    for l in range(lmin, lmax, lstep):

        g = StrategyClass('MovingAverageCrossover', k, l)
        g.Run(dat)

        returns[int((k - kmin)/kstep), int((l - lmin)/lstep)] = g.value[-1]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
K, L = np.meshgrid(range(kmin, kmax, kstep), range(lmin, lmax, lstep))
ax.plot_surface(K, L, returns, cmap = 'plasma')
plt.show()
