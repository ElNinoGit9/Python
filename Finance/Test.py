import pandas as pd
import pandas_datareader as pdr
import quandl
import fix_yahoo_finance as yf
data = yf.download("SPY", start="2017-01-01", end="2017-04-30")
import datetime
import numpy
import matplotlib.pyplot as plt

def get(tickers, startdate, enddate):
  def data(ticker):
    return (yf.download(ticker, start=startdate, end=enddate))
  datas = map (data, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'LATO-B.ST', 'LUND-B.ST']
all_data = get(tickers, datetime.datetime(2012, 10, 1), datetime.datetime(2018, 1, 1))


# Isolate the `Adj Close` values and transform the DataFrame
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')

# Calculate the daily percentage change for `daily_close_px`
daily_pct_change = daily_close_px.pct_change()

# Plot the distributions
# daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))

# Show the resulting plot
plt.show()
#aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01")
# Plot the closing prices for `aapl`
a = all_data[['Close']].reset_index().pivot('Date', 'Ticker', 'Close')
print(a)
a.plot()
plt.show()
# Show the plot
#plt.show()
