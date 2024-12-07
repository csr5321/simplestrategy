import backtrader as bt

class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=50), bt.ind.SMA(period=200)
        self.signal_add(bt.SIGNAL_LONG, sma1 > sma2)

cerebro = bt.Cerebro()
data = bt.feeds.PandasData(dataname=yf.download("AAPL", "2020-01-01", "2023-01-01"))
cerebro.adddata(data)
cerebro.addstrategy(SmaCross)
cerebro.run()
cerebro.plot()
