import yfinance as yf

# Fetch historical data for a stock
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Calculate moving averages
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Buy/Sell signals
data['Signal'] = 0
data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1
data.loc[data['SMA50'] <= data['SMA200'], 'Signal'] = -1

print(data[['Close', 'SMA50', 'SMA200', 'Signal']].tail())