import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

stock = yf.Ticker("AAPL")

print(stock.info['currentPrice'])
print(stock.info['marketCap'])
print(stock.info['sector'])
print(stock.info['fiftyTwoWeekHigh'])
print(stock.info['fiftyTwoWeekLow'])

history = stock.history(period='5y' )

df = pd.DataFrame(history)
df = df.drop(columns=['Open','Dividends','Stock Splits'])
print(df)

df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

df[['Close', 'MA50', 'MA200']].plot(kind='line')
plt.title("Apple Stock - Moving Average")
plt.xlabel("Date")
plt.ylabel("$")
plt.show()

desc = df['Close'].describe()
print(desc)

df.to_csv('Applestock.csv')