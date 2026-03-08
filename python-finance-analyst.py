import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

name_to_ticker = {
    'APPLE': 'AAPL',
    'GOOGLE': 'GOOGL',
    'MICROSOFT': 'MSFT',
    'TESLA': 'TSLA',
    'AMAZON': 'AMZN',
    'META': 'META',
    'NETFLIX': 'NFLX',
    'NVIDIA': 'NVDA',
    'AMD': 'AMD',
    'INTEL': 'INTC',
    'IBM': 'IBM',
    'ORACLE': 'ORCL',
    'ADOBE': 'ADBE',
    'SPOTIFY': 'SPOT',
    'UBER': 'UBER',
    'AIRBNB': 'ABNB',
    'PAYPAL': 'PYPL',
    'JPMORGAN': 'JPM',
    'GOLDMAN': 'GS',
    'VISA': 'V',
    'MASTERCARD': 'MA',
    'BLACKROCK': 'BLK',
    'PFIZER': 'PFE',
    'JOHNSON': 'JNJ',
    'MODERNA': 'MRNA',
    'EXXON': 'XOM',
    'CHEVRON': 'CVX',
    'COCACOLA': 'KO',
    'PEPSI': 'PEP',
    'MCDONALDS': 'MCD',
    'NIKE': 'NKE',
    'DISNEY': 'DIS',
    'WALMART': 'WMT',
    'STARBUCKS': 'SBUX',
}

while True:
    try:
        num = int(input("Enter How many stocks do you want to analyze?: "))
        break
    except ValueError:
        print("Please enter a valid number!")

tickers = []

for i in range(num):
    while True:
        name = input(f"Enter company {i+1} name or ticker: ").upper()
        ticker = name_to_ticker.get(name, name)
        try:
            stock = yf.Ticker(ticker)
            _ = stock.info['currentPrice']
            tickers.append(ticker)
            break
        except:
            print(f"'{name}' not found! Please enter a valid company name.")

print(tickers)

for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info
    print(f"\n--- {ticker} ---")
    print(f"Price: ${info['currentPrice']}")
    print(f"Market Cap: ${info['marketCap']:,}")
    print(f"52W High: ${info['fiftyTwoWeekHigh']}")
    print(f"52W Low: ${info['fiftyTwoWeekLow']}")

all_data = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    history = stock.history(period='5y')
    df = pd.DataFrame(history)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    all_data[ticker] = df

fig, axes = plt.subplots(1, len(tickers), figsize=(15, 5))
for i, ticker in enumerate(tickers):
    all_data[ticker][['Close', 'MA50', 'MA200']].plot(ax=axes[i], title=ticker)

plt.tight_layout()
plt.show()
