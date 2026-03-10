import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt 
import json 
import pandas_ta as ta
from scipy import stats

with open('companies.json', 'r') as f:
    name_to_ticker = json.load(f)

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

for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info
    print(f"\n--- {ticker} ---")
    print(f"Price: ${info['currentPrice']}")
    print(f"Market Cap: ${info['marketCap']:,}")
    print(f"52W High: ${info['fiftyTwoWeekHigh']}")
    print(f"52W Low: ${info['fiftyTwoWeekLow']}")
    print(f'Beta: {info.get('beta','N/A')}')
    print(f"Dividend Yield: {info.get('dividendYield','N/A')}")
    print(f"Price to Book: {info.get("priceToBook",'N/A')}")
    print(f"P/E Ratio: {info.get('trailingPE','N/A')}")

all_data = {}
market = yf.Ticker('^GSPC')
market_history = market.history(period='5y')
market_df = pd.DataFrame(market_history)


for ticker in tickers:
    stock = yf.Ticker(ticker)
    history = stock.history(period='5y')
    df = pd.DataFrame(history)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    print(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
    df['Stock_Return'] = df['Close'].pct_change()
    market_df['Market_Return'] = market_df['Close'].pct_change() 
    combined = pd.concat([df['Stock_Return'], market_df['Market_Return']], axis=1).dropna()
    combined.columns = ['Stock_Return', 'Market_Return']
    slope, intercept, r_value, p_value, std_err = stats.linregress(
    combined['Market_Return'],
    combined['Stock_Return']
    )
    r_squared  = r_value ** 2
    print(f"Beta (calculated): {slope:.4f}")
    print(f"R² (Systematic Risk): {r_squared:.4f}")
    print(f"Unsystematic Risk:{(1-r_squared):.4f}")

    all_data[ticker] = df

for ticker in tickers:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    all_data[ticker][['Close', 'MA200','MA50']].plot(ax = ax1 , title = ticker)
    all_data[ticker]['RSI'].plot(ax = ax2 , title = 'RSI', color = 'purple')
    
    ax2.axhline(y=70, color='red', linestyle='--')    
    ax2.axhline(y=30, color='green', linestyle='--')  
    
    plt.tight_layout()
    plt.show()
