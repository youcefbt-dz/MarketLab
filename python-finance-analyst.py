import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt 
import json 
import pandas_ta as ta
from scipy import stats
import numpy as np

with open('companies.json', 'r') as f:
    name_to_ticker = json.load(f)

while True:
    try:
        num = int(input("Enter How many stocks do you want to analyze?: "))
        break
    except ValueError:
        print("Please enter a valid number!")

while True:
    try:
        perd = int(input("Enter How many years do you want to analyze?: "))
        perd_str = str(perd) + 'y'
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

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    print(f"\n--- {ticker} ---")
    print(f"Price: ${info['currentPrice']}")
    print(f"Market Cap: ${info['marketCap']:,}")
    print(f"52W High: ${info['fiftyTwoWeekHigh']}")
    print(f"52W Low: ${info['fiftyTwoWeekLow']}")
    print(f"Beta: {info.get('beta','N/A')}")
    print(f"Dividend Yield: {info.get('dividendYield','N/A')}")
    print(f"Price to Book: {info.get('priceToBook','N/A')}")
    print(f"P/E Ratio: {info.get('trailingPE','N/A')}")
    return info


def analyze_seasonality(ticker,df):
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    monthly_returns = df.groupby('Month')['Stock_Return'].mean() * 100
    print(f"\n -- Monthly Seasonality for {ticker} --")
    print(monthly_returns.round(3))
    best_month = monthly_returns.idxmax()
    worst_month = monthly_returns.idxmin()
    print(f"Best Month: {best_month}")
    print(f"Worst Month: {worst_month}")

    monthly_returns.plot(kind= 'bar', title = f'{ticker} - monthly Seasonality')
    plt.xlabel('Month')
    plt.ylabel('Averge Return %')
    plt.axhline(y=0,color='red',linestyle = '--')
    plt.tight_layout()
    plt.show()

all_data = {}
market = yf.Ticker('^GSPC')
market_history = market.history(period=perd_str)
market_df = pd.DataFrame(market_history)
market_df['Market_Return'] = market_df['Close'].pct_change() 

def calculate_indicators(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period=perd_str)
    df = pd.DataFrame(history)
    print(f"\n--Statistics for {ticker} over the last {perd_str}--")
    print(df['Close'].describe())
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    print(f"\n-- MA and EMA anf RSI for {ticker} over the last {perd_str}--")
    print(f'MA50: {df["MA50"].iloc[-1]:.2f}')
    print(f'MA200: {df["MA200"].iloc[-1]:.2f}')
    print(f"EMA20: {df['EMA20'].iloc[-1]:.2f}")
    print(f"EMA50: {df['EMA50'].iloc[-1]:.2f}")
    print(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
    df['Stock_Return'] = df['Close'].pct_change()
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
    return df

for ticker in tickers:
    get_stock_info(ticker)
    
for ticker in tickers:
    all_data[ticker] = calculate_indicators(ticker)

for ticker in tickers:
    analyze_seasonality(ticker, all_data[ticker])

def calculate_financial_metrics(stock_returns, market_returns, beta):
    annual_rf = 0.04
    daily_rf = annual_rf/252
    sharp_ratio = (stock_returns.mean() - daily_rf)/(stock_returns.std() + 1e-9)
    sharp_annualized = sharp_ratio* np.sqrt(252)
    daily_expected_return = daily_rf + beta * (stock_returns.mean() - daily_rf)
    annualized_return = ((1 + daily_expected_return)**(252)) - 1
    return  {
        'Sharpe Annualized': sharp_annualized,
         'Annualized Return': annualized_return,
    }

def prepare_data(stock_prices, market_prices):
    df = pd.concat([stock_prices,market_prices],axis=1)
    df.columns = ['Stock','Market']
    returns  = df.pct_change().dropna()
    return returns['Stock'], returns['Market']


for ticker in tickers:
    df = all_data[ticker]
    stock_returns, market_returns = prepare_data(df['Close'], market_df['Close'])
    beta = stats.linregress(market_returns, stock_returns).slope
    metrics = calculate_financial_metrics(stock_returns, market_returns, beta)
    print(f'\n--- {ticker} Financial Metrics ---')
    print(f"Sharpe Ratio: {metrics['Sharpe Annualized']:.4f}")
    print(f"Annualized Return: {metrics['Annualized Return']:.4f}")


def plot_stock(ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    all_data[ticker][['Close', 'MA200','MA50','EMA20', 'EMA50']].plot(ax = ax1 , title = ticker)
    all_data[ticker]['RSI'].plot(ax = ax2 , title = 'RSI', color = 'purple')
    
    ax2.axhline(y=70, color='red', linestyle='--')    
    ax2.axhline(y=30, color='green', linestyle='--')  
    
    plt.tight_layout()
    plt.show()

for ticker in tickers:
    plot_stock(ticker)
