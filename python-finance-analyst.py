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
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df ['BB_middle'] + (df['BB_std']*2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std']*2)
    df['MACD'] = df['Close'].ewm(span = 12).mean() - df['Close'].ewm(span = 26).mean()
    df['Signal'] = df["MACD"].ewm(span = 9).mean()
    df["Histogram"] =df['MACD'] - df['Signal']
    df['L14'] = df['Low'].rolling(window=14).min()
    df['H14'] = df['High'].rolling(window=14).max()
    df['%K'] = ((df['Close'] - df['L14']) / (df['H14'] - df['L14'])) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()


    print(f"\n-- MA and EMA anf RSI for {ticker} over the last {perd_str}--")
    print(f"BB_middle: {df['BB_middle'].iloc[-1]:.2f}")
    print(f"BB_std: {df['BB_std'].iloc[-1]:.2f}")
    print(f"BB_upper: {df['BB_upper'].iloc[-1]:.2f}")
    print(f"BB_lower: {df['BB_lower'].iloc[-1]:.2f}")
    print(f"MA50: {df["MA50"].iloc[-1]:.2f}")
    print(f"MA200: {df["MA200"].iloc[-1]:.2f}")
    print(f"EMA20: {df['EMA20'].iloc[-1]:.2f}")
    print(f"EMA50: {df['EMA50'].iloc[-1]:.2f}")
    print(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
    print(f"MACD: {df['MACD'].iloc[-1]: .4f}")
    print(f"Signal: {df['Signal'].iloc[-1]:.4f}")
    print(f"Histogram: {df['Histogram'].iloc[-1]:.4f}")
    print(f"%K: {df['%K'].iloc[-1]:.2f}")
    print(f"%D: {df['%D'].iloc[-1]:.2f}")


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

def calculate_correlation(all_data):
    if len(all_data) < 2:
        print("Need at least 2 stocks for correlation!")
        return
    returns_df = pd.DataFrame()
    for ticker in all_data:
        returns_df[ticker] = all_data[ticker]['Stock_Return']

    corr_matrix = returns_df.corr()
    print("\n -- Correlation Matrix --")
    print(corr_matrix.round(3))

    plt.figure(figsize=(8,6))
    plt.imshow(corr_matrix, cmap= 'RdYlGn' , vmin=-1,vmax=1 )
    plt.colorbar( )
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title("Cross-Correlation Matrix")
    plt.tight_layout()
    plt.show()


for ticker in tickers:
    df = all_data[ticker]
    stock_returns, market_returns = prepare_data(df['Close'], market_df['Close'])
    beta = stats.linregress(market_returns, stock_returns).slope
    metrics = calculate_financial_metrics(stock_returns, market_returns, beta)
    print(f'\n--- {ticker} Financial Metrics ---')
    print(f"Sharpe Ratio: {metrics['Sharpe Annualized']:.4f}")
    print(f"Annualized Return: {metrics['Annualized Return']:.4f}")


def plot_stock(ticker):
    df = all_data[ticker]

    plt.figure(figsize =(12,5))
    df[['Close', 'MA200','MA50','EMA20', 'EMA50']].plot(ax=plt.gca())
    plt.title(f' {ticker} - Price & Moving Averages')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = (12,5))
    df[['Close' , 'BB_middle' , 'BB_lower' , 'BB_upper']].plot(ax=plt.gca())
    plt.title("Bollinger Bands")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = (12,5))
    df['RSI'].plot(color = 'purple',ax=plt.gca())
    plt.axhline(y=70, color = 'green', linestyle = '--')
    plt.axhline(y = 30, color = 'red', linestyle = '--')
    plt.title(f'{ticker} - RSI')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,5))
    df['MACD'].plot(label= 'MACD', color ='blue')
    df['Signal'].plot(label = 'Signal', color = 'orange')
    plt.bar(df.index, df['Histogram'],label = 'Histogram', color = 'black',alpha = 0.3)
    plt.axhline(y=0,color = 'red', linestyle = '--')
    plt.title(f"{ticker} - MACD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    df['%K'].plot(label='%K', color='blue')
    df['%D'].plot(label='%D', color='orange')
    plt.axhline(y=80, color='red', linestyle='--', label='Overbought')
    plt.axhline(y=20, color='green', linestyle='--', label='Oversold')
    plt.title(f'{ticker} - Stochastic Oscillator')
    plt.legend()
    plt.tight_layout()
    plt.show()


for ticker in tickers:
    plot_stock(ticker)

for ticker in tickers:
    analyze_seasonality(ticker, all_data[ticker])

calculate_correlation(all_data)
