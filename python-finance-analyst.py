import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import json
import pandas_ta as ta
from scipy import stats
import numpy as np
import warnings
from signals import generate_signal
from report_generator import  generate_pdf_report
 
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['lines.linewidth'] = 1.5

try:
    with open('companies.json', 'r') as f:
        name_to_ticker = json.load(f)
except FileNotFoundError:
    name_to_ticker = {}
    print("Warning: 'companies.json' not found. You can still enter tickers manually.")

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
            _ = stock.fast_info['lastPrice'] 
            tickers.append(ticker)
            break
        except Exception:
            print(f"'{name}' not found or network error! Please enter a valid company name.")

print("\nFetching Market Data (^GSPC)...")
market = yf.Ticker('^GSPC')
market_history = market.history(period=perd_str)
market_df = pd.DataFrame(market_history)
market_df['Market_Return'] = market_df['Close'].pct_change()

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    fi = stock.fast_info
    
    info = {
        'currentPrice': fi.get('lastPrice', 'N/A'),
        'marketCap': fi.get('marketCap', 'N/A'),
        'fiftyTwoWeekHigh': fi.get('yearHigh', 'N/A'),
        'fiftyTwoWeekLow': fi.get('yearLow', 'N/A'),
    }
    
    print(f"\n--- {ticker} Basic Info ---")
    print(f"Price: ${info['currentPrice']}")
    if info['marketCap'] != 'N/A':
        print(f"Market Cap: ${info['marketCap']:,}")
    print(f"52W High: ${info['fiftyTwoWeekHigh']}")
    print(f"52W Low: ${info['fiftyTwoWeekLow']}")
    
    return info

def analyze_seasonality(ticker, df):
    df_copy = df.copy()
    df_copy['Month'] = df_copy.index.month
    monthly_returns = df_copy.groupby('Month')['Stock_Return'].mean() * 100
    
    print(f"\n-- Monthly Seasonality for {ticker} --")
    print(monthly_returns.round(3))
    print(f"Best Month: {monthly_returns.idxmax()}")
    print(f"Worst Month: {monthly_returns.idxmin()}")

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_returns.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
    ax.set_title(f'{ticker} - Average Monthly Return (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Return %', fontsize=12)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def calculate_indicators(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period=perd_str)
    df = pd.DataFrame(history)
    df['Stock_Return'] = df['Close'].pct_change()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df["MACD"].ewm(span=9).mean()
    df["Histogram"] = df['MACD'] - df['Signal']
    df['L14'] = df['Low'].rolling(window=14).min()
    df['H14'] = df['High'].rolling(window=14).max()
    df['%K'] = ((df['Close'] - df['L14']) / (df['H14'] - df['L14'])) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    df.dropna(inplace=True) 

    if len(df) == 0:
        print(f"\n Warning: Not enough data for {ticker} after applying indicators (need at least 200 days for MA200).")
        return None

    combined = pd.concat([df['Stock_Return'], market_df['Market_Return']], axis=1).dropna()
    combined.columns = ['Stock_Return', 'Market_Return']
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        combined['Market_Return'],
        combined['Stock_Return']
    )
    r_squared = r_value ** 2
    df.attrs['beta'] = slope 
    df.attrs['r_squared'] = r_squared

    print(f"\n-- Latest Indicator Values for {ticker} --")
    print(f"MA50: {df['MA50'].iloc[-1]:.2f} | MA200: {df['MA200'].iloc[-1]:.2f}")
    print(f"RSI: {df['RSI'].iloc[-1]:.2f} | MACD: {df['MACD'].iloc[-1]:.4f}")
    print(f"Beta (Calculated Once): {slope:.4f}")
    print(f"R² (Systematic Risk): {r_squared:.4f}")
    
    return df

def calculate_financial_metrics(stock_returns, beta):
    annual_rf = 0.04
    daily_rf = annual_rf / 252
    mean_daily_return = stock_returns.mean()
    annualized_return = ((1 + mean_daily_return) ** 252) - 1
    excess_returns = stock_returns - daily_rf
    sharpe_annualized = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)
    
    return {
        'Sharpe Annualized': sharpe_annualized,
        'Annualized Return': annualized_return,
        'Beta': beta
    }

def calculate_correlation(all_data):
    valid_data = {k: v for k, v in all_data.items() if v is not None}
    
    if len(valid_data) < 2:
        return
    
    returns_df = pd.DataFrame()
    for ticker, df in valid_data.items():
        returns_df[ticker] = df['Stock_Return']

    corr_matrix = returns_df.corr()
    print("\n -- Correlation Matrix --")
    print(corr_matrix.round(3))

    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')

    plt.title("Cross-Correlation Matrix", fontsize=14, fontweight='bold')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_stock(ticker, df):
    if df is None or len(df) == 0: return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=2)
    ax.plot(df.index, df['MA50'], label='MA50', linestyle='--')
    ax.plot(df.index, df['MA200'], label='MA200', linestyle='--')
    ax.plot(df.index, df['EMA20'], label='EMA20')
    ax.set_title(f'{ticker} - Price & Moving Averages', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Close Price', color='black')
    ax.plot(df.index, df['BB_upper'], label='Upper Band', color='red', alpha=0.5)
    ax.plot(df.index, df['BB_lower'], label='Lower Band', color='green', alpha=0.5)
    ax.fill_between(df.index, df['BB_lower'], df['BB_upper'], color='grey', alpha=0.1)
    ax.set_title(f"{ticker} - Bollinger Bands", fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['RSI'], color='purple', label='RSI')
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, label='Overbought (70)')
    ax.axhline(y=30, color='green', linestyle='--', linewidth=1.5, label='Oversold (30)')
    ax.fill_between(df.index, y1=70, y2=df['RSI'], where=(df['RSI'] >= 70), color='red', alpha=0.3)
    ax.fill_between(df.index, y1=30, y2=df['RSI'], where=(df['RSI'] <= 30), color='green', alpha=0.3)
    ax.set_title(f'{ticker} - RSI', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['Signal'], label='Signal', color='orange')
    colors = ['green' if val >= 0 else 'red' for val in df['Histogram']]
    ax.bar(df.index, df['Histogram'], color=colors, alpha=0.5, label='Histogram')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title(f"{ticker} - MACD", fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

all_data = {}
stock_info = {}
all_metrics = {}

for ticker in tickers:
    stock_info[ticker] = get_stock_info(ticker)
    
    df = calculate_indicators(ticker)
    all_data[ticker] = df
    
    if df is not None:
        beta = df.attrs.get('beta', 1.0)
        
        metrics = calculate_financial_metrics(df['Stock_Return'], beta)
        all_metrics[ticker] = metrics
        
        print(f'\n--- {ticker} Financial Metrics ---')
        print(f"Sharpe Ratio: {metrics['Sharpe Annualized']:.4f}")
        print(f"Annualized Return: {metrics['Annualized Return'] * 100:.2f}%")

for ticker in tickers:
    if all_data[ticker] is not None:
        plot_stock(ticker, all_data[ticker])
        analyze_seasonality(ticker, all_data[ticker])

calculate_correlation(all_data)


try:
    print("\n" + "="*50)
    for ticker in tickers:
        df = all_data.get(ticker)
        if df is not None:
            info = stock_info[ticker]
            metrics = all_metrics[ticker]
            
            result = generate_signal(df, info, metrics)
            print(f"\n  {ticker} → {result['signal']}")
            print(f"  Score: {result['score']}/{result.get('max_score', 'N/A')}")
            print(f"  {'─'*40}")
            for reason in result.get('reasons', []):
                print(f"  - {reason}")
    print("="*50)
except NameError:
    pass


generate_pdf_report(all_data, stock_info, all_metrics, tickers)
