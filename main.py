import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import json
import pandas_ta as ta
from scipy import stats
import numpy as np
import warnings
from thefuzz import process
from signals import generate_signal, BUY_THRESHOLD, SELL_THRESHOLD
from report_generator import generate_pdf_report
from sentiment import analyze_sentiment, print_sentiment

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["lines.linewidth"] = 1.5

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

ANNUAL_RF    = 0.04
BAR          = "═" * 58
THIN_BAR     = "─" * 58
SIGNAL_ICONS = {
    "STRONG BUY":  "🟢",
    "BUY":         "🟩",
    "HOLD":        "🟡",
    "SELL":        "🟥",
    "STRONG SELL": "🔴",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{THIN_BAR}\n  {title}\n{THIN_BAR}")

def prompt_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}]: ").strip()
        if raw.lstrip("-").isdigit() and lo <= (v := int(raw)) <= hi:
            return v
        print(f"    ✗  Enter a whole number between {lo} and {hi}.")

def load_companies(path: str = "companies.json") -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"  ✔  Loaded {len(data):,} companies from {path}")
        return data
    except FileNotFoundError:
        print("  ⚠  companies.json not found — enter tickers directly (e.g. AAPL).")
        return {}

def resolve_ticker(raw: str, name_to_ticker: dict) -> str:
    name = raw.upper().strip()
    if name in name_to_ticker:
        ticker = name_to_ticker[name]
        print(f"    ✔  Resolved '{name}' → {ticker}")
        return ticker
    if name_to_ticker:
        best, score = process.extractOne(name, list(name_to_ticker.keys()))
        if score > 75:
            ans = input(f"    ❓ Did you mean '{best}'? (y/n): ").lower()
            if ans in ("y", "yes"):
                return name_to_ticker[best]
    return name

def validate_ticker(ticker: str) -> bool:
    try:
        _ = yf.Ticker(ticker).fast_info["lastPrice"]
        return True
    except Exception:
        return False

# ─── DATA & INDICATORS ────────────────────────────────────────────────────────

def get_market_returns(period: str) -> pd.Series:
    df = yf.Ticker("^GSPC").history(period=period)
    return df["Close"].pct_change().rename("Market_Return")

def get_stock_info(ticker: str) -> dict:
    fi = yf.Ticker(ticker).fast_info
    return {
        "currentPrice":     round(fi.get("lastPrice", 0), 2),
        "marketCap":        fi.get("marketCap", "N/A"),
        "fiftyTwoWeekHigh": round(fi.get("yearHigh", 0), 2),
        "fiftyTwoWeekLow":  round(fi.get("yearLow",  0), 2),
    }

def calculate_indicators(ticker: str, period: str, market_returns: pd.Series) -> pd.DataFrame | None:
    df = pd.DataFrame(yf.Ticker(ticker).history(period=period))

    df["Stock_Return"] = df["Close"].pct_change()
    df["MA50"]         = df["Close"].rolling(50).mean()
    df["MA200"]        = df["Close"].rolling(200).mean()
    df["EMA20"]        = df["Close"].ewm(span=20).mean()
    df["EMA50"]        = df["Close"].ewm(span=50).mean()
    df["RSI"]          = ta.rsi(df["Close"], length=14)
    df["BB_middle"]    = df["Close"].rolling(20).mean()
    df["BB_std"]       = df["Close"].rolling(20).std()
    df["BB_upper"]     = df["BB_middle"] + df["BB_std"] * 2
    df["BB_lower"]     = df["BB_middle"] - df["BB_std"] * 2
    df["MACD"]         = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["Signal"]       = df["MACD"].ewm(span=9).mean()
    df["Histogram"]    = df["MACD"] - df["Signal"]
    df["L14"]          = df["Low"].rolling(14).min()
    df["H14"]          = df["High"].rolling(14).max()
    df["%K"]           = (df["Close"] - df["L14"]) / (df["H14"] - df["L14"]) * 100
    df["%D"]           = df["%K"].rolling(3).mean()
    df.dropna(inplace=True)

    if df.empty:
        print(f"  ⚠  Not enough data for {ticker} after computing indicators.")
        return None

    combined = pd.concat([df["Stock_Return"], market_returns], axis=1).dropna()
    combined.columns = ["Stock_Return", "Market_Return"]
    slope, _, r_value, _, _ = stats.linregress(combined["Market_Return"], combined["Stock_Return"])
    df.attrs["beta"]      = round(slope, 4)
    df.attrs["r_squared"] = round(r_value ** 2, 4)

    return df

def calculate_financial_metrics(returns: pd.Series, beta: float) -> dict:
    daily_rf          = ANNUAL_RF / 252
    annualized_return = (1 + returns.mean()) ** 252 - 1
    excess            = returns - daily_rf
    sharpe            = (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)
    return {
        "Sharpe Annualized": round(sharpe, 4),
        "Annualized Return": round(annualized_return, 4),
        "Beta":              beta,
    }

# ─── PRINTING ─────────────────────────────────────────────────────────────────

def print_stock_info(ticker: str, info: dict) -> None:
    section(f"{ticker}  ·  Basic Info")
    cap_str = f"${info['marketCap']:,}" if info["marketCap"] != "N/A" else "N/A"
    print(f"    Price         : ${info['currentPrice']}")
    print(f"    Market Cap    : {cap_str}")
    print(f"    52W High      : ${info['fiftyTwoWeekHigh']}")
    print(f"    52W Low       : ${info['fiftyTwoWeekLow']}")

def print_indicators(ticker: str, df: pd.DataFrame) -> None:
    section(f"{ticker}  ·  Latest Indicators")
    print(f"    MA50 / MA200  : {df['MA50'].iloc[-1]:.2f}  /  {df['MA200'].iloc[-1]:.2f}")
    print(f"    EMA20         : {df['EMA20'].iloc[-1]:.2f}")
    print(f"    RSI           : {df['RSI'].iloc[-1]:.2f}")
    print(f"    MACD          : {df['MACD'].iloc[-1]:.4f}  |  Signal: {df['Signal'].iloc[-1]:.4f}")
    print(f"    Stoch %K/%D   : {df['%K'].iloc[-1]:.2f}  /  {df['%D'].iloc[-1]:.2f}")
    print(f"    Beta          : {df.attrs['beta']:.3f}  |  R²: {df.attrs['r_squared']:.4f}")

def print_metrics(ticker: str, metrics: dict) -> None:
    section(f"{ticker}  ·  Financial Metrics")
    print(f"    Sharpe Ratio  : {metrics['Sharpe Annualized']:.4f}")
    print(f"    Ann. Return   : {metrics['Annualized Return'] * 100:.2f}%")
    print(f"    Beta          : {metrics['Beta']:.4f}")

# ─── CHARTS ───────────────────────────────────────────────────────────────────

def analyze_seasonality(ticker: str, df: pd.DataFrame) -> None:
    monthly        = df.copy()
    monthly["Month"] = monthly.index.month
    monthly_ret    = monthly.groupby("Month")["Stock_Return"].mean() * 100

    section(f"{ticker}  ·  Monthly Seasonality")
    print(monthly_ret.round(3).to_string())
    print(f"\n    Best month  : {monthly_ret.idxmax()}")
    print(f"    Worst month : {monthly_ret.idxmin()}")

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_ret.plot(kind="bar", ax=ax, color="teal", edgecolor="black")
    ax.set_title(f"{ticker} — Average Monthly Return (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Return (%)")
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.xticks(rotation=0); plt.tight_layout(); plt.show()

def calculate_correlation(all_data: dict) -> None:
    valid = {k: v for k, v in all_data.items() if v is not None}
    if len(valid) < 2:
        return

    corr = pd.DataFrame({t: d["Stock_Return"] for t, d in valid.items()}).corr()
    section("Portfolio  ·  Correlation Matrix")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im)
    ticks = range(len(corr.columns))
    ax.set_xticks(ticks); ax.set_xticklabels(corr.columns)
    ax.set_yticks(ticks); ax.set_yticklabels(corr.columns)
    for i in ticks:
        for j in ticks:
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Cross-Correlation Matrix", fontsize=14, fontweight="bold")
    plt.grid(False); plt.tight_layout(); plt.show()

def plot_stock(ticker: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"],  label="Close",  color="black", linewidth=2)
    ax.plot(df.index, df["MA50"],   label="MA50",   linestyle="--")
    ax.plot(df.index, df["MA200"],  label="MA200",  linestyle="--")
    ax.plot(df.index, df["EMA20"],  label="EMA20")
    ax.set_title(f"{ticker} — Price & Moving Averages", fontsize=14, fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"],    label="Close",      color="black")
    ax.plot(df.index, df["BB_upper"], label="Upper Band", color="red",   alpha=0.5)
    ax.plot(df.index, df["BB_lower"], label="Lower Band", color="green", alpha=0.5)
    ax.fill_between(df.index, df["BB_lower"], df["BB_upper"], color="grey", alpha=0.1)
    ax.set_title(f"{ticker} — Bollinger Bands", fontsize=14, fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["RSI"], color="purple", label="RSI")
    ax.axhline(70, color="red",   linestyle="--", linewidth=1.5, label="Overbought (70)")
    ax.axhline(30, color="green", linestyle="--", linewidth=1.5, label="Oversold (30)")
    ax.fill_between(df.index, 70, df["RSI"], where=(df["RSI"] >= 70), color="red",   alpha=0.3)
    ax.fill_between(df.index, 30, df["RSI"], where=(df["RSI"] <= 30), color="green", alpha=0.3)
    ax.set_title(f"{ticker} — RSI", fontsize=14, fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["MACD"],   label="MACD",   color="blue")
    ax.plot(df.index, df["Signal"], label="Signal", color="orange")
    colors = ["green" if v >= 0 else "red" for v in df["Histogram"]]
    ax.bar(df.index, df["Histogram"], color=colors, alpha=0.5, label="Histogram")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} — MACD", fontsize=14, fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.show()

# ─── SIGNAL LOGIC ─────────────────────────────────────────────────────────────

def compute_final_signal(df: pd.DataFrame, info: dict, metrics: dict, sentiment: dict) -> dict:
    result         = generate_signal(df, info, metrics)
    base_score     = result.get("score", 0)
    sent_score     = sentiment.get("score", 0)
    adjusted_score = base_score + sent_score

    if sent_score != 0:
        label   = sentiment.get("label", "NEUTRAL")
        compound = sentiment.get("compound", 0)
        result["reasons"].append(
            f"Sentiment: {label} "
            f"(compound: {compound:.4f}, "
            f"score adjustment: {'+' if sent_score > 0 else ''}{sent_score})."
        )

    is_bullish = df["Close"].iloc[-1] > df["MA200"].iloc[-1]

    if adjusted_score >= BUY_THRESHOLD:
        final = "STRONG BUY" if (adjusted_score >= 10 and is_bullish) else "BUY"
    elif adjusted_score <= SELL_THRESHOLD:
        final = "STRONG SELL" if (adjusted_score <= -10 and not is_bullish) else "SELL"
    else:
        final = "HOLD"

    return {
        "signal":          final,
        "original_signal": result.get("signal", "UNKNOWN"),
        "base_score":      base_score,
        "adjusted_score":  adjusted_score,
        "sent_score":      sent_score,
        "reasons":         result.get("reasons", []),
    }

def print_signal(ticker: str, res: dict) -> None:
    icon   = SIGNAL_ICONS.get(res["signal"], "⬜")
    change = f"  (was {res['original_signal']} before sentiment)" if res["signal"] != res["original_signal"] else ""
    print(f"\n  {icon}  {ticker:<6}  →  {res['signal']}{change}")
    print(f"       Score : {res['base_score']} → {res['adjusted_score']}  (after sentiment)")

    if res["signal"] == "HOLD":
        if res["sent_score"] >= 2:
            print("       ⚠  Sentiment bullish — watch for technical confirmation.")
        elif res["sent_score"] <= -2:
            print("       ⚠  Sentiment bearish — defensive stance advised.")

    for reason in res["reasons"]:
        print(f"       · {reason}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BAR}")
    print("       Python Finance Analyst  ·  Portfolio Analyzer  v2.0")
    print(BAR)

    name_to_ticker = load_companies()

    print()
    num      = prompt_int("Number of stocks to analyze", 1, 20)
    perd     = prompt_int("Years of historical data",    1, 10)
    perd_str = f"{perd}y"

    section("Stock Selection")
    tickers = []
    for i in range(num):
        while True:
            raw = input(f"  [{i+1}/{num}] Company name or ticker: ").strip()
            if not raw:
                print("    ✗  Input cannot be empty."); continue
            ticker = resolve_ticker(raw, name_to_ticker)
            if validate_ticker(ticker):
                tickers.append(ticker)
                print(f"    ✔  {ticker} added.")
                break
            print(f"    ✗  '{ticker}' not found. Please try again.")

    print(f"\n  Loading market benchmark (S&P 500)…")
    market_returns = get_market_returns(perd_str)

    # ── Per-ticker analysis ───────────────────────────────────────────────────
    all_data, stock_info, all_metrics, all_sentiment = {}, {}, {}, {}

    for ticker in tickers:
        print(f"\n{BAR}")
        print(f"  Analyzing  {ticker}")
        print(BAR)

        info               = get_stock_info(ticker)
        stock_info[ticker] = info
        print_stock_info(ticker, info)

        df               = calculate_indicators(ticker, perd_str, market_returns)
        all_data[ticker] = df

        if df is not None:
            print_indicators(ticker, df)
            metrics              = calculate_financial_metrics(df["Stock_Return"], df.attrs["beta"])
            all_metrics[ticker]  = metrics
            print_metrics(ticker, metrics)

        sentiment              = analyze_sentiment(ticker)
        all_sentiment[ticker]  = sentiment
        print_sentiment(ticker, sentiment)

    # ── Charts ────────────────────────────────────────────────────────────────
    for ticker in tickers:
        if all_data[ticker] is not None:
            plot_stock(ticker, all_data[ticker])
            analyze_seasonality(ticker, all_data[ticker])

    calculate_correlation(all_data)

    # ── Signal Summary ────────────────────────────────────────────────────────
    section("Signal Summary")
    for ticker in tickers:
        df = all_data.get(ticker)
        if df is None:
            continue
        res = compute_final_signal(
            df, stock_info[ticker],
            all_metrics.get(ticker, {}),
            all_sentiment.get(ticker, {})
        )
        print_signal(ticker, res)

    # ── PDF Report ────────────────────────────────────────────────────────────
    print(f"\n{THIN_BAR}")
    print("  Generating PDF report…")
    generate_pdf_report(all_data, stock_info, all_metrics, tickers, all_sentiment)
    print("  ✔  Report saved successfully.")
    print(f"{THIN_BAR}\n")


if __name__ == "__main__":
    main()        perd_str = str(perd) + 'y'
        break
    except ValueError:
        print("Please enter a valid number!")

tickers = []

for i in range(num):
    while True:
        name = input(f'Enter company {i+1} name or ticker: ').upper().strip()
        
        if name in name_to_ticker:
            ticker = name_to_ticker[name]
        elif name_to_ticker:
            best_match, score = process.extractOne(name, list(name_to_ticker.keys()))
            if score > 75:
                question = input(f"Did you mean '{best_match}'? (y/n): ").lower()
                if question in ('y', 'yes'):
                    ticker = name_to_ticker[best_match]
                else:
                    ticker = name
            else:
                ticker = name
        else:
            ticker = name
        try:
            stock = yf.Ticker(ticker)
            _ = stock.fast_info['lastPrice']
            tickers.append(ticker)
            print(f" Successfully added ")
            break
        except Exception:
            print(f"'{ticker}' not found! Please try again.")


market = yf.Ticker('^GSPC')
market_history = market.history(period=perd_str)
market_df = pd.DataFrame(market_history)
market_df['Market_Return'] = market_df['Close'].pct_change()

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    fi = stock.fast_info
    
    info = {
        'currentPrice': round(fi.get('lastPrice', 0), 2),
        'marketCap': fi.get('marketCap', 'N/A'),
        'fiftyTwoWeekHigh': round(fi.get('yearHigh', 0), 2),
        'fiftyTwoWeekLow': round(fi.get('yearLow', 0), 2),
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
        print(f"\n Warning: Not enough data for {ticker} after applying indicators.")
        return None

    combined = pd.concat([df['Stock_Return'], market_df['Market_Return']], axis=1).dropna()
    combined.columns = ['Stock_Return', 'Market_Return']
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        combined['Market_Return'],
        combined['Stock_Return']
    )
    r_squared = r_value ** 2
    df.attrs['beta'] = round(slope, 4)
    df.attrs['r_squared'] = r_squared

    print(f"\n-- Latest Indicator Values for {ticker} --")
    print(f"MA50: {df['MA50'].iloc[-1]:.2f} | MA200: {df['MA200'].iloc[-1]:.2f}")
    print(f"RSI: {df['RSI'].iloc[-1]:.2f} | MACD: {df['MACD'].iloc[-1]:.4f}")
    print(f"Beta (Calculated Once): {slope:.3f}")
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


# ─── MAIN ANALYSIS LOOP ───────────────────────────────────────────────────────

all_data      = {}
stock_info    = {}
all_metrics   = {}
all_sentiment = {}   # ← NEW: store sentiment per ticker

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

    # ── Sentiment Analysis ──────────────────────────────────────────────────
    sentiment = analyze_sentiment(ticker)          # ← NEW
    all_sentiment[ticker] = sentiment              # ← NEW
    print_sentiment(ticker, sentiment)             # ← NEW

for ticker in tickers:
    if all_data[ticker] is not None:
        plot_stock(ticker, all_data[ticker])
        analyze_seasonality(ticker, all_data[ticker])

calculate_correlation(all_data)


# ─── SIGNAL GENERATION (with Sentiment boost) ─────────────────────────────────

try:
    print("\n --- Result ---")
    for ticker in tickers:
        df = all_data.get(ticker)
        if df is not None:
            info      = stock_info[ticker]
            metrics   = all_metrics[ticker]
            sentiment = all_sentiment.get(ticker, {})

            result = generate_signal(df, info, metrics)

            # ── Sentiment override ──────────────────────────────────────────
            # Sentiment adjusts the final score and can upgrade/downgrade signal
            sentiment_score  = sentiment.get('score', 0)
            sentiment_label  = sentiment.get('label', 'NEUTRAL')
            base_score       = result.get('score', 0)
            adjusted_score   = base_score + sentiment_score

            if sentiment_score != 0:
                result['reasons'].append(
                    f"Sentiment: News is {sentiment_label} "
                    f"(compound: {sentiment.get('compound', 0):.4f}, "
                    f"score adjustment: {'+' if sentiment_score > 0 else ''}{sentiment_score})."
                )

            # Re-evaluate signal with adjusted score
            from signals import BUY_THRESHOLD, SELL_THRESHOLD
            is_bullish_trend = df['Close'].iloc[-1] > df['MA200'].iloc[-1]

            if adjusted_score >= BUY_THRESHOLD:
                final_signal = "STRONG BUY" if (adjusted_score >= 10 and is_bullish_trend) else "BUY"
            elif adjusted_score <= SELL_THRESHOLD:
                final_signal = "STRONG SELL" if (adjusted_score <= -10 and not is_bullish_trend) else "SELL"
            else:
                final_signal = "HOLD"

            signal = result.get('signal', 'UNKNOWN')

            print(f"\n  {ticker} → {final_signal}", end="")
            if final_signal != signal:
                print(f"  (was {signal} before sentiment)", end="")

            # ── Sentiment Divergence Warning ──────────────────────────
            sent_score = sentiment.get('score', 0)
            if final_signal == 'HOLD' and sent_score >= 2:
                print(f"  ⚠️  Sentiment Bullish — watch for technical confirmation", end="")
            elif final_signal == 'HOLD' and sent_score <= -2:
                print(f"  ⚠️  Sentiment Bearish — defensive stance advised", end="")
            print()

            if final_signal not in ('ERROR', 'WAIT'):
                print(f"  Score: {base_score} → {adjusted_score} (after sentiment)")

            for reason in result.get('reasons', []):
                print(f"  - {reason}")

except NameError:
    pass


generate_pdf_report(all_data, stock_info, all_metrics, tickers, all_sentiment)
