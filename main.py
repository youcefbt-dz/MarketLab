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

BAR      = "═" * 58
THIN_BAR = "─" * 58

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

def prompt_float(prompt: str, lo: float, hi: float) -> float:
    """Used for the risk-free rate input."""
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}]: ").strip()
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"    ✗  Enter a number between {lo} and {hi}.")

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

# ─── DATA FETCHING ────────────────────────────────────────────────────────────

def get_market_returns(period: str) -> pd.Series:
    df = yf.Ticker("^GSPC").history(period=period)
    return df["Close"].pct_change().rename("Market_Return")

def fetch_ticker_data(ticker: str, period: str) -> tuple[dict, pd.DataFrame | None]:
    """
    Fetch all data for a ticker in a single call.
    Returns (info_dict, history_df) to avoid multiple round-trips.
    """
    stock = yf.Ticker(ticker)

    # ── fast_info (one request) ──────────────────────────────────────────────
    fi = stock.fast_info
    info = {
        "currentPrice":     round(fi.get("lastPrice", 0), 2),
        "marketCap":        fi.get("marketCap", "N/A"),
        "fiftyTwoWeekHigh": round(fi.get("yearHigh", 0), 2),
        "fiftyTwoWeekLow":  round(fi.get("yearLow",  0), 2),
    }

    # ── history (one request) ────────────────────────────────────────────────
    try:
        history = stock.history(period=period)
        if history.empty:
            print(f"  ⚠  No price history returned for {ticker}.")
            return info, None
    except Exception as e:
        print(f"  ✗  Failed to fetch history for {ticker}: {e}")
        return info, None

    return info, pd.DataFrame(history)

# ─── INDICATORS ───────────────────────────────────────────────────────────────

def calculate_indicators(
    ticker: str,
    history: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.DataFrame | None:
    """
    Compute all technical indicators on an already-fetched DataFrame.
    Accepts history from fetch_ticker_data() so no extra network call is made.
    """
    try:
        df = history.copy()

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
        slope, _, r_value, _, _ = stats.linregress(
            combined["Market_Return"], combined["Stock_Return"]
        )
        df.attrs["beta"]      = round(slope, 4)
        df.attrs["r_squared"] = round(r_value ** 2, 4)

        return df

    except Exception as e:
        print(f"  ✗  Error computing indicators for {ticker}: {e}")
        return None

# ─── METRICS ──────────────────────────────────────────────────────────────────

def calculate_financial_metrics(
    returns: pd.Series,
    beta: float,
    annual_rf: float,
) -> dict:
    daily_rf          = annual_rf / 252
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
    try:
        monthly          = df.copy()
        monthly["Month"] = monthly.index.month
        monthly_ret      = monthly.groupby("Month")["Stock_Return"].mean() * 100

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

    except Exception as e:
        print(f"  ⚠  Could not plot seasonality for {ticker}: {e}")

def calculate_correlation(all_data: dict) -> None:
    valid = {k: v for k, v in all_data.items() if v is not None}
    if len(valid) < 2:
        return

    # Build in one shot — no per-ticker loop appending
    corr = pd.concat(
        [d["Stock_Return"].rename(t) for t, d in valid.items()], axis=1
    ).corr()

    section("Portfolio  ·  Correlation Matrix")
    print(corr.round(3).to_string())

    try:
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
    except Exception as e:
        print(f"  ⚠  Could not plot correlation matrix: {e}")

def plot_stock(ticker: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return

    try:
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

    except Exception as e:
        print(f"  ⚠  Could not plot charts for {ticker}: {e}")

# ─── SIGNAL LOGIC ─────────────────────────────────────────────────────────────

def compute_final_signal(df: pd.DataFrame, info: dict, metrics: dict, sentiment: dict) -> dict:
    result         = generate_signal(df, info, metrics)
    base_score     = result.get("score", 0)
    sent_score     = sentiment.get("score", 0)
    adjusted_score = base_score + sent_score

    if sent_score != 0:
        result["reasons"].append(
            f"Sentiment: {sentiment.get('label', 'NEUTRAL')} "
            f"(compound: {sentiment.get('compound', 0):.4f}, "
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

    # ── Risk-free rate (4) ────────────────────────────────────────────────────
    print(f"\n  Risk-Free Rate  (current 10Y US Treasury ≈ 4.2%)")
    print(f"  Press Enter to use 4.0% default, or type a value (e.g. 4.2).")
    raw_rf = input("  Annual RF rate [%]: ").strip()
    try:
        annual_rf = float(raw_rf) / 100 if raw_rf else 0.04
        annual_rf = max(0.0, min(annual_rf, 0.20))   # clamp to [0%, 20%]
    except ValueError:
        annual_rf = 0.04
    print(f"    ✔  Using risk-free rate: {annual_rf * 100:.2f}%")

    # ── Ticker collection ─────────────────────────────────────────────────────
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

        # Single fetch per ticker (1)
        info, history          = fetch_ticker_data(ticker, perd_str)
        stock_info[ticker]     = info
        print_stock_info(ticker, info)

        if history is None:
            all_data[ticker] = None
            continue

        # Indicators computed on already-fetched data (1)
        df               = calculate_indicators(ticker, history, market_returns)
        all_data[ticker] = df

        if df is not None:
            print_indicators(ticker, df)
            metrics             = calculate_financial_metrics(df["Stock_Return"], df.attrs["beta"], annual_rf)  # (4)
            all_metrics[ticker] = metrics
            print_metrics(ticker, metrics)

        sentiment              = analyze_sentiment(ticker)
        all_sentiment[ticker]  = sentiment
        print_sentiment(ticker, sentiment)

    # ── Charts ────────────────────────────────────────────────────────────────
    for ticker in tickers:
        if all_data.get(ticker) is not None:
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
    main()
