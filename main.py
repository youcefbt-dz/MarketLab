"""
main.py — MarketLab v3.0
Entry point for all analysis modes.

Modes:
    [1] Portfolio Analysis  — signals + sentiment + PDF report
    [2] Backtesting         — historical simulation + logger
    [3] Watchlist Scanner   — parallel scan of 250+ tickers
    [4] ML Predictor        — train & evaluate quality predictor
    [5] Warehouse Manager   — update / inspect local data
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats
from thefuzz import process

from report_generator  import generate_pdf_report
from sentiment         import analyze_sentiment, print_sentiment
from signals           import generate_signal, BUY_THRESHOLD, SELL_THRESHOLD
from stock_warehouse   import load_local, warehouse_status, load_companies

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["lines.linewidth"] = 1.5

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

VERSION  = "3.0"
BAR      = "═" * 62
THIN_BAR = "─" * 62

SIGNAL_ICONS = {
    "STRONG BUY" : "🟢",
    "BUY"        : "🟩",
    "HOLD"       : "🟡",
    "SELL"       : "🟥",
    "STRONG SELL": "🔴",
}

MODES = {
    "1": ("Portfolio Analysis",  "signals · sentiment · charts · PDF report"),
    "2": ("Backtesting",         "historical simulation · walk-forward · logger"),
    "3": ("Watchlist Scanner",   "parallel scan of 250+ tickers · ranked output"),
    "4": ("ML Predictor",        "train · evaluate · feature importance"),
    "5": ("Warehouse Manager",   "update · status · inspect local data"),
}


# ─── UI HELPERS ───────────────────────────────────────────────────────────────

def _banner() -> None:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    print(f"\n{BAR}")
    print(f"  MarketLab  v{VERSION}  ·  Quantitative Finance Analytics")
    print(f"  {now}")
    print(BAR)


def _section(title: str) -> None:
    print(f"\n{THIN_BAR}\n  {title}\n{THIN_BAR}")


def _ok(msg: str)   -> None: print(f"  ✔  {msg}")
def _warn(msg: str) -> None: print(f"  ⚠  {msg}")
def _err(msg: str)  -> None: print(f"  ✗  {msg}")


def _prompt_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}]: ").strip()
        if raw.lstrip("-").isdigit() and lo <= (v := int(raw)) <= hi:
            return v
        _err(f"Enter a whole number between {lo} and {hi}.")


def _prompt_choice(prompt: str, valid: set[str]) -> str:
    while True:
        ans = input(f"  {prompt} [{'/'.join(sorted(valid))}]: ").strip()
        if ans in valid:
            return ans
        _err(f"Enter one of: {', '.join(sorted(valid))}")


def _mode_menu() -> str:
    print()
    for key, (name, desc) in MODES.items():
        print(f"    [{key}]  {name:<22}  {desc}")
    print()
    return _prompt_choice("Select mode", set(MODES))


# ─── TICKER HELPERS ───────────────────────────────────────────────────────────

def _resolve_ticker(raw: str, name_to_ticker: dict) -> str:
    name = raw.upper().strip()
    if name in name_to_ticker:
        ticker = name_to_ticker[name]
        _ok(f"Resolved '{name}' → {ticker}")
        return ticker
    if name_to_ticker:
        best, score = process.extractOne(name, list(name_to_ticker.keys()))
        if score > 75:
            ans = input(f"    ❓ Did you mean '{best}'? (y/n): ").lower()
            if ans in ("y", "yes"):
                return name_to_ticker[best]
    return name


def _validate_ticker(ticker: str) -> bool:
    try:
        load_local(ticker)
        return True
    except FileNotFoundError:
        return False


def _collect_tickers(num: int, name_to_ticker: dict) -> list[str]:
    _section("Stock Selection")
    tickers = []
    for i in range(num):
        while True:
            raw = input(f"  [{i+1}/{num}] Company name or ticker: ").strip()
            if not raw:
                _err("Input cannot be empty."); continue
            ticker = _resolve_ticker(raw, name_to_ticker)
            if _validate_ticker(ticker):
                tickers.append(ticker)
                _ok(f"{ticker} added.")
                break
            _err(f"'{ticker}' not found in warehouse. Run mode [5] to update.")
    return tickers


# ─── DATA & INDICATORS ────────────────────────────────────────────────────────

def _get_market_returns() -> pd.Series | None:
    try:
        df = load_local("SPY").set_index("Date")
        return df["Close"].pct_change().rename("Market_Return")
    except FileNotFoundError:
        _warn("SPY not in warehouse — market regime filter disabled.")
        return None


def _fetch_ticker_data(ticker: str, years: int) -> tuple[dict, pd.DataFrame | None]:
    try:
        df = load_local(ticker).set_index("Date")
        df.index = pd.to_datetime(df.index)
        info = {
            "currentPrice"    : round(df["Close"].iloc[-1], 2),
            "marketCap"       : "N/A",
            "fiftyTwoWeekHigh": round(df["High"].tail(252).max(), 2),
            "fiftyTwoWeekLow" : round(df["Low"].tail(252).min(),  2),
        }
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        df = df[df.index >= cutoff]
        return (info, None) if df.empty else (info, df)
    except FileNotFoundError:
        _err(f"{ticker} not found in local warehouse.")
        return {}, None
    except Exception as e:
        _err(f"Error loading {ticker}: {e}")
        return {}, None


def _calculate_indicators(
    ticker: str,
    history: pd.DataFrame,
    market_returns: pd.Series | None,
) -> pd.DataFrame | None:
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
            _warn(f"Not enough data for {ticker} after computing indicators.")
            return None

        if market_returns is not None:
            combined = pd.concat([df["Stock_Return"], market_returns], axis=1).dropna()
            combined.columns = ["Stock_Return", "Market_Return"]
            slope, _, r_value, _, _ = stats.linregress(
                combined["Market_Return"], combined["Stock_Return"]
            )
            df.attrs["beta"]      = round(slope, 4)
            df.attrs["r_squared"] = round(r_value ** 2, 4)
        else:
            df.attrs["beta"]      = 1.0
            df.attrs["r_squared"] = 0.0

        return df
    except Exception as e:
        _err(f"Error computing indicators for {ticker}: {e}")
        return None


def _calculate_metrics(returns: pd.Series, beta: float, annual_rf: float) -> dict:
    daily_rf          = annual_rf / 252
    annualized_return = (1 + returns.mean()) ** 252 - 1
    excess            = returns - daily_rf
    sharpe            = (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)
    return {
        "Sharpe Annualized": round(sharpe, 4),
        "Annualized Return": round(annualized_return, 4),
        "Beta"             : beta,
    }


# ─── PRINTING ─────────────────────────────────────────────────────────────────

def _print_stock_info(ticker: str, info: dict) -> None:
    _section(f"{ticker}  ·  Basic Info")
    cap = f"${info['marketCap']:,}" if info["marketCap"] != "N/A" else "N/A"
    print(f"    Price         : ${info['currentPrice']}")
    print(f"    Market Cap    : {cap}")
    print(f"    52W High      : ${info['fiftyTwoWeekHigh']}")
    print(f"    52W Low       : ${info['fiftyTwoWeekLow']}")


def _print_indicators(ticker: str, df: pd.DataFrame) -> None:
    _section(f"{ticker}  ·  Latest Indicators")
    print(f"    MA50 / MA200  : {df['MA50'].iloc[-1]:.2f}  /  {df['MA200'].iloc[-1]:.2f}")
    print(f"    EMA20         : {df['EMA20'].iloc[-1]:.2f}")
    print(f"    RSI           : {df['RSI'].iloc[-1]:.2f}")
    print(f"    MACD          : {df['MACD'].iloc[-1]:.4f}  |  Signal: {df['Signal'].iloc[-1]:.4f}")
    print(f"    Stoch %K/%D   : {df['%K'].iloc[-1]:.2f}  /  {df['%D'].iloc[-1]:.2f}")
    print(f"    Beta          : {df.attrs['beta']:.3f}  |  R²: {df.attrs['r_squared']:.4f}")


def _print_metrics(ticker: str, metrics: dict) -> None:
    _section(f"{ticker}  ·  Financial Metrics")
    print(f"    Sharpe Ratio  : {metrics['Sharpe Annualized']:.4f}")
    print(f"    Ann. Return   : {metrics['Annualized Return'] * 100:.2f}%")
    print(f"    Beta          : {metrics['Beta']:.4f}")


# ─── CHARTS ───────────────────────────────────────────────────────────────────

def _plot_stock(ticker: str, df: pd.DataFrame) -> None:
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
        _warn(f"Could not plot charts for {ticker}: {e}")


def _analyze_seasonality(ticker: str, df: pd.DataFrame) -> str | None:
    try:
        monthly          = df.copy()
        monthly["Month"] = monthly.index.month
        monthly_ret      = monthly.groupby("Month")["Stock_Return"].mean() * 100

        _section(f"{ticker}  ·  Monthly Seasonality")
        print(monthly_ret.round(3).to_string())
        print(f"\n    Best month  : {monthly_ret.idxmax()}")
        print(f"    Worst month : {monthly_ret.idxmin()}")

        fig, ax = plt.subplots(figsize=(10, 5))
        monthly_ret.plot(kind="bar", ax=ax, color="teal", edgecolor="black")
        ax.set_title(f"{ticker} — Average Monthly Return (%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Avg Return (%)")
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        plt.xticks(rotation=0); plt.tight_layout()

        path = f"{ticker}_seasonality.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
    except Exception as e:
        _warn(f"Could not plot seasonality for {ticker}: {e}")
        return None


def _calculate_correlation(all_data: dict) -> None:
    valid = {k: v for k, v in all_data.items() if v is not None}
    if len(valid) < 2:
        return
    corr = pd.concat(
        [d["Stock_Return"].rename(t) for t, d in valid.items()], axis=1
    ).corr()
    _section("Portfolio  ·  Correlation Matrix")
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
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
        ax.set_title("Cross-Correlation Matrix", fontsize=14, fontweight="bold")
        plt.grid(False); plt.tight_layout(); plt.show()
    except Exception as e:
        _warn(f"Could not plot correlation matrix: {e}")


# ─── SIGNAL LOGIC ─────────────────────────────────────────────────────────────

def _compute_final_signal(
    df            : pd.DataFrame,
    info          : dict,
    metrics       : dict,
    sentiment     : dict,
    market_returns: pd.Series | None = None,
) -> dict:
    result         = generate_signal(df, info, metrics, market_returns)
    base_score     = result.get("score", 0)
    sent_score     = sentiment.get("score", 0)
    adjusted_score = base_score + sent_score

    if sent_score != 0:
        result["reasons"].append(
            f"Sentiment: {sentiment.get('label', 'NEUTRAL')} "
            f"(compound: {sentiment.get('compound', 0):.4f}, "
            f"adjustment: {'+' if sent_score > 0 else ''}{sent_score})."
        )

    is_bullish = df["Close"].iloc[-1] > df["MA200"].iloc[-1]

    if adjusted_score >= BUY_THRESHOLD:
        final = "STRONG BUY" if (adjusted_score >= 10 and is_bullish) else "BUY"
    elif adjusted_score <= SELL_THRESHOLD:
        final = "STRONG SELL" if (adjusted_score <= -10 and not is_bullish) else "SELL"
    else:
        final = "HOLD"

    return {
        "signal"         : final,
        "original_signal": result.get("signal", "UNKNOWN"),
        "base_score"     : base_score,
        "adjusted_score" : adjusted_score,
        "sent_score"     : sent_score,
        "reasons"        : result.get("reasons", []),
    }


def _print_signal(ticker: str, res: dict) -> None:
    icon   = SIGNAL_ICONS.get(res["signal"], "⬜")
    change = (f"  (was {res['original_signal']} before sentiment)"
              if res["signal"] != res["original_signal"] else "")
    print(f"\n  {icon}  {ticker:<6}  →  {res['signal']}{change}")
    print(f"       Score : {res['base_score']} → {res['adjusted_score']}  (after sentiment)")
    if res["signal"] == "HOLD":
        if res["sent_score"] >= 2:
            _warn("Sentiment bullish — watch for technical confirmation.")
        elif res["sent_score"] <= -2:
            _warn("Sentiment bearish — defensive stance advised.")
    for reason in res["reasons"]:
        print(f"       · {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Portfolio Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _run_analysis(name_to_ticker: dict) -> None:
    _section("Mode 1 — Portfolio Analysis")

    num   = _prompt_int("Number of stocks to analyze", 1, 20)
    years = _prompt_int("Years of historical data",    1, 10)

    print(f"\n  Risk-Free Rate  (current 10Y US Treasury ≈ 4.2%)")
    print(f"  Press Enter for 4.0% default, or type a value (e.g. 4.2).")
    raw_rf = input("  Annual RF rate [%]: ").strip()
    try:
        annual_rf = max(0.0, min(float(raw_rf) / 100 if raw_rf else 0.04, 0.20))
    except ValueError:
        annual_rf = 0.04
    _ok(f"Using risk-free rate: {annual_rf * 100:.2f}%")

    tickers        = _collect_tickers(num, name_to_ticker)
    market_returns = _get_market_returns()

    all_data, stock_info, all_metrics, all_sentiment = {}, {}, {}, {}

    for ticker in tickers:
        print(f"\n{BAR}\n  Analyzing  {ticker}\n{BAR}")

        info, history      = _fetch_ticker_data(ticker, years)
        stock_info[ticker] = info
        _print_stock_info(ticker, info)

        if history is None:
            all_data[ticker] = None
            continue

        df               = _calculate_indicators(ticker, history, market_returns)
        all_data[ticker] = df

        if df is not None:
            _print_indicators(ticker, df)
            metrics             = _calculate_metrics(df["Stock_Return"], df.attrs["beta"], annual_rf)
            all_metrics[ticker] = metrics
            _print_metrics(ticker, metrics)

        sentiment             = analyze_sentiment(ticker)
        all_sentiment[ticker] = sentiment
        print_sentiment(ticker, sentiment)

    # Charts
    seasonality_charts = {}
    for ticker in tickers:
        if all_data.get(ticker) is not None:
            _plot_stock(ticker, all_data[ticker])
            path = _analyze_seasonality(ticker, all_data[ticker])
            if path:
                seasonality_charts[ticker] = path

    _calculate_correlation(all_data)

    # Signal Summary
    _section("Signal Summary")
    signal_results = {}
    for ticker in tickers:
        df = all_data.get(ticker)
        if df is None:
            continue
        res = _compute_final_signal(
            df, stock_info[ticker],
            all_metrics.get(ticker, {}),
            all_sentiment.get(ticker, {}),
            market_returns,
        )
        signal_results[ticker] = res
        _print_signal(ticker, res)

    # PDF Report
    print(f"\n{THIN_BAR}")
    print("  Generating PDF report…")
    generate_pdf_report(
        all_data, stock_info, all_metrics,
        tickers, all_sentiment, seasonality_charts,
    )
    _ok("Report saved successfully.")
    print(THIN_BAR)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Backtesting
# ═══════════════════════════════════════════════════════════════════════════════

def _run_backtest() -> None:
    _section("Mode 2 — Backtesting")
    from backtest import main as _backtest_main
    _backtest_main()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — Watchlist Scanner
# ═══════════════════════════════════════════════════════════════════════════════

def _run_scanner(name_to_ticker: dict) -> None:
    _section("Mode 3 — Watchlist Scanner")
    from watchlist_scanner import scan_watchlist

    print("  Scan options (press Enter to use defaults):")

    raw_top = input("  Top N results     [default 20]: ").strip()
    top_n   = int(raw_top) if raw_top.isdigit() else 20

    raw_min = input("  Min signal score  [default  4]: ").strip()
    min_score = int(raw_min) if raw_min.lstrip("-").isdigit() else 4

    raw_tickers = input(
        "  Specific tickers  [Enter = all companies.json]: "
    ).strip()
    tickers = None
    if raw_tickers:
        raw_list = [t.strip() for t in raw_tickers.replace(",", " ").split()]
        tickers  = [name_to_ticker.get(t.upper(), t.upper()) for t in raw_list]

    raw_export = input("  Export CSV path   [Enter = skip]: ").strip()
    export     = raw_export if raw_export else None

    scan_watchlist(
        tickers   = tickers,
        top_n     = top_n,
        min_score = min_score,
        export    = export,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 4 — ML Predictor
# ═══════════════════════════════════════════════════════════════════════════════

def _run_ml_predictor() -> None:
    _section("Mode 4 — ML Predictor")
    from ml_predictor import BacktestPredictor

    print("  Options:")
    print("    [1]  Full report   (train + evaluate + feature importance)")
    print("    [2]  Train only")
    print("    [3]  Evaluate only (uses existing trained model)")
    print("    [4]  Feature importance")
    print()
    sub = _prompt_choice("Select", {"1", "2", "3", "4"})

    history_file = input(
        f"  History file [default: backtest_history.json]: "
    ).strip() or "backtest_history.json"

    if not Path(history_file).exists():
        _err(f"'{history_file}' not found. Run mode [2] first to generate backtest data.")
        return

    predictor = BacktestPredictor(history_file)

    if sub == "1":
        predictor.full_report()
    elif sub == "2":
        predictor.train(verbose=True)
    elif sub == "3":
        predictor.train(verbose=False)
        predictor.evaluate()
    elif sub == "4":
        raw_n  = input("  Top N features [default 10]: ").strip()
        top_n  = int(raw_n) if raw_n.isdigit() else 10
        predictor.train(verbose=False)
        predictor.feature_importance(top_n=top_n)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 5 — Warehouse Manager
# ═══════════════════════════════════════════════════════════════════════════════

def _run_warehouse() -> None:
    _section("Mode 5 — Warehouse Manager")
    from stock_warehouse import weekly_update

    print("  Options:")
    print("    [1]  Status       — show all stored symbols")
    print("    [2]  Update       — download / refresh all symbols")
    print("    [3]  Inspect      — preview rows for a specific ticker")
    print()
    sub = _prompt_choice("Select", {"1", "2", "3"})

    if sub == "1":
        warehouse_status()

    elif sub == "2":
        print()
        _warn("This will download data for all symbols in companies.json.")
        confirm = input("  Continue? (y/n): ").strip().lower()
        if confirm in ("y", "yes"):
            weekly_update()
        else:
            print("  Cancelled.")

    elif sub == "3":
        ticker = input("  Ticker symbol: ").strip().upper()
        try:
            df = load_local(ticker)
            raw_n = input("  Rows to preview [default 10]: ").strip()
            n     = int(raw_n) if raw_n.isdigit() else 10
            _section(f"{ticker}  ·  Last {n} rows")
            print(df.tail(n).to_string(index=False))
            print(f"\n  Total rows: {len(df):,}")
            print(f"  Date range: {df['Date'].iloc[0]}  →  {df['Date'].iloc[-1]}")
        except FileNotFoundError:
            _err(f"'{ticker}' not found in warehouse. Run update first.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _banner()
    warehouse_status()

    name_to_ticker = load_companies()
    _ok(f"Loaded {len(name_to_ticker):,} companies from companies.json")

    while True:
        mode = _mode_menu()

        if   mode == "1": _run_analysis(name_to_ticker)
        elif mode == "2": _run_backtest()
        elif mode == "3": _run_scanner(name_to_ticker)
        elif mode == "4": _run_ml_predictor()
        elif mode == "5": _run_warehouse()

        print(f"\n{THIN_BAR}")
        again = input("  Return to main menu? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print(f"\n  MarketLab session ended.  {datetime.now().strftime('%H:%M:%S')}")
            print(f"{BAR}\n")
            break


if __name__ == "__main__":
    main()
