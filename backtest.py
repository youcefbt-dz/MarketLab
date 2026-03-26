import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas_ta as ta
from scipy import stats
from datetime import datetime
import json
import csv
import os

from thefuzz import process
import yfinance as yf
from signals import generate_signal, BUY_THRESHOLD, SELL_THRESHOLD

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

COMMISSION   = 0.001   # 0.1% per trade
SLIPPAGE     = 0.001   # 0.1% slippage on entry/exit
COOLDOWN     = 5       # days to wait after a loss before re-entering
RISK_REWARD  = 2.3     # Take Profit = risk × 2.3
RESULTS_DIR  = "backtest_results"

# ── Dynamic Position Sizing based on Signal Strength ─────────────────────────
POSITION_SIZE_MAP = {
    "STRONG BUY":  0.4,   # 35% — إشارة قوية جداً (score ≥10)
    "BUY":         0.22,   # 22% — إشارة جيدة (score 5-9)
    "STRONG SELL": 0.4,   # 35% — Short position (future feature)
    "SELL":        0.22,   # 22% — Weak short
}
DEFAULT_POSITION_PCT = 0.15  # 15% — Fallback for weak signals
MAX_PORTFOLIO_EXPOSURE = 0.70  # Maximum 70% of capital deployed at once

BAR      = "═" * 58
THIN_BAR = "─" * 58


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_position_size(signal: str, score: int, available_cash: float, 
                      current_exposure: float = 0.0) -> float:

    # Base allocation from signal strength
    base_pct = POSITION_SIZE_MAP.get(signal, DEFAULT_POSITION_PCT)
    
    # Check portfolio-level cap
    remaining_capacity = MAX_PORTFOLIO_EXPOSURE - current_exposure
    if remaining_capacity <= 0:
        return 0.0  # Portfolio fully deployed
    
    # Use minimum of (signal allocation, remaining capacity)
    final_pct = min(base_pct, remaining_capacity)
    
    # Safety: never exceed 40% in a single position
    final_pct = min(final_pct, 0.40)
    
    return available_cash * final_pct


def section(title: str) -> None:
    print(f"\n{THIN_BAR}\n  {title}\n{THIN_BAR}")

def prompt_float(prompt: str, default: float) -> float:
    raw = input(f"  {prompt} [default {default}]: ").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default

def prompt_int(prompt: str, lo: int, hi: int, default: int) -> int:
    raw = input(f"  {prompt} [{lo}–{hi}, default {default}]: ").strip()
    try:
        v = int(raw) if raw else default
        return v if lo <= v <= hi else default
    except ValueError:
        return default

def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── DATA PREPARATION ─────────────────────────────────────────────────────────

def fetch_and_prepare(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download OHLCV data and compute all indicators without look-ahead bias.
    All rolling calculations use only past data at each point in time.
    """
    try:
        df = yf.Ticker(ticker).history(start=start, end=end)
        if df.empty or len(df) < 250:
            print(f"  ⚠  Not enough data for {ticker}.")
            return None

        df["Stock_Return"] = df["Close"].pct_change()
        df["MA50"]         = df["Close"].rolling(50).mean()
        df["MA200"]        = df["Close"].rolling(200).mean()
        df["EMA20"]        = df["Close"].ewm(span=20, adjust=False).mean()
        df["RSI"]          = ta.rsi(df["Close"], length=14)
        df["BB_middle"]    = df["Close"].rolling(20).mean()
        df["BB_std"]       = df["Close"].rolling(20).std()
        df["BB_upper"]     = df["BB_middle"] + df["BB_std"] * 2
        df["BB_lower"]     = df["BB_middle"] - df["BB_std"] * 2
        df["MACD"]         = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
        df["Signal"]       = df["MACD"].ewm(span=9, adjust=False).mean()
        df["Histogram"]    = df["MACD"] - df["Signal"]
        df["L14"]          = df["Low"].rolling(14).min()
        df["H14"]          = df["High"].rolling(14).max()
        df["%K"]           = (df["Close"] - df["L14"]) / (df["H14"] - df["L14"]) * 100
        df["%D"]           = df["%K"].rolling(3).mean()

        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"  ✗  Failed to fetch {ticker}: {e}")
        return None

def fetch_market_returns(start: str, end: str) -> pd.Series:
    df = yf.Ticker("^GSPC").history(start=start, end=end)
    return df["Close"].pct_change().rename("Market_Return")


# ─── CORE SIMULATION ──────────────────────────────────────────────────────────

def _precompute_rolling_metrics(
    df: pd.DataFrame,
    mkt_returns: pd.Series,
    annual_rf: float,
    window: int = 252,
) -> pd.DataFrame:
    """
    Pre-compute rolling Sharpe, Beta, and Annualized Return using vectorized
    rolling windows — eliminates the O(N²) linregress loop.

    window = 252 trading days (1 year) for all rolling calculations.
    """
    ret   = df["Stock_Return"]
    mkt   = mkt_returns.reindex(df.index)
    daily_rf = annual_rf / 252

    # Rolling Sharpe
    excess        = ret - daily_rf
    roll_mean     = excess.rolling(window).mean()
    roll_std      = excess.rolling(window).std()
    rolling_sharpe = (roll_mean / (roll_std + 1e-9)) * np.sqrt(252)

    # Rolling Annualized Return
    rolling_ann_ret = (1 + ret.rolling(window).mean()) ** 252 - 1

    # Rolling Beta via rolling covariance / variance
    roll_cov  = ret.rolling(window).cov(mkt)
    roll_var  = mkt.rolling(window).var()
    rolling_beta = roll_cov / (roll_var + 1e-9)

    return pd.DataFrame({
        "rolling_sharpe":  rolling_sharpe,
        "rolling_ann_ret": rolling_ann_ret,
        "rolling_beta":    rolling_beta,
    }, index=df.index)


def run_backtest(
    ticker:      str,
    df:          pd.DataFrame,
    mkt_returns: pd.Series,
    initial_cash: float,
    annual_rf:   float = 0.04,
) -> dict:
    """
    Walk-forward simulation — at each bar only data up to that bar is visible.
    Manages open positions, SL/TP hits, commissions, and slippage.

    Optimizations applied:
      (1) Rolling metrics pre-computed vectorially — O(N) instead of O(N²).
      (2) Gap-down / gap-up detection for realistic SL/TP exit prices.
      (3) Dynamic risk-free rate passed as parameter.
    """
    # ── Pre-compute rolling metrics once (O(N)) ───────────────────────────────
    rolling = _precompute_rolling_metrics(df, mkt_returns, annual_rf)

    cash          = initial_cash
    position      = None
    equity_curve  = []
    trades        = []
    cooldown_left = 0

    dummy_info = {
        "currentPrice": 0, "marketCap": "N/A",
        "fiftyTwoWeekHigh": 0, "fiftyTwoWeekLow": 0,
    }

    warmup = 252   # aligned with rolling window

    for i in range(warmup, len(df)):
        today      = df.index[i]
        open_price = df["Open"].iloc[i]
        price      = df["Close"].iloc[i]
        high       = df["High"].iloc[i]
        low        = df["Low"].iloc[i]
        equity     = cash + (position["shares"] * price if position else 0)
        equity_curve.append({"date": today, "equity": equity})

        if cooldown_left > 0:
            cooldown_left -= 1

        # ── Check open position for SL / TP ──────────────────────────────────
        if position:
            sl = position["sl"]
            tp = position["tp"]

            # Gap-down check (2): if Open gaps below SL, exit at Open (not SL)
            gapped_below_sl = open_price < sl
            gapped_above_tp = open_price > tp

            if gapped_below_sl:
                exit_price = open_price          # realistic gap-down exit
                reason     = "Hit SL (Gap Down)"
            elif gapped_above_tp:
                exit_price = open_price          # realistic gap-up exit
                reason     = "Hit TP (Gap Up)"
            elif low <= sl:
                exit_price = sl
                reason     = "Hit SL"
            elif high >= tp:
                exit_price = tp
                reason     = "Hit TP"
            else:
                exit_price = None
                reason     = None

            if exit_price is not None:
                exit_price *= (1 - SLIPPAGE) * (1 - COMMISSION)
                proceeds    = position["shares"] * exit_price
                pnl         = proceeds - position["cost"]
                pnl_pct     = pnl / position["cost"] * 100
                result      = "WIN" if pnl > 0 else "LOSS"

                trades.append({
                    "ticker":       ticker,
                    "entry_date":   position["entry_date"].strftime("%Y-%m-%d"),
                    "entry_price":  round(position["entry_price"], 4),
                    "signal":       position["signal"],
                    "score":        position["score"],
                    "position_pct": position.get("position_pct", 0),
                    "stop_loss":    round(sl, 4),
                    "take_profit":  round(tp, 4),
                    "exit_date":    today.strftime("%Y-%m-%d"),
                    "exit_price":   round(exit_price, 4),
                    "exit_reason":  reason,
                    "pnl":          round(pnl, 2),
                    "pnl_pct":      round(pnl_pct, 2),
                    "result":       result,
                })

                cash         += proceeds
                position      = None
                if result == "LOSS":
                    cooldown_left = COOLDOWN

        # ── Generate signal using pre-computed rolling metrics (O(1) lookup) ──
        if position is None and cooldown_left == 0:
            row = rolling.iloc[i]
            if pd.isna(row["rolling_sharpe"]):
                continue

            metrics = {
                "Sharpe Annualized": round(row["rolling_sharpe"],  4),
                "Annualized Return": round(row["rolling_ann_ret"], 4),
                "Beta":              round(row["rolling_beta"],    4),
            }

            # Pass only current-bar slice to avoid look-ahead in generate_signal
            slice_df  = df.iloc[: i + 1]
            mkt_slice = mkt_returns.reindex(slice_df.index)

            result = generate_signal(slice_df, dummy_info, metrics, mkt_slice)
            sig    = result.get("signal", "HOLD")
            score  = result.get("score",  0)

            if sig in ("BUY", "STRONG BUY"):
                entry_price  = price * (1 + SLIPPAGE)               # slippage on entry
                entry_price *= (1 + COMMISSION)                     # commission on entry
                
                # ── DYNAMIC POSITION SIZING ───────────────────────────────────────
                # Calculate current portfolio exposure (always 0 since position is None here)
                current_exposure = 0.0
                alloc = get_position_size(sig, score, cash, current_exposure)
                shares = alloc / entry_price

                if shares > 0 and cash >= alloc:
                    # ── EXIT STRATEGY from signals.py ────────────────────────────
                    exit_levels = result.get("exit_levels", {})
                    sl = exit_levels.get("stop_loss",   price * 0.95)
                    tp = exit_levels.get("take_profit", price * (1 + (price - sl) / price * RISK_REWARD))

                    position = {
                        "entry_price": entry_price,
                        "entry_date":  today,
                        "shares":      shares,
                        "cost":        shares * entry_price,
                        "sl":          sl,
                        "tp":          tp,
                        "signal":      sig,
                        "score":       score,
                        "position_pct": round(alloc / (cash + alloc) * 100, 1),  # Track allocation %
                    }
                    cash -= shares * entry_price

    # ── Close any open position at end ───────────────────────────────────────
    if position:
        last_price  = df["Close"].iloc[-1] * (1 - SLIPPAGE) * (1 - COMMISSION)
        proceeds    = position["shares"] * last_price
        pnl         = proceeds - position["cost"]
        pnl_pct     = pnl / position["cost"] * 100

        trades.append({
            "ticker":       ticker,
            "entry_date":   position["entry_date"].strftime("%Y-%m-%d"),
            "entry_price":  round(position["entry_price"], 4),
            "signal":       position["signal"],
            "score":        position["score"],
            "position_pct": position.get("position_pct", 0),
            "stop_loss":    round(position["sl"], 4),
            "take_profit":  round(position["tp"], 4),
            "exit_date":    df.index[-1].strftime("%Y-%m-%d"),
            "exit_price":   round(last_price, 4),
            "exit_reason":  "End of Period",
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "result":       "WIN" if pnl > 0 else "LOSS",
        })
        cash += proceeds

    final_equity = cash
    return {
        "ticker":        ticker,
        "initial_cash":  initial_cash,
        "final_equity":  round(final_equity, 2),
        "total_return":  round((final_equity - initial_cash) / initial_cash * 100, 2),
        "trades":        trades,
        "equity_curve":  equity_curve,
    }


# ─── PERFORMANCE METRICS ──────────────────────────────────────────────────────

def compute_metrics(result: dict) -> dict:
    trades = result["trades"]
    if not trades:
        return {"verdict": "NO TRADES", "win_rate": 0, "profit_factor": 0,
                "max_drawdown": 0, "total_trades": 0, "total_return": result["total_return"]}

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]

    win_rate      = len(wins) / len(trades) * 100
    gross_profit  = sum(t["pnl"] for t in wins)
    gross_loss    = abs(sum(t["pnl"] for t in losses)) + 1e-9
    profit_factor = round(gross_profit / gross_loss, 2)

    # Max Drawdown from equity curve
    eq     = pd.Series([e["equity"] for e in result["equity_curve"]])
    peak   = eq.cummax()
    dd     = (eq - peak) / peak * 100
    max_dd = round(dd.min(), 2)

    # Sharpe on daily equity returns
    eq_returns = eq.pct_change().dropna()
    sharpe     = round((eq_returns.mean() / (eq_returns.std() + 1e-9)) * np.sqrt(252), 2)

    # ── NEW METRICS ───────────────────────────────────────────────────────────
    
    # 1. Average R-Multiple (Risk/Reward realized)
    r_multiples = []
    for t in trades:
        risk = abs(t["entry_price"] - t["stop_loss"])
        if risk > 0:
            r_multiple = t["pnl"] / (risk * (t.get("position_pct", 20) / 100) * result["initial_cash"])
            r_multiples.append(r_multiple)
    avg_r_multiple = round(np.mean(r_multiples), 2) if r_multiples else 0.0
    
    # 2. Max Consecutive Losses
    max_consecutive_losses = 0
    current_streak = 0
    for t in trades:
        if t["result"] == "LOSS":
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0
    
    # 3. Average Position Size by Signal Type
    signal_sizes = {}
    for t in trades:
        sig = t["signal"]
        pct = t.get("position_pct", 20)
        if sig not in signal_sizes:
            signal_sizes[sig] = []
        signal_sizes[sig].append(pct)
    
    avg_position_by_signal = {
        sig: round(np.mean(sizes), 1) 
        for sig, sizes in signal_sizes.items()
    }
    
    # 4. Exit Reason Breakdown
    exit_reasons = {}
    for t in trades:
        reason = t.get("exit_reason", "Unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Verdict
    passed = (
        win_rate      >= 50  and
        profit_factor >= 1.3 and
        result["total_return"] > 0
    )
    verdict = "PASS ✅" if passed else "FAIL ❌"

    return {
        "verdict":                verdict,
        "passed":                 passed,
        "total_trades":           len(trades),
        "win_rate":               round(win_rate, 1),
        "profit_factor":          profit_factor,
        "max_drawdown":           max_dd,
        "sharpe":                 sharpe,
        "total_return":           result["total_return"],
        "avg_r_multiple":         avg_r_multiple,
        "max_consecutive_losses": max_consecutive_losses,
        "avg_position_by_signal": avg_position_by_signal,
        "exit_reasons":           exit_reasons,
        "sharpe":         sharpe,
        "total_return":   result["total_return"],
        "gross_profit":   round(gross_profit, 2),
        "gross_loss":     round(gross_loss,   2),
    }


# ─── BENCHMARK ────────────────────────────────────────────────────────────────

def compute_benchmark(ticker: str, start: str, end: str, initial_cash: float) -> float:
    """Buy & Hold return for the same ticker over the same period."""
    try:
        df    = yf.Ticker(ticker).history(start=start, end=end)
        p_start = df["Close"].iloc[0]
        p_end   = df["Close"].iloc[-1]
        return round((p_end - p_start) / p_start * 100, 2)
    except Exception:
        return 0.0


# ─── SAVING RESULTS ───────────────────────────────────────────────────────────

def save_trades_csv(all_trades: list, run_date: str) -> str:
    path = os.path.join(RESULTS_DIR, f"trades_{run_date}.csv")
    if not all_trades:
        return path
    keys = all_trades[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_trades)
    return path

class _JSONEncoder(json.JSONEncoder):
    """Handle numpy scalar types that standard json cannot serialize."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

def save_summary_json(summary: dict, run_date: str) -> str:
    path = os.path.join(RESULTS_DIR, f"summary_{run_date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=_JSONEncoder)
    return path

def save_summary_txt(summary: dict, run_date: str) -> str:
    path   = os.path.join(RESULTS_DIR, f"report_{run_date}.txt")
    passed = summary["strategy"]["passed_count"]
    total  = summary["strategy"]["total_tickers"]
    acc    = summary["strategy"]["accuracy_pct"]
    verdict = summary["strategy"]["overall_verdict"]

    lines = [
        "=" * 58,
        "  MarketLab — Backtest Strategy Report",
        f"  Date: {run_date}",
        "=" * 58,
        "",
        f"  Tickers Tested   : {total}",
        f"  Period           : {summary['period']['start']} → {summary['period']['end']}",
        f"  Initial Capital  : ${summary['settings']['initial_cash']:,.0f}",
        f"  Commission       : {COMMISSION*100:.1f}%  |  Slippage: {SLIPPAGE*100:.1f}%",
        "",
        "─" * 58,
        "  Per-Ticker Results",
        "─" * 58,
    ]

    for t, m in summary["tickers"].items():
        bh = summary["benchmarks"].get(t, 0)
        lines.append(
            f"  {t:<6}  {m['verdict']:<10}  "
            f"Return: {m['total_return']:+.1f}%  "
            f"B&H: {bh:+.1f}%  "
            f"WR: {m['win_rate']:.0f}%  "
            f"PF: {m['profit_factor']:.2f}  "
            f"DD: {m['max_drawdown']:.1f}%  "
            f"Trades: {m['total_trades']}"
        )

    lines += [
        "",
        "─" * 58,
        "  Strategy Accuracy",
        "─" * 58,
        f"  Signals Correct  : {passed}/{total}",
        f"  Accuracy         : {acc:.1f}%",
        f"  Overall Verdict  : {verdict}",
        "=" * 58,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ─── EQUITY CURVE PLOT ────────────────────────────────────────────────────────

def plot_equity_curves(results: dict, benchmarks: dict, initial_cash: float, run_date: str) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))

    for ticker, res in results.items():
        if not res["equity_curve"]:
            continue
        eq   = pd.Series(
            [e["equity"] for e in res["equity_curve"]],
            index=[e["date"]  for e in res["equity_curve"]]
        )
        norm = eq / initial_cash * 100
        ax.plot(norm, label=f"{ticker} (strategy)", linewidth=1.5)

    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="Starting Capital")
    ax.set_title("Equity Curves — Strategy vs Starting Capital", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (% of initial)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"equity_curve_{run_date}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─── PRINTING ─────────────────────────────────────────────────────────────────

def print_ticker_result(ticker: str, metrics: dict, benchmark: float) -> None:
    section(f"{ticker}  ·  Backtest Result")
    print(f"    Verdict        : {metrics['verdict']}")
    print(f"    Total Return   : {metrics['total_return']:+.2f}%  (Buy & Hold: {benchmark:+.2f}%)")
    print(f"    Total Trades   : {metrics['total_trades']}")
    print(f"    Win Rate       : {metrics['win_rate']:.1f}%")
    print(f"    Profit Factor  : {metrics['profit_factor']:.2f}")
    print(f"    Max Drawdown   : {metrics['max_drawdown']:.2f}%")
    print(f"    Sharpe Ratio   : {metrics['sharpe']:.2f}")
    
    # ── NEW METRICS ───────────────────────────────────────────────────────────
    print(f"    Avg R-Multiple : {metrics.get('avg_r_multiple', 0):.2f}  (Risk/Reward realized)")
    print(f"    Max Consec Loss: {metrics.get('max_consecutive_losses', 0)}")
    
    # Position Sizes by Signal
    if "avg_position_by_signal" in metrics and metrics["avg_position_by_signal"]:
        print("    Avg Position %:")
        for sig, pct in metrics["avg_position_by_signal"].items():
            print(f"        {sig:12s} : {pct:.1f}%")
    
    # Exit Reasons
    if "exit_reasons" in metrics and metrics["exit_reasons"]:
        print("    Exit Reasons:")
        for reason, count in sorted(metrics["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"        {reason:15s} : {count}")

def print_strategy_summary(summary: dict) -> None:
    s       = summary["strategy"]
    passed  = s["passed_count"]
    total   = s["total_tickers"]
    acc     = s["accuracy_pct"]
    verdict = s["overall_verdict"]

    print(f"\n{BAR}")
    print("  Strategy Accuracy Report")
    print(BAR)
    print(f"  Signals Correct  : {passed}/{total}")
    print(f"  Accuracy         : {acc:.1f}%")
    print(f"  Overall Verdict  : {verdict}")
    print(BAR)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BAR}")
    print("  MarketLab  ·  Backtesting Engine")
    print(BAR)

    ensure_results_dir()
    run_date = datetime.today().strftime("%Y-%m-%d")

    # ── User settings ─────────────────────────────────────────────────────────
    print()
    initial_cash = prompt_float("Initial capital ($)", 10000)
    start_year   = prompt_int("Backtest start year", 2015, 2023, 2020)
    start        = f"{start_year}-01-01"
    end          = datetime.today().strftime("%Y-%m-%d")

    print(f"  Risk-Free Rate  (current 10Y US Treasury ≈ 4.2%)")
    print(f"  Press Enter to use 4.0% default, or type a value (e.g. 4.2).")
    raw_rf    = input("  Annual RF rate [%]: ").strip()
    try:
        annual_rf = float(raw_rf) / 100 if raw_rf else 0.04
        annual_rf = max(0.0, min(annual_rf, 0.20))
    except ValueError:
        annual_rf = 0.04
    print(f"    ✔  Using risk-free rate: {annual_rf * 100:.2f}%")

    # ── Stock Selection — same UX as main.py ────────────────────────────────
    from main import load_companies, resolve_ticker, validate_ticker

    name_to_ticker = load_companies()

    num = prompt_int("Number of stocks to backtest", 1, 20, 3)

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

    print(f"\n  Loading S&P 500 benchmark data…")
    mkt_returns = fetch_market_returns(start, end)

    # ── Run backtest per ticker ───────────────────────────────────────────────
    all_results  = {}
    all_metrics  = {}
    all_trades   = []
    benchmarks   = {}

    for ticker in tickers:
        print(f"\n{BAR}")
        print(f"  Backtesting  {ticker}  ({start} → {end})")
        print(BAR)

        df = fetch_and_prepare(ticker, start, end)
        if df is None:
            continue

        result     = run_backtest(ticker, df, mkt_returns, initial_cash, annual_rf)
        metrics    = compute_metrics(result)
        benchmark  = compute_benchmark(ticker, start, end, initial_cash)

        all_results[ticker] = result
        all_metrics[ticker] = metrics
        benchmarks[ticker]  = benchmark
        all_trades.extend(result["trades"])

        print_ticker_result(ticker, metrics, benchmark)

    if not all_metrics:
        print("\n  ✗  No results to summarize.")
        return

    # ── Strategy-level summary ────────────────────────────────────────────────
    passed_count = sum(1 for m in all_metrics.values() if m.get("passed", False))
    total        = len(all_metrics)
    accuracy     = passed_count / total * 100
    overall      = "CREDIBLE STRATEGY ✅" if accuracy >= 60 else "NEEDS IMPROVEMENT ⚠️"

    summary = {
        "period":    {"start": start, "end": end},
        "settings":  {"initial_cash": initial_cash, "commission": COMMISSION, "slippage": SLIPPAGE},
        "tickers":   all_metrics,
        "benchmarks": benchmarks,
        "strategy":  {
            "passed_count":   passed_count,
            "total_tickers":  total,
            "accuracy_pct":   round(accuracy, 1),
            "overall_verdict": overall,
        },
    }

    print_strategy_summary(summary)

    # ── Save results ──────────────────────────────────────────────────────────
    section("Saving Results")
    csv_path  = save_trades_csv(all_trades,  run_date)
    json_path = save_summary_json(summary,   run_date)
    txt_path  = save_summary_txt(summary,    run_date)
    chart_path = plot_equity_curves(all_results, benchmarks, initial_cash, run_date)

    print(f"    ✔  Trades log   : {csv_path}")
    print(f"    ✔  Summary JSON : {json_path}")
    print(f"    ✔  Report TXT   : {txt_path}")
    print(f"    ✔  Equity chart : {chart_path}")
    print(f"\n{THIN_BAR}\n")


if __name__ == "__main__":
    main()    raw = input(f"  {prompt} [default {default}]: ").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default

def prompt_int(prompt: str, lo: int, hi: int, default: int) -> int:
    raw = input(f"  {prompt} [{lo}–{hi}, default {default}]: ").strip()
    try:
        v = int(raw) if raw else default
        return v if lo <= v <= hi else default
    except ValueError:
        return default

def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── DATA PREPARATION ─────────────────────────────────────────────────────────

def fetch_and_prepare(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download OHLCV data and compute all indicators without look-ahead bias.
    All rolling calculations use only past data at each point in time.
    """
    try:
        df = yf.Ticker(ticker).history(start=start, end=end)
        if df.empty or len(df) < 250:
            print(f"  ⚠  Not enough data for {ticker}.")
            return None

        df["Stock_Return"] = df["Close"].pct_change()
        df["MA50"]         = df["Close"].rolling(50).mean()
        df["MA200"]        = df["Close"].rolling(200).mean()
        df["EMA20"]        = df["Close"].ewm(span=20, adjust=False).mean()
        df["RSI"]          = ta.rsi(df["Close"], length=14)
        df["BB_middle"]    = df["Close"].rolling(20).mean()
        df["BB_std"]       = df["Close"].rolling(20).std()
        df["BB_upper"]     = df["BB_middle"] + df["BB_std"] * 2
        df["BB_lower"]     = df["BB_middle"] - df["BB_std"] * 2
        df["MACD"]         = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
        df["Signal"]       = df["MACD"].ewm(span=9, adjust=False).mean()
        df["Histogram"]    = df["MACD"] - df["Signal"]
        df["L14"]          = df["Low"].rolling(14).min()
        df["H14"]          = df["High"].rolling(14).max()
        df["%K"]           = (df["Close"] - df["L14"]) / (df["H14"] - df["L14"]) * 100
        df["%D"]           = df["%K"].rolling(3).mean()

        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"  ✗  Failed to fetch {ticker}: {e}")
        return None

def fetch_market_returns(start: str, end: str) -> pd.Series:
    df = yf.Ticker("^GSPC").history(start=start, end=end)
    return df["Close"].pct_change().rename("Market_Return")


# ─── CORE SIMULATION ──────────────────────────────────────────────────────────

def _precompute_rolling_metrics(
    df: pd.DataFrame,
    mkt_returns: pd.Series,
    annual_rf: float,
    window: int = 252,
) -> pd.DataFrame:
    """
    Pre-compute rolling Sharpe, Beta, and Annualized Return using vectorized
    rolling windows — eliminates the O(N²) linregress loop.

    window = 252 trading days (1 year) for all rolling calculations.
    """
    ret   = df["Stock_Return"]
    mkt   = mkt_returns.reindex(df.index)
    daily_rf = annual_rf / 252

    # Rolling Sharpe
    excess        = ret - daily_rf
    roll_mean     = excess.rolling(window).mean()
    roll_std      = excess.rolling(window).std()
    rolling_sharpe = (roll_mean / (roll_std + 1e-9)) * np.sqrt(252)

    # Rolling Annualized Return
    rolling_ann_ret = (1 + ret.rolling(window).mean()) ** 252 - 1

    # Rolling Beta via rolling covariance / variance
    roll_cov  = ret.rolling(window).cov(mkt)
    roll_var  = mkt.rolling(window).var()
    rolling_beta = roll_cov / (roll_var + 1e-9)

    return pd.DataFrame({
        "rolling_sharpe":  rolling_sharpe,
        "rolling_ann_ret": rolling_ann_ret,
        "rolling_beta":    rolling_beta,
    }, index=df.index)


def run_backtest(
    ticker:      str,
    df:          pd.DataFrame,
    mkt_returns: pd.Series,
    initial_cash: float,
    annual_rf:   float = 0.04,
) -> dict:
    """
    Walk-forward simulation — at each bar only data up to that bar is visible.
    Manages open positions, SL/TP hits, commissions, and slippage.

    Optimizations applied:
      (1) Rolling metrics pre-computed vectorially — O(N) instead of O(N²).
      (2) Gap-down / gap-up detection for realistic SL/TP exit prices.
      (3) Dynamic risk-free rate passed as parameter.
    """
    # ── Pre-compute rolling metrics once (O(N)) ───────────────────────────────
    rolling = _precompute_rolling_metrics(df, mkt_returns, annual_rf)

    cash          = initial_cash
    position      = None
    equity_curve  = []
    trades        = []
    cooldown_left = 0

    dummy_info = {
        "currentPrice": 0, "marketCap": "N/A",
        "fiftyTwoWeekHigh": 0, "fiftyTwoWeekLow": 0,
    }

    warmup = 252   # aligned with rolling window

    for i in range(warmup, len(df)):
        today      = df.index[i]
        open_price = df["Open"].iloc[i]
        price      = df["Close"].iloc[i]
        high       = df["High"].iloc[i]
        low        = df["Low"].iloc[i]
        equity     = cash + (position["shares"] * price if position else 0)
        equity_curve.append({"date": today, "equity": equity})

        if cooldown_left > 0:
            cooldown_left -= 1

        # ── Check open position for SL / TP ──────────────────────────────────
        if position:
            sl = position["sl"]
            tp = position["tp"]

            # Gap-down check (2): if Open gaps below SL, exit at Open (not SL)
            gapped_below_sl = open_price < sl
            gapped_above_tp = open_price > tp

            if gapped_below_sl:
                exit_price = open_price          # realistic gap-down exit
                reason     = "Hit SL (Gap Down)"
            elif gapped_above_tp:
                exit_price = open_price          # realistic gap-up exit
                reason     = "Hit TP (Gap Up)"
            elif low <= sl:
                exit_price = sl
                reason     = "Hit SL"
            elif high >= tp:
                exit_price = tp
                reason     = "Hit TP"
            else:
                exit_price = None
                reason     = None

            if exit_price is not None:
                exit_price *= (1 - SLIPPAGE) * (1 - COMMISSION)
                proceeds    = position["shares"] * exit_price
                pnl         = proceeds - position["cost"]
                pnl_pct     = pnl / position["cost"] * 100
                result      = "WIN" if pnl > 0 else "LOSS"

                trades.append({
                    "ticker":       ticker,
                    "entry_date":   position["entry_date"].strftime("%Y-%m-%d"),
                    "entry_price":  round(position["entry_price"], 4),
                    "signal":       position["signal"],
                    "score":        position["score"],
                    "stop_loss":    round(sl, 4),
                    "take_profit":  round(tp, 4),
                    "exit_date":    today.strftime("%Y-%m-%d"),
                    "exit_price":   round(exit_price, 4),
                    "exit_reason":  reason,
                    "pnl":          round(pnl, 2),
                    "pnl_pct":      round(pnl_pct, 2),
                    "result":       result,
                })

                cash         += proceeds
                position      = None
                if result == "LOSS":
                    cooldown_left = COOLDOWN

        # ── Generate signal using pre-computed rolling metrics (O(1) lookup) ──
        if position is None and cooldown_left == 0:
            row = rolling.iloc[i]
            if pd.isna(row["rolling_sharpe"]):
                continue

            metrics = {
                "Sharpe Annualized": round(row["rolling_sharpe"],  4),
                "Annualized Return": round(row["rolling_ann_ret"], 4),
                "Beta":              round(row["rolling_beta"],    4),
            }

            # Pass only current-bar slice to avoid look-ahead in generate_signal
            slice_df  = df.iloc[: i + 1]
            mkt_slice = mkt_returns.reindex(slice_df.index)

            result = generate_signal(slice_df, dummy_info, metrics, mkt_slice)
            sig    = result.get("signal", "HOLD")
            score  = result.get("score",  0)

            if sig in ("BUY", "STRONG BUY"):
                entry_price  = price * (1 + SLIPPAGE)               # slippage on entry
                entry_price *= (1 + COMMISSION)                     # commission on entry
                alloc        = cash * POSITION_PCT                  # Fixed Fractional
                shares       = alloc / entry_price

                if shares > 0 and cash >= alloc:
                    exit_levels = result.get("exit_levels", {})
                    sl = exit_levels.get("stop_loss",   price * 0.95)
                    tp = exit_levels.get("take_profit", price * (1 + (price - sl) / price * RISK_REWARD))

                    position = {
                        "entry_price": entry_price,
                        "entry_date":  today,
                        "shares":      shares,
                        "cost":        shares * entry_price,
                        "sl":          sl,
                        "tp":          tp,
                        "signal":      sig,
                        "score":       score,
                    }
                    cash -= shares * entry_price

    # ── Close any open position at end ───────────────────────────────────────
    if position:
        last_price  = df["Close"].iloc[-1] * (1 - SLIPPAGE) * (1 - COMMISSION)
        proceeds    = position["shares"] * last_price
        pnl         = proceeds - position["cost"]
        pnl_pct     = pnl / position["cost"] * 100

        trades.append({
            "ticker":       ticker,
            "entry_date":   position["entry_date"].strftime("%Y-%m-%d"),
            "entry_price":  round(position["entry_price"], 4),
            "signal":       position["signal"],
            "score":        position["score"],
            "stop_loss":    round(position["sl"], 4),
            "take_profit":  round(position["tp"], 4),
            "exit_date":    df.index[-1].strftime("%Y-%m-%d"),
            "exit_price":   round(last_price, 4),
            "exit_reason":  "End of Period",
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "result":       "WIN" if pnl > 0 else "LOSS",
        })
        cash += proceeds

    final_equity = cash
    return {
        "ticker":        ticker,
        "initial_cash":  initial_cash,
        "final_equity":  round(final_equity, 2),
        "total_return":  round((final_equity - initial_cash) / initial_cash * 100, 2),
        "trades":        trades,
        "equity_curve":  equity_curve,
    }


# ─── PERFORMANCE METRICS ──────────────────────────────────────────────────────

def compute_metrics(result: dict) -> dict:
    trades = result["trades"]
    if not trades:
        return {"verdict": "NO TRADES", "win_rate": 0, "profit_factor": 0,
                "max_drawdown": 0, "total_trades": 0, "total_return": result["total_return"]}

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]

    win_rate      = len(wins) / len(trades) * 100
    gross_profit  = sum(t["pnl"] for t in wins)
    gross_loss    = abs(sum(t["pnl"] for t in losses)) + 1e-9
    profit_factor = round(gross_profit / gross_loss, 2)

    # Max Drawdown from equity curve
    eq     = pd.Series([e["equity"] for e in result["equity_curve"]])
    peak   = eq.cummax()
    dd     = (eq - peak) / peak * 100
    max_dd = round(dd.min(), 2)

    # Sharpe on daily equity returns
    eq_returns = eq.pct_change().dropna()
    sharpe     = round((eq_returns.mean() / (eq_returns.std() + 1e-9)) * np.sqrt(252), 2)

    # Verdict
    passed = (
        win_rate      >= 50  and
        profit_factor >= 1.3 and
        result["total_return"] > 0
    )
    verdict = "PASS ✅" if passed else "FAIL ❌"

    return {
        "verdict":        verdict,
        "passed":         passed,
        "total_trades":   len(trades),
        "win_rate":       round(win_rate, 1),
        "profit_factor":  profit_factor,
        "max_drawdown":   max_dd,
        "sharpe":         sharpe,
        "total_return":   result["total_return"],
        "gross_profit":   round(gross_profit, 2),
        "gross_loss":     round(gross_loss,   2),
    }


# ─── BENCHMARK ────────────────────────────────────────────────────────────────

def compute_benchmark(ticker: str, start: str, end: str, initial_cash: float) -> float:
    """Buy & Hold return for the same ticker over the same period."""
    try:
        df    = yf.Ticker(ticker).history(start=start, end=end)
        p_start = df["Close"].iloc[0]
        p_end   = df["Close"].iloc[-1]
        return round((p_end - p_start) / p_start * 100, 2)
    except Exception:
        return 0.0


# ─── SAVING RESULTS ───────────────────────────────────────────────────────────

def save_trades_csv(all_trades: list, run_date: str) -> str:
    path = os.path.join(RESULTS_DIR, f"trades_{run_date}.csv")
    if not all_trades:
        return path
    keys = all_trades[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_trades)
    return path

class _JSONEncoder(json.JSONEncoder):
    """Handle numpy scalar types that standard json cannot serialize."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

def save_summary_json(summary: dict, run_date: str) -> str:
    path = os.path.join(RESULTS_DIR, f"summary_{run_date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=_JSONEncoder)
    return path

def save_summary_txt(summary: dict, run_date: str) -> str:
    path   = os.path.join(RESULTS_DIR, f"report_{run_date}.txt")
    passed = summary["strategy"]["passed_count"]
    total  = summary["strategy"]["total_tickers"]
    acc    = summary["strategy"]["accuracy_pct"]
    verdict = summary["strategy"]["overall_verdict"]

    lines = [
        "=" * 58,
        "  MarketLab — Backtest Strategy Report",
        f"  Date: {run_date}",
        "=" * 58,
        "",
        f"  Tickers Tested   : {total}",
        f"  Period           : {summary['period']['start']} → {summary['period']['end']}",
        f"  Initial Capital  : ${summary['settings']['initial_cash']:,.0f}",
        f"  Commission       : {COMMISSION*100:.1f}%  |  Slippage: {SLIPPAGE*100:.1f}%",
        "",
        "─" * 58,
        "  Per-Ticker Results",
        "─" * 58,
    ]

    for t, m in summary["tickers"].items():
        bh = summary["benchmarks"].get(t, 0)
        lines.append(
            f"  {t:<6}  {m['verdict']:<10}  "
            f"Return: {m['total_return']:+.1f}%  "
            f"B&H: {bh:+.1f}%  "
            f"WR: {m['win_rate']:.0f}%  "
            f"PF: {m['profit_factor']:.2f}  "
            f"DD: {m['max_drawdown']:.1f}%  "
            f"Trades: {m['total_trades']}"
        )

    lines += [
        "",
        "─" * 58,
        "  Strategy Accuracy",
        "─" * 58,
        f"  Signals Correct  : {passed}/{total}",
        f"  Accuracy         : {acc:.1f}%",
        f"  Overall Verdict  : {verdict}",
        "=" * 58,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ─── EQUITY CURVE PLOT ────────────────────────────────────────────────────────

def plot_equity_curves(results: dict, benchmarks: dict, initial_cash: float, run_date: str) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))

    for ticker, res in results.items():
        if not res["equity_curve"]:
            continue
        eq   = pd.Series(
            [e["equity"] for e in res["equity_curve"]],
            index=[e["date"]  for e in res["equity_curve"]]
        )
        norm = eq / initial_cash * 100
        ax.plot(norm, label=f"{ticker} (strategy)", linewidth=1.5)

    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="Starting Capital")
    ax.set_title("Equity Curves — Strategy vs Starting Capital", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (% of initial)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"equity_curve_{run_date}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─── PRINTING ─────────────────────────────────────────────────────────────────

def print_ticker_result(ticker: str, metrics: dict, benchmark: float) -> None:
    section(f"{ticker}  ·  Backtest Result")
    print(f"    Verdict        : {metrics['verdict']}")
    print(f"    Total Return   : {metrics['total_return']:+.2f}%  (Buy & Hold: {benchmark:+.2f}%)")
    print(f"    Total Trades   : {metrics['total_trades']}")
    print(f"    Win Rate       : {metrics['win_rate']:.1f}%")
    print(f"    Profit Factor  : {metrics['profit_factor']:.2f}")
    print(f"    Max Drawdown   : {metrics['max_drawdown']:.2f}%")
    print(f"    Sharpe Ratio   : {metrics['sharpe']:.2f}")

def print_strategy_summary(summary: dict) -> None:
    s       = summary["strategy"]
    passed  = s["passed_count"]
    total   = s["total_tickers"]
    acc     = s["accuracy_pct"]
    verdict = s["overall_verdict"]

    print(f"\n{BAR}")
    print("  Strategy Accuracy Report")
    print(BAR)
    print(f"  Signals Correct  : {passed}/{total}")
    print(f"  Accuracy         : {acc:.1f}%")
    print(f"  Overall Verdict  : {verdict}")
    print(BAR)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BAR}")
    print("  MarketLab  ·  Backtesting Engine")
    print(BAR)

    ensure_results_dir()
    run_date = datetime.today().strftime("%Y-%m-%d")

    # ── User settings ─────────────────────────────────────────────────────────
    print()
    initial_cash = prompt_float("Initial capital ($)", 10000)
    start_year   = prompt_int("Backtest start year", 2015, 2023, 2020)
    start        = f"{start_year}-01-01"
    end          = datetime.today().strftime("%Y-%m-%d")

    print(f"  Risk-Free Rate  (current 10Y US Treasury ≈ 4.2%)")
    print(f"  Press Enter to use 4.0% default, or type a value (e.g. 4.2).")
    raw_rf    = input("  Annual RF rate [%]: ").strip()
    try:
        annual_rf = float(raw_rf) / 100 if raw_rf else 0.04
        annual_rf = max(0.0, min(annual_rf, 0.20))
    except ValueError:
        annual_rf = 0.04
    print(f"    ✔  Using risk-free rate: {annual_rf * 100:.2f}%")

    # ── Stock Selection — same UX as main.py ────────────────────────────────
    from main import load_companies, resolve_ticker, validate_ticker

    name_to_ticker = load_companies()

    num = prompt_int("Number of stocks to backtest", 1, 20, 3)

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

    print(f"\n  Loading S&P 500 benchmark data…")
    mkt_returns = fetch_market_returns(start, end)

    # ── Run backtest per ticker ───────────────────────────────────────────────
    all_results  = {}
    all_metrics  = {}
    all_trades   = []
    benchmarks   = {}

    for ticker in tickers:
        print(f"\n{BAR}")
        print(f"  Backtesting  {ticker}  ({start} → {end})")
        print(BAR)

        df = fetch_and_prepare(ticker, start, end)
        if df is None:
            continue

        result     = run_backtest(ticker, df, mkt_returns, initial_cash, annual_rf)
        metrics    = compute_metrics(result)
        benchmark  = compute_benchmark(ticker, start, end, initial_cash)

        all_results[ticker] = result
        all_metrics[ticker] = metrics
        benchmarks[ticker]  = benchmark
        all_trades.extend(result["trades"])

        print_ticker_result(ticker, metrics, benchmark)

    if not all_metrics:
        print("\n  ✗  No results to summarize.")
        return

    # ── Strategy-level summary ────────────────────────────────────────────────
    passed_count = sum(1 for m in all_metrics.values() if m.get("passed", False))
    total        = len(all_metrics)
    accuracy     = passed_count / total * 100
    overall      = "CREDIBLE STRATEGY ✅" if accuracy >= 60 else "NEEDS IMPROVEMENT ⚠️"

    summary = {
        "period":    {"start": start, "end": end},
        "settings":  {"initial_cash": initial_cash, "commission": COMMISSION, "slippage": SLIPPAGE},
        "tickers":   all_metrics,
        "benchmarks": benchmarks,
        "strategy":  {
            "passed_count":   passed_count,
            "total_tickers":  total,
            "accuracy_pct":   round(accuracy, 1),
            "overall_verdict": overall,
        },
    }

    print_strategy_summary(summary)

    # ── Save results ──────────────────────────────────────────────────────────
    section("Saving Results")
    csv_path  = save_trades_csv(all_trades,  run_date)
    json_path = save_summary_json(summary,   run_date)
    txt_path  = save_summary_txt(summary,    run_date)
    chart_path = plot_equity_curves(all_results, benchmarks, initial_cash, run_date)

    print(f"    ✔  Trades log   : {csv_path}")
    print(f"    ✔  Summary JSON : {json_path}")
    print(f"    ✔  Report TXT   : {txt_path}")
    print(f"    ✔  Equity chart : {chart_path}")
    print(f"\n{THIN_BAR}\n")


if __name__ == "__main__":
    main()
