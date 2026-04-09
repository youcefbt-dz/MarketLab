"""
backtest.py — MarketLab Backtesting Engine
Walk-forward simulation with warehouse-first data loading.

Changes from previous version:
  - fetch_and_prepare()    → uses load_local() instead of yfinance
  - fetch_market_returns() → uses load_local('SPY') instead of yfinance
  - compute_benchmark()    → uses load_local() instead of yfinance
  - Removed duplicate code blocks
  - Fixed POSITION_PCT undefined variable
  - AUTO_PLANS mode fully integrated with warehouse
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta

from core.signals          import generate_signal, BUY_THRESHOLD, SELL_THRESHOLD
from analysis.backtest_logger import log_backtest_run
from core.stock_warehouse  import load_local, load_companies


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

COMMISSION  = 0.001   # 0.1% per trade
SLIPPAGE    = 0.001   # 0.1% slippage on entry/exit
COOLDOWN    = 5       # days to wait after a loss before re-entering
RISK_REWARD = 2.3     # Take Profit = risk × 2.3
RESULTS_DIR = "backtest_results"

POSITION_SIZE_MAP = {
    "STRONG BUY" : 0.35,
    "BUY"        : 0.22,
    "STRONG SELL": 0.35,
    "SELL"       : 0.22,
}
DEFAULT_POSITION_PCT   = 0.15
MAX_PORTFOLIO_EXPOSURE = 0.70

BAR      = "═" * 58
THIN_BAR = "─" * 58

AUTO_PLANS = [
    {
        "label"  : "Tech Giants — 5Y",
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
        "start"  : "2020-01-01",
    },
    {
        "label"  : "High Volatility — 3Y",
        "tickers": ["TSLA", "COIN", "PLTR", "MSTR", "RIVN"],
        "start"  : "2022-01-01",
    },
    {
        "label"  : "Diversified — 7Y",
        "tickers": ["AAPL", "JPM", "XOM", "JNJ", "WMT"],
        "start"  : "2018-01-01",
    },
    {
        "label"  : "Growth — 4Y",
        "tickers": ["NVDA", "AMZN", "CRM", "NOW", "CRWD"],
        "start"  : "2021-01-01",
    },
    {
        "label"  : "Finance & Pharma — 5Y",
        "tickers": ["GS", "JPM", "LLY", "ABBV", "PFE"],
        "start"  : "2020-01-01",
    },
    {
        "label"  : "Bull Run — 2Y",
        "tickers": ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA"],
        "start"  : "2023-01-01",
    },
    {
        "label"  : "Bear Test — crisis period",
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
        "start"  : "2022-01-01",
    },
    {
        "label"  : "Broad Market — 10Y",
        "tickers": ["SPY", "QQQ", "AAPL", "MSFT", "JPM"],
        "start"  : "2015-01-01",
    },
]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_position_size(
    signal: str,
    score: int,
    available_cash: float,
    current_exposure: float = 0.0,
) -> float:
    base_pct           = POSITION_SIZE_MAP.get(signal, DEFAULT_POSITION_PCT)
    remaining_capacity = MAX_PORTFOLIO_EXPOSURE - current_exposure
    if remaining_capacity <= 0:
        return 0.0
    final_pct = min(base_pct, remaining_capacity, 0.40)
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


def validate_ticker(ticker: str) -> bool:
    """Check if ticker exists in warehouse."""
    try:
        load_local(ticker)
        return True
    except FileNotFoundError:
        return False


def resolve_ticker(query: str, name_to_ticker: dict) -> str:
    query_upper = query.strip().upper()
    if query_upper in name_to_ticker.values():
        return query_upper
    for name, ticker in name_to_ticker.items():
        if query.lower() in name.lower():
            return ticker
    return query_upper


# ─── DATA PREPARATION — WAREHOUSE FIRST ──────────────────────────────────────

def fetch_and_prepare(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Load OHLCV from local warehouse and compute all indicators.
    Falls back to yfinance only if ticker not in warehouse.
    """
    try:
        df = load_local(ticker).set_index("Date")
        df.index = pd.to_datetime(df.index)
    except FileNotFoundError:
        # Fallback: try yfinance
        try:
            import yfinance as yf
            print(f"  ⚠  {ticker} not in warehouse — fetching from yfinance…")
            df = yf.Ticker(ticker).history(start=start, end=end)
            if df.empty:
                print(f"  ✗  No data for {ticker}.")
                return None
        except Exception as e:
            print(f"  ✗  Failed to fetch {ticker}: {e}")
            return None
    except Exception as e:
        print(f"  ✗  Error loading {ticker}: {e}")
        return None

    # Filter to requested date range
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    if len(df) < 250:
        print(f"  ⚠  Not enough data for {ticker} in {start} → {end} ({len(df)} rows).")
        return None

    # Compute indicators
    df["Stock_Return"] = df["Close"].pct_change()
    df["MA50"]         = df["Close"].rolling(50).mean()
    df["MA200"]        = df["Close"].rolling(200).mean()
    df["EMA20"]        = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI"]          = ta.rsi(df["Close"], length=14)
    df["BB_middle"]    = df["Close"].rolling(20).mean()
    df["BB_std"]       = df["Close"].rolling(20).std()
    df["BB_upper"]     = df["BB_middle"] + df["BB_std"] * 2
    df["BB_lower"]     = df["BB_middle"] - df["BB_std"] * 2
    df["MACD"]         = (df["Close"].ewm(span=12, adjust=False).mean()
                          - df["Close"].ewm(span=26, adjust=False).mean())
    df["Signal_line"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"]    = df["MACD"] - df["Signal_line"]
    df["L14"]          = df["Low"].rolling(14).min()
    df["H14"]          = df["High"].rolling(14).max()
    df["%K"]           = (df["Close"] - df["L14"]) / (df["H14"] - df["L14"] + 1e-9) * 100
    df["%D"]           = df["%K"].rolling(3).mean()

    df.dropna(inplace=True)

    if df.empty:
        print(f"  ⚠  No data left for {ticker} after dropna.")
        return None

    return df


def fetch_market_returns(start: str, end: str) -> pd.Series:
    """Load SPY from warehouse for market regime calculations."""
    try:
        df = load_local("SPY").set_index("Date")
        df.index = pd.to_datetime(df.index)
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        return df["Close"].pct_change().rename("Market_Return")
    except FileNotFoundError:
        print("  ⚠  SPY not in warehouse — fetching from yfinance…")
        try:
            import yfinance as yf
            df = yf.Ticker("^GSPC").history(start=start, end=end)
            return df["Close"].pct_change().rename("Market_Return")
        except Exception:
            return pd.Series(dtype=float, name="Market_Return")


def compute_benchmark(ticker: str, start: str, end: str, initial_cash: float) -> float:
    """Buy & Hold return — uses warehouse, no internet call needed."""
    try:
        df = load_local(ticker).set_index("Date")
        df.index = pd.to_datetime(df.index)
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        if len(df) < 2:
            return 0.0
        p_start = df["Close"].iloc[0]
        p_end   = df["Close"].iloc[-1]
        return round((p_end - p_start) / p_start * 100, 2)
    except FileNotFoundError:
        try:
            import yfinance as yf
            df    = yf.Ticker(ticker).history(start=start, end=end)
            return round((df["Close"].iloc[-1] - df["Close"].iloc[0])
                         / df["Close"].iloc[0] * 100, 2)
        except Exception:
            return 0.0


# ─── CORE SIMULATION ──────────────────────────────────────────────────────────

def _precompute_rolling_metrics(
    df: pd.DataFrame,
    mkt_returns: pd.Series,
    annual_rf: float,
    window: int = 252,
) -> pd.DataFrame:
    ret      = df["Stock_Return"]
    mkt      = mkt_returns.reindex(df.index)
    daily_rf = annual_rf / 252

    excess         = ret - daily_rf
    roll_mean      = excess.rolling(window).mean()
    roll_std       = excess.rolling(window).std()
    rolling_sharpe = (roll_mean / (roll_std + 1e-9)) * np.sqrt(252)

    rolling_ann_ret = (1 + ret.rolling(window).mean()) ** 252 - 1

    roll_cov     = ret.rolling(window).cov(mkt)
    roll_var     = mkt.rolling(window).var()
    rolling_beta = roll_cov / (roll_var + 1e-9)

    return pd.DataFrame({
        "rolling_sharpe" : rolling_sharpe,
        "rolling_ann_ret": rolling_ann_ret,
        "rolling_beta"   : rolling_beta,
    }, index=df.index)


def run_backtest(
    ticker:       str,
    df:           pd.DataFrame,
    mkt_returns:  pd.Series,
    initial_cash: float,
    annual_rf:    float = 0.04,
) -> dict:
    rolling = _precompute_rolling_metrics(df, mkt_returns, annual_rf)

    cash          = initial_cash
    position      = None
    equity_curve  = []
    trades        = []
    cooldown_left = 0
    warmup        = 252

    dummy_info = {
        "currentPrice"    : 0,
        "marketCap"       : "N/A",
        "fiftyTwoWeekHigh": 0,
        "fiftyTwoWeekLow" : 0,
    }

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

        # ── SL / TP check ─────────────────────────────────────────────────────
        if position:
            sl = position["sl"]
            tp = position["tp"]

            if open_price < sl:
                exit_price, reason = open_price, "Hit SL (Gap Down)"
            elif open_price > tp:
                exit_price, reason = open_price, "Hit TP (Gap Up)"
            elif low <= sl:
                exit_price, reason = sl, "Hit SL"
            elif high >= tp:
                exit_price, reason = tp, "Hit TP"
            else:
                exit_price, reason = None, None

            if exit_price is not None:
                exit_price *= (1 - SLIPPAGE) * (1 - COMMISSION)
                proceeds    = position["shares"] * exit_price
                pnl         = proceeds - position["cost"]
                pnl_pct     = pnl / position["cost"] * 100
                result      = "WIN" if pnl > 0 else "LOSS"

                hold_days = (today - position["entry_date"]).days

                trades.append({
                    "ticker"      : ticker,
                    "entry_date"  : position["entry_date"].strftime("%Y-%m-%d"),
                    "entry_price" : round(position["entry_price"], 4),
                    "signal"      : position["signal"],
                    "score"       : position["score"],
                    "position_pct": position.get("position_pct", 0),
                    "stop_loss"   : round(sl, 4),
                    "take_profit" : round(tp, 4),
                    "exit_date"   : today.strftime("%Y-%m-%d"),
                    "exit_price"  : round(exit_price, 4),
                    "exit_reason" : reason,
                    "hold_days"   : hold_days,
                    "pnl"         : round(pnl, 2),
                    "pnl_pct"     : round(pnl_pct, 2),
                    "result"      : result,
                })

                cash     += proceeds
                position  = None
                if result == "LOSS":
                    cooldown_left = COOLDOWN

        # ── Signal generation ─────────────────────────────────────────────────
        if position is None and cooldown_left == 0:
            row = rolling.iloc[i]
            if pd.isna(row["rolling_sharpe"]):
                continue

            metrics = {
                "Sharpe Annualized": round(row["rolling_sharpe"],  4),
                "Annualized Return": round(row["rolling_ann_ret"], 4),
                "Beta"             : round(row["rolling_beta"],    4),
            }

            slice_df  = df.iloc[: i + 1]
            mkt_slice = mkt_returns.reindex(slice_df.index)

            result = generate_signal(slice_df, dummy_info, metrics, mkt_slice)
            sig    = result.get("signal", "HOLD")
            score  = result.get("score",  0)

            if sig in ("BUY", "STRONG BUY"):
                entry_price = price * (1 + SLIPPAGE) * (1 + COMMISSION)
                alloc       = get_position_size(sig, score, cash, 0.0)
                shares      = alloc / entry_price

                if shares > 0 and cash >= alloc:
                    exit_levels = result.get("exit_levels", {})
                    sl = exit_levels.get("stop_loss",   price * 0.95)
                    tp = exit_levels.get("take_profit",
                         price * (1 + (price - sl) / price * RISK_REWARD))

                    position = {
                        "entry_price" : entry_price,
                        "entry_date"  : today,
                        "shares"      : shares,
                        "cost"        : shares * entry_price,
                        "sl"          : sl,
                        "tp"          : tp,
                        "signal"      : sig,
                        "score"       : score,
                        "position_pct": round(alloc / (cash + alloc) * 100, 1),
                    }
                    cash -= shares * entry_price

    # ── Close open position at end ────────────────────────────────────────────
    if position:
        last_price = df["Close"].iloc[-1] * (1 - SLIPPAGE) * (1 - COMMISSION)
        proceeds   = position["shares"] * last_price
        pnl        = proceeds - position["cost"]
        pnl_pct    = pnl / position["cost"] * 100

        hold_days = (df.index[-1] - position["entry_date"]).days

        trades.append({
            "ticker"      : ticker,
            "entry_date"  : position["entry_date"].strftime("%Y-%m-%d"),
            "entry_price" : round(position["entry_price"], 4),
            "signal"      : position["signal"],
            "score"       : position["score"],
            "position_pct": position.get("position_pct", 0),
            "stop_loss"   : round(position["sl"], 4),
            "take_profit" : round(position["tp"], 4),
            "exit_date"   : df.index[-1].strftime("%Y-%m-%d"),
            "exit_price"  : round(last_price, 4),
            "exit_reason" : "End of Period",
            "hold_days"   : hold_days,
            "pnl"         : round(pnl, 2),
            "pnl_pct"     : round(pnl_pct, 2),
            "result"      : "WIN" if pnl > 0 else "LOSS",
        })
        cash += proceeds

    return {
        "ticker"      : ticker,
        "initial_cash": initial_cash,
        "final_equity": round(cash, 2),
        "total_return": round((cash - initial_cash) / initial_cash * 100, 2),
        "trades"      : trades,
        "equity_curve": equity_curve,
    }


# ─── PERFORMANCE METRICS ──────────────────────────────────────────────────────

def compute_metrics(result: dict) -> dict:
    trades = result["trades"]
    if not trades:
        return {
            "verdict": "NO TRADES", "win_rate": 0, "profit_factor": 0,
            "max_drawdown": 0, "total_trades": 0, "total_return": result["total_return"],
            "sharpe": 0, "avg_r_multiple": 0, "max_consecutive_losses": 0,
            "avg_position_by_signal": {}, "exit_reasons": {}, "passed": False,
        }

    wins  = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]

    win_rate      = len(wins) / len(trades) * 100
    gross_profit  = sum(t["pnl"] for t in wins)
    gross_loss    = abs(sum(t["pnl"] for t in losses)) + 1e-9
    profit_factor = round(gross_profit / gross_loss, 2)

    eq     = pd.Series([e["equity"] for e in result["equity_curve"]])
    peak   = eq.cummax()
    dd     = (eq - peak) / peak * 100
    max_dd = round(dd.min(), 2)

    eq_returns = eq.pct_change().dropna()
    sharpe     = round((eq_returns.mean() / (eq_returns.std() + 1e-9)) * np.sqrt(252), 2)

    # Avg R-Multiple
    r_multiples = []
    for t in trades:
        risk = abs(t["entry_price"] - t["stop_loss"])
        if risk > 0:
            r_multiples.append(
                t["pnl"] / (risk * (t.get("position_pct", 20) / 100) * result["initial_cash"])
            )
    avg_r_multiple = round(np.mean(r_multiples), 2) if r_multiples else 0.0

    # Max Consecutive Losses
    max_consec = cur = 0
    for t in trades:
        cur = cur + 1 if t["result"] == "LOSS" else 0
        max_consec = max(max_consec, cur)

    # Avg position by signal
    sig_sizes: dict = {}
    for t in trades:
        sig_sizes.setdefault(t["signal"], []).append(t.get("position_pct", 20))
    avg_position_by_signal = {s: round(np.mean(v), 1) for s, v in sig_sizes.items()}

    # Exit reasons
    exit_reasons: dict = {}
    for t in trades:
        r = t.get("exit_reason", "Unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Avg Hold Days
    hold_days_list = [t.get("hold_days", 0) for t in trades]
    avg_hold_days  = round(np.mean(hold_days_list), 1) if hold_days_list else 0.0

    passed  = win_rate >= 50 and profit_factor >= 1.3 and result["total_return"] > 0
    verdict = "PASS ✅" if passed else "FAIL ❌"

    return {
        "verdict"               : verdict,
        "passed"                : passed,
        "total_trades"          : len(trades),
        "win_rate"              : round(win_rate, 1),
        "profit_factor"         : profit_factor,
        "max_drawdown"          : max_dd,
        "sharpe"                : sharpe,
        "total_return"          : result["total_return"],
        "avg_r_multiple"        : avg_r_multiple,
        "max_consecutive_losses": max_consec,
        "avg_hold_days"         : avg_hold_days,
        "avg_position_by_signal": avg_position_by_signal,
        "exit_reasons"          : exit_reasons,
        "gross_profit"          : round(gross_profit, 2),
        "gross_loss"            : round(gross_loss,   2),
    }


# ─── SAVING ───────────────────────────────────────────────────────────────────

class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def save_trades_csv(all_trades: list, run_date: str) -> str:
    path = os.path.join(RESULTS_DIR, f"trades_{run_date}.csv")
    if not all_trades:
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_trades[0].keys())
        writer.writeheader()
        writer.writerows(all_trades)
    return path


def save_summary_json(summary: dict, run_date: str) -> str:
    path = os.path.join(RESULTS_DIR, f"summary_{run_date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=_JSONEncoder)
    return path


def save_summary_txt(summary: dict, run_date: str) -> str:
    path    = os.path.join(RESULTS_DIR, f"report_{run_date}.txt")
    s       = summary["strategy"]
    passed  = s["passed_count"]
    total   = s["total_tickers"]
    acc     = s["accuracy_pct"]
    verdict = s["overall_verdict"]

    lines = [
        "=" * 58,
        "  MarketLab — Backtest Strategy Report",
        f"  Date: {run_date}",
        "=" * 58,
        "",
        f"  Tickers Tested  : {total}",
        f"  Period          : {summary['period']['start']} → {summary['period']['end']}",
        f"  Initial Capital : ${summary['settings']['initial_cash']:,.0f}",
        f"  Commission      : {COMMISSION*100:.1f}%  |  Slippage: {SLIPPAGE*100:.1f}%",
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
            f"Sharpe: {m['sharpe']:.2f}  "
            f"AvgHold: {m.get('avg_hold_days', 0):.0f}d  "
            f"Trades: {m['total_trades']}"
        )
    lines += [
        "",
        "─" * 58,
        "  Strategy Accuracy",
        "─" * 58,
        f"  Signals Correct : {passed}/{total}",
        f"  Accuracy        : {acc:.1f}%",
        f"  Overall Verdict : {verdict}",
        "=" * 58,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ─── EQUITY CURVE ─────────────────────────────────────────────────────────────

def plot_equity_curves(
    results: dict, benchmarks: dict, initial_cash: float, run_date: str
) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    for ticker, res in results.items():
        if not res["equity_curve"]:
            continue
        eq   = pd.Series(
            [e["equity"] for e in res["equity_curve"]],
            index=[e["date"] for e in res["equity_curve"]],
        )
        norm = eq / initial_cash * 100
        ax.plot(norm, label=f"{ticker} (strategy)", linewidth=1.5)

    ax.axhline(100, color="black", linestyle="--", linewidth=1, label="Starting Capital")
    ax.set_title("Equity Curves — Strategy vs Starting Capital",
                 fontsize=14, fontweight="bold")
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
    print(f"    Verdict         : {metrics['verdict']}")
    print(f"    Total Return    : {metrics['total_return']:+.2f}%"
          f"  (Buy & Hold: {benchmark:+.2f}%)")
    print(f"    Total Trades    : {metrics['total_trades']}")
    print(f"    Win Rate        : {metrics['win_rate']:.1f}%")
    print(f"    Profit Factor   : {metrics['profit_factor']:.2f}")
    print(f"    Max Drawdown    : {metrics['max_drawdown']:.2f}%")
    print(f"    Sharpe Ratio    : {metrics['sharpe']:.2f}")
    print(f"    Avg R-Multiple  : {metrics.get('avg_r_multiple', 0):.2f}")
    print(f"    Max Consec Loss : {metrics.get('max_consecutive_losses', 0)}")
    print(f"    Avg Hold Days   : {metrics.get('avg_hold_days', 0):.0f} days")

    if metrics.get("avg_position_by_signal"):
        print("    Avg Position %  :")
        for sig, pct in metrics["avg_position_by_signal"].items():
            print(f"        {sig:12s} : {pct:.1f}%")

    if metrics.get("exit_reasons"):
        print("    Exit Reasons    :")
        for reason, count in sorted(
            metrics["exit_reasons"].items(), key=lambda x: -x[1]
        ):
            print(f"        {reason:20s} : {count}")


def print_strategy_summary(summary: dict) -> None:
    s = summary["strategy"]
    print(f"\n{'═'*58}")
    print("  Strategy Accuracy Report")
    print(f"{'═'*58}")
    print(f"  Signals Correct  : {s['passed_count']}/{s['total_tickers']}")
    print(f"  Accuracy         : {s['accuracy_pct']:.1f}%")
    print(f"  Overall Verdict  : {s['overall_verdict']}")
    print(f"{'═'*58}")


# ─── STOCK SELECTION HELPERS ──────────────────────────────────────────────────

def _select_tickers_manual(name_to_ticker: dict) -> tuple[list[str], str, str]:
    num        = prompt_int("Number of stocks to backtest", 1, 20, 3)
    start_year = prompt_int("Backtest start year", 2015, 2024, 2020)
    start      = f"{start_year}-01-01"
    end        = datetime.today().strftime("%Y-%m-%d")

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
            print(f"    ✗  '{ticker}' not found in warehouse. Run Mode [5] to update.")
    return tickers, start, end


def _select_tickers_auto(name_to_ticker: dict) -> list[tuple[list[str], str, str, str]]:
    end = datetime.today().strftime("%Y-%m-%d")

    print(f"\n  {THIN_BAR}")
    print("  Available Plans:")
    print(f"  {THIN_BAR}")
    for i, plan in enumerate(AUTO_PLANS, 1):
        t_str = ", ".join(plan["tickers"])
        print(f"    [{i}] {plan['label']:<35} {plan['start']}  {t_str}")
    print(f"    [A] Run All ({len(AUTO_PLANS)} plans)")
    print(f"  {THIN_BAR}")

    raw = input("  Select a plan [1-8 or A]: ").strip().upper()
    if raw == "A":
        selected = AUTO_PLANS
    elif raw.isdigit() and 1 <= int(raw) <= len(AUTO_PLANS):
        selected = [AUTO_PLANS[int(raw) - 1]]
    else:
        print("  Invalid selection — running first plan by default.")
        selected = [AUTO_PLANS[0]]

    plans_out = []
    for plan in selected:
        valid   = [t for t in plan["tickers"] if validate_ticker(t)]
        skipped = set(plan["tickers"]) - set(valid)
        if skipped:
            print(f"  ⚠  {plan['label']}: skipping {skipped} (not in warehouse).")
        if not valid:
            print(f"  ✗  {plan['label']}: no valid tickers — skipped.")
            continue
        plans_out.append((valid, plan["start"], end, plan["label"]))
    return plans_out


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═'*58}")
    print("  MarketLab  ·  Backtesting Engine")
    print(f"{'═'*58}")

    ensure_results_dir()
    run_date = datetime.today().strftime("%Y-%m-%d")

    initial_cash = prompt_float("Initial capital ($)", 10000)

    print("  Risk-Free Rate  (10Y US Treasury ≈ 4.2%)")
    print("  Press Enter for 4.0% default.")
    raw_rf = input("  Annual RF rate [%]: ").strip()
    try:
        annual_rf = max(0.0, min(float(raw_rf) / 100 if raw_rf else 0.04, 0.20))
    except ValueError:
        annual_rf = 0.04
    print(f"    ✔  Risk-free rate: {annual_rf * 100:.2f}%")

    name_to_ticker = load_companies()

    print(f"\n  {THIN_BAR}")
    print("  Stock selection:")
    print("    [1]  Manual  — choose companies and dates")
    print("    [2]  Auto    — run predefined diverse plans")
    print(f"  {THIN_BAR}")
    mode_choice = input("  Your choice [1/2]: ").strip()

    if mode_choice == "2":
        plans = _select_tickers_auto(name_to_ticker)
        if not plans:
            print("  ✗  No valid plans. Update warehouse first (Mode 5).")
            return
    else:
        tickers, start, end = _select_tickers_manual(name_to_ticker)
        plans = [(tickers, start, end, "Manual")]

    grand_metrics    = {}
    grand_trades     = []
    grand_results    = {}
    grand_benchmarks = {}

    for plan_idx, (tickers, start, end, label) in enumerate(plans, 1):
        print(f"\n{'═'*58}")
        print(f"  Plan [{plan_idx}/{len(plans)}]: {label}")
        print(f"  Tickers : {', '.join(tickers)}")
        print(f"  Period  : {start}  →  {end}")
        print(f"{'═'*58}")

        mkt_returns     = fetch_market_returns(start, end)
        plan_results    = {}
        plan_metrics    = {}
        plan_trades     = []
        plan_benchmarks = {}

        for ticker in tickers:
            print(f"\n{'─'*58}")
            print(f"  Backtesting  {ticker}  ({start} → {end})")
            print(f"{'─'*58}")

            df = fetch_and_prepare(ticker, start, end)
            if df is None:
                continue

            result    = run_backtest(ticker, df, mkt_returns, initial_cash, annual_rf)
            metrics   = compute_metrics(result)
            benchmark = compute_benchmark(ticker, start, end, initial_cash)

            key = f"{ticker}|{label}"
            plan_results[key]    = result
            plan_metrics[key]    = metrics
            plan_benchmarks[key] = benchmark
            plan_trades.extend(result["trades"])

            print_ticker_result(ticker, metrics, benchmark)

            log_backtest_run(
                ticker    = ticker,
                start     = start,
                end       = end,
                metrics   = metrics,
                benchmark = benchmark,
                settings  = {
                    "initial_cash": initial_cash,
                    "commission"  : COMMISSION,
                    "slippage"    : SLIPPAGE,
                },
                trades = result["trades"],
            )

        grand_metrics.update(plan_metrics)
        grand_trades.extend(plan_trades)
        grand_results.update(plan_results)
        grand_benchmarks.update(plan_benchmarks)

        if plan_metrics:
            p_pass = sum(1 for m in plan_metrics.values() if m.get("passed", False))
            p_tot  = len(plan_metrics)
            print(f"\n  ✦  {label}: {p_pass}/{p_tot} passed"
                  f" ({p_pass/p_tot*100:.0f}%)")

    if not grand_metrics:
        print("\n  ✗  No results to summarize.")
        return

    passed_count = sum(1 for m in grand_metrics.values() if m.get("passed", False))
    total        = len(grand_metrics)
    accuracy     = passed_count / total * 100
    overall      = "CREDIBLE STRATEGY ✅" if accuracy >= 60 else "NEEDS IMPROVEMENT ⚠️"

    summary = {
        "period"    : {"start": plans[0][1], "end": plans[0][2]},
        "settings"  : {"initial_cash": initial_cash,
                       "commission"  : COMMISSION,
                       "slippage"    : SLIPPAGE},
        "tickers"   : grand_metrics,
        "benchmarks": grand_benchmarks,
        "strategy"  : {
            "passed_count"   : passed_count,
            "total_tickers"  : total,
            "accuracy_pct"   : round(accuracy, 1),
            "overall_verdict": overall,
        },
    }

    print_strategy_summary(summary)

    section("Saving Results")
    csv_path   = save_trades_csv(grand_trades,  run_date)
    json_path  = save_summary_json(summary,     run_date)
    txt_path   = save_summary_txt(summary,      run_date)
    chart_path = plot_equity_curves(
        grand_results, grand_benchmarks, initial_cash, run_date
    )

    # Invalidate ML cache so next predict_quality() retrains on fresh data
    try:
        from ml_predictor import invalidate_cache
        invalidate_cache()
        print("    ✔  ML cache invalidated.")
    except ImportError:
        pass

    print(f"    ✔  Trades log    : {csv_path}")
    print(f"    ✔  Summary JSON  : {json_path}")
    print(f"    ✔  Report TXT    : {txt_path}")
    print(f"    ✔  Equity chart  : {chart_path}")
    print(f"\n{THIN_BAR}\n")


if __name__ == "__main__":
    main()
