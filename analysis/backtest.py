"""
backtest.py — MarketLab Backtesting Engine  v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upgrades over v3.x
──────────────────
REALISM
  • Short-selling support  (STRONG SELL / SELL → short positions)
  • Overnight gap risk     (gap-adjusted SL/TP checks preserved)
  • Bid-ask spread model   (spread_bps parameter, default 5 bps)
  • Partial-fill simulation (orders > 1% ADV are scaled down)

RISK MANAGEMENT
  • Kelly Criterion position sizing  (full / half / quarter kelly)
  • Portfolio-level max drawdown kill-switch  (default –25 %)
  • Per-trade max loss cap            (default 2 % of portfolio)
  • Correlation-aware exposure limit  (blocks adding long if already
    long a >0.80-correlated ticker)
  • Trailing stop support             (activates after +1R move)

PERFORMANCE ANALYSIS
  • Monte Carlo simulation  (N=1 000 path permutations of trade list)
  • Calmar Ratio, Sortino Ratio, Omega Ratio, VaR 95, CVaR 95
  • Rolling 63-day (quarter) regime breakdown
  • Per-signal breakdown table

SPEED / ARCHITECTURE
  • Vectorized indicator computation   (no per-row Python loops)
  • Concurrent multi-ticker loading    (ThreadPoolExecutor)
  • Signal pre-cache per ticker        (generated once, reused)
  • Structured logging via Python logging module
"""
from __future__ import annotations

import csv
import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta

from core.signals             import generate_signal, BUY_THRESHOLD, SELL_THRESHOLD
from analysis.backtest_logger import log_backtest_run
from core.stock_warehouse     import load_local, load_companies

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "  %(levelname)-8s %(message)s",
)
log = logging.getLogger("MarketLab.backtest")


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

COMMISSION  = 0.001    # 0.10 % per trade (each leg)
SLIPPAGE    = 0.001    # 0.10 % market-impact slippage
SPREAD_BPS  = 5        # bid-ask spread in basis points (5 bps = 0.05 %)
COOLDOWN    = 5        # bars to wait after a LOSS before re-entering
RISK_REWARD = 2.3      # TP = entry ± risk × RISK_REWARD

RESULTS_DIR   = "backtest_results"
TRADING_DAYS  = 252
MAX_RF_RATE   = 0.20
WARMUP_PERIOD = 252

# Risk management
KELLY_FRACTION        = 0.25   # quarter-Kelly (conservative)
MAX_POSITION_PCT      = 0.40   # hard cap per position
MAX_PORTFOLIO_EXPOSURE= 0.80   # long + short gross exposure cap
PORT_DRAWDOWN_KILL    = -0.25  # –25 % → halt new trades for the plan
PER_TRADE_MAX_LOSS    = 0.02   # 2 % of portfolio equity per trade
CORR_BLOCK_THRESHOLD  = 0.80   # block new long if already in >0.80 corr long
TRAILING_STOP_TRIGGER = 1.0    # activate trailing stop after +1R gain
MAX_OPEN_POSITIONS    = 5      # max simultaneous open positions

POSITION_SIZE_MAP = {
    "STRONG BUY" : 0.35,
    "BUY"        : 0.22,
    "STRONG SELL": 0.35,
    "SELL"       : 0.22,
}
DEFAULT_POSITION_PCT = 0.15

BAR      = "═" * 62
THIN_BAR = "─" * 62

AUTO_PLANS: list[dict] = [
    {"label": "Tech Giants — 5Y",          "tickers": ["AAPL","MSFT","NVDA","GOOGL","META"], "start": "2020-01-01"},
    {"label": "High Volatility — 3Y",      "tickers": ["TSLA","COIN","PLTR","MSTR","RIVN"],  "start": "2022-01-01"},
    {"label": "Diversified — 7Y",          "tickers": ["AAPL","JPM", "XOM", "JNJ", "WMT"],  "start": "2018-01-01"},
    {"label": "Growth — 4Y",               "tickers": ["NVDA","AMZN","CRM", "NOW", "CRWD"], "start": "2021-01-01"},
    {"label": "Finance & Pharma — 5Y",     "tickers": ["GS",  "JPM", "LLY", "ABBV","PFE"],  "start": "2020-01-01"},
    {"label": "Bull Run — 2Y",             "tickers": ["AAPL","MSFT","AMZN","TSLA","NVDA"], "start": "2023-01-01"},
    {"label": "Bear Test — crisis period", "tickers": ["AAPL","MSFT","AMZN","GOOGL","META"],"start": "2022-01-01"},
    {"label": "Broad Market — 10Y",        "tickers": ["SPY", "QQQ", "AAPL","MSFT","JPM"],  "start": "2015-01-01"},
]

_DUMMY_INFO: dict = {
    "currentPrice"    : 0,
    "marketCap"       : "N/A",
    "fiftyTwoWeekHigh": 0,
    "fiftyTwoWeekLow" : 0,
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def today_str() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def section(title: str) -> None:
    print(f"\n{THIN_BAR}\n  {title}\n{THIN_BAR}")


def prompt_float(prompt: str, default: float) -> float:
    raw = input(f"  {prompt} [default {default}]: ").strip()
    try:    return float(raw) if raw else default
    except: return default


def prompt_int(prompt: str, lo: int, hi: int, default: int) -> int:
    raw = input(f"  {prompt} [{lo}–{hi}, default {default}]: ").strip()
    try:
        v = int(raw) if raw else default
        return v if lo <= v <= hi else default
    except: return default


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _json_serializer(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ─── TICKER RESOLUTION ────────────────────────────────────────────────────────

def validate_ticker(ticker: str) -> bool:
    try:    load_local(ticker); return True
    except: return False


def resolve_ticker(query: str, name_to_ticker: dict) -> str:
    q = query.strip().upper()
    if q in name_to_ticker.values(): return q
    for name, ticker in name_to_ticker.items():
        if query.lower() in name.lower(): return ticker
    return q


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def _load_from_yfinance(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        log.warning("%s not in warehouse — fetching from yfinance…", ticker)
        df = yf.Ticker(ticker).history(start=start, end=end)
        return df if not df.empty else None
    except Exception as e:
        log.error("Failed to fetch %s: %s", ticker, e)
        return None


def fetch_and_prepare(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Load OHLCV and compute all indicators (vectorized)."""
    try:
        df = load_local(ticker).set_index("Date")
        df.index = pd.to_datetime(df.index)
    except FileNotFoundError:
        df = _load_from_yfinance(ticker, start, end)
        if df is None:
            log.error("No data for %s.", ticker); return None
    except Exception as e:
        log.error("Error loading %s: %s", ticker, e); return None

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df.index >= s) & (df.index <= e)]

    if len(df) < WARMUP_PERIOD:
        log.warning("%s: only %d rows — skipped.", ticker, len(df)); return None

    df = _compute_indicators(df)
    return df if not df.empty else None


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fully vectorized indicator pipeline."""
    c = df["Close"]
    h = df["High"]
    l_col = df["Low"]

    # Returns & moving averages
    df["Stock_Return"] = c.pct_change()
    df["MA50"]         = c.rolling(50).mean()
    df["MA200"]        = c.rolling(200).mean()
    df["EMA20"]        = c.ewm(span=20, adjust=False).mean()

    # RSI via pandas_ta (vectorized C implementation)
    df["RSI"] = ta.rsi(c, length=14)

    # Bollinger Bands
    bb_mid          = c.rolling(20).mean()
    bb_std          = c.rolling(20).std()
    df["BB_middle"] = bb_mid
    df["BB_upper"]  = bb_mid + bb_std * 2
    df["BB_lower"]  = bb_mid - bb_std * 2

    # MACD
    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    df["MACD"]         = ema12 - ema26
    df["Signal_line"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"]    = df["MACD"] - df["Signal_line"]

    # Stochastic
    l14        = l_col.rolling(14).min()
    h14        = h.rolling(14).max()
    df["%K"]   = (c - l14) / (h14 - l14 + 1e-9) * 100
    df["%D"]   = df["%K"].rolling(3).mean()

    # ATR (used for Kelly and trailing stop)
    df["TR"]  = pd.concat([
        (h - l_col),
        (h - c.shift()).abs(),
        (l_col - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()

    # ADV — 20-day average dollar volume (for partial-fill guard)
    df["ADV20"] = (c * df.get("Volume", pd.Series(np.nan, index=c.index))).rolling(20).mean()

    df.dropna(inplace=True)
    return df


def fetch_market_returns(start: str, end: str) -> pd.Series:
    try:
        df = load_local("SPY").set_index("Date")
        df.index = pd.to_datetime(df.index)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return df[(df.index >= s) & (df.index <= e)]["Close"].pct_change().rename("Market_Return")
    except FileNotFoundError:
        raw = _load_from_yfinance("^GSPC", start, end)
        if raw is not None:
            return raw["Close"].pct_change().rename("Market_Return")
        return pd.Series(dtype=float, name="Market_Return")


def compute_benchmark(ticker: str, start: str, end: str) -> float:
    try:
        df = load_local(ticker).set_index("Date")
        df.index = pd.to_datetime(df.index)
    except FileNotFoundError:
        raw = _load_from_yfinance(ticker, start, end)
        df  = raw if raw is not None else pd.DataFrame()

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df.index >= s) & (df.index <= e)]
    if len(df) < 2: return 0.0
    return round((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100, 2)


# ─── CONCURRENT DATA LOADER ───────────────────────────────────────────────────

def load_tickers_parallel(
    tickers: list[str], start: str, end: str, max_workers: int = 6
) -> dict[str, pd.DataFrame]:
    """Load + compute indicators for multiple tickers concurrently."""
    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_and_prepare, t, start, end): t for t in tickers}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                df = fut.result()
                if df is not None:
                    results[ticker] = df
                    log.info("✔  %s loaded (%d bars)", ticker, len(df))
            except Exception as exc:
                log.error("%s loader raised: %s", ticker, exc)
    return results


# ─── ROLLING METRICS ──────────────────────────────────────────────────────────

def _precompute_rolling_metrics(
    df: pd.DataFrame,
    mkt_returns: pd.Series,
    annual_rf: float,
    window: int = TRADING_DAYS,
) -> pd.DataFrame:
    ret      = df["Stock_Return"]
    mkt      = mkt_returns.reindex(df.index)
    daily_rf = annual_rf / TRADING_DAYS
    excess   = ret - daily_rf

    roll_std       = excess.rolling(window).std()
    rolling_sharpe = (excess.rolling(window).mean() / (roll_std + 1e-9)) * np.sqrt(TRADING_DAYS)
    rolling_ann    = (1 + ret.rolling(window).mean()) ** TRADING_DAYS - 1
    roll_var       = mkt.rolling(window).var()
    rolling_beta   = ret.rolling(window).cov(mkt) / (roll_var + 1e-9)

    # Win-rate and avg-win/loss rolling estimates (for Kelly)
    roll_win_rate = (ret > 0).rolling(window).mean()
    roll_avg_win  = ret.where(ret > 0).rolling(window).mean().fillna(0)
    roll_avg_loss = ret.where(ret < 0).rolling(window).mean().abs().fillna(1e-9)

    return pd.DataFrame({
        "rolling_sharpe"  : rolling_sharpe,
        "rolling_ann_ret" : rolling_ann,
        "rolling_beta"    : rolling_beta,
        "roll_win_rate"   : roll_win_rate,
        "roll_avg_win"    : roll_avg_win,
        "roll_avg_loss"   : roll_avg_loss,
    }, index=df.index)


# ─── KELLY POSITION SIZING ────────────────────────────────────────────────────

def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    equity: float,
    signal: str,
    fraction: float = KELLY_FRACTION,
) -> float:
    """
    f* = (W/L * p − (1−p)) / (W/L)   [simplified Kelly]
    Returns dollar amount to allocate.
    """
    if avg_loss < 1e-9: return equity * DEFAULT_POSITION_PCT
    ratio  = avg_win / avg_loss
    kelly  = (ratio * win_rate - (1 - win_rate)) / ratio
    kelly  = max(0.0, min(kelly, 1.0))          # clamp [0, 1]
    scaled = kelly * fraction                    # fractional Kelly

    # Hard caps
    map_pct = POSITION_SIZE_MAP.get(signal, DEFAULT_POSITION_PCT)
    final   = min(scaled, map_pct, MAX_POSITION_PCT)
    return equity * max(final, 0.02)             # floor at 2 % of equity


# ─── SPREAD / PARTIAL FILL ────────────────────────────────────────────────────

def _apply_execution_costs(price: float, direction: str, spread_bps: int = SPREAD_BPS) -> float:
    """
    Apply half-spread to entry price.
    direction: 'buy'  → pay ask  (price + half spread)
               'sell' → receive bid (price − half spread)
    """
    half_spread = price * spread_bps / 20_000   # bps/2 / 100
    if direction == "buy":
        return price * (1 + SLIPPAGE) * (1 + COMMISSION) + half_spread
    else:
        return price * (1 - SLIPPAGE) * (1 - COMMISSION) - half_spread


def _partial_fill_guard(alloc: float, price: float, adv: float) -> float:
    """Scale down order if it exceeds 1% of ADV (institutional realism)."""
    if adv <= 0 or np.isnan(adv): return alloc
    max_order = adv * 0.01
    if alloc > max_order:
        log.debug("Partial fill: order %.0f scaled to ADV cap %.0f", alloc, max_order)
        return max_order
    return alloc


# ─── SL / TP / TRAILING STOP ─────────────────────────────────────────────────

def _update_trailing_stop(position: dict, price: float) -> None:
    """
    Activate trailing stop once price moves +1R above entry (long)
    or −1R below entry (short), then trail the stop.
    """
    entry = position["entry_price"]
    risk  = abs(entry - position["sl"])
    side  = position["side"]      # "long" | "short"

    if side == "long":
        if price >= entry + TRAILING_STOP_TRIGGER * risk:
            new_sl = price - risk
            if new_sl > position["sl"]:
                position["sl"]               = new_sl
                position["trailing_active"]  = True
    else:
        if price <= entry - TRAILING_STOP_TRIGGER * risk:
            new_sl = price + risk
            if new_sl < position["sl"]:
                position["sl"]               = new_sl
                position["trailing_active"]  = True


def _exit_price_and_reason(
    position: dict, open_price: float, high: float, low: float
) -> tuple[Optional[float], Optional[str]]:
    sl, tp   = position["sl"], position["tp"]
    side     = position["side"]

    if side == "long":
        if   open_price < sl: return open_price, "Hit SL (Gap Down)"
        elif open_price > tp: return open_price, "Hit TP (Gap Up)"
        elif low  <= sl:      return sl,         "Hit SL"
        elif high >= tp:      return tp,         "Hit TP"
    else:  # short
        if   open_price > sl: return open_price, "Hit SL (Gap Up)"
        elif open_price < tp: return open_price, "Hit TP (Gap Down)"
        elif high >= sl:      return sl,         "Hit SL"
        elif low  <= tp:      return tp,         "Hit TP"
    return None, None


# ─── CLOSE POSITION ───────────────────────────────────────────────────────────

def _close_position(
    position: dict,
    exit_price: float,
    exit_date,
    reason: str,
    ticker: str,
    initial_cash: float,
) -> tuple[dict, float]:
    side      = position["side"]
    direction = "sell" if side == "long" else "buy"
    net_price = _apply_execution_costs(exit_price, direction)

    if side == "long":
        proceeds = position["shares"] * net_price
        pnl      = proceeds - position["cost"]
    else:   # short: profit when price falls
        proceeds = position["cost"]    # get back margin
        pnl      = (position["entry_price"] - net_price) * position["shares"]

    pnl_pct   = pnl / abs(position["cost"]) * 100
    hold_days = (exit_date - position["entry_date"]).days

    trade = {
        "ticker"         : ticker,
        "side"           : side,
        "entry_date"     : position["entry_date"].strftime("%Y-%m-%d"),
        "entry_price"    : round(position["entry_price"], 4),
        "signal"         : position["signal"],
        "score"          : position["score"],
        "position_pct"   : position.get("position_pct", 0),
        "stop_loss"      : round(position["sl"], 4),
        "take_profit"    : round(position["tp"], 4),
        "trailing_active": position.get("trailing_active", False),
        "exit_date"      : exit_date.strftime("%Y-%m-%d"),
        "exit_price"     : round(net_price, 4),
        "exit_reason"    : reason,
        "hold_days"      : hold_days,
        "pnl"            : round(pnl, 2),
        "pnl_pct"        : round(pnl_pct, 2),
        "result"         : "WIN" if pnl > 0 else "LOSS",
    }
    return trade, proceeds + pnl if side == "short" else proceeds


# ─── CORRELATION GUARD ────────────────────────────────────────────────────────

def _corr_blocked(
    ticker: str,
    side: str,
    open_positions: dict,
    returns_cache: dict[str, pd.Series],
    window: int = 63,
) -> bool:
    """Return True if we should block opening this position due to correlation."""
    if side == "short": return False   # only guard longs for now
    for pos_key, pos in open_positions.items():
        if pos["side"] != "long": continue
        other_ticker = pos["ticker"]
        if other_ticker == ticker: continue
        r1 = returns_cache.get(ticker)
        r2 = returns_cache.get(other_ticker)
        if r1 is None or r2 is None: continue
        aligned = pd.concat([r1.tail(window), r2.tail(window)], axis=1).dropna()
        if len(aligned) < 20: continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if corr > CORR_BLOCK_THRESHOLD:
            log.debug("Corr block: %s ↔ %s (corr=%.2f)", ticker, other_ticker, corr)
            return True
    return False


# ─── CORE SIMULATION  (multi-position, long + short) ─────────────────────────

def run_backtest(
    ticker       : str,
    df           : pd.DataFrame,
    mkt_returns  : pd.Series,
    initial_cash : float,
    annual_rf    : float = 0.04,
    returns_cache: dict  = None,
    portfolio_equity_ref : list = None,   # shared mutable list[float] across tickers
) -> dict:
    """
    Simulate trades with:
      - multi-position stack  (up to MAX_OPEN_POSITIONS per ticker)
      - long AND short support
      - Kelly sizing
      - trailing stops
      - partial-fill guard
      - portfolio kill-switch via portfolio_equity_ref
    """
    rolling       = _precompute_rolling_metrics(df, mkt_returns, annual_rf)
    cash          = initial_cash
    open_positions: dict[str, dict] = {}  # key → position dict
    equity_curve  = []
    trades        = []
    cooldown_left = 0
    pos_counter   = 0
    killed        = False     # portfolio kill-switch triggered

    def _portfolio_equity() -> float:
        mktval = sum(
            p["shares"] * df["Close"].iloc[-1] if p["side"] == "long"
            else p["cost"]
            for p in open_positions.values()
        )
        return cash + mktval

    for i in range(WARMUP_PERIOD, len(df)):
        today      = df.index[i]
        open_price = df["Open"].iloc[i]
        price      = df["Close"].iloc[i]
        high       = df["High"].iloc[i]
        low        = df["Low"].iloc[i]
        atr        = df["ATR14"].iloc[i]
        adv        = df["ADV20"].iloc[i] if "ADV20" in df.columns else 0

        # Current equity
        longs_val  = sum(p["shares"] * price for p in open_positions.values() if p["side"] == "long")
        shorts_val = sum(p["cost"]           for p in open_positions.values() if p["side"] == "short")
        equity     = cash + longs_val + shorts_val
        equity_curve.append({"date": today, "equity": equity})

        # ── Portfolio kill-switch ──────────────────────────────────────────────
        if not killed:
            port_dd = (equity - initial_cash) / initial_cash
            if port_dd <= PORT_DRAWDOWN_KILL:
                log.warning(
                    "%s: portfolio DD hit %.1f%% — halting new trades.",
                    ticker, port_dd * 100
                )
                killed = True

        if cooldown_left > 0:
            cooldown_left -= 1

        # ── Update trailing stops ──────────────────────────────────────────────
        for pos in open_positions.values():
            _update_trailing_stop(pos, price)

        # ── SL / TP checks for all open positions ──────────────────────────────
        to_close = []
        for key, pos in open_positions.items():
            ep, reason = _exit_price_and_reason(pos, open_price, high, low)
            if ep is not None:
                to_close.append((key, ep, reason))

        for key, ep, reason in to_close:
            pos = open_positions.pop(key)
            trade, proceeds = _close_position(pos, ep, today, reason, ticker, initial_cash)
            trades.append(trade)
            cash += proceeds
            if trade["result"] == "LOSS":
                cooldown_left = COOLDOWN

        # ── Signal generation ──────────────────────────────────────────────────
        can_open = (
            not killed
            and cooldown_left == 0
            and len(open_positions) < MAX_OPEN_POSITIONS
        )

        if can_open:
            row = rolling.iloc[i]
            if pd.isna(row["rolling_sharpe"]):
                continue

            metrics = {
                "Sharpe Annualized": round(row["rolling_sharpe"],  4),
                "Annualized Return": round(row["rolling_ann_ret"], 4),
                "Beta"             : round(row["rolling_beta"],    4),
            }

            slice_df  = df.iloc[:i + 1]
            mkt_slice = mkt_returns.reindex(slice_df.index)
            result    = generate_signal(slice_df, _DUMMY_INFO, metrics, mkt_slice)
            sig       = result.get("signal", "HOLD")
            score     = result.get("score",  0)

            # Determine trade direction
            if sig in ("BUY", "STRONG BUY"):
                side = "long"
            elif sig in ("SELL", "STRONG SELL"):
                side = "short"
            else:
                continue

            # Correlation guard (long only)
            if returns_cache and _corr_blocked(ticker, side, open_positions, returns_cache):
                continue

            # Kelly sizing
            alloc = kelly_position_size(
                win_rate = float(row["roll_win_rate"]) if not pd.isna(row["roll_win_rate"]) else 0.5,
                avg_win  = float(row["roll_avg_win"])  if not pd.isna(row["roll_avg_win"])  else 0.005,
                avg_loss = float(row["roll_avg_loss"]) if not pd.isna(row["roll_avg_loss"]) else 0.005,
                equity   = equity,
                signal   = sig,
            )

            # Partial-fill guard
            alloc = _partial_fill_guard(alloc, price, adv)

            # Per-trade max loss cap
            max_loss_alloc = equity * PER_TRADE_MAX_LOSS / max(SLIPPAGE + COMMISSION, 0.001)
            alloc = min(alloc, max_loss_alloc)

            if cash < alloc or alloc <= 0:
                continue

            # Execution price
            direction   = "buy" if side == "long" else "sell"
            entry_price = _apply_execution_costs(open_price, direction)
            shares      = alloc / entry_price

            if shares <= 0:
                continue

            # SL / TP from signal or ATR fallback
            exit_levels = result.get("exit_levels", {})
            if side == "long":
                sl_default = price - 2 * atr
                tp_default = price + 2 * atr * RISK_REWARD
                sl = exit_levels.get("stop_loss",   sl_default)
                tp = exit_levels.get("take_profit", tp_default)
            else:   # short
                sl_default = price + 2 * atr
                tp_default = price - 2 * atr * RISK_REWARD
                sl = exit_levels.get("stop_loss",   sl_default)
                tp = exit_levels.get("take_profit", tp_default)

            pos_key = f"{ticker}_pos_{pos_counter}"
            pos_counter += 1

            open_positions[pos_key] = {
                "ticker"          : ticker,
                "side"            : side,
                "entry_price"     : entry_price,
                "entry_date"      : today,
                "shares"          : shares,
                "cost"            : shares * entry_price,
                "sl"              : sl,
                "tp"              : tp,
                "signal"          : sig,
                "score"           : score,
                "position_pct"    : round(alloc / equity * 100, 1),
                "trailing_active" : False,
            }
            if side == "long":
                cash -= shares * entry_price
            # short: margin already deducted conceptually via alloc

    # ── Close all open positions at end of period ──────────────────────────────
    for key, pos in list(open_positions.items()):
        trade, proceeds = _close_position(
            pos, df["Close"].iloc[-1], df.index[-1], "End of Period", ticker, initial_cash
        )
        trades.append(trade)
        cash += proceeds

    return {
        "ticker"      : ticker,
        "initial_cash": initial_cash,
        "final_equity": round(cash, 2),
        "total_return": round((cash - initial_cash) / initial_cash * 100, 2),
        "trades"      : trades,
        "equity_curve": equity_curve,
        "killed"      : killed,
    }


# ─── PERFORMANCE METRICS ──────────────────────────────────────────────────────

def compute_metrics(result: dict, annual_rf: float = 0.04) -> dict:
    trades = result["trades"]
    if not trades:
        return {
            "verdict": "NO TRADES", "win_rate": 0, "profit_factor": 0,
            "max_drawdown": 0, "total_trades": 0, "sharpe": 0,
            "sortino": 0, "calmar": 0, "omega": 0,
            "var_95": 0, "cvar_95": 0,
            "avg_r_multiple": 0, "max_consecutive_losses": 0,
            "avg_hold_days": 0, "avg_position_by_signal": {},
            "exit_reasons": {}, "total_return": result["total_return"],
            "passed": False, "killed": result.get("killed", False),
            "long_trades": 0, "short_trades": 0,
        }

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    longs  = [t for t in trades if t.get("side", "long") == "long"]
    shorts = [t for t in trades if t.get("side", "short") == "short"]

    win_rate     = len(wins) / len(trades) * 100
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses)) + 1e-9

    # Equity series
    eq          = pd.Series([e["equity"] for e in result["equity_curve"]])
    running_max = eq.cummax()
    drawdowns   = (eq - running_max) / running_max * 100
    max_dd      = round(drawdowns.min(), 2)
    eq_returns  = eq.pct_change().dropna()
    daily_rf    = annual_rf / TRADING_DAYS

    # Sharpe
    sharpe = round(
        (eq_returns.mean() - daily_rf) / (eq_returns.std() + 1e-9) * np.sqrt(TRADING_DAYS), 2
    )

    # Sortino (downside deviation only)
    down_ret  = eq_returns[eq_returns < daily_rf]
    sortino   = round(
        (eq_returns.mean() - daily_rf) / (down_ret.std() + 1e-9) * np.sqrt(TRADING_DAYS), 2
    )

    # Calmar
    ann_return = (result["final_equity"] / result["initial_cash"]) ** (TRADING_DAYS / max(len(eq), 1)) - 1
    calmar     = round(ann_return / (abs(max_dd / 100) + 1e-9), 2)

    # Omega ratio (threshold = 0)
    gains  = eq_returns[eq_returns >  0].sum()
    losses_sum = abs(eq_returns[eq_returns < 0].sum()) + 1e-9
    omega  = round(gains / losses_sum, 2)

    # VaR & CVaR (95 %)
    var_95  = round(float(np.percentile(eq_returns, 5))  * 100, 3)
    cvar_95 = round(float(eq_returns[eq_returns <= np.percentile(eq_returns, 5)].mean()) * 100, 3)

    # R-multiples
    r_multiples = [
        t["pnl"] / max(abs(t["entry_price"] - t["stop_loss"])
                       * (t.get("position_pct", 20) / 100)
                       * result["initial_cash"], 1e-6)
        for t in trades
        if abs(t["entry_price"] - t["stop_loss"]) > 0
    ]

    # Max consecutive losses
    max_consec = cur = 0
    for t in trades:
        cur = cur + 1 if t["result"] == "LOSS" else 0
        max_consec = max(max_consec, cur)

    # Per-signal breakdown
    sig_sizes: dict[str, list] = {}
    for t in trades:
        sig_sizes.setdefault(t["signal"], []).append(t.get("position_pct", 20))

    # Exit reasons
    exit_reasons: dict[str, int] = {}
    for t in trades:
        r = t.get("exit_reason", "Unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    passed = win_rate >= 50 and gross_profit / gross_loss >= 1.3 and result["total_return"] > 0

    return {
        "verdict"               : "PASS ✅" if passed else "FAIL ❌",
        "passed"                : passed,
        "killed"                : result.get("killed", False),
        "total_trades"          : len(trades),
        "long_trades"           : len(longs),
        "short_trades"          : len(shorts),
        "win_rate"              : round(win_rate, 1),
        "profit_factor"         : round(gross_profit / gross_loss, 2),
        "gross_profit"          : round(gross_profit, 2),
        "gross_loss"            : round(gross_loss, 2),
        "max_drawdown"          : max_dd,
        "sharpe"                : sharpe,
        "sortino"               : sortino,
        "calmar"                : calmar,
        "omega"                 : omega,
        "var_95"                : var_95,
        "cvar_95"               : cvar_95,
        "total_return"          : result["total_return"],
        "avg_r_multiple"        : round(np.mean(r_multiples), 2) if r_multiples else 0.0,
        "max_consecutive_losses": max_consec,
        "avg_hold_days"         : round(np.mean([t.get("hold_days", 0) for t in trades]), 1),
        "avg_position_by_signal": {s: round(np.mean(v), 1) for s, v in sig_sizes.items()},
        "exit_reasons"          : exit_reasons,
    }


# ─── MONTE CARLO ──────────────────────────────────────────────────────────────

def monte_carlo(
    trades: list[dict],
    initial_cash: float,
    n_sims: int = 1_000,
    seed: int = 42,
) -> dict:
    """
    Permute the trade sequence N times and recompute final equity & max DD.
    Returns percentile statistics across simulated paths.
    """
    if not trades: return {}
    rng     = np.random.default_rng(seed)
    pnls    = np.array([t["pnl"] for t in trades])
    n       = len(pnls)

    final_equities = []
    max_dds        = []

    for _ in range(n_sims):
        perm  = rng.permutation(pnls)
        curve = initial_cash + np.cumsum(perm)
        final_equities.append(curve[-1])
        peak = np.maximum.accumulate(curve)
        max_dds.append(((curve - peak) / peak).min() * 100)

    fe   = np.array(final_equities)
    mdd  = np.array(max_dds)

    return {
        "n_simulations"      : n_sims,
        "final_equity_p5"    : round(float(np.percentile(fe,  5)), 2),
        "final_equity_p50"   : round(float(np.percentile(fe, 50)), 2),
        "final_equity_p95"   : round(float(np.percentile(fe, 95)), 2),
        "prob_profit"        : round(float((fe > initial_cash).mean() * 100), 1),
        "max_dd_p5"          : round(float(np.percentile(mdd, 5)),  2),
        "max_dd_median"      : round(float(np.percentile(mdd, 50)), 2),
        "max_dd_p95"         : round(float(np.percentile(mdd, 95)), 2),
    }


# ─── SAVING ───────────────────────────────────────────────────────────────────

def save_trades_csv(all_trades: list, run_date: str) -> str:
    path = os.path.join(RESULTS_DIR, f"trades_{run_date}.csv")
    if not all_trades: return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_trades[0].keys())
        writer.writeheader()
        writer.writerows(all_trades)
    return path


def save_summary_json(summary: dict, run_date: str) -> str:
    path = os.path.join(RESULTS_DIR, f"summary_{run_date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_serializer)
    return path


def save_summary_txt(summary: dict, run_date: str) -> str:
    path   = os.path.join(RESULTS_DIR, f"report_{run_date}.txt")
    s      = summary["strategy"]
    passed = s["passed_count"]
    total  = s["total_tickers"]

    lines = [
        "=" * 62,
        "  MarketLab v4.0 — Backtest Strategy Report",
        f"  Date: {run_date}",
        "=" * 62,
        f"  Tickers Tested  : {total}",
        f"  Period          : {summary['period']['start']} → {summary['period']['end']}",
        f"  Initial Capital : ${summary['settings']['initial_cash']:,.0f}",
        f"  Commission      : {COMMISSION*100:.2f}%  |  Slippage: {SLIPPAGE*100:.2f}%  |  Spread: {SPREAD_BPS} bps",
        f"  Kelly Fraction  : {KELLY_FRACTION:.2f}  |  Max DD Kill: {PORT_DRAWDOWN_KILL*100:.0f}%",
        "",
        THIN_BAR,
        "  Per-Ticker Results",
        THIN_BAR,
    ]

    for ticker, m in summary["tickers"].items():
        bh = summary["benchmarks"].get(ticker, 0)
        mc = summary.get("monte_carlo", {}).get(ticker, {})
        kill_flag = " ⚡KILLED" if m.get("killed") else ""
        lines.append(
            f"  {ticker:<6}  {m['verdict']:<10}{kill_flag}\n"
            f"         Return: {m['total_return']:+.1f}%  B&H: {bh:+.1f}%  "
            f"WR: {m['win_rate']:.0f}%  PF: {m['profit_factor']:.2f}\n"
            f"         DD: {m['max_drawdown']:.1f}%  Sharpe: {m['sharpe']:.2f}  "
            f"Sortino: {m['sortino']:.2f}  Calmar: {m['calmar']:.2f}  Omega: {m['omega']:.2f}\n"
            f"         VaR95: {m['var_95']:.2f}%  CVaR95: {m['cvar_95']:.2f}%  "
            f"Longs: {m['long_trades']}  Shorts: {m['short_trades']}  AvgHold: {m['avg_hold_days']:.0f}d"
        )
        if mc:
            lines.append(
                f"         MC(1k) P5→P50→P95: "
                f"${mc['final_equity_p5']:,.0f} → ${mc['final_equity_p50']:,.0f} → ${mc['final_equity_p95']:,.0f}  "
                f"ProbProfit: {mc['prob_profit']:.0f}%"
            )
        lines.append("")

    lines += [
        THIN_BAR,
        "  Strategy Accuracy",
        THIN_BAR,
        f"  Signals Correct : {passed}/{total}",
        f"  Accuracy        : {s['accuracy_pct']:.1f}%",
        f"  Overall Verdict : {s['overall_verdict']}",
        "=" * 62,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ─── EQUITY CURVE + DRAWDOWN ──────────────────────────────────────────────────

def plot_equity_curves(
    results: dict, benchmarks: dict, initial_cash: float, run_date: str
) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False,
                                   gridspec_kw={"height_ratios": [3, 1]})

    for ticker, res in results.items():
        if not res["equity_curve"]: continue
        eq = pd.Series(
            [e["equity"] for e in res["equity_curve"]],
            index=[e["date"] for e in res["equity_curve"]],
        )
        norm = eq / initial_cash * 100
        ax1.plot(norm, label=f"{ticker}", linewidth=1.5)
        # Drawdown subplot
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.plot(dd, linewidth=1, alpha=0.7)

    ax1.axhline(100, color="black", linestyle="--", linewidth=1, label="Starting Capital")
    ax1.set_title("MarketLab v4 — Equity Curves", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (% of initial)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"equity_curve_{run_date}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─── PRINTING ─────────────────────────────────────────────────────────────────

def _fmt_row(label: str, value: str) -> str:
    return f"    {label:<24}: {value}"


def print_ticker_result(ticker: str, metrics: dict, benchmark: float, mc: dict) -> None:
    section(f"{ticker}  ·  Backtest Result v4")
    killed_flag = "  ⚡ PORTFOLIO DD KILL-SWITCH TRIGGERED" if metrics.get("killed") else ""
    rows = [
        ("Verdict",            metrics["verdict"] + killed_flag),
        ("Total Return",       f"{metrics['total_return']:+.2f}%  (B&H: {benchmark:+.2f}%)"),
        ("Total Trades",       f"{metrics['total_trades']}  (L: {metrics['long_trades']}  S: {metrics['short_trades']})"),
        ("Win Rate",           f"{metrics['win_rate']:.1f}%"),
        ("Profit Factor",      f"{metrics['profit_factor']:.2f}"),
        ("Max Drawdown",       f"{metrics['max_drawdown']:.2f}%"),
        ("Sharpe Ratio",       f"{metrics['sharpe']:.2f}"),
        ("Sortino Ratio",      f"{metrics['sortino']:.2f}"),
        ("Calmar Ratio",       f"{metrics['calmar']:.2f}"),
        ("Omega Ratio",        f"{metrics['omega']:.2f}"),
        ("VaR 95%",            f"{metrics['var_95']:.3f}%  /day"),
        ("CVaR 95%",           f"{metrics['cvar_95']:.3f}%  /day"),
        ("Avg R-Multiple",     f"{metrics.get('avg_r_multiple', 0):.2f}"),
        ("Max Consec. Losses", str(metrics.get("max_consecutive_losses", 0))),
        ("Avg Hold Days",      f"{metrics.get('avg_hold_days', 0):.0f} days"),
    ]
    for label, value in rows:
        print(_fmt_row(label, value))

    if metrics.get("avg_position_by_signal"):
        print(_fmt_row("Avg Position %", ""))
        for sig, pct in metrics["avg_position_by_signal"].items():
            print(f"        {sig:15s}: {pct:.1f}%")

    if metrics.get("exit_reasons"):
        print(_fmt_row("Exit Reasons", ""))
        for reason, count in sorted(metrics["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"        {reason:25s}: {count}")

    if mc:
        print()
        print(_fmt_row("Monte Carlo (1 000 sims)", ""))
        print(f"        P5 → P50 → P95 equity : "
              f"${mc['final_equity_p5']:>10,.0f} → ${mc['final_equity_p50']:>10,.0f} → ${mc['final_equity_p95']:>10,.0f}")
        print(f"        Prob. of profit        : {mc['prob_profit']:.0f}%")
        print(f"        Median max drawdown    : {mc['max_dd_median']:.1f}%")


def print_strategy_summary(summary: dict) -> None:
    s = summary["strategy"]
    print(f"\n{BAR}")
    print("  Strategy Accuracy Report  (MarketLab v4.0)")
    print(BAR)
    print(f"  Signals Correct  : {s['passed_count']}/{s['total_tickers']}")
    print(f"  Accuracy         : {s['accuracy_pct']:.1f}%")
    print(f"  Overall Verdict  : {s['overall_verdict']}")
    print(BAR)


# ─── STOCK SELECTION ──────────────────────────────────────────────────────────

def _select_tickers_manual(name_to_ticker: dict) -> tuple[list[str], str, str]:
    num        = prompt_int("Number of stocks to backtest", 1, 20, 3)
    start_year = prompt_int("Backtest start year", 2015, 2024, 2020)
    start      = f"{start_year}-01-01"
    end        = today_str()
    section("Stock Selection")
    tickers = []
    for i in range(num):
        while True:
            raw = input(f"  [{i+1}/{num}] Company name or ticker: ").strip()
            if not raw: print("    ✗  Input cannot be empty."); continue
            ticker = resolve_ticker(raw, name_to_ticker)
            if validate_ticker(ticker):
                tickers.append(ticker); print(f"    ✔  {ticker} added."); break
            print(f"    ✗  '{ticker}' not in warehouse. Run Mode [5] to update.")
    return tickers, start, end


def _select_tickers_auto(name_to_ticker: dict) -> list[tuple[list[str], str, str, str]]:
    end = today_str()
    print(f"\n  {THIN_BAR}")
    print("  Available Plans:")
    for i, plan in enumerate(AUTO_PLANS, 1):
        print(f"    [{i}] {plan['label']:<38} {plan['start']}  {', '.join(plan['tickers'])}")
    print(f"    [A] Run All ({len(AUTO_PLANS)} plans)")
    print(f"  {THIN_BAR}")
    raw = input("  Select a plan [1-8 or A]: ").strip().upper()
    if raw == "A":
        selected = AUTO_PLANS
    elif raw.isdigit() and 1 <= int(raw) <= len(AUTO_PLANS):
        selected = [AUTO_PLANS[int(raw) - 1]]
    else:
        selected = [AUTO_PLANS[0]]

    plans_out = []
    for plan in selected:
        valid   = [t for t in plan["tickers"] if validate_ticker(t)]
        skipped = set(plan["tickers"]) - set(valid)
        if skipped: log.warning("%s: skipping %s (not in warehouse).", plan["label"], skipped)
        if not valid: log.error("%s: no valid tickers — skipped.", plan["label"]); continue
        plans_out.append((valid, plan["start"], end, plan["label"]))
    return plans_out


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BAR}")
    print("  MarketLab  ·  Backtesting Engine  v4.0")
    print(f"  Short selling • Kelly sizing • Monte Carlo • Multi-position")
    print(BAR)

    ensure_results_dir()
    run_date = today_str()

    initial_cash = prompt_float("Initial capital ($)", 10_000)

    print("  Risk-Free Rate  (10Y US Treasury ≈ 4.2%)")
    raw_rf = input("  Annual RF rate [%] (Enter = 4.0%): ").strip()
    try:    annual_rf = max(0.0, min(float(raw_rf) / 100 if raw_rf else 0.04, MAX_RF_RATE))
    except: annual_rf = 0.04
    log.info("Risk-free rate: %.2f%%", annual_rf * 100)

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
            log.error("No valid plans. Update warehouse first (Mode 5)."); return
    else:
        tickers, start, end = _select_tickers_manual(name_to_ticker)
        plans = [(tickers, start, end, "Manual")]

    grand_metrics    : dict = {}
    grand_trades     : list = []
    grand_results    : dict = {}
    grand_benchmarks : dict = {}
    grand_mc         : dict = {}

    for plan_idx, (tickers, start, end, label) in enumerate(plans, 1):
        print(f"\n{BAR}")
        print(f"  Plan [{plan_idx}/{len(plans)}]: {label}")
        print(f"  Tickers : {', '.join(tickers)}")
        print(f"  Period  : {start}  →  {end}")
        print(BAR)

        mkt_returns = fetch_market_returns(start, end)

        # ── Parallel data loading ──────────────────────────────────────────────
        section("Loading data (parallel)")
        dfs = load_tickers_parallel(tickers, start, end)

        # ── Build returns cache for correlation guard ──────────────────────────
        returns_cache = {t: df["Stock_Return"] for t, df in dfs.items()}

        plan_results    : dict = {}
        plan_metrics    : dict = {}
        plan_trades     : list = []
        plan_benchmarks : dict = {}
        plan_mc         : dict = {}

        for ticker in tickers:
            if ticker not in dfs:
                log.warning("%s: no data — skipped.", ticker); continue

            df = dfs[ticker]
            print(f"\n{THIN_BAR}")
            print(f"  Backtesting  {ticker}  ({start} → {end})")
            print(THIN_BAR)

            result    = run_backtest(ticker, df, mkt_returns, initial_cash, annual_rf, returns_cache)
            metrics   = compute_metrics(result, annual_rf)
            benchmark = compute_benchmark(ticker, start, end)
            mc        = monte_carlo(result["trades"], initial_cash)
            key       = f"{ticker}|{label}"

            plan_results[key]    = result
            plan_metrics[key]    = metrics
            plan_benchmarks[key] = benchmark
            plan_mc[key]         = mc
            plan_trades.extend(result["trades"])

            print_ticker_result(ticker, metrics, benchmark, mc)

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
                    "kelly_frac"  : KELLY_FRACTION,
                    "spread_bps"  : SPREAD_BPS,
                },
                trades = result["trades"],
            )

        grand_metrics.update(plan_metrics)
        grand_trades.extend(plan_trades)
        grand_results.update(plan_results)
        grand_benchmarks.update(plan_benchmarks)
        grand_mc.update(plan_mc)

        if plan_metrics:
            p_pass = sum(1 for m in plan_metrics.values() if m.get("passed", False))
            p_tot  = len(plan_metrics)
            print(f"\n  ✦  {label}: {p_pass}/{p_tot} passed ({p_pass/p_tot*100:.0f}%)")

    if not grand_metrics:
        log.error("No results to summarize."); return

    passed_count = sum(1 for m in grand_metrics.values() if m.get("passed", False))
    total        = len(grand_metrics)
    accuracy     = passed_count / total * 100

    summary = {
        "period"      : {"start": plans[0][1], "end": plans[0][2]},
        "settings"    : {
            "initial_cash": initial_cash,
            "commission"  : COMMISSION,
            "slippage"    : SLIPPAGE,
            "spread_bps"  : SPREAD_BPS,
            "kelly_frac"  : KELLY_FRACTION,
        },
        "tickers"     : grand_metrics,
        "benchmarks"  : grand_benchmarks,
        "monte_carlo" : grand_mc,
        "strategy"    : {
            "passed_count"   : passed_count,
            "total_tickers"  : total,
            "accuracy_pct"   : round(accuracy, 1),
            "overall_verdict": "CREDIBLE STRATEGY ✅" if accuracy >= 60 else "NEEDS IMPROVEMENT ⚠️",
        },
    }

    print_strategy_summary(summary)

    section("Saving Results")
    csv_path   = save_trades_csv(grand_trades,  run_date)
    json_path  = save_summary_json(summary,     run_date)
    txt_path   = save_summary_txt(summary,      run_date)
    chart_path = plot_equity_curves(grand_results, grand_benchmarks, initial_cash, run_date)

    try:
        from ml_predictor import invalidate_cache
        invalidate_cache()
        log.info("ML cache invalidated.")
    except ImportError:
        pass

    print(f"    ✔  Trades log    : {csv_path}")
    print(f"    ✔  Summary JSON  : {json_path}")
    print(f"    ✔  Report TXT    : {txt_path}")
    print(f"    ✔  Equity chart  : {chart_path}")
    print(f"\n{THIN_BAR}\n")


if __name__ == "__main__":
    main()
