"""
MarketLab — Strategy Optimizer  v1.0
=====================================
Uses Optuna (Bayesian optimization) to find the best signal parameters
by running walk-forward backtests on a fixed basket of stocks.

Usage:
    python strategy_optimizer.py            # runs optimization
    python strategy_optimizer.py --apply    # applies best_params.json to signals.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

N_TRIALS        = 100          # Bayesian trials (≈10-15 min on typical hardware)
N_STARTUP       = 20           # random exploration before Bayesian kicks in
BEST_PARAMS_FILE = "best_params.json"
HISTORY_FILE    = "backtest_history.json"
SIGNALS_FILE = "core/signals.py"

# Fixed basket — diverse enough to avoid overfitting to one regime
OPTIMIZATION_BASKET = [
    ("AAPL",  "2020-01-01"),
    ("MSFT",  "2020-01-01"),
    ("NVDA",  "2021-01-01"),
    ("TSLA",  "2021-01-01"),
    ("JPM",   "2020-01-01"),
    ("XOM",   "2020-01-01"),
    ("AMZN",  "2020-01-01"),
]

END_DATE = datetime.today().strftime("%Y-%m-%d")

# ─── PARAMETER SEARCH SPACE ───────────────────────────────────────────────────

SEARCH_SPACE: dict[str, tuple] = {
    # Signal thresholds
    "BUY_THRESHOLD"         : ("int",   4,   9,  1),
    "SELL_THRESHOLD"        : ("int",  -9,  -4,  1),   # will be stored as negative

    # ATR
    "atr_multiplier"        : ("float", 1.0, 2.5, 0.1),

    # Risk/Reward
    "rr_base"               : ("float", 1.8, 3.0, 0.1),
    "rr_strong_bull"        : ("float", 2.5, 4.0, 0.1),
    "rr_bull"               : ("float", 2.0, 3.5, 0.1),

    # ADX
    "adx_strong"            : ("float", 20.0, 30.0, 1.0),
    "adx_weak"              : ("float", 15.0, 25.0, 1.0),

    # RSI
    "rsi_oversold"          : ("float", 20.0, 35.0, 1.0),
    "rsi_overbought"        : ("float", 65.0, 80.0, 1.0),

    # Stochastic
    "stoch_oversold"        : ("float", 15.0, 30.0, 1.0),
    "stoch_overbought"      : ("float", 70.0, 85.0, 1.0),

    # Volume
    "volume_multiplier"     : ("float", 1.2, 2.5, 0.1),

    # Score weights (integer points added per signal)
    "w_trend"               : ("int",   1,   3,  1),
    "w_golden_cross"        : ("int",   1,   2,  1),
    "w_divergence"          : ("int",   2,   4,  1),
    "w_double_oversold"     : ("int",   3,   5,  1),
    "w_macd"                : ("int",   1,   3,  1),
    "w_bb_touch"            : ("int",   1,   3,  1),
    "w_volume"              : ("int",   1,   3,  1),

    # Volatility filter thresholds
    "vol_extreme_pct"       : ("float", 4.0, 7.0, 0.5),
    "vol_high_pct"          : ("float", 2.0, 4.0, 0.5),
    "vol_low_pct"           : ("float", 0.5, 1.5, 0.25),

    # Relative Strength
    "rs_outperform_pct"     : ("float", 3.0, 10.0, 1.0),
    "rs_underperform_pct"   : ("float",-15.0,-5.0, 1.0),

    # Cooldown (backtest.py constant — passed via env)
    "cooldown_days"         : ("int",   3,   10,  1),
}


# ─── OBJECTIVE FUNCTION ───────────────────────────────────────────────────────

def objective(trial, name_to_ticker: dict) -> float:
    """
    Optuna objective — lower is better (we return -score).
    Score = mean across basket of: Sharpe × (WinRate/100) / max(|MaxDD|, 1)
    Penalize if avg_trades < 5 (too few signals = overfitting).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    params = _suggest_params(trial)

    scores = []
    for ticker, start in OPTIMIZATION_BASKET:
        try:
            s = _run_single(ticker, start, END_DATE, params, name_to_ticker)
            if s is not None:
                scores.append(s)
        except Exception:
            pass

    if not scores:
        return -999.0   # completely failed

    return float(np.mean(scores))


def _suggest_params(trial) -> dict:
    p = {}
    for name, spec in SEARCH_SPACE.items():
        kind, lo, hi, step = spec
        if kind == "int":
            p[name] = trial.suggest_int(name, lo, hi, step=step)
        else:
            p[name] = trial.suggest_float(name, lo, hi, step=step)

    # Enforce: adx_weak < adx_strong
    if p["adx_weak"] >= p["adx_strong"]:
        p["adx_weak"] = p["adx_strong"] - 1

    # Enforce: rsi_oversold < rsi_overbought
    if p["rsi_oversold"] >= p["rsi_overbought"]:
        p["rsi_oversold"] = p["rsi_overbought"] - 10

    # Enforce: stoch_oversold < stoch_overbought
    if p["stoch_oversold"] >= p["stoch_overbought"]:
        p["stoch_oversold"] = p["stoch_overbought"] - 10

    # Enforce: vol_high < vol_extreme
    if p["vol_high_pct"] >= p["vol_extreme_pct"]:
        p["vol_high_pct"] = p["vol_extreme_pct"] - 0.5

    # Enforce: rr_bull < rr_strong_bull
    if p["rr_bull"] >= p["rr_strong_bull"]:
        p["rr_bull"] = p["rr_strong_bull"] - 0.2

    return p


def _run_single(
    ticker: str,
    start: str,
    end: str,
    params: dict,
    name_to_ticker: dict,
) -> float | None:
    """
    Inject params into a patched version of signals + run backtest.
    Reads from local data/ warehouse — no internet calls.
    Returns composite score or None on failure (never raises).
    """
    try:
        import analysis.backtest as bt
        import core.signals as sig_module

        _patch_signals(sig_module, params)

        # ── Load from local warehouse — no internet ───────────────────────
        try:
            from core.stock_warehouse import load_local
            raw = load_local(ticker, start=start, end=end)
        except FileNotFoundError:
            return None
        except Exception:
            return None

        if raw is None or len(raw) < 300:
            return None

        # ── Prepare indicators ────────────────────────────────────────────
        try:
            import pandas_ta as ta

            raw = raw.copy()
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw.set_index("Date").sort_index()

            raw["Stock_Return"] = raw["Close"].pct_change()
            raw["MA50"]         = raw["Close"].rolling(50).mean()
            raw["MA200"]        = raw["Close"].rolling(200).mean()
            raw["EMA20"]        = raw["Close"].ewm(span=20, adjust=False).mean()
            raw["RSI"]          = ta.rsi(raw["Close"], length=14)
            raw["BB_middle"]    = raw["Close"].rolling(20).mean()
            raw["BB_std"]       = raw["Close"].rolling(20).std()
            raw["BB_upper"]     = raw["BB_middle"] + raw["BB_std"] * 2
            raw["BB_lower"]     = raw["BB_middle"] - raw["BB_std"] * 2
            raw["MACD"]         = (raw["Close"].ewm(span=12, adjust=False).mean()
                                 - raw["Close"].ewm(span=26, adjust=False).mean())
            raw["Signal"]       = raw["MACD"].ewm(span=9, adjust=False).mean()
            raw["Histogram"]    = raw["MACD"] - raw["Signal"]
            raw["L14"]          = raw["Low"].rolling(14).min()
            raw["H14"]          = raw["High"].rolling(14).max()
            raw["%K"]           = (raw["Close"] - raw["L14"]) / (raw["H14"] - raw["L14"] + 1e-9) * 100
            raw["%D"]           = raw["%K"].rolling(3).mean()
            raw.dropna(inplace=True)

            if len(raw) < 300:
                return None

            df = raw

        except Exception:
            return None

        # ── Market returns from local SPY ─────────────────────────────────
        try:
            from core.stock_warehouse import load_local as _ll
            spy = _ll("SPY", start=start, end=end)
            spy["Date"] = pd.to_datetime(spy["Date"])
            spy = spy.set_index("Date").sort_index()
            mkt = spy["Close"].pct_change().rename("Market_Return")
            mkt = mkt.reindex(df.index)
        except Exception:
            mkt = pd.Series(0.0, index=df.index, name="Market_Return")

        # ── Run backtest ──────────────────────────────────────────────────
        result  = bt.run_backtest(ticker, df, mkt, initial_cash=10_000, annual_rf=0.04)
        metrics = bt.compute_metrics(result)

        trades     = metrics.get("total_trades", 0)
        sharpe     = metrics.get("sharpe",       0)
        win_rate   = metrics.get("win_rate",     0)
        max_dd     = abs(metrics.get("max_drawdown", 100))
        tot_return = metrics.get("total_return", 0)

        if trades < 4:
            return -50.0

        dd_denom = max(max_dd, 1.0)
        score = (sharpe * (win_rate / 100) / dd_denom) * 100 + tot_return * 0.1
        return float(score)

    except Exception:
        return None


def _patch_signals(sig_module, params: dict) -> None:
    """
    Directly overwrite module-level variables and monkey-patch
    the generate_signal function to use new params.
    """
    sig_module.BUY_THRESHOLD  =  params["BUY_THRESHOLD"]
    sig_module.SELL_THRESHOLD = -abs(params["SELL_THRESHOLD"])

    # Store params as module attrs so generate_signal can read them
    sig_module._OPT_PARAMS = params


# ─── PATCHED generate_signal ──────────────────────────────────────────────────
# We do NOT modify signals.py on disk during optimization.
# Instead we override the function at runtime via monkeypatching.

def _build_patched_generate_signal(original_module):
    """
    Returns a new generate_signal that reads _OPT_PARAMS from the module
    when present, otherwise falls back to original hard-coded values.
    """
    import core.signals as sig

    def patched_generate_signal(df, info, metrics, market_returns=None):
        p = getattr(sig, "_OPT_PARAMS", None)
        if p is None:
            return sig._original_generate_signal(df, info, metrics, market_returns)

        # ── Re-implement generate_signal with injected params ─────────────────
        reasons = []
        try:
            if len(df) < 200:
                return {"signal": "WAIT", "score": 0, "reasons": []}

            required_cols = ["Close", "RSI", "Histogram", "MA50", "MA200",
                             "Volume", "BB_upper", "BB_lower", "%K", "%D"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                return {"signal": "ERROR", "score": 0, "reasons": []}

            if df[required_cols].tail(2).isnull().any().any():
                return {"signal": "ERROR", "score": 0, "reasons": []}

            price      = df["Close"].iloc[-1]
            volume     = df["Volume"].iloc[-1]
            avg_volume = df["Volume"].tail(20).mean()
            rsi        = df["RSI"].iloc[-1]
            stoch_k    = df["%K"].iloc[-1]
            stoch_d    = df["%D"].iloc[-1]
            prev_k     = df["%K"].iloc[-2]
            prev_d     = df["%D"].iloc[-2]
            hist       = df["Histogram"].iloc[-1]
            prev_hist  = df["Histogram"].iloc[-2]
            ma50       = df["MA50"].iloc[-1]
            ma200      = df["MA200"].iloc[-1]
            upper_bb   = df["BB_upper"].iloc[-1]
            lower_bb   = df["BB_lower"].iloc[-1]
            sharpe_m   = metrics.get("Sharpe Annualized", 0)

            atr = sig.calculate_atr(df)
            adx = sig.calculate_adx(df)

            score          = 0
            volume_high    = volume > avg_volume * p["volume_multiplier"]
            confidence     = "Normal"
            trigger_active = False
            trigger_bias   = 0

            is_bullish_trend = price > ma200
            is_golden_cross  = ma50  > ma200

            score += p["w_trend"] if is_bullish_trend else -p["w_trend"]
            reasons.append(f"Trend: {'Bullish' if is_bullish_trend else 'Bearish'}.")

            score += p["w_golden_cross"] if is_golden_cross else -p["w_golden_cross"]
            reasons.append(f"Structure: {'Golden' if is_golden_cross else 'Death'} Cross.")

            # ADX
            if adx >= p["adx_strong"]:
                score += 1
                reasons.append(f"ADX={adx:.1f} strong (+1).")
            elif adx < p["adx_weak"]:
                score -= 1
                reasons.append(f"ADX={adx:.1f} weak (−1).")

            # Divergence
            divergence = sig.detect_divergence(df)
            if divergence == "bullish":
                score += p["w_divergence"]
                trigger_active = True; trigger_bias += 1
                reasons.append("Bullish divergence.")
            elif divergence == "bearish":
                score -= p["w_divergence"]
                trigger_active = True; trigger_bias -= 1
                reasons.append("Bearish divergence.")

            # Oversold / Overbought
            if rsi < p["rsi_oversold"] and stoch_k < p["stoch_oversold"]:
                if is_bullish_trend:
                    score += p["w_double_oversold"]
                    trigger_active = True; trigger_bias += 1
                    reasons.append("Double Oversold in Bull trend.")
            elif rsi > p["rsi_overbought"] and stoch_k > p["stoch_overbought"]:
                if not is_bullish_trend:
                    score -= p["w_double_oversold"]
                    trigger_active = True; trigger_bias -= 1
                    reasons.append("Double Overbought in Bear trend.")
                else:
                    score -= 1

            # MA200 support
            margin = (price - ma200) / ma200
            if 0 < margin < 0.02 and p["rsi_oversold"] < rsi < p["rsi_oversold"] + 10:
                score += 1
                reasons.append("MA200 support test.")

            # Stochastic crossover
            if stoch_k > stoch_d and prev_k <= prev_d and stoch_k < p["stoch_oversold"]:
                score += 1; trigger_active = True; trigger_bias += 1
            elif stoch_k < stoch_d and prev_k >= prev_d and stoch_k > p["stoch_overbought"]:
                score -= 1; trigger_active = True; trigger_bias -= 1

            # MACD
            if hist > 0 and prev_hist <= 0:
                score += p["w_macd"]; trigger_active = True; trigger_bias += 1
                reasons.append("MACD Bullish Crossover.")
            elif hist < 0 and prev_hist >= 0:
                score -= p["w_macd"]; trigger_active = True; trigger_bias -= 1
                reasons.append("MACD Bearish Crossover.")

            # BB touch
            if price <= lower_bb and is_bullish_trend:
                score += p["w_bb_touch"]; trigger_active = True; trigger_bias += 1
                reasons.append("Lower BB touch.")
            elif price >= upper_bb and not is_bullish_trend:
                score -= p["w_bb_touch"]; trigger_active = True; trigger_bias -= 1
                reasons.append("Upper BB touch.")

            # Volume
            if volume_high and trigger_active and trigger_bias != 0:
                vol_bonus = p["w_volume"] if trigger_bias > 0 else -p["w_volume"]
                score += vol_bonus
                reasons.append(f"Volume confirms ({'Bull' if trigger_bias > 0 else 'Bear'}).")

            # Sharpe quality
            if sharpe_m > 1.5:
                confidence = "High"
            elif sharpe_m < 0:
                confidence = "Low"
                score -= 2

            # Volatility filter
            atr_pct = (atr / price) * 100
            if atr_pct > p["vol_extreme_pct"]:
                score -= 3
            elif atr_pct > p["vol_high_pct"]:
                score -= 1
            elif atr_pct < p["vol_low_pct"]:
                score += 1

            # Market regime
            regime_info = sig.assess_market_regime(market_returns)
            if regime_info["score_penalty"] != 0:
                score += regime_info["score_penalty"]

            # Relative strength
            rs_info = sig.assess_relative_strength(df, market_returns)
            if rs_info["rs_score"] != 0:
                score += rs_info["rs_score"]

            # Signal
            buy_thresh  =  p["BUY_THRESHOLD"]
            sell_thresh = -abs(p["SELL_THRESHOLD"])

            if score >= buy_thresh:
                signal = "STRONG BUY" if (score >= buy_thresh + 2 and is_bullish_trend) else "BUY"
            elif score <= sell_thresh:
                signal = "STRONG SELL" if (score <= sell_thresh - 2 and not is_bullish_trend) else "SELL"
            else:
                signal = "HOLD"

            # Exit levels with optimized params
            if signal in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
                rr = (
                    p["rr_strong_bull"] if (is_bullish_trend and score >= buy_thresh + 2) else
                    p["rr_bull"]        if (is_bullish_trend and score >= buy_thresh) else
                    p["rr_base"]
                )
                atr_mult = p["atr_multiplier"]
                if "BUY" in signal:
                    sl = round(price - atr_mult * atr, 2)
                    tp = round(price + (price - sl) * rr, 2)
                else:
                    sl = round(price + atr_mult * atr, 2)
                    tp = round(price - (sl - price) * rr, 2)
                exit_levels = {"stop_loss": sl, "take_profit": tp,
                               "risk_reward": rr, "atr_used": round(atr, 4)}
            else:
                exit_levels = {}

            return {
                "signal":           signal,
                "score":            score,
                "confidence_level": confidence,
                "reasons":          reasons,
                "price_at_signal":  round(price, 4),
                "exit_levels":      exit_levels,
                "market_regime":    regime_info["regime"],
                "rs_pct":           rs_info["rs_pct"],
                "adx":              adx,
                "atr":              round(atr, 4),
                "atr_pct":          round(atr_pct, 2),
                "time_exit": {"enabled": True, "days": 5, "min_profit_pct": 1.5},
            }

        except Exception as e:
            return {"signal": "ERROR", "score": 0, "reasons": [str(e)], "exit_levels": {}}

    return patched_generate_signal


# ─── APPLY BEST PARAMS TO signals.py ─────────────────────────────────────────

PARAM_PATTERNS: dict[str, str] = {
    "BUY_THRESHOLD"     : r"^BUY_THRESHOLD\s*=\s*.*",
    "SELL_THRESHOLD"    : r"^SELL_THRESHOLD\s*=\s*.*",
    "atr_multiplier"    : r"atr_multiplier\s*=\s*[\d.]+",
    "rr_base"           : r"risk_reward\s*=\s*[\d.]+",
    "adx_strong"        : r"adx\s*>=\s*[\d.]+",
    "adx_weak"          : r"adx\s*<\s*[\d.]+",
    "rsi_oversold"      : r"rsi\s*<\s*[\d.]+",
    "rsi_overbought"    : r"rsi\s*>\s*[\d.]+",
    "stoch_oversold"    : r"stoch_k\s*<\s*[\d.]+",
    "stoch_overbought"  : r"stoch_k\s*>\s*[\d.]+",
    "volume_multiplier" : r"avg_volume\s*\*\s*[\d.]+",
}


def apply_best_params(params_file: str = BEST_PARAMS_FILE, signals_file: str = "core/signals.py") -> None:
    if not os.path.exists(params_file):
        print(f"  ✗  {params_file} not found. Run optimization first.")
        return

    with open(params_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = data.get("params", {})
    if not params:
        print("  ✗  No params found in file.")
        return

    with open(signals_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Backup
    backup = signals_file.replace(".py", "_backup.py")
    with open(backup, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✔  Backup saved: {backup}")

    # Apply top-level constants
    for key, val in [
        ("BUY_THRESHOLD",  params.get("BUY_THRESHOLD",  6)),
        ("SELL_THRESHOLD", -abs(params.get("SELL_THRESHOLD", 6))),
    ]:
        content = re.sub(
            rf"^({key}\s*=\s*).*",
            lambda m, v=val: f"{m.group(1)}{v}",
            content,
            flags=re.MULTILINE,
        )

    # Apply atr_multiplier
    atr_m = round(params.get("atr_multiplier", 1.5), 2)
    content = re.sub(
        r"(atr_multiplier\s*=\s*)[\d.]+",
        lambda m: f"{m.group(1)}{atr_m}",
        content,
    )

    # Apply risk_reward base
    rr = round(params.get("rr_base", 2.3), 2)
    content = re.sub(
        r"(risk_reward\s*:\s*float\s*=\s*)[\d.]+",
        lambda m: f"{m.group(1)}{rr}",
        content,
    )

    # Apply ADX thresholds
    adx_s = round(params.get("adx_strong", 25), 1)
    adx_w = round(params.get("adx_weak",   20), 1)
    content = re.sub(r"(adx\s*>=\s*)[\d.]+",  lambda m: f"{m.group(1)}{adx_s}", content)
    content = re.sub(r"(adx\s*<\s*)[\d.]+",   lambda m: f"{m.group(1)}{adx_w}", content)

    # Apply RSI
    rsi_os = round(params.get("rsi_oversold",   30), 1)
    rsi_ob = round(params.get("rsi_overbought", 70), 1)
    content = re.sub(r"(rsi\s*<\s*)[\d.]+",  lambda m: f"{m.group(1)}{rsi_os}", content)
    content = re.sub(r"(rsi\s*>\s*)[\d.]+",  lambda m: f"{m.group(1)}{rsi_ob}", content)

    # Apply volume multiplier
    vol_m = round(params.get("volume_multiplier", 1.5), 2)
    content = re.sub(
        r"(avg_volume\s*\*\s*)[\d.]+",
        lambda m: f"{m.group(1)}{vol_m}",
        content,
    )

    with open(signals_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✔  {signals_file} updated with optimized parameters.")
    print(f"\n  Applied parameters:")
    for k, v in params.items():
        print(f"    {k:<28} = {v}")


# ─── MAIN OPTIMIZATION LOOP ───────────────────────────────────────────────────

def run_optimization() -> None:
    try:
        import optuna
    except ImportError:
        print("  Installing optuna...")
        os.system(f"{sys.executable} -m pip install optuna -q")
        import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Monkey-patch generate_signal before importing backtest
    import core.signals as sig_module
    if not hasattr(sig_module, "_original_generate_signal"):
        sig_module._original_generate_signal = sig_module.generate_signal
    sig_module.generate_signal = _build_patched_generate_signal(sig_module)

    # Load company map
    try:
        with open("companies.json", "r", encoding="utf-8") as f:
            name_to_ticker = json.load(f)
    except FileNotFoundError:
        name_to_ticker = {}

    # Verify basket data exists
    global OPTIMIZATION_BASKET
    available = []
    for ticker, start in OPTIMIZATION_BASKET:
        path = os.path.join("data", f"{ticker}.csv")
        if os.path.exists(path):
            available.append((ticker, start))
        else:
            print(f"  ⚠  No local data for {ticker} — skipping from basket.")

    if len(available) < 3:
        print("  ✗  Need at least 3 stocks with local data. Run weekly_update() first.")
        return

    OPTIMIZATION_BASKET = available

    print(f"\n{'═'*58}")
    print(f"  MarketLab — Strategy Optimizer  v1.0")
    print(f"{'═'*58}")
    print(f"  Basket       : {', '.join(t for t,_ in OPTIMIZATION_BASKET)}")
    print(f"  Trials       : {N_TRIALS}  (first {N_STARTUP} random)")
    print(f"  Parameters   : {len(SEARCH_SPACE)}")
    print(f"  Objective    : maximize Sharpe × WinRate / |MaxDD|")
    print(f"{'═'*58}\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=N_STARTUP,
            seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    best_so_far = [-999.0]

    def callback(study, trial):
        val = trial.value if trial.value is not None else -999
        if val > best_so_far[0]:
            best_so_far[0] = val
            print(f"  Trial {trial.number:>3}  ✦ New best: {val:+.4f}  "
                  f"BUY_THR={trial.params.get('BUY_THRESHOLD')}  "
                  f"ATR_mult={trial.params.get('atr_multiplier'):.1f}  "
                  f"RSI_os={trial.params.get('rsi_oversold'):.0f}")
        elif trial.number % 10 == 0:
            print(f"  Trial {trial.number:>3}  score={val:+.4f}  "
                  f"(best={best_so_far[0]:+.4f})")

    study.optimize(
        lambda trial: objective(trial, name_to_ticker),
        n_trials=N_TRIALS,
        callbacks=[callback],
        show_progress_bar=False,
    )

    best = study.best_trial
    best_params = best.params.copy()

    # Enforce SELL_THRESHOLD sign
    best_params["SELL_THRESHOLD"] = -abs(best_params.get("SELL_THRESHOLD", 6))

    print(f"\n{'═'*58}")
    print(f"  Optimization Complete")
    print(f"{'═'*58}")
    print(f"  Best Score   : {best.value:+.4f}")
    print(f"  Best Trial   : #{best.number}")
    print(f"\n  Best Parameters:")
    for k, v in best_params.items():
        default_val = _get_default(k)
        change = ""
        if isinstance(v, float):
            change = f"  (default: {default_val})" if abs(v - default_val) > 0.05 else ""
        else:
            change = f"  (default: {default_val})" if v != default_val else ""
        print(f"    {k:<28} = {v}{change}")

    # Save
    output = {
        "timestamp"   : datetime.now().isoformat(),
        "best_score"  : round(best.value, 6),
        "best_trial"  : best.number,
        "n_trials"    : N_TRIALS,
        "basket"      : [t for t, _ in OPTIMIZATION_BASKET],
        "params"      : best_params,
    }
    with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✔  Saved to {BEST_PARAMS_FILE}")
    print(f"\n  To apply to signals.py:")
    print(f"    python strategy_optimizer.py --apply")
    print(f"{'═'*58}\n")


def _get_default(key: str) -> Any:
    defaults = {
        "BUY_THRESHOLD": 6, "SELL_THRESHOLD": -6,
        "atr_multiplier": 1.5, "rr_base": 2.3,
        "rr_strong_bull": 3.0, "rr_bull": 2.5,
        "adx_strong": 25.0, "adx_weak": 20.0,
        "rsi_oversold": 30.0, "rsi_overbought": 70.0,
        "stoch_oversold": 20.0, "stoch_overbought": 80.0,
        "volume_multiplier": 1.5,
        "w_trend": 2, "w_golden_cross": 1, "w_divergence": 3,
        "w_double_oversold": 4, "w_macd": 2, "w_bb_touch": 2, "w_volume": 2,
        "vol_extreme_pct": 5.0, "vol_high_pct": 3.0, "vol_low_pct": 1.0,
        "rs_outperform_pct": 5.0, "rs_underperform_pct": -10.0,
        "cooldown_days": 5,
    }
    return defaults.get(key, "?")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketLab Strategy Optimizer")
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply best_params.json to signals.py without running optimization"
    )
    args = parser.parse_args()

    if args.apply:
        apply_best_params()
    else:
        run_optimization()
