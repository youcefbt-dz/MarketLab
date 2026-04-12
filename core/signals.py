import pandas as pd
import numpy as np

# ─── THRESHOLDS ───────────────────────────────────────────────────────────────
BUY_THRESHOLD  =  6   
SELL_THRESHOLD = -6
MIN_ADX_ENTRY  = 18

# ─── ATR CALCULATOR ───────────────────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:

    high  = df["High"].values  if "High"  in df.columns else df["Close"].values
    low   = df["Low"].values   if "Low"   in df.columns else df["Close"].values
    close = df["Close"].values

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1]),
        ),
    )
    if len(tr) < period:
        return float(close[-1] * 0.02)  
    return float(np.mean(tr[-period:]))

# ─── ADX CALCULATOR ───────────────────────────────────────────────────────────

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:

    if "High" not in df.columns or "Low" not in df.columns:
        return 25.0  # fallback 

    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(close)

    if n < period + 1:
        return 25.0

    plus_dm  = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                         np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                         np.maximum(low[:-1] - low[1:], 0), 0)
    tr       = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]),
                   np.abs(low[1:]  - close[:-1])),
    )

    atr14    = np.mean(tr[-period:])
    if atr14 == 0:
        return 25.0

    plus_di  = 100 * np.mean(plus_dm[-period:])  / atr14
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr14
    dx_denom = plus_di + minus_di
    if dx_denom == 0:
        return 25.0

    dx  = 100 * abs(plus_di - minus_di) / dx_denom
    return round(float(dx), 2)


# ─── DIVERGENCE DETECTION ─────────────────────────────────────────────────────

def detect_divergence(df: pd.DataFrame, lookback: int = 30) -> str | None:
    if len(df) < lookback:
        return None

    window = df.tail(lookback)
    prices = window["Close"].values
    rsi    = window["RSI"].values

    low_idx = [
        i for i in range(1, len(prices) - 1)
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]
    ]
    if len(low_idx) >= 2:
        i1, i2 = low_idx[-2], low_idx[-1]
        if prices[i2] < prices[i1] and rsi[i2] > rsi[i1]:
            return "bullish"

    high_idx = [
        i for i in range(1, len(prices) - 1)
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]
    ]
    if len(high_idx) >= 2:
        i1, i2 = high_idx[-2], high_idx[-1]
        if prices[i2] > prices[i1] and rsi[i2] < rsi[i1]:
            return "bearish"

    return None


# ─── VOLATILITY FILTER ────────────────────────────────────────────────────────

def assess_volatility(df: pd.DataFrame, atr: float) -> dict:

    price     = df["Close"].iloc[-1]
    atr_pct   = (atr / price) * 100

    if atr_pct > 5.0:
        return {
            "vol_score": -3,
            "atr_pct":   round(atr_pct, 2),
            "reason":    f"Volatility: ATR is {atr_pct:.1f}% of price — Extreme volatility, high risk (−3).",
        }
    elif atr_pct > 3.0:
        return {
            "vol_score": -1,
            "atr_pct":   round(atr_pct, 2),
            "reason":    f"Volatility: ATR is {atr_pct:.1f}% of price — Elevated volatility (−1).",
        }
    elif atr_pct < 1.0:
        return {
            "vol_score": 1,
            "atr_pct":   round(atr_pct, 2),
            "reason":    f"Volatility: ATR is {atr_pct:.1f}% of price — Low volatility, stable conditions (+1).",
        }
    else:
        return {
            "vol_score": 0,
            "atr_pct":   round(atr_pct, 2),
            "reason":    f"Volatility: ATR is {atr_pct:.1f}% of price — Normal range.",
        }


# ─── EXIT STRATEGY (ATR-Based SL + Dynamic RR) ───────────────────────────────

def calculate_exit_levels(
    df: pd.DataFrame,
    signal: str,
    atr: float,
    score: int,
    is_bullish_trend: bool,
    risk_reward: float = 2.3,
) -> dict:

    if signal not in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
        return {}

    price = df["Close"].iloc[-1]

    # ── Dynamic Risk/Reward ───────────────────────────────────────────────────
    if is_bullish_trend and score >= 8:
        rr = 3.0
    elif is_bullish_trend and score >= 6:
        rr = 2.5
    elif score <= -8:
        rr = 3.0
    elif score <= -6:
        rr = 2.5
    else:
        rr = 2.3

    atr_multiplier = 2.5  

    if "BUY" in signal:
        stop_loss   = round(price - atr_multiplier * atr, 2)
        risk        = price - stop_loss
        take_profit = round(price + risk * rr, 2)
    else:
        stop_loss   = round(price + atr_multiplier * atr, 2)
        risk        = stop_loss - price
        take_profit = round(price - risk * rr, 2)

    return {
        "stop_loss":        stop_loss,
        "take_profit":      take_profit,
        "risk_reward":      rr,
        "atr_used":         round(atr, 4),
        "atr_multiplier":   atr_multiplier,
    }


# ─── MARKET REGIME FILTER ─────────────────────────────────────────────────────

def assess_market_regime(market_returns: pd.Series | None) -> dict:
    if market_returns is None or len(market_returns) < 200:
        return {"regime": "Unknown", "score_penalty": 0, "reason": "Insufficient market data for regime filter."}

    market_index = (1 + market_returns.dropna()).cumprod()
    ma200        = market_index.rolling(200).mean().iloc[-1]
    current      = market_index.iloc[-1]
    pct_vs_ma200 = (current - ma200) / ma200 * 100

    if current < ma200:
        return {
            "regime":        "Risk-Off",
            "score_penalty": -3,
            "reason":        f"Market Regime: S&P 500 is {abs(pct_vs_ma200):.1f}% below its MA200 — Risk-Off mode (score −3).",
        }
    return {
        "regime":        "Risk-On",
        "score_penalty": 0,
        "reason":        f"Market Regime: S&P 500 is {pct_vs_ma200:.1f}% above its MA200 — Risk-On mode.",
    }


# ─── RELATIVE STRENGTH FILTER ────────────────────────────────────────────────

def assess_relative_strength(df: pd.DataFrame, market_returns: pd.Series | None, lookback: int = 60) -> dict:
    if market_returns is None or len(df) < lookback or len(market_returns) < lookback:
        return {"rs_score": 0, "rs_pct": 0.0, "reason": "Relative Strength: insufficient data for comparison."}

    stock_return = (df["Close"].iloc[-1] / df["Close"].iloc[-lookback] - 1) * 100
    mkt_aligned  = market_returns.reindex(df.index).dropna()

    if len(mkt_aligned) < lookback:
        return {"rs_score": 0, "rs_pct": 0.0, "reason": "Relative Strength: market data alignment failed."}

    mkt_return = ((1 + mkt_aligned.iloc[-lookback:]).cumprod().iloc[-1] - 1) * 100
    rs_diff    = round(stock_return - mkt_return, 2)

    if rs_diff >= 5:
        return {
            "rs_score": 1,
            "rs_pct":   rs_diff,
            "reason":   f"Relative Strength: Stock outperforms S&P 500 by {rs_diff:+.1f}% (last {lookback}d) — RS Bonus (+1).",
        }
    elif rs_diff <= -10:
        return {
            "rs_score": -2,
            "rs_pct":   rs_diff,
            "reason":   f"Relative Strength: Stock underperforms S&P 500 by {abs(rs_diff):.1f}% (last {lookback}d) — RS Penalty (−2).",
        }
    else:
        return {
            "rs_score": 0,
            "rs_pct":   rs_diff,
            "reason":   f"Relative Strength: Stock vs S&P 500 = {rs_diff:+.1f}% (last {lookback}d) — Neutral.",
        }


# ─── MAIN SIGNAL FUNCTION ─────────────────────────────────────────────────────

def generate_signal(
    df: pd.DataFrame,
    info: dict,
    metrics: dict,
    market_returns: pd.Series | None = None,
) -> dict:
    reasons = []

    try:
        # ── Data validation ───────────────────────────────────────────────────
        if len(df) < 200:
            return {"signal": "WAIT", "score": 0, "reasons": [f"Insufficient data: need 200 rows, have {len(df)}."]}

        required_cols = ["Close", "RSI", "Histogram", "MA50", "MA200", "Volume", "BB_upper", "BB_lower", "%K", "%D"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return {"signal": "ERROR", "score": 0, "reasons": [f"Missing columns: {', '.join(missing)}"]}

        if df[required_cols].tail(2).isnull().any().any():
            return {"signal": "ERROR", "score": 0, "reasons": ["Latest rows contain NaN values."]}

        # ── Raw values ────────────────────────────────────────────────────────
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
        sharpe     = metrics.get("Sharpe Annualized", 0)

        # ── ATR + ADX ─────────────────────────────────────────────────────────
        atr = calculate_atr(df)
        adx = calculate_adx(df)

        score          = 0
        volume_high    = volume > avg_volume * 1.2
        confidence     = "Normal"
        trigger_active = False
        trigger_bias   = 0

        # ── 1. Trend ──────────────────────────────────────────────────────────
        is_bullish_trend = price > ma200
        is_golden_cross  = ma50  > ma200

        score += 2 if is_bullish_trend else -2
        reasons.append(f"Trend: {'Bullish' if is_bullish_trend else 'Bearish'} (Price vs MA200).")

        score += 1 if is_golden_cross else -1
        reasons.append(f"Structure: {'Golden' if is_golden_cross else 'Death'} Cross active.")

        if adx < MIN_ADX_ENTRY:
            score -= 3
            reasons.append(f"Trend Strength: ADX = {adx:.1f} — No clear trend, choppy market (−3).")
        elif adx >= 25:
            score += 1
            reasons.append(f"Trend Strength: ADX = {adx:.1f} — Strong trend confirmed (+1).")
        else:
            reasons.append(f"Trend Strength: ADX = {adx:.1f} — Moderate trend.")

        # ── 3. Divergence ─────────────────────────────────────────────────────
        divergence = detect_divergence(df)
        if divergence == "bullish":
            score += 3
            trigger_active = True
            trigger_bias  += 1
            reasons.append("Divergence: Bullish RSI divergence — price lower low, RSI higher low.")
        elif divergence == "bearish":
            score -= 3
            trigger_active = True
            trigger_bias  -= 1
            reasons.append("Divergence: Bearish RSI divergence — price higher high, RSI lower high.")

        # ── 4. Oversold / Overbought ──────────────────────────────────────────
        if rsi < 25 and stoch_k < 20:
            if is_bullish_trend:
                score += 4
                trigger_active = True
                trigger_bias  += 1
                reasons.append(f"Trigger: Double Oversold (RSI {rsi:.1f} & Stoch {stoch_k:.1f}) in Bullish trend.")
            else:
                reasons.append(f"Trigger: Oversold detected ({rsi:.1f}) — ignored in Bearish trend.")
        elif rsi > 70 and stoch_k > 80:
            if not is_bullish_trend:
                score -= 4
                trigger_active = True
                trigger_bias  -= 1
                reasons.append(f"Trigger: Double Overbought (RSI {rsi:.1f} & Stoch {stoch_k:.1f}) in Bearish trend.")
            else:
                score -= 1
                reasons.append("Trigger: Overbought in Bullish trend (potential minor pullback).")

        # ── 5. MA200 Support Test + Bear Penalty ────────────────────────────
        margin = (price - ma200) / ma200
        if 0 < margin < 0.02 and 30 < rsi < 40:
            score += 1
            reasons.append(
                f"Structure: Price testing MA200 support ({margin*100:.1f}% above) with RSI near oversold ({rsi:.1f})."
            )

        # Bear market deep penalty:
        if margin < -0.15:
            score -= 2
            reasons.append(
                f"Structure: Price is {abs(margin)*100:.1f}% below MA200 — Deep Bear trend (−2)."
            )

                # ── 6. Stochastic Crossover ───────────────────────────────────────────
        if stoch_k > stoch_d and prev_k <= prev_d and stoch_k < 30:
            score += 1
            trigger_active = True
            trigger_bias  += 1
            reasons.append("Momentum: Stochastic Bullish Crossover in Oversold zone.")
        elif stoch_k < stoch_d and prev_k >= prev_d and stoch_k > 70:
            score -= 1
            trigger_active = True
            trigger_bias  -= 1
            reasons.append("Momentum: Stochastic Bearish Crossover in Overbought zone.")

        # ── 7. MACD Crossover ─────────────────────────────────────────────────
        if hist > 0 and prev_hist <= 0:
            score += 2
            trigger_active = True
            trigger_bias  += 1
            reasons.append("Momentum: MACD Bullish Crossover.")
        elif hist < 0 and prev_hist >= 0:
            score -= 2
            trigger_active = True
            trigger_bias  -= 1
            reasons.append("Momentum: MACD Bearish Crossover.")

        # ── 8. Bollinger Band Touch ───────────────────────────────────────────
        if price <= lower_bb and is_bullish_trend:
            score += 2
            trigger_active = True
            trigger_bias  += 1
            reasons.append("Volatility: Price touched Lower BB (ideal pullback entry).")
        elif price >= upper_bb and not is_bullish_trend:
            score -= 2
            trigger_active = True
            trigger_bias  -= 1
            reasons.append("Volatility: Price touched Upper BB (overextended bearish).")

        # ── 9. Volume Confirmation ────────────────────────────────────────────
        if volume_high and trigger_active and trigger_bias != 0:
            vol_bonus  = 2 if trigger_bias > 0 else -2
            score     += vol_bonus
            direction  = "Bullish" if trigger_bias > 0 else "Bearish"
            reasons.append(f"Volume: High activity ({volume/avg_volume:.1f}x) confirms {direction} triggers.")

        # ── 10. Sharpe Quality ────────────────────────────────────────────────
        if sharpe > 1.5:
            confidence = "High"
            reasons.append(f"Quality: Solid Sharpe Ratio ({sharpe:.2f}).")
        elif sharpe < 0:
            confidence = "Low"
            score     -= 2
            reasons.append(f"Quality: Negative Sharpe ({sharpe:.2f}) — elevated risk.")

        # ── 11. Volatility Filter (NEW) ───────────────────────────────────────
        vol_info = assess_volatility(df, atr)
        if vol_info["vol_score"] != 0:
            score += vol_info["vol_score"]
        reasons.append(vol_info["reason"])

        # ── 12. Market Regime Filter ──────────────────────────────────────────
        regime_info = assess_market_regime(market_returns)
        if regime_info["score_penalty"] != 0:
            score += regime_info["score_penalty"]
        reasons.append(regime_info["reason"])

        # ── 13. Relative Strength Filter ─────────────────────────────────────
        rs_info = assess_relative_strength(df, market_returns)
        if rs_info["rs_score"] != 0:
            score += rs_info["rs_score"]
        reasons.append(rs_info["reason"])

        # ── Signal determination ──────────────────────────────────────────────
        if score >= BUY_THRESHOLD:
            signal = "STRONG BUY" if (score >= 8 and is_bullish_trend) else "BUY"
        elif score <= SELL_THRESHOLD:
            signal = "STRONG SELL" if (score <= -10 and not is_bullish_trend) else "SELL"
        else:
            signal = "HOLD"

        # ── 14. Exit Strategy (ATR-Based) ─────────────────────────────────────
        exit_levels = calculate_exit_levels(df, signal, atr, score, is_bullish_trend)
        if exit_levels:
            reasons.append(
                f"Exit Strategy: Stop Loss ${exit_levels['stop_loss']}  |  "
                f"Take Profit ${exit_levels['take_profit']}  "
                f"(Risk/Reward 1:{exit_levels['risk_reward']} | ATR={exit_levels['atr_used']})."
            )

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
            "atr_pct":          vol_info["atr_pct"],
            # ── Time Exit  ─────────────────────────────────────────────
            "time_exit": {
                "enabled":        True,
                "days":           5,     
                "min_profit_pct": 1.5,    
            },
        }

    except Exception as e:
        return {"signal": "ERROR", "score": 0, "reasons": [f"Logic Failure: {e}"]}
