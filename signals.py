import pandas as pd
import numpy as np

BUY_THRESHOLD  =  7
SELL_THRESHOLD = -7


# ─── DIVERGENCE DETECTION ─────────────────────────────────────────────────────

def detect_divergence(df: pd.DataFrame, lookback: int = 30) -> str | None:
    """
    Compare the last two swing highs/lows of price vs RSI.

    Returns:
        'bullish'  — price makes lower low, RSI makes higher low  (buy signal)
        'bearish'  — price makes higher high, RSI makes lower high (sell signal)
        None       — no divergence detected
    """
    if len(df) < lookback:
        return None

    window  = df.tail(lookback)
    prices  = window["Close"].values
    rsi     = window["RSI"].values

    # ── Find last two local lows (bullish divergence) ─────────────────────────
    low_idx = [
        i for i in range(1, len(prices) - 1)
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]
    ]
    if len(low_idx) >= 2:
        i1, i2 = low_idx[-2], low_idx[-1]          # older → newer
        price_lower_low = prices[i2] < prices[i1]
        rsi_higher_low  = rsi[i2]   > rsi[i1]
        if price_lower_low and rsi_higher_low:
            return "bullish"

    # ── Find last two local highs (bearish divergence) ────────────────────────
    high_idx = [
        i for i in range(1, len(prices) - 1)
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]
    ]
    if len(high_idx) >= 2:
        i1, i2 = high_idx[-2], high_idx[-1]
        price_higher_high = prices[i2] > prices[i1]
        rsi_lower_high    = rsi[i2]    < rsi[i1]
        if price_higher_high and rsi_lower_high:
            return "bearish"

    return None


# ─── EXIT STRATEGY ────────────────────────────────────────────────────────────

def calculate_exit_levels(df: pd.DataFrame, signal: str, risk_reward: float = 2.3) -> dict:
    """
    Compute Stop Loss and Take Profit levels.

    Stop Loss  : lower Bollinger Band  (support-based, dynamic)
    Take Profit: SL distance × risk_reward ratio (default 1:2.3)

    Returns a dict with 'stop_loss', 'take_profit', and 'risk_reward'.
    Returns empty dict if signal is HOLD / ERROR / WAIT.
    """
    if signal not in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
        return {}

    price     = df["Close"].iloc[-1]
    lower_bb  = df["BB_lower"].iloc[-1]
    upper_bb  = df["BB_upper"].iloc[-1]

    if "BUY" in signal:
        stop_loss   = round(lower_bb, 2)
        risk        = price - stop_loss
        take_profit = round(price + risk * risk_reward, 2)
    else:  # SELL
        stop_loss   = round(upper_bb, 2)
        risk        = stop_loss - price
        take_profit = round(price - risk * risk_reward, 2)

    return {
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
    }


# ─── MARKET REGIME FILTER ─────────────────────────────────────────────────────

def assess_market_regime(market_returns: pd.Series | None) -> dict:
    """
    Evaluate whether the broader market (S&P 500) is in Risk-On or Risk-Off mode.

    Logic:
        - Reconstruct a cumulative price index from daily returns.
        - Compute its 200-day moving average.
        - If price < MA200 → Risk-Off  → penalty applied to all scores.
        - If price > MA200 → Risk-On   → no adjustment.

    Returns:
        {
            'regime'       : 'Risk-On' | 'Risk-Off',
            'score_penalty': int   (0 or negative),
            'reason'       : str,
        }
    """
    if market_returns is None or len(market_returns) < 200:
        return {"regime": "Unknown", "score_penalty": 0, "reason": "Insufficient market data for regime filter."}

    # Reconstruct index level from returns
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
    else:
        return {
            "regime":        "Risk-On",
            "score_penalty": 0,
            "reason":        f"Market Regime: S&P 500 is {pct_vs_ma200:.1f}% above its MA200 — Risk-On mode.",
        }


# ─── MAIN SIGNAL FUNCTION ─────────────────────────────────────────────────────

def generate_signal(df: pd.DataFrame, info: dict, metrics: dict, market_returns: pd.Series | None = None) -> dict:
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

        score        = 0
        volume_high  = volume > avg_volume * 1.5
        confidence   = "Normal"
        trigger_active = False
        trigger_bias   = 0

        # ── 1. Trend ──────────────────────────────────────────────────────────
        is_bullish_trend = price > ma200
        is_golden_cross  = ma50  > ma200

        score += 2 if is_bullish_trend else -2
        reasons.append(f"Trend: {'Bullish' if is_bullish_trend else 'Bearish'} (Price vs MA200).")

        score += 1 if is_golden_cross else -1
        reasons.append(f"Structure: {'Golden' if is_golden_cross else 'Death'} Cross active.")

        # ── 2. Divergence (NEW) ───────────────────────────────────────────────
        divergence = detect_divergence(df)
        if divergence == "bullish":
            score += 3
            trigger_active = True
            trigger_bias  += 1
            reasons.append("Divergence: Bullish RSI divergence — price lower low, RSI higher low (strong reversal signal).")
        elif divergence == "bearish":
            score -= 3
            trigger_active = True
            trigger_bias  -= 1
            reasons.append("Divergence: Bearish RSI divergence — price higher high, RSI lower high (strong reversal signal).")

        # ── 3. Oversold / Overbought ──────────────────────────────────────────
        if rsi < 30 and stoch_k < 20:
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

        # ── 4. MA200 Support Test ─────────────────────────────────────────────
        margin = (price - ma200) / ma200
        if 0 < margin < 0.02 and 30 < rsi < 40:
            score += 1
            reasons.append(
                f"Structure: Price testing MA200 support "
                f"({margin * 100:.1f}% above) with RSI near oversold ({rsi:.1f})."
            )

        # ── 5. Stochastic Crossover ───────────────────────────────────────────
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

        # ── 6. MACD Crossover ─────────────────────────────────────────────────
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

        # ── 7. Bollinger Band Touch ───────────────────────────────────────────
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

        # ── 8. Volume Confirmation ────────────────────────────────────────────
        if volume_high and trigger_active and trigger_bias != 0:
            vol_bonus  = 2 if trigger_bias > 0 else -2
            score     += vol_bonus
            direction  = "Bullish" if trigger_bias > 0 else "Bearish"
            reasons.append(f"Volume: High activity ({volume / avg_volume:.1f}x) confirms {direction} triggers.")

        # ── 9. Sharpe Quality ─────────────────────────────────────────────────
        if sharpe > 1.5:
            confidence = "High"
            reasons.append(f"Quality: Solid Sharpe Ratio ({sharpe:.2f}).")
        elif sharpe < 0:
            confidence = "Low"
            score     -= 2
            reasons.append(f"Quality: Negative Sharpe ({sharpe:.2f}) — elevated risk.")

        # ── 10. Market Regime Filter (NEW) ────────────────────────────────────
        regime_info = assess_market_regime(market_returns)
        if regime_info["score_penalty"] != 0:
            score += regime_info["score_penalty"]
        reasons.append(regime_info["reason"])

        # ── Signal determination ──────────────────────────────────────────────
        if score >= BUY_THRESHOLD:
            signal = "STRONG BUY" if (score >= 10 and is_bullish_trend) else "BUY"
        elif score <= SELL_THRESHOLD:
            signal = "STRONG SELL" if (score <= -10 and not is_bullish_trend) else "SELL"
        else:
            signal = "HOLD"

        # ── 11. Exit Strategy (NEW) ───────────────────────────────────────────
        exit_levels = calculate_exit_levels(df, signal)
        if exit_levels:
            reasons.append(
                f"Exit Strategy: Stop Loss ${exit_levels['stop_loss']}  |  "
                f"Take Profit ${exit_levels['take_profit']}  "
                f"(Risk/Reward 1:{exit_levels['risk_reward']})."
            )

        return {
            "signal":          signal,
            "score":           score,
            "confidence_level": confidence,
            "reasons":         reasons,
            "price_at_signal": round(price, 4),
            "exit_levels":     exit_levels,
            "market_regime":   regime_info["regime"],
        }

    except Exception as e:
        return {"signal": "ERROR", "score": 0, "reasons": [f"Logic Failure: {e}"]}        
    prev_hist = df['Histogram'].iloc[-2]
    if histogram > 0 and prev_hist < 0: 
        score += 3
        reasons.append("MACD Histogram crossed above zero: Strong Bullish reversal.")
    elif histogram > 0:
        score += 1
        reasons.append("MACD stays positive: Bullish momentum continues.")
    else:
        score -= 2
        reasons.append("MACD is negative: Bearish momentum.")
    if ma50 > ma200:
        score += 2
        reasons.append("Golden Cross (MA50 > MA200) active.")
    else:
        score -= 2
        reasons.append("Death Cross (MA50 < MA200) active.")
    if sharpe > 1.5:
        score += 2
        reasons.append(f"Excellent Sharpe Ratio ({sharpe:.2f}): High quality returns.")
    elif sharpe < 0:
        score -= 2
        reasons.append(f"Negative Sharpe ({sharpe:.2f}): Poor risk-adjusted performance.")
    if score >= BUY_THRESHOLD:
        signal = "STRONG BUY" if score > 8 else "BUY"
    elif score <= SELL_THRESHOLD:
        signal = "STRONG SELL" if score < -6 else "SELL"
    else:
        signal = "HOLD"
    return {
        'signal': signal,
        'score': score,
        'max_score': 15, 
        'reasons': reasons
    }
