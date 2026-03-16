import pandas as pd

# Config - يمكن تعديلها بسهولة
BUY_THRESHOLD  = 5
SELL_THRESHOLD = -2

def generate_signal(df, info, metrics):
    score = 0
    reasons = []
    
    rsi      = df['RSI'].iloc[-1]
    histogram= df['Histogram'].iloc[-1]
    k        = df['%K'].iloc[-1]
    d        = df['%D'].iloc[-1]
    ma50     = df['MA50'].iloc[-1]
    ma200    = df['MA200'].iloc[-1]
    ema20    = df['EMA20'].iloc[-1]
    ema50    = df['EMA50'].iloc[-1]
    bb_upper = df['BB_upper'].iloc[-1]
    bb_lower = df['BB_lower'].iloc[-1]
    price    = info.get('currentPrice')
    beta     = info.get('beta', 1)
    sharpe   = metrics['Sharpe Annualized']

    # فحص القيم المفقودة
    if pd.isna(rsi):
        reasons.append("RSI data is missing, skipping RSI analysis.")
        rsi = 50
    if pd.isna(ma50) or pd.isna(ma200):
        reasons.append("MA data is missing, skipping MA analysis.")
        ma50 = ma200 = 0
    if pd.isna(ema20) or pd.isna(ema50):
        reasons.append("EMA data is missing, skipping EMA analysis.")
        ema20 = ema50 = 0
    if pd.isna(k) or pd.isna(d):
        reasons.append("Stochastic data is missing, skipping Stochastic analysis.")
        k = d = 50

    # RSI
    if rsi < 30:
        score += 2
        reasons.append(f"RSI is at {rsi:.1f}, indicating the stock is Oversold. A price reversal upward is likely.")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI is at {rsi:.1f}, indicating the stock is Overbought. A price correction downward is likely.")
    else:
        reasons.append(f"RSI is at {rsi:.1f}, which is in the Neutral zone. No clear signal.")

    # MACD Histogram
    if pd.isna(histogram):
        reasons.append("MACD data is missing, skipping MACD analysis.")
    elif histogram > 0:
        score += 2
        reasons.append(f"MACD Histogram is positive ({histogram:.4f}), confirming Bullish momentum in the market.")
    else:
        score -= 1
        reasons.append(f"MACD Histogram is negative ({histogram:.4f}), indicating Bearish momentum and selling pressure.")

    # Stochastic %K
    if k < 20:
        score += 2
        reasons.append(f"Stochastic %K is at {k:.1f}, which is in the Oversold zone. This is a potential Buy opportunity.")
    elif k > 80:
        score -= 2
        reasons.append(f"Stochastic %K is at {k:.1f}, which is in the Overbought zone. Consider taking profits.")
    else:
        reasons.append(f"Stochastic %K is at {k:.1f}, which is Neutral. No strong signal at this level.")

    # Stochastic %D
    if k < 20 and d < 20:
        score += 1
        reasons.append(f"Stochastic %D is at {d:.1f}, confirming the Oversold signal from %K. Buy signal is stronger.")
    elif k > 80 and d > 80:
        score -= 1
        reasons.append(f"Stochastic %D is at {d:.1f}, confirming the Overbought signal from %K. Sell signal is stronger.")
    else:
        reasons.append(f"Stochastic %D is at {d:.1f}. No confirmation of %K signal.")

    # MA50 vs MA200 ← وزن أكبر لأنه اتجاه هيكلي طويل الأمد
    if ma50 > ma200:
        score += 3
        reasons.append(f"MA50 ({ma50:.2f}) is above MA200 ({ma200:.2f}), forming a Golden Cross. Long term uptrend is confirmed.")
    else:
        score -= 2
        reasons.append(f"MA50 ({ma50:.2f}) is below MA200 ({ma200:.2f}), forming a Death Cross. Long term downtrend is in place.")

    # EMA20 vs EMA50
    if ema20 > ema50:
        score += 1
        reasons.append(f"EMA20 ({ema20:.2f}) is above EMA50 ({ema50:.2f}), indicating a Short term uptrend.")
    else:
        score -= 1
        reasons.append(f"EMA20 ({ema20:.2f}) is below EMA50 ({ema50:.2f}), indicating a Short term downtrend.")

    # Bollinger Bands
    if pd.isna(bb_lower) or pd.isna(bb_upper):
        reasons.append("Bollinger Bands data is missing, skipping BB analysis.")
    elif price <= bb_lower:
        score += 1
        reasons.append(f"Price (${price}) is at or below the Lower Bollinger Band (${bb_lower:.2f}). A bounce back is expected.")
    elif price >= bb_upper:
        score -= 1
        reasons.append(f"Price (${price}) is at or above the Upper Bollinger Band (${bb_upper:.2f}). A pullback is expected.")
    else:
        reasons.append(f"Price (${price}) is inside the Bollinger Bands. The market is in a normal range.")

    # Beta
    if pd.isna(beta):
        reasons.append("Beta data is missing, skipping Beta analysis.")
    elif beta < 1:
        score += 1
        reasons.append(f"Beta is {beta}, which is below 1. This stock is less volatile than the market, meaning lower risk.")
    elif beta > 1.5:
        score -= 1
        reasons.append(f"Beta is {beta}, which is above 1.5. This stock is highly volatile and carries higher risk.")
    else:
        reasons.append(f"Beta is {beta}, indicating moderate risk compared to the overall market.")

    # Sharpe Ratio
    if pd.isna(sharpe):
        reasons.append("Sharpe Ratio data is missing, skipping Sharpe analysis.")
    elif sharpe > 1:
        score += 1
        reasons.append(f"Sharpe Ratio is {sharpe:.2f}, which is above 1. The stock offers a good risk-adjusted return.")
    elif sharpe < 0:
        score -= 1
        reasons.append(f"Sharpe Ratio is {sharpe:.2f}, which is negative. The stock is underperforming the risk-free rate.")
    else:
        reasons.append(f"Sharpe Ratio is {sharpe:.2f}, which is acceptable but not outstanding.")

    # القرار النهائي باستخدام Config
    max_score = 14
    if score >= BUY_THRESHOLD:
        signal = "BUY"
    elif score <= SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        'signal': signal,
        'score': score,
        'max_score': max_score,
        'reasons': reasons
    }