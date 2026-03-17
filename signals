import pandas as pd

BUY_THRESHOLD  = 6  
SELL_THRESHOLD = -3

def generate_signal(df, info, metrics):
    score = 0
    reasons = []
    
    rsi       = df['RSI'].iloc[-1]
    histogram = df['Histogram'].iloc[-1]
    k         = df['%K'].iloc[-1]
    d         = df['%D'].iloc[-1]
    ma50      = df['MA50'].iloc[-1]
    ma200     = df['MA200'].iloc[-1]
    price     = info.get('currentPrice')
    beta      = metrics.get('Beta', 1) 
    sharpe    = metrics.get('Sharpe Annualized', 0)
    is_bullish_trend = price > ma200
    
    if is_bullish_trend:
        score += 1
        reasons.append("Price is above MA200: Long-term trend is Bullish.")
    else:
        score -= 1
        reasons.append("Price is below MA200: Long-term trend is Bearish. Be careful with BUY signals.")
    if rsi < 30:
        weight = 3 if is_bullish_trend else 1 
        score += weight
        reasons.append(f"RSI is Oversold ({rsi:.1f}). Potential reversal.")
    elif rsi > 70:
        weight = 3 if not is_bullish_trend else 1 
        score -= weight
        reasons.append(f"RSI is Overbought ({rsi:.1f}). Potential correction.")
    else:
        reasons.append(f"RSI is Neutral ({rsi:.1f}). No clear signal.")
        
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
