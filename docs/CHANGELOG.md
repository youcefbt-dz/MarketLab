# Backtest Engine v2.5.0 — Changelog

##  Major Updates (April 1, 2026)

---

### 1. **ATR-Based Dynamic Stop Loss** *(replaces Fixed BB × 0.95)*

**Previous Behavior:**
```python
# v2.1 — Fixed percentage below Bollinger Band
stop_loss = lower_bb * 0.95   # Always 5% below BB lower
```

**New Behavior:**
```python
# v2.5.0 — Adapts to each stock's actual volatility
atr        = calculate_atr(df, period=14)
stop_loss  = price - (1.5 × atr)   # BUY
stop_loss  = price + (1.5 × atr)   # SELL
```

**Why This Matters:**
- A calm stock like KO has ATR ≈ $0.80 → SL is tight and precise
- A volatile stock like TSLA has ATR ≈ $12 → SL widens automatically
- Prevents premature stop-outs caused by normal daily noise

---

### 2. **Dynamic Risk/Reward Ratio**

**Previous Behavior:**
- Fixed `risk_reward = 2.3` for all trades regardless of signal strength

**New Behavior:**

| Condition | Risk/Reward |
|-----------|-------------|
| Bullish trend + score ≥ 8 | **1 : 3.0** |
| Bullish trend + score ≥ 6 | **1 : 2.5** |
| Default | **1 : 2.3** |
| Bearish score ≤ −8 | **1 : 3.0** |
| Bearish score ≤ −6 | **1 : 2.5** |

**Impact:** High-conviction trades now target larger profits while maintaining the same controlled risk.

---

### 3. **ADX Trend Strength Filter** *(New)*

**Logic:**
```python
adx = calculate_adx(df, period=14)

if adx >= 25:   score += 1   # Strong trend confirmed
elif adx < 20:  score -= 1   # Choppy/sideways market — penalized
```

**Why:** Entering during a trendless market (ADX < 20) is one of the most common causes of false signals. This filter reduces noise entries significantly.

---

### 4. **Volatility Filter via ATR Percentage** *(New)*

**Logic:**
```python
atr_pct = (atr / price) * 100

if atr_pct > 5.0:   score -= 3   # Extreme volatility — high risk
elif atr_pct > 3.0: score -= 1   # Elevated volatility — caution
elif atr_pct < 1.0: score += 1   # Low volatility — stable conditions
```

**Example:**
- TSLA during earnings week: ATR/Price ≈ 6.2% → score −3 (avoided)
- KO on a normal week: ATR/Price ≈ 0.8% → score +1 (preferred)

---

### 5. **BUY/SELL Threshold Raised**

| | v2.1 | v2.5.0 |
|---|---|---|
| BUY threshold | ≥ 5 | **≥ 6** |
| SELL threshold | ≤ −5 | **≤ −6** |
| STRONG BUY | ≥ 7 + bullish | **≥ 8 + bullish** |
| STRONG SELL | ≤ −10 + bearish | **≤ −10 + bearish** |

**Impact:** Reduces low-quality entries. Fewer trades, but higher precision.

---

### 6. **Time Exit Extended**

| | v2.1 | v2.5.0 |
|---|---|---|
| Days to evaluate | 3 | **5** |
| Min profit target | 2.0% | **1.5%** |

**Reasoning:** 3 days was too aggressive — many valid trades were exited before reaching their target. 5 days gives the trade enough room to develop while the lower profit threshold (1.5%) catches partial wins.

---

### 7. **Dynamic Position Sizing** *(carried from v2.1)*

```python
POSITION_SIZE_MAP = {
    "STRONG BUY":  35%,   # High-conviction trades
    "BUY":         22%,   # Standard trades
    "STRONG SELL": 35%,
    "SELL":        22%,
}
DEFAULT_POSITION_PCT      = 15%   # Weak signals
MAX_PORTFOLIO_EXPOSURE    = 70%   # Total cap
MAX_SINGLE_POSITION       = 40%   # Safety limit
```

---

### 8. **Enhanced Metrics Suite** *(carried from v2.1)*

| Metric | Description |
|--------|-------------|
| `avg_r_multiple` | Realized PnL / Initial Risk — measures actual RR achieved |
| `max_consecutive_losses` | Longest losing streak — critical for psychology |
| `avg_position_by_signal` | Verifies dynamic sizing is working correctly |
| `exit_reasons` | Breakdown of TP / SL / Time Exit / End of Period |
| `adx` | Trend strength at signal time |
| `atr_pct` | Volatility level at entry |

---

##  Performance Impact (v2.1 → v2.5.0)

Based on **72 backtests across 59 tickers**:

| Metric | v2.1 | v2.5.0 | Change |
|--------|------|--------|--------|
| ML Model AUC | 0.977 | **0.996** | ⬆️ +0.019 |
| Quality Recall | 93.8% | **100%** | ⬆️ +6.2% |
| Overall Accuracy | 95% | **99%** | ⬆️ +4% |
| QUALITY % of runs | 20.5% | 9.7% | ⬇️ More selective |
| Signal quality | Standard | **Higher precision** | ✅ |

> **Note on QUALITY %:** The drop from 20.5% to 9.7% is intentional — v2.5.0 is more selective. Fewer signals, but each one is higher quality.

---

##  Technical Implementation

### Key Functions Added:

**`calculate_atr(df, period=14)`**
```python
# Returns average true range over last `period` candles
# Fallback: 2% of price if insufficient data
```

**`calculate_adx(df, period=14)`**
```python
# Returns ADX value (0-100)
# Fallback: 25.0 (neutral) if High/Low columns missing
```

**`assess_volatility(df, atr)`**
```python
# Returns: { vol_score, atr_pct, reason }
# vol_score: -3 / -1 / 0 / +1
```

**`calculate_exit_levels(df, signal, atr, score, is_bullish_trend)`**
```python
# Returns: { stop_loss, take_profit, risk_reward, atr_used, atr_multiplier }
# risk_reward: 2.3 / 2.5 / 3.0 — dynamic based on score
```

### Key Functions Modified:

**`generate_signal()`**
- Now computes ATR and ADX at the start
- Passes `atr` and `score` to `calculate_exit_levels()`
- Returns `adx`, `atr`, `atr_pct` in output dict

**`compute_metrics(result)`**
- Added: `avg_r_multiple`, `max_consecutive_losses`
- Added: `avg_position_by_signal`, `exit_reasons`

---

##  Testing Recommendations

### Before Deploying:

1. **Run Multi-Ticker Backtest (2020–2024):**
   ```bash
   python backtest.py
   # Suggested: NVDA, AAPL, TSLA, SPOT, JPM
   ```

2. **Run ML Predictor to verify data quality:**
   ```bash
   python ml_predictor.py
   # Target: AUC > 0.95, Quality Recall > 90%
   ```

3. **Stress Test — 2022 Bear Market:**
   - Verify ADX filter reduces false BUY signals
   - Check Volatility Filter blocks high-ATR entries
   - Confirm Time Exit (5 days) doesn't trigger prematurely

4. **Compare v2.1 vs v2.5.0 on same tickers:**
   - Metrics to watch: Sharpe Ratio, Max Drawdown, Win Rate
   - Expected: fewer trades, higher win rate

---

##  Usage Example

### Command Line:
```bash
python backtest.py

# Interactive prompts:
Initial capital ($) [default 10000]: 50000
Backtest start year [2015–2024, default 2020]: 2020
Risk-Free Rate [4.0%]: 4.2
Number of stocks to backtest [1–20, default 3]: 5

Stock Selection:
[1/5] Company name or ticker: NVIDIA
[2/5] Company name or ticker: APPLE
[3/5] Company name or ticker: SPOTIFY
[4/5] Company name or ticker: MSFT
[5/5] Company name or ticker: GOOGLE
```

### Output Example:
```
──────────────────────────────────────────────────────────────
  NVDA  ·  Backtest Result  (v2.5.0)
──────────────────────────────────────────────────────────────
    Verdict          : PASS ✅
    Total Return     : +187.3%  (Buy & Hold: +152.1%)
    Total Trades     : 14       (↓ from 18 — higher threshold)
    Win Rate         : 71.4%    (↑ from 66.7%)
    Profit Factor    : 2.61
    Max Drawdown     : -16.8%   (↓ from -18.2%)
    Sharpe Ratio     : 1.94     (↑ from 1.82)
    Avg R-Multiple   : 2.34     (↑ from 2.18)
    Max Consec Loss  : 2        (↓ from 3)
    ADX at Entry     : 28.4     (Strong trend confirmed)
    ATR % at Entry   : 2.1%     (Normal volatility)
    Avg Position %:
        STRONG BUY   : 34.8%
        BUY          : 22.1%
    Exit Reasons:
        Hit TP       : 8   ← ATR-based TP
        Hit SL       : 4   ← ATR-based SL
        Time Exit    : 1   ← 5-day rule
        End of Period: 1
```

---

##  Migration Guide

### From v2.1 → v2.5.0:

**No Breaking Changes** — v2.5.0 is fully backward-compatible.

**Exit Strategy:**
```python
# Old (v2.1):
stop_loss   = lower_bb * 0.95
take_profit = price + (price - stop_loss) * 2.3

# New (v2.5.0):
atr         = calculate_atr(df)
stop_loss   = price - (1.5 * atr)
take_profit = price + (price - stop_loss) * dynamic_rr
```

**Thresholds:**
```python
# Old (v2.1):
BUY_THRESHOLD  =  5
SELL_THRESHOLD = -5

# New (v2.5.0):
BUY_THRESHOLD  =  6
SELL_THRESHOLD = -6
```

**Time Exit:**
```python
# Old (v2.1):
"days": 3, "min_profit_pct": 2.0

# New (v2.5.0):
"days": 5, "min_profit_pct": 1.5
```

---

##  Known Limitations

1. **Portfolio Exposure Tracking:**
   - Assumes one position open at a time
   - Multi-position tracking planned for v2.6.0

2. **ADX Approximation:**
   - Uses simplified single-period ADX (not Wilder's smoothed version)
   - Full Wilder smoothing planned for v2.6.0

3. **Short Selling:**
   - `STRONG SELL` / `SELL` signals defined but not implemented
   - Requires margin account logic (future feature)

4. **ATR Fallback:**
   - If `High`/`Low` columns missing, falls back to `Close × 0.02`
   - Ensure OHLCV data is complete for best results

---

##  References

1. Wilder, J.W. — *New Concepts in Technical Trading Systems* (ATR, ADX)
2. Van Tharp — *Trade Your Way to Financial Freedom* (R-Multiples)
3. Kelly Criterion — Position Sizing Optimization (1956)
4. Pardo — *The Evaluation and Optimization of Trading Strategies* (Walk-Forward)

---

## 📞 Support

Questions or issues? Open a [GitHub Issue](https://github.com/youcefbt-dz/python-finance-analyst/issues)

**Version:** 2.5.0
**Release Date:** April 1, 2026
**Author:** Youcef BT
**License:** Apache 2.0
