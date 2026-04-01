# System Release History

All notable changes to the **MarketLab** quantitative framework are documented in this technical evolution log.

## v2.5.0 - (Latest)
###  Machine Learning & Advanced Risk
- **ML Pipeline:** Implemented `ml_predictor.py` with SMOTE for class balancing and Data Purge for training hygiene.
- **Performance:** Achieved **AUC = 0.996** and **100% Quality Recall** on 72 real records.
- **ATR-Based Risk:** Replaced fixed exits with **ATR-based Dynamic Stop Loss** (1.5 × ATR14).
- **Filters:** Added **ADX Trend Strength Filter** and **Volatility Filter via ATR%**.
- **Precision:** Raised BUY/SELL thresholds to ±6 for higher signal quality.

## v2.4.0 
###  Black Box & Reliability
- **Data Logger:** Launched `backtest_logger.py` with persistent JSON history.
- **Reliability Engine:** New (0–100) scoring system per ticker and market regime.
- **Market Guard:** Added **S&P 500 MA200 Regime Filter** and **Relative Strength Filter**.

## v2.3.0 
###  Advanced Signal Logic
- **Divergence:** Implemented RSI/Price divergence detection.
- **Dynamic Exit:** First integration of dynamic SL/TP strategy.
- **NLP Sentiment:** Improved news analysis with **Tail Risk detection** and time-weighting.

## v2.2.0 
###  Modular Architecture
- **Rebranding:** Officially rebranded to **MarketLab**.
- **Refactoring:** Modularized the engine into `main.py`, `signals.py`, and `sentiment.py`.
- **Infrastructure:** Initial implementation of the PDF report generator.

## v2.1.0 
###  Backtesting Core
- **Engine:** Added the first full backtesting simulator with walk-forward logic.
- **Position Sizing:** Implemented **Dynamic Position Sizing** (35% for strong signals).
- **Analytics:** Added Sharpe Ratio, Max Drawdown, and R-Multiple metrics.

## v2.0.0 
###  Data & Logistics
- **Warehouse:** Implementation of local CSV data caching (Offline-First).
- **Fuzzy Search:** Added `companies.json` with fuzzy name matching for 240+ companies.

## v1.x.x -     2026-03-07 to 2026-03-15
### 🌱 Foundations
- **Indicators:** Added MACD, Stochastic, Bollinger Bands, and RSI.
- **Fundamentals:** Basic Beta, R², and Seasonality analysis.
- **Initial Commit:** Project birth on March 7, 2026.

---
<div align="center">
  <i>For previous minor updates, check the Git commit history.</i>
</div>
