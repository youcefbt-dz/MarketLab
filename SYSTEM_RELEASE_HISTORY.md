# System Release History

All notable changes to the **MarketLab** quantitative finance framework are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/), and the project adheres to Semantic Versioning.

## [Unreleased] / Next
- Trailing stop loss and partial exits
- Swing trading module (shorter timeframes)
- Crypto support via CCXT (planned)
- React dashboard for live monitoring

## v3.0.0 - 2026-04-04
### Major Features & Improvements
- **Bayesian Strategy Optimizer**: Added `strategy_optimizer.py` using **Optuna** (26 parameters, 100 trials with TPE sampler).
- **Enhanced Signal Engine**: Optimized parameters, added `MIN_ADX_ENTRY=18` filter, bear market deep penalty (−2 when price >15% below MA200), and improved volatility handling.
- **CLI Overhaul**: Restructured `main.py` into a clean **5-mode interactive loop** with Warehouse Manager.
- **Hybrid Data Loading**: Improved data handling in backtesting with automated plans.
- **Screenshots & Documentation**: Major README updates with better table layout for screenshots.

### Bug Fixes & Polish
- Fixed multiple issues in sentiment analysis (7 bugs improved scoring accuracy).
- Fixed data leakage, caching, and zero-division errors in `ml_predictor.py`.

## v2.5.0 - 2026-04-02
### Machine Learning & Risk Management
- Implemented `ml_predictor.py` with SMOTE for class balancing and Data Purge to prevent leakage.
- Achieved **AUC = 0.996** and **100% Quality Recall** on real test records.
- Replaced fixed exits with **ATR-based Dynamic Stop Loss** (now 2.5 × ATR14).
- Added **ADX Trend Strength Filter**, **Volatility Filter (ATR%)**, and raised signal thresholds to ±6.

## v2.4.0
### Black Box & Market Filters
- Launched `backtest_logger.py` with persistent JSON history and **Reliability Score (0–100)**.
- Added **S&P 500 MA200 Regime Filter** and **Relative Strength vs Market Filter**.

## v2.3.0
### Signal & Sentiment Improvements
- Added RSI/Price **divergence detection**.
- Introduced dynamic SL/TP strategy.
- Enhanced news sentiment with **tail risk detection** and time-weighted scoring.

## v2.2.0
### Architecture & Branding
- Official rebranding to **MarketLab**.
- Modular refactoring: separated `signals.py`, `sentiment.py`, `report_generator.py`.
- Initial implementation of professional PDF report generator (ReportLab).

## v2.1.0
### Backtesting Engine
- Full walk-forward backtesting simulator with gap-aware exits.
- Dynamic position sizing (35% for STRONG BUY, etc.).
- Added core metrics: Sharpe Ratio, Max Drawdown, Profit Factor, R-Multiple.

## v2.0.0
### Data Foundation
- Local **Data Warehouse** (`stock_warehouse.py`) — Offline-First with 250+ symbols.
- Added `companies.json` with fuzzy matching for 240+ companies.

## v1.x.x (March 2026)
### Initial Foundation
- Core technical indicators: MACD, Stochastic, Bollinger Bands, RSI, MA crossovers.
- Basic backtesting and signal logic.
- Project initialization on March 7, 2026.

---

<div align="center">
  <sub>For full commit history, see the <a href="https://github.com/youcefbt-dz/MarketLab/commits/main">GitHub commit log</a>.</sub>
</div>
