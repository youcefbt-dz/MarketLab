<div align="center">
  <img src="logo.svg" alt="MarketLab" width="440"/>
  <br/><br/>

[![Version](https://img.shields.io/badge/version-2.5.0-378ADD?style=for-the-badge)](https://github.com/youcefbt-dz/MarketLab)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ML Accuracy](https://img.shields.io/badge/ML%20Accuracy-99%25-blueviolet?style=for-the-badge)](https://github.com/youcefbt-dz/MarketLab)
[![Stars](https://img.shields.io/github/stars/youcefbt-dz/MarketLab?style=for-the-badge&color=yellow)](https://github.com/youcefbt-dz/MarketLab/stargazers)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)

**An open-source quantitative research and trading framework for stock analysis, strategy validation, and financial decision-making.**

[Quick Start](#quick-start) · [Features](#features) · [Backtesting](#backtesting-engine) · [Black Box Logger](#black-box-logger) · [Screenshots](#screenshots) · [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

**MarketLab** is a quantitative research and trading framework built for finance students, researchers, and aspiring quants.

It integrates technical analysis, risk modeling, NLP-driven sentiment analysis, a rule-based signals engine, a full backtesting system, and a **self-improving reliability tracker** into a single modular pipeline — transforming raw market data into actionable intelligence.

```
Raw Market Data  →  Technical Indicators  →  Trading Signals  →  Backtesting  →  Black Box Logger  →  ML Predictor
```

>  **Disclaimer:** MarketLab is for educational and research purposes only. It does not constitute financial advice.

---

## Screenshots

<div align="center">

### Executive Summary Report
<img src="docs/screenshots/screenshot_report.png" width="650"/>

<br>

### News Sentiment Analysis
<img src="docs/screenshots/screenshot_sentiment.png" width="650"/>

<br>

### Technical Charts
<img src="docs/screenshots/screenshot_charts.png" width="650"/>

</div>

---

## Features

###  Technical Analysis
| Indicator | Description |
|-----------|-------------|
| Moving Averages | MA50, MA200, EMA20, EMA50 |
| Momentum | RSI (14), MACD, Stochastic %K/%D |
| Volatility | Bollinger Bands (20, ±2σ), ATR (14) |
| Trend Strength | ADX (14) |
| Divergence | RSI/Price Bullish & Bearish Divergence |

###  Signals Engine
- **13-indicator scoring system** producing BUY / HOLD / SELL signals
- **ATR-based Stop Loss** — dynamic SL at 1.5× ATR instead of fixed BB percentage
- **Dynamic Risk/Reward** — 2.3 / 2.5 / 3.0 based on signal strength and trend
- **ADX Trend Strength Filter** — only enters when trend is confirmed (ADX ≥ 25)
- **Volatility Filter** — penalizes or blocks entry during extreme ATR conditions
- **Market Regime Filter** — S&P 500 MA200 used to switch between Risk-On / Risk-Off
- **Relative Strength Filter** — only enter stocks outperforming the S&P 500
- **Sentiment Integration** — news score adjusts the final signal
- **Divergence Detection** — RSI/Price swing high-low comparison over configurable lookback
- **Time Exit** — exit if < 1.5% gain in 5 days

###  Sentiment Analysis
- **VADER NLP** with custom financial keyword boosting (`beats`, `misses`, `downgrade`, etc.)
- **Time-weighted scoring** — recent news carries more weight (decay over 72h)
- **Tail Risk detection** — single strong negative news triggers a score adjustment
- **Confidence scoring** with positive/negative ratio breakdown

###  Risk & Performance Metrics
- Beta, R², Sharpe Ratio (annualized), Annualized Return
- Configurable risk-free rate (default: 4%)
- Cross-asset correlation matrix

###  Seasonality Analysis
- Best and worst month detection per ticker
- Monthly average return bar chart exported as PNG

###  Local Data Warehouse
- **250+ symbols** stored as local CSV files — no repeated API calls
- Smart incremental updates — only fetches new rows since last download
- 7-day update interval with automatic staleness detection
- ~1.8 million rows total across all symbols
- `load_local(ticker, start, end)` — instant data access for analysis and backtesting

###  Backtesting Engine
- Walk-forward simulation with zero look-ahead bias
- Gap-down / gap-up realistic exit pricing
- Dynamic position sizing (35% for STRONG BUY, 22% for BUY)
- ATR-based Stop Loss, Dynamic Take Profit, Trailing Stop, Partial Exit, and Time Exit
- Full metrics: Win Rate, Profit Factor, Sharpe, Max Drawdown, R-Multiple

###  Black Box Logger *(v2.4.0)*
- Persistent JSON history of every backtest run
- **Reliability Score (0–100)** computed from accumulated results
- Per-ticker and per-market-regime breakdown
- Trend detection: Improving / Stable / Declining
- ML-ready dataset export for future model training

###  ML Predictor *(v2.5.0)*
- **RandomForest / GradientBoosting / LogisticRegression** — auto-selects best model via CV AUC
- **SMOTE** — handles class imbalance by generating synthetic minority samples
- **Data Purge** — automatically removes demo runs and duplicates before training
- **quality_trade** target: Sharpe > 0.5 AND Max Drawdown > −12% AND Win Rate > 50%
- Achieved **AUC = 0.996**, **Quality Recall = 100%** on 72 real backtest records

###  PDF Report Generation
- Professional executive summary with all indicators, signals, sentiment, and charts
- Goldman Sachs-inspired color palette with embedded sparklines

---

## Real-World Results

After **72 backtests** across 59 tickers (2017–2024):

| Metric | Value |
|--------|-------|
| ML Model AUC | **0.996** |
| Quality Recall | **100%** |
| Overall Accuracy | **99%** |
| Best Model | RandomForest |

**Per-ticker reliability (top performers):**

| Ticker | Score | Runs |
|--------|-------|------|
| AAPL | 🟢 92.8 / 100 | 12 |
| ADBE | 🟢 85.0 / 100 | 1 |
| GOOGL | 🟢 85.0 / 100 | 2 |
| NVDA | 🟢 82.1 / 100 | 2 |
| JPM | 🟢 84.7 / 100 | 1 |

> Results accumulate automatically — each new backtest run refines the reliability model.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/youcefbt-dz/python-finance-analyst.git
cd python-finance-analyst
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analyzer

```bash
python main.py
```

You will be prompted to:
- Choose a mode: **Live Analysis** or **Backtesting**
- Enter the number of stocks and years of historical data
- Input company names (e.g. `Apple`, `TSLA`, `NVIDIA`) or ticker symbols

> The fuzzy search engine will resolve names automatically using `companies.json` (240+ companies supported).

### 4. (Recommended) Build the local data warehouse

```bash
python stock_warehouse.py
```

This downloads **250+ symbols** from Yahoo Finance and stores them as local CSV files. All subsequent analysis and backtesting runs read from disk — no internet required per run, no API rate limits, and dramatically faster execution.

```
data/
├── AAPL.csv      # ~7,000 rows (full history)
├── NVDA.csv
├── MSFT.csv
├── ...
└── _metadata.json   # tracks last update date per symbol
```

> The warehouse auto-updates only symbols older than 7 days — subsequent runs are near-instant.

### 5. Run the backtesting engine

```bash
python backtest.py
```

Results are saved to `backtest_results/` and logged automatically to `backtest_history.json`.

### 6. View the reliability report

```bash
python backtest_logger.py
```

### 7. Run the ML predictor

```bash
python ml_predictor.py
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                             │
│                                                             │
│  1. Fetch data           local warehouse→yfinance fallback  │
│  2. Compute indicators   pandas-ta, scipy, numpy            │
│  3. Analyze sentiment    VADER + financial booster          │
│  4. Generate signal      signals.py (13 rules)              │
│  5. Export charts        matplotlib                         │
│  6. Generate PDF         reportlab                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       backtest.py                           │
│                                                             │
│  1. Walk-forward simulation    no look-ahead bias           │
│  2. Dynamic position sizing    35% / 22% per signal         │
│  3. Exit management            ATR-SL / TP / Time Exit      │
│  4. Auto-log results     →     backtest_logger.py           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   backtest_logger.py                        │
│                                                             │
│  1. Persist run to JSON        backtest_history.json        │
│  2. Compute Reliability Score  weighted 4-factor model      │
│  3. Per-ticker breakdown       score + pass rate + regime   │
│  4. Export ML dataset          ready for training           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ml_predictor.py                          │
│                                                             │
│  1. Data Purge         remove demos + duplicates            │
│  2. Feature Engineering  16 features from backtest history  │
│  3. SMOTE              balance quality_trade classes        │
│  4. Train & Select     3 models → best CV AUC wins          │
│  5. Predict            probability + confidence + checks    │
└─────────────────────────────────────────────────────────────┘
```

### Signal Scoring Logic

```
Score  ≥  6  →  BUY        (≥ 8 + bullish trend → STRONG BUY)
Score  ≤ -6  →  SELL       (≤ -10 + bearish trend → STRONG SELL)
Otherwise    →  HOLD
```

| Rule | Weight |
|------|--------|
| Price vs MA200 (trend) | ±2 |
| Golden / Death Cross | ±1 |
| ADX Trend Strength | ±1 |
| RSI/Price Divergence | ±3 |
| Double Oversold / Overbought | ±4 |
| MA200 Support Test | ±1 |
| Stochastic Crossover | ±1 |
| MACD Crossover | ±2 |
| Bollinger Band Touch | ±2 |
| Volume Confirmation | ±2 |
| Sharpe Quality Filter | ±2 |
| Volatility Filter (ATR) | ±1 to ±3 |
| Market Regime (S&P 500 MA200) | ±3 |
| Relative Strength vs S&P 500 | ±2 |
| News Sentiment | ±1 to ±3 |

### Exit Strategy Logic

| Condition | Risk/Reward |
|-----------|-------------|
| Bullish trend + score ≥ 8 | 1 : 3.0 |
| Bullish trend + score ≥ 6 | 1 : 2.5 |
| Default | 1 : 2.3 |
| Bearish score ≤ −8 | 1 : 3.0 |
| Bearish score ≤ −6 | 1 : 2.5 |

Stop Loss = `price ± (1.5 × ATR14)` — adapts to each stock's actual volatility.

### Reliability Score Formula

```
Reliability Score = Σ (component × weight) × 100

  Pass Rate           × 40%
  Avg Win Rate        × 25%   (normalized to 65% target)
  Avg Profit Factor   × 20%   (normalized to 2.5 target)
  Beat Benchmark Rate × 15%
```

### ML quality_trade Formula

```
quality_trade = 1  if:
    Sharpe Ratio  > 0.5   AND
    Max Drawdown  > −12%  AND
    Win Rate      > 50%

quality_trade = 0  otherwise
```

---

## Project Structure

```
python-finance-analyst/
│
├── main.py               # Entry point — Live Analysis mode
├── backtest.py           # Backtesting engine (walk-forward)
├── backtest_logger.py    # Black Box Logger + Reliability Engine
├── signals.py            # Signal generation (13-rule scoring)
├── ml_predictor.py       # ML Pipeline (SMOTE + 3 models)
├── sentiment.py          # NLP sentiment (VADER + boosters)
├── report_generator.py   # PDF report builder (ReportLab)
├── stock_warehouse.py    # Local data warehouse (250+ symbols)
├── companies.json        # 240+ company name → ticker mappings
├── requirements.txt      # Python dependencies
├── logo.svg              # Project logo
│
└── docs/
    └── screenshots/      # Screenshots for README
```

---

## Dependencies

```
yfinance>=0.2.40        # Market data
pandas>=2.0.0           # Data manipulation
pandas-ta>=0.3.14b      # Technical indicators
scipy>=1.11.0           # Linear regression (Beta, R²)
numpy>=1.26.0           # Numerical operations
matplotlib>=3.8.0       # Chart generation
reportlab>=4.0.0        # PDF report generation
vaderSentiment>=3.3.2   # NLP sentiment analysis
thefuzz>=0.22.0         # Fuzzy company name matching
scikit-learn>=1.3.0     # ML models (RandomForest, etc.)
imbalanced-learn>=0.11  # SMOTE for class balancing
flask>=3.0.0            # Optional web interface
flask-cors>=4.0.0       # CORS for Flask API
```

---

## Supported Companies

MarketLab ships with `companies.json` containing **240+ pre-mapped companies** across sectors:

| Sector | Examples |
|--------|----------|
| Tech | Apple, Microsoft, NVIDIA, AMD, Google |
| Finance | JPMorgan, Goldman Sachs, Visa, Mastercard |
| Healthcare | Pfizer, Moderna, Eli Lilly, J&J |
| Consumer | Tesla, Amazon, Nike, Disney, McDonald's |
| Energy | ExxonMobil, Chevron, Shell, BP |
| ETFs | SPY, QQQ, GLD, IBIT (Bitcoin ETF) |

---

## Roadmap

- [x] ATR-based dynamic Stop Loss
- [x] ML model trained on accumulated backtest history
- [ ] Trailing Stop + Partial Exit implementation
- [ ] Streamlit dashboard for reliability visualization
- [ ] Parameter auto-adjustment via feedback loop

---

## Changelog

[![History](https://img.shields.io/badge/Evolution-History-FF5733?style=for-the-badge&logo=gitbook&logoColor=white)](./SYSTEM_RELEASE_HISTORY.md)

---
## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

```bash
git checkout -b feature/your-feature-name
```

---

## License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ for the quant community
</div>
