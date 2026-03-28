<div align="center">
  <img src="logo.svg" alt="MarketLab" width="440"/>
  <br/><br/>

[![Version](https://img.shields.io/badge/version-2.3.0-378ADD?style=flat-square)](https://github.com/youcefbt-dz/python-finance-analyst)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Open Source](https://img.shields.io/badge/open%20source-yes-brightgreen?style=flat-square)](https://github.com/youcefbt-dz)
[![Stars](https://img.shields.io/github/stars/youcefbt-dz/python-finance-analyst?style=flat-square&color=yellow)](https://github.com/youcefbt-dz/python-finance-analyst/stargazers)

**An open-source quantitative research and trading framework for stock analysis, strategy validation, and financial decision-making.**

[Quick Start](#quick-start) · [Features](#features) · [How It Works](#how-it-works) · [Backtesting](#backtesting-engine) · [Screenshots](#screenshots) · [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

**MarketLab** is a quantitative research and trading framework built for finance students, researchers, and aspiring quants.

It integrates technical analysis, risk modeling, NLP-driven sentiment analysis, a rule-based signals engine, and a full backtesting system into a single modular pipeline — transforming raw market data into actionable intelligence.

```
Raw Market Data  →  Technical Indicators  →  Trading Signals  →  Backtesting  →  PDF Report
```

> ⚠️ **Disclaimer:** MarketLab is for educational and research purposes only. It does not constitute financial advice.

---

## Screenshots

<div align="center">

### 📄 Executive Summary Report
![Report](docs/screenshots/screenshot_report.png)

### 🧠 News Sentiment Analysis
![Sentiment](docs/screenshots/screenshot_sentiment.png)

### 📈 Technical Charts
![Charts](docs/screenshots/screenshot_charts.png)

### 📊 Seasonality Analysis
![Seasonality](docs/screenshots/AAPL_seasonality.png)

</div>

---

## Features

### 📐 Technical Analysis
| Indicator | Description |
|-----------|-------------|
| Moving Averages | MA50, MA200, EMA20, EMA50 |
| Momentum | RSI (14), MACD, Stochastic %K/%D |
| Volatility | Bollinger Bands (20, ±2σ) |
| Divergence | RSI/Price Bullish & Bearish Divergence |

### 📡 Signals Engine
- **9-indicator scoring system** producing BUY / HOLD / SELL signals
- **Market Regime Filter** — S&P 500 MA200 used to switch between Risk-On / Risk-Off
- **Sentiment Integration** — news score adjusts the final signal
- **Exit Strategy** — auto-calculated Stop Loss & Take Profit (default Risk/Reward 1:2.3)
- **Divergence Detection** — RSI/Price swing high-low comparison over configurable lookback

### 📰 Sentiment Analysis
- **VADER NLP** with custom financial keyword boosting (`beats`, `misses`, `downgrade`, etc.)
- **Time-weighted scoring** — recent news carries more weight (decay over 72h)
- **Tail Risk detection** — single strong negative news triggers a score adjustment
- **Confidence scoring** with positive/negative ratio breakdown

### 📊 Risk & Performance Metrics
- Beta, R², Sharpe Ratio (annualized), Annualized Return
- Configurable risk-free rate (default: 4%)
- Cross-asset correlation matrix

### 📅 Seasonality Analysis
- Best and worst month detection per ticker
- Monthly average return bar chart exported as PNG

### 🔁 Backtesting Engine
- Historical strategy simulation with rule-based entry/exit
- Dynamic position sizing
- Performance summary with key metrics

### 📄 PDF Report Generation
- Professional executive summary with all indicators, signals, sentiment, and charts
- Auto-generated per analysis run

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

### 4. Output

- Terminal output with indicators, metrics, and signals
- `{TICKER}_seasonality.png` saved locally
- `Financial_Analysis_Report_{DATE}.pdf` generated automatically

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                      main.py                        │
│                                                     │
│  1. Fetch data          yfinance (1 call/ticker)    │
│  2. Compute indicators  pandas-ta, scipy, numpy     │
│  3. Analyze sentiment   VADER + financial booster   │
│  4. Generate signal     signals.py (9 rules)        │
│  5. Export charts       matplotlib                  │
│  6. Generate PDF        reportlab                   │
└─────────────────────────────────────────────────────┘
```

### Signal Scoring Logic

```
Score  ≥  5  →  BUY        (≥ 10 + bullish trend → STRONG BUY)
Score  ≤ -5  →  SELL       (≤ -10 + bearish trend → STRONG SELL)
Otherwise    →  HOLD
```

**Scoring components:**
| Rule | Weight |
|------|--------|
| Price vs MA200 (trend) | ±2 |
| Golden / Death Cross | ±1 |
| RSI/Price Divergence | ±3 |
| Double Oversold / Overbought | ±4 |
| Stochastic Crossover | ±1 |
| MACD Crossover | ±2 |
| Bollinger Band Touch | ±2 |
| Volume Confirmation | ±2 |
| Sharpe Quality Filter | ±2 |
| Market Regime (S&P 500) | ±3 |
| News Sentiment | ±1 to ±3 |

---

## Backtesting Engine

Run with mode `[2]` from the main menu, or directly:

```bash
python backtest.py
```

**Features:**
- Simulates BUY/SELL signals over historical data
- Dynamic capital allocation per trade (position sizing)
- Rule-based exit system (Stop Loss / Take Profit)
- Exports results to `BACKTEST_LOGS.md`

---

## Project Structure

```
python-finance-analyst/
│
├── main.py               # Entry point — Live Analysis mode
├── backtest.py           # Backtesting engine
├── signals.py            # Signal generation (9-rule scoring engine)
├── sentiment.py          # NLP sentiment analysis (VADER + boosters)
├── report_generator.py   # PDF report builder (ReportLab)
├── flask_app.py          # Optional web interface
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
flask>=3.0.0            # Optional web interface
yfinance>=0.2.40        # Market data
pandas>=2.0.0           # Data manipulation
pandas-ta>=0.3.14b      # Technical indicators
scipy>=1.11.0           # Linear regression (Beta, R²)
numpy>=1.26.0           # Numerical operations
matplotlib>=3.8.0       # Chart generation
reportlab>=4.0.0        # PDF report generation
vaderSentiment>=3.3.2   # NLP sentiment analysis
thefuzz>=0.22.0         # Fuzzy company name matching
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

You can also enter any valid Yahoo Finance ticker directly.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

**v2.3.0** — Latest
- Added RSI/Price divergence detection
- Added Market Regime Filter (S&P 500 MA200)
- Added dynamic exit strategy (Stop Loss / Take Profit)
- Improved sentiment with Tail Risk detection & time-weighting
- Refactored main entry point (Live & Backtest dual-mode)

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

```bash
# Fork → Branch → Commit → Pull Request
git checkout -b feature/your-feature-name
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ by <a href="https://github.com/youcefbt-dz">youcefbt-dz</a>

⭐ Star the repo if you find it useful!

</div>
