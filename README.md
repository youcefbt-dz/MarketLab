<div align="center">
<img src="logo.svg" alt="MarketLab" width="420"/>
<br/>
<br/>

[![Version](https://img.shields.io/badge/version-2.3.0-378ADD?style=flat-square)](https://github.com/youcefbt-dz/python-finance-analyst)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=flat-square\&logo=python\&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Open Source](https://img.shields.io/badge/open%20source-yes-brightgreen?style=flat-square)](https://github.com/youcefbt-dz)
[![Stars](https://img.shields.io/github/stars/youcefbt-dz/python-finance-analyst?style=flat-square\&color=yellow)](https://github.com/youcefbt-dz/python-finance-analyst/stargazers)

**An open-source quantitative research and trading framework for stock analysis, strategy validation, and financial decision-making.**

[Getting Started](#quick-start) · [Features](#features) · [Backtesting Engine](#backtesting-engine) · [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

MarketLab is a **quantitative research and trading framework** designed for finance students, researchers, and aspiring quants.

It integrates technical analysis, risk modeling, NLP-driven sentiment analysis, a rule-based signals engine, and a complete backtesting system into a single modular pipeline.

The framework transforms raw market data into:

* technical indicators
* trading signals
* strategy performance evaluation
* risk metrics
* professional PDF reports

allowing users to move from **analysis → signals → backtesting → decision-making** in one workflow.

---

## Preview

<div align="center">

### 📄 Executive Summary Report

![Report](docs/screenshots/screenshot_report.png)

---

### 🧠 News Sentiment Analysis

![Sentiment](docs/screenshots/screenshot_sentiment.png)

---

### 📈 Technical Charts

![Charts](docs/screenshots/screenshot_charts.png)

</div>

---

## Features

```
Technical Analysis     → MA50/200, EMA20/50, RSI, MACD, Bollinger Bands, Stochastic
Risk Metrics           → Beta, R², Sharpe Ratio, Annualized Return

Signals System         → BUY / HOLD / SELL with 9-indicator scoring engine
Market Regime Filter   → Bullish / Bearish trend filtering

Backtesting Engine     → Historical simulation of trading strategies
Position Sizing        → Dynamic capital allocation per trade
Exit Strategy          → Rule-based exit system
Divergence Detection   → RSI / Price divergence signals

News Sentiment         → NLP analysis with Tail Risk detection & time-weighted scoring

Seasonality Analysis   → Best/Worst month detection
Correlation Matrix     → Cross-asset heatmap

PDF Reports            → Automated institutional-grade reports
Dual Mode Execution    → Live mode + Backtest mode
50+ Companies          → Preloaded company-to-ticker database
```

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/youcefbt-dz/python-finance-analyst.git

cd python-finance-analyst

# Install dependencies
pip install -r requirements.txt

# Live analysis
python main.py

# Backtesting
python main.py --mode backtest
```

---

## Project Structure

```
MarketLab/
│
├── main.py              ← Main entry point (Live & Backtest modes)
├── indicators.py        ← Technical analysis engine
├── signals.py           ← Signal scoring system
├── sentiment.py         ← News sentiment analysis
├── backtest.py          ← Backtesting engine
├── report_generator.py  ← PDF report generation
├── companies.json       ← Company database
├── requirements.txt
└── CONTRIBUTING.md
```

---

## Signals System

The scoring engine evaluates **9 independent market conditions**.

| Condition            | Max Score |
| -------------------- | --------- |
| Price vs MA200       | ±2        |
| Golden / Death Cross | ±1        |
| RSI + Stochastic     | ±4        |
| Stochastic crossover | ±1        |
| MACD Histogram       | ±2        |
| Bollinger Band touch | ±2        |
| Volume confirmation  | ±2        |
| Sharpe Ratio         | ±2        |
| News Sentiment       | ±3        |

```
Score ≥ 7  → BUY
Score ≤ -7 → SELL
Otherwise  → HOLD
```

Market regime and sentiment adjust the final decision.

---

## Backtesting Engine

MarketLab includes a **complete backtesting system** for evaluating strategies on historical data.

### Capabilities

* historical trade simulation
* dynamic position sizing
* entry and exit logic
* regime-based filtering
* divergence-based signals
* performance tracking

### Performance Metrics

```
Total Return
Win Rate
Sharpe Ratio
Max Drawdown
Number of Trades
Average Trade
```

This allows realistic validation of trading strategies before live execution.

---

## Execution Modes

### Live Mode

```
python main.py
```

Runs:

* technical indicators
* signals
* sentiment
* PDF report

---

### Backtest Mode

```
python main.py --mode backtest
```

Runs:

* historical simulation
* trade execution
* performance evaluation

---

## News Sentiment Engine

Features:

* VADER NLP
* financial keyword boosting
* time-weighted scoring
* tail risk detection
* confidence scoring
* retry system

Sentiment adjusts signal strength and risk exposure.

---

## Built With

* Python
* Pandas
* NumPy
* Matplotlib
* SciPy
* ReportLab
* VADER NLP

---

## Roadmap

* [x] Technical indicators

* [x] Risk metrics

* [x] Signals system

* [x] News sentiment

* [x] PDF reports

* [x] Backtesting engine

* [x] Position sizing

* [x] Market regime filter

* [x] Exit strategy

* [x] Divergence detection

* [ ] Portfolio optimization

* [ ] Efficient frontier

* [ ] VaR & Sortino

* [ ] FinBERT integration

* [ ] Streamlit dashboard

* [ ] API support

---

## Contributing

Contributions are welcome.

Read **CONTRIBUTING.md** to get started.

---

<div align="center">

**Built by Youcef Boutemedjet**
Finance Student | Quantitative Research Enthusiast

</div>

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/youcefbt-dz/python-finance-analyst.git
cd python-finance-analyst

# Install dependencies
pip install -r requirements.txt

# Run
python indicators.py
```

---

## Project Structure

```
MarketLab/
├── indicators.py        ← Main analysis engine
├── signals.py           ← BUY / HOLD / SELL scoring system
├── sentiment.py         ← News sentiment analysis (VADER + Financial Boost)
├── report_generator.py  ← PDF report generation
├── companies.json       ← 50+ supported companies
├── requirements.txt     ← Dependencies
└── CONTRIBUTING.md      ← Contribution guide
```

---

## Signals System

The scoring engine evaluates **9 independent conditions** and returns a weighted score:

| Condition | Max Score |
|-----------|-----------|
| Price vs MA200 (Trend) | ±2 |
| Golden / Death Cross | ±1 |
| Double Oversold / Overbought (RSI + Stoch) | ±4 |
| Stochastic Crossover in extreme zone | ±1 |
| MACD Histogram Crossover | ±2 |
| Bollinger Band Touch | ±2 |
| Volume Confirmation | ±2 |
| Sharpe Ratio Quality | ±2 |
| **News Sentiment** | **±3** |

```
Score ≥  7  →  BUY      |   Score ≥ 10 + Bullish trend  →  STRONG BUY
Score ≤ -7  →  SELL     |   Score ≤ -10 + Bearish trend →  STRONG SELL
Otherwise   →  HOLD
```

---

## News Sentiment Engine

The `sentiment.py` module goes beyond simple averaging:

- **Financial Keyword Boost** — compensates for VADER's weakness on financial language (`beats`, `upgrade`, `misses`, `downgrade`, etc.)
- **Time-Weighted Scoring** — recent news carries more weight (last 6h = 1.0x, 3 days old = 0.55x)
- **Tail Risk Detection** — a single strong negative headline (compound ≤ -0.5) adjusts the overall score via a 40/60 blend
- **Retry System** — 3 automatic retries if the API returns empty results
- **Confidence Score** — 0–100% based on signal strength

---

## Example Output

```
--- News Sentiment for AAPL ---
Overall   : 🔴 BEARISH
Compound  : -0.2818  |  Confidence: 28.2%  |  News: 10
Positive  : 30%  |  Negative: 70%
⚠️  TAIL RISK: Strong negative news detected — result adjusted.

--- Result ---
  AAPL → HOLD  (was BUY before sentiment)  ⚠️ Sentiment Bearish — defensive stance advised
  Score: 7 → 4 (after sentiment)
  - Trend: Bullish (Price vs MA200).
  - Structure: Golden Cross active.
  - Structure: Price testing MA200 support (0.6% above) with RSI near oversold (34.9).
  - Momentum: Stochastic Bullish Crossover in Oversold zone.
  - Volume: High activity (2.1x) confirms the Bullish triggers.
  - Sentiment: News is BEARISH (compound: -0.2818, score adjustment: -3).
```

---

## Built With

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![ReportLab](https://img.shields.io/badge/ReportLab-FF0000?style=for-the-badge)
![VADER](https://img.shields.io/badge/VADER_NLP-4B8BBE?style=for-the-badge)

</div>

---

## Roadmap

- [x] Technical indicators (MA, EMA, RSI, MACD, BB, Stochastic)
- [x] Risk metrics (Beta, R², Sharpe Ratio, Annualized Return)
- [x] Signals scoring system with 9 conditions
- [x] News Sentiment Analysis (VADER + Financial Boost + Tail Risk)
- [x] PDF report generation (Goldman Sachs style)
- [x] Modular architecture
- [ ] Backtesting engine
- [ ] Portfolio Optimization (Efficient Frontier)
- [ ] VaR & Sortino Ratio
- [ ] FinBERT integration (replace VADER)
- [ ] Streamlit web interface update

---

## Contributing

Contributions are welcome! Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

<div align="center">

**Built by [Youcef Boutemedjet](https://github.com/youcefbt-dz) — Finance Student | Quantitative Research Enthusiast**

</div>
