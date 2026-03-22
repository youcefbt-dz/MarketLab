<div align="center">
<img src="logo.svg" alt="MarketLab" width="420"/>
<br/>
<br/>

[![Version](https://img.shields.io/badge/version-2.2.0-378ADD?style=flat-square)](https://github.com/youcefbt-dz/python-finance-analyst)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Open Source](https://img.shields.io/badge/open%20source-yes-brightgreen?style=flat-square)](https://github.com/youcefbt-dz)
[![Stars](https://img.shields.io/github/stars/youcefbt-dz/python-finance-analyst?style=flat-square&color=yellow)](https://github.com/youcefbt-dz/python-finance-analyst/stargazers)

**An open-source quantitative research framework for stock analysis, risk modeling, and financial decision-making.**

[Getting Started](#quick-start) · [Features](#features) · [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

MarketLab is built for **finance students, researchers, and analysts** who want to process complex market data efficiently. It combines technical analysis, risk metrics, a signals scoring system, and NLP-based news sentiment into a single modular framework — turning raw market data into clear, actionable insights with professional PDF reports.

---

## Preview

<div align="center">

| Executive Summary Report | News Sentiment Analysis |
|:---:|:---:|
| ![Report](docs/screenshots/screenshot_report.png) | ![Sentiment](docs/screenshots/screenshot_sentiment.png) |
| *Signal badge · Score adjustment · Market data table* | *VADER NLP · Tail Risk detection · Time-weighted scoring* |

<br/>

![Charts](docs/screenshots/screenshot_charts.png)
*Technical charts embedded directly in the PDF report — Price Action, Bollinger Bands, RSI, MACD, Stochastic*

</div>

---

## Features

```
 Technical Analysis     →  MA50/200, EMA20/50, RSI, MACD, Bollinger Bands, Stochastic
 Risk Metrics           →  Beta, R², Sharpe Ratio, Annualized Return
 Signals System         →  BUY / HOLD / SELL with 9-indicator scoring engine
 News Sentiment         →  NLP analysis with Tail Risk detection & time-weighted scoring
 Seasonality Analysis   →  Best/Worst month detection
 Correlation Matrix     →  Cross-asset heatmap
 PDF Reports            →  Automated professional report (Goldman Sachs style)
 50+ Companies          →  Pre-loaded company name-to-ticker database
```

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
