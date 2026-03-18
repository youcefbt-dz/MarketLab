<div align="center">

<img src="logo.svg" alt="MarketLab" width="420"/>



[![Version](https://img.shields.io/badge/version-2.1.0-378ADD?style=flat-square)](https://github.com/youcefbt-dz/python-finance-analyst)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Open Source](https://img.shields.io/badge/open%20source-yes-brightgreen?style=flat-square)](https://github.com/youcefbt-dz)
[![Stars](https://img.shields.io/github/stars/youcefbt-dz/python-finance-analyst?style=flat-square&color=yellow)](https://github.com/youcefbt-dz/python-finance-analyst/stargazers)

**An open-source quantitative research framework for stock analysis, risk modeling, and financial decision-making.**

[Getting Started](#quick-start) · [Features](#features) · [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

MarketLab is built for **finance students, researchers, and analysts** who want to process complex market data efficiently. It combines technical analysis, risk metrics, and a signals system into a single modular framework — turning raw data into clear, actionable insights.

---

## Features
```
📊 Technical Analysis     →  MA, EMA, RSI, MACD, Bollinger Bands, Stochastic
📈 Risk Metrics           →  Beta, R², Sharpe Ratio, Volatility
🤖 Signals System         →  BUY / HOLD / SELL with trend filter & scoring
📅 Seasonality Analysis   →  Best/Worst month detection
🔗 Correlation Matrix     →  Cross-asset heatmap
📄 PDF Reports            →  Automated professional report generation
🏢 50+ Companies          →  Pre-loaded company database
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
├── signals.py           ← BUY / HOLD / SELL system
├── report_generator.py  ← PDF report generation
├── companies.json       ← 50+ supported companies
├── requirements.txt     ← Dependencies
└── CONTRIBUTING.md      ← Contribution guide
```

---

## Example Output
```
==================================================
  AAPL → STRONG BUY
  Score: 8/15
  ────────────────────────────────────────
  - Price is above MA200: Long-term trend is Bullish.
  - RSI is Oversold (28.3). Potential reversal.
  - MACD Histogram crossed above zero: Strong Bullish reversal.
  - Golden Cross (MA50 > MA200) active.
==================================================
```

---

## Built With

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

</div>

---

## Roadmap

- [x] Technical indicators (MA, EMA, RSI, MACD, BB, Stochastic)
- [x] Risk metrics (Beta, R², Sharpe Ratio, Volatility)
- [x] Signals system with trend filter
- [x] PDF report generation
- [x] Modular architecture
- [ ] Portfolio Optimization (Efficient Frontier)
- [ ] VaR & Sortino Ratio
- [ ] Machine Learning price predictions

---

## Contributing

Contributions are welcome! Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

<div align="center">

**Built by [Youcef Boutemedjet](https://github.com/youcefbt-dz) — Finance Student | Quantitative Research Enthusiast**

</div>
