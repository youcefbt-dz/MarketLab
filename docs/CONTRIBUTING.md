# Contributing to MarketLab

Thank you for your interest in contributing to **MarketLab**! Whether you're fixing a bug, adding a feature, or improving documentation — every contribution matters.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Ideas for Contribution](#ideas-for-contribution)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Questions](#questions)

---

## Getting Started

### 1. Fork the Repository

Click the **Fork** button at the top right of the repository page.

### 2. Clone Your Fork

```bash
git clone https://github.com/your-username/python-finance-analyst.git
cd python-finance-analyst
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 5. Make Your Changes

Test your changes locally before committing:

```bash
python main.py
python backtest.py
python ml_predictor.py
```

### 6. Commit and Push

```bash
git add .
git commit -m "feat: describe your change clearly"
git push origin feature/your-feature-name
```

---

## Project Structure

```
python-finance-analyst/
│
├── main.py               # Entry point — Live Analysis mode
├── backtest.py           # Backtesting engine (walk-forward, ATR-based exits)
├── backtest_logger.py    # Black Box Logger + Reliability Score engine
├── signals.py            # Signal generation (13-rule scoring system)
├── ml_predictor.py       # ML Pipeline (SMOTE + RandomForest/GB/LR)
├── sentiment.py          # NLP sentiment analysis (VADER + financial boosters)
├── report_generator.py   # PDF report builder (ReportLab)
├── stock_warehouse.py    # Local data warehouse (250+ symbols)
├── companies.json        # 240+ company name → ticker mappings
├── requirements.txt      # Python dependencies
└── logo.svg              # Project logo
```

---

## Ideas for Contribution

###  Technical Indicators
- Add OBV (On-Balance Volume) to the scoring system
- Add Williams %R as an overbought/oversold confirmation
- Add Ichimoku Cloud for multi-layered trend analysis
- Add Volume Profile or VWAP for intraday-aware signals

###  ML & Data
- Expand `quality_trade` thresholds with configurable YAML/JSON config
- Add walk-forward ML validation (train on past, test on future)
- Export feature importance as a chart/PNG
- Add model persistence (`joblib` save/load) to avoid retraining each run

###  Risk & Metrics
- Add VaR (Value at Risk) and CVaR to the risk report
- Add Sortino Ratio (downside deviation only)
- Add Calmar Ratio (return / max drawdown)
- Add portfolio-level metrics across multiple tickers

###  Market Coverage
- Add support for non-US markets (LSE, TSX, Euronext, Tadawul)
- Expand `companies.json` with European and Asian tickers
- Add currency conversion for multi-market portfolios

###  Strategy Improvements
- Add portfolio optimization (Efficient Frontier, Minimum Variance)
- Add regime-aware position sizing (reduce exposure in Risk-Off)
- Add sector rotation logic

###  Sentiment
- Integrate FinBERT for deeper financial NLP
- Add Reddit/StockTwits sentiment as an additional source
- Add earnings surprise detection from news headlines

###  Visualization
- Build a Streamlit dashboard for reliability and ML results
- Add interactive Plotly charts to replace static matplotlib exports
- Add a live signal scanner across the full 250+ symbol universe

---

## Code Style

- Use **clear, descriptive variable names** — avoid single letters except in loops
- Follow the **existing module structure** — each file has one responsibility
- Add **Arabic or English comments** for non-obvious logic (both are welcome)
- Keep functions **short and focused** — prefer helper functions over long blocks
- Always handle edge cases: empty DataFrames, missing columns, zero-division

---

## Submitting a Pull Request

1. Make sure your branch is up to date with `main`
2. Write a clear PR title: `feat: add Sortino Ratio to metrics` or `fix: handle NaN in ADX calculator`
3. In the PR description, explain:
   - **What** you changed
   - **Why** the change is useful
   - **How** to test it
4. Link any related issues if applicable

---

## Questions?

Open an [issue on GitHub](https://github.com/youcefbt-dz/python-finance-analyst/issues) and we'll be happy to help!
