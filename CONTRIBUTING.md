# Contributing to MarketLab

Thank you for your interest in contributing!

## How to Contribute

### 1. Fork the Repository
Click the **Fork** button at the top right of this page.

### 2. Clone Your Fork

```bash
git clone https://github.com/your-username/MarketLab.git
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Make Your Changes

- Add new indicators
- Improve existing analysis
- Fix bugs
- Improve documentation

### 5. Submit a Pull Request

- Describe what you changed
- Explain why the change is useful

---

## Project Structure

```
main.py              ← Main analysis engine
signals.py           ← Buy/Sell/Hold scoring system
report_generator.py  ← PDF report generation
backtest.py          ← MACD crossover backtesting
strategy_test.py     ← Historical signal validation
companies.json       ← Supported companies database
requirements.txt     ← Python dependencies
```

---

## Ideas for Contribution

- Add new technical indicators (ATR, OBV, Williams %R...)
- Add more companies to `companies.json`
- Improve the signals scoring system
- Add new financial metrics (VaR, Sortino Ratio...)
- Improve PDF report design
- Add support for non-US markets

---

## Code Style

- Use clear variable names
- Follow existing code structure
- Test your changes before submitting

---

## Questions?

Open an issue on GitHub and we'll be happy to help!
