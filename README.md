<div align="center">

<img src="assets/logo.svg" alt="MarketLab" width="380"/>

<br/><br/>

<p>
  <img src="https://img.shields.io/badge/version-3.1.0-0F172A?style=flat-square&labelColor=0F172A&color=F59E0B" alt="Version"/>
  &nbsp;
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0F172A" alt="Python"/>
  &nbsp;
  <img src="https://img.shields.io/badge/license-Apache_2.0-2563EB?style=flat-square&labelColor=0F172A" alt="License"/>
  &nbsp;
  <img src="https://img.shields.io/github/stars/youcefbt-dz/MarketLab?style=flat-square&color=F59E0B&labelColor=0F172A" alt="Stars"/>
  &nbsp;
  <img src="https://img.shields.io/badge/status-active-22C55E?style=flat-square&labelColor=0F172A" alt="Status"/>
</p>

<p>
  <strong>Open-source quantitative research framework for systematic trading.</strong><br/>
  Signals · Walk-forward backtesting · Sentiment analysis · Bayesian optimization — all offline.
</p>

<p>
  <a href="https://youcefbt-dz.github.io/MarketLab/"><strong>Website</strong></a>
  &nbsp;·&nbsp;
  <a href="#-quick-start"><strong>Quick Start</strong></a>
  &nbsp;·&nbsp;
  <a href="#-architecture"><strong>Architecture</strong></a>
  &nbsp;·&nbsp;
  <a href="#-features"><strong>Features</strong></a>
  &nbsp;·&nbsp;
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

</div>

---

> [!NOTE]
> MarketLab is built for **educational and research purposes only**. It does not constitute financial advice.

---

## Overview

MarketLab is a modular quantitative finance framework built for finance students, researchers, and aspiring quants. It transforms raw market data into structured trading intelligence through a reproducible research pipeline — with **no dependency on paid APIs or proprietary data**.

```
Raw Data  ──►  Indicators  ──►  Signal Scoring  ──►  Backtest  ──►  Optimize  ──►  PDF Report
```

**What makes it different:**

- **Fully offline** — 250+ symbols stored locally as CSV (~1.8M rows), no API calls at runtime
- **No black boxes** — every signal rule is scored transparently, every trade is logged
- **Research-grade rigor** — walk-forward simulation, gap-aware exits, zero look-ahead bias
- **Self-improving** — Bayesian optimizer tunes 26 signal parameters automatically

---

## Screenshots

<div align="center">
  <img src="docs/screenshots/terminal_UI.png" width="860" alt="MarketLab CLI"/>
  <br/><sub>Interactive 5-mode CLI — portfolio analysis, backtesting, watchlist scanning, optimization, warehouse management</sub>
</div>

<br/>

<div align="center">
  <img src="docs/screenshots/screenshot_report.png" width="265" alt="Strategic Report"/>
  &nbsp;
  <img src="docs/screenshots/screenshot_sentiment.png" width="265" alt="Sentiment Analysis"/>
  &nbsp;
  <img src="docs/screenshots/screenshot_charts.png" width="265" alt="Technical Charts"/>
  <br/>
  <sub>PDF Report &nbsp;·&nbsp; Sentiment Analysis &nbsp;·&nbsp; Technical Charts</sub>
</div>

---

##  Quick Start

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/youcefbt-dz/MarketLab.git
cd MarketLab

# 2. Install dependencies
pip install -r config/requirements.txt

# 3. Build the local data warehouse (run once)
python core/stock_warehouse.py

# 4. Launch MarketLab
python main.py
```

> The warehouse step downloads 250+ symbols as local CSV files. All subsequent runs are fully offline.

**Optional — tune signal parameters with Bayesian optimization:**

```bash
python analysis/strategy_optimizer.py           # 100 trials
python analysis/strategy_optimizer.py --apply   # apply best params to signals.py
```

---

##  Architecture

```
main.py  ──  5-mode interactive CLI
│
├── Mode 1  Portfolio Analysis    signals + sentiment + PDF report
├── Mode 2  Backtesting           walk-forward simulation
├── Mode 3  Watchlist Scanner     250+ tickers in parallel
├── Mode 4  Strategy Optimizer    Bayesian parameter search (Optuna)
└── Mode 5  Warehouse Manager     local data download & updates
```

```
core/
├── signals.py              13-rule scoring engine · ATR exits · ADX/regime filter
├── sentiment.py            VADER NLP · financial keyword booster · tail risk detect
└── stock_warehouse.py      Local CSV warehouse · 250+ symbols · smart updates

analysis/
├── backtest.py             Walk-forward engine · gap-aware exits · dynamic sizing
├── backtest_logger.py      Persistent run history · reliability score (0–100)
├── strategy_optimizer.py   Optuna · 26 params · 100 Bayesian trials
└── watchlist_scanner.py    Parallel scanner · ranked signal output

ui/
└── report_generator.py     PDF builder · ReportLab · executive summary format
```

**Data layer:**

```
stock_warehouse.py  →  data/AAPL.csv, MSFT.csv, ...
                        250+ symbols · ~1.8M rows · load_local(ticker) — no internet at runtime
```

---

##  Features

### Signal Engine

A transparent 13-rule scoring system producing `STRONG BUY` / `BUY` / `HOLD` / `SELL` / `STRONG SELL` signals with dynamic ATR-based exit levels.

| Rule | Weight |
|---|---|
| Price vs MA200 — trend direction | ±2 |
| Golden Cross / Death Cross (MA50 × MA200) | ±1 |
| ADX trend strength gate | −3 / +1 |
| RSI / Price divergence | ±3 |
| Double oversold / overbought | ±4 |
| MA200 support test | +1 |
| Bear market deep penalty | −2 |
| Stochastic crossover | ±1 |
| MACD crossover | ±2 |
| Bollinger Band touch | ±2 |
| Volume confirmation | ±2 |
| Volatility filter (ATR%) | ±1 to ±3 |
| Market regime (S&P 500 vs MA200) | 0 / −3 |
| Relative strength vs S&P 500 | ±1 to ±2 |

**Thresholds:**

```
score ≥  6                      →  BUY
score ≥  8  +  bullish trend    →  STRONG BUY
score ≤ −6                      →  SELL
score ≤ −10 +  bearish trend    →  STRONG SELL
```

**Active filters (v3.1):**
- `MIN_ADX_ENTRY = 18` — blocks entry in choppy, directionless markets
- Bear market penalty of −2 when price sits 15%+ below MA200
- Stop loss dynamically sized to `ATR14 × 2.5`

---

### Backtesting Engine

Walk-forward simulation with strict zero look-ahead bias.

| Feature | Detail |
|---|---|
| Gap-aware exits | Open gaps below SL exit at Open price, not SL |
| Position sizing | STRONG BUY 35% · BUY 22% · fallback 15% |
| Portfolio cap | Max 70% deployed · 40% per position |
| Post-loss cooldown | Configurable idle bars after a losing trade |
| Rolling metrics | Sharpe · Beta · annualized return in O(N) |
| Output | CSV trade log · equity curve PNG · JSON summary · TXT report |

**Metrics reported per ticker:**

| Metric | Description |
|---|---|
| Total Return | Strategy vs Buy & Hold |
| Win Rate | % of profitable trades |
| Profit Factor | Gross profit / gross loss |
| Sharpe Ratio | Annualized risk-adjusted return |
| Max Drawdown | Peak-to-trough equity decline |
| Avg R-Multiple | Realized vs planned risk/reward |
| Exit Breakdown | TP hits · SL hits · gap exits · end-of-period |

---

### Strategy Optimizer

Bayesian search over 26 signal parameters using Optuna's TPE sampler.

```
Trial N
 └─ sample 26 params  (BUY_THRESHOLD, ATR_mult, RSI bands, ADX floor, rule weights…)
      └─ run full backtest on 7-stock basket  (offline, ~3s/trial)
           └─ objective = Sharpe × WinRate / |MaxDrawdown|
                └─ Optuna updates surrogate model → next trial smarter
                     └─ after 100 trials → best_params.json saved
```

- Trials 1–20: random exploration
- Trials 21–100: TPE Bayesian search
- `--apply` flag patches `signals.py` in-place with a timestamped backup
- Optimization basket: `AAPL MSFT NVDA TSLA JPM XOM AMZN`

---

### Watchlist Scanner

Parallel scan of 250+ tickers ranked by signal score.

```bash
python analysis/watchlist_scanner.py                       # top 20, min score 4
python analysis/watchlist_scanner.py --top 10 --min-score 6
python analysis/watchlist_scanner.py --export results.csv
```

Output columns: `signal · score · price · RSI · ADX · ATR% · RS% · stop_loss · take_profit`

---

### Sentiment Analysis

- **VADER NLP** with financial keyword booster — `beats`, `misses`, `downgrade`, `rally`, `cuts guidance`, and more
- **72-hour time decay** — recent headlines weighted exponentially higher
- **Tail risk detection** — single strong negative event overrides the aggregate score
- **Confidence scoring** — positive/negative ratio with headline count breakdown

---

### Black Box Logger

Every backtest run is automatically appended to `backtest_history.json`.

**Reliability Score formula (0–100):**

```
Reliability = Pass Rate        × 40
            + Avg Win Rate     × 25   (normalized to 65% target)
            + Avg PF           × 20   (normalized to 2.5 target)
            + Beat Benchmark   × 15
```

Scores are broken down per ticker and per market regime: Bull · Sideways · Bear.

---

### PDF Report

Executive summary generated with ReportLab on every portfolio analysis run:

- Signal badge with score and confidence level
- Live market data table + full indicator snapshot
- Scored rule breakdown — which of the 13 triggered and why
- Sentiment badge with top headlines
- 5 embedded charts: Price/MA · Bollinger Bands · RSI · MACD · Stochastic
- Narrative qualitative interpretation

---

##  Project Structure

```
MarketLab/
│
├── main.py                         Entry point — 5-mode interactive CLI
│
├── core/
│   ├── signals.py                  Signal engine — 13 rules, ATR exits, regime filter
│   ├── sentiment.py                NLP sentiment — VADER + financial keyword booster
│   ├── stock_warehouse.py          Local data warehouse — 250+ symbols, smart sync
│   └── crypto_warehouse.py         Crypto data layer
│
├── analysis/
│   ├── backtest.py                 Walk-forward backtesting engine
│   ├── backtest_logger.py          Black Box Logger + reliability score
│   ├── batch_backtest.py           Batch runner for ML training data generation
│   ├── ml_predictor.py             XGBoost signal quality predictor (+ SMOTE)
│   ├── strategy_optimizer.py       Bayesian parameter optimizer — Optuna, 26 params
│   └── watchlist_scanner.py        Parallel 250+ ticker scanner
│
├── ui/
│   └── report_generator.py         PDF report builder — ReportLab
│
├── config/
│   ├── companies.json              240+ company name → ticker mappings
│   ├── crypto_symbols.json
│   └── requirements.txt
│
├── assets/
│   └── logo.svg
│
├── data/                           Local CSV warehouse — gitignored
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── _metadata.json
│
├── docs/screenshots/
├── backtest_results/               Trade logs · equity curves · reports
├── backtest_history.json           Accumulated backtest runs
└── best_params.json                Latest optimizer output
```

---

##  Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `numpy` | ≥ 1.26.0 | Numerical computation |
| `pandas-ta` | ≥ 0.3.14b | Technical indicators |
| `yfinance` | ≥ 0.2.40 | Initial data download |
| `matplotlib` | ≥ 3.8.0 | Charts and equity curves |
| `reportlab` | ≥ 4.0.0 | PDF report generation |
| `vaderSentiment` | ≥ 3.3.2 | NLP sentiment scoring |
| `optuna` | ≥ 3.0.0 | Bayesian optimization |
| `thefuzz` | ≥ 0.22.0 | Fuzzy company name matching |
| `scipy` | ≥ 1.11.0 | Statistical utilities |

---

##  Supported Symbols

240+ pre-mapped companies across sectors via `config/companies.json`:

| Sector | Tickers |
|---|---|
| Technology | AAPL · MSFT · NVDA · AMD · GOOGL · META · ORCL |
| Finance | JPM · GS · V · MA · BLK · BAC · MS |
| Healthcare | LLY · ABBV · JNJ · PFE · MRNA · UNH |
| Consumer | AMZN · TSLA · NKE · DIS · WMT · SBUX |
| Energy | XOM · CVX · SHEL · BP · TTE · COP |
| ETFs | SPY · QQQ · GLD · IBIT · ETHA · IWM |

---

##  Roadmap

- [x] 13-rule signal engine with ADX gate and bear market filter
- [x] ATR-based dynamic stop loss
- [x] Walk-forward backtesting with gap-aware exits
- [x] Bayesian strategy optimizer (Optuna · 26 params)
- [x] Local data warehouse (250+ symbols · ~1.8M rows)
- [x] Parallel watchlist scanner
- [x] Black Box reliability logger
- [x] GitHub Pages landing page
- [ ] Trailing stop + partial position exit
- [ ] Multi-timeframe signal confirmation
- [ ] Crypto module (CCXT integration)
- [ ] React dashboard for live signal monitoring

---

##  Changelog

Full history in [`SYSTEM_RELEASE_HISTORY.md`](./SYSTEM_RELEASE_HISTORY.md).

**v3.1.0** — Signal Filters + CLI Refactor
- Refactored `main.py` into a clean 5-mode interactive CLI
- Added `MIN_ADX_ENTRY = 18` — blocks signals in choppy markets (−3 penalty)
- Added bear market deep penalty (−2 when price is 15%+ below MA200)
- ATR stop loss multiplier raised from 1.5 → 2.5 for more realistic exits
- Launched [project website](https://youcefbt-dz.github.io/MarketLab/)

**v3.0.0** — Strategy Optimizer
- Added `strategy_optimizer.py` — 100-trial Bayesian search via Optuna
- Replaced ML Predictor (Mode 4) with Bayesian Strategy Optimizer

**v2.8.0** — Watchlist Scanner + Warehouse
**v2.5.0** — ML Predictor (XGBoost + SMOTE)
**v2.4.0** — Black Box Logger + Reliability Score

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

```bash
git checkout -b feature/your-feature-name
```

---

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
<sub>Built by <a href="https://github.com/youcefbt-dz">youcefbt-dz</a> · for the quant community</sub>
</div>
