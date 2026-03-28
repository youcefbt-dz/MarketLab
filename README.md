<div align="center">
<img src="logo.svg" alt="MarketLab" width="420"/>



An open-source quantitative research and trading framework for stock analysis, strategy validation, and financial decision-making.

Getting Started • Architecture • Features • Contributing

</div>

📖 Overview

MarketLab is a comprehensive quantitative research and trading framework tailored for finance students, researchers, and aspiring quants.

It bridges the gap between raw financial data and actionable trading strategies by integrating technical analysis, risk modeling, NLP-driven sentiment analysis, a rule-based signals engine, and a robust backtesting environment.

The framework transforms raw market data into institutional-grade outputs, allowing users to seamlessly transition from Data Analysis → Signal Generation → Backtest Validation → Execution Decision within a single, modular pipeline.

👁️ Visual Previews

<div align="center">

📄 Executive Summary Report

Automated, print-ready PDF reports summarizing technicals, risk, and sentiment.

🧠 News Sentiment Analysis

NLP engine parsing financial headlines to detect market sentiment and tail risks.

📈 Technical Charts

Visual representation of moving averages, momentum oscillators, and volatility bands.

</div>

⚙️ System Architecture

MarketLab is built on a highly modular architecture, ensuring separation of concerns between data fetching, analysis, and reporting.

[ Market Data API ] ──┐
                      ▼
[ NLP News Engine ] ──┼──▶ [ Quantitative Engine ] ──▶ [ Scoring System (0-10) ]
                      │        (MA, RSI, MACD)               │
[ Risk Free Rates ] ──┘                                      ▼
                                                     [ Final Decision ]
                                                       (BUY/HOLD/SELL)
                                                             │
      ┌──────────────────────────────────────────────────────┴──┐
      ▼                                                         ▼
[ Backtesting Mode ]                                   [ Live Execution Mode ]
Simulates historical performance                       Generates Institutional PDF Report


⚡ Core Capabilities

Instead of isolated scripts, MarketLab provides a unified FinTech toolkit:

📊 1. Quantitative & Technical Analysis

Trend & Momentum: MA50/200, EMA20/50, MACD, RSI, Stochastic Oscillator.

Volatility: Bollinger Bands, Standard Deviation.

Risk Metrics: Portfolio Beta, R-Squared, Sharpe Ratio, Annualized Return.

🧠 2. NLP-Driven Sentiment Engine (VADER + Financial Boost)

Time-Weighted Scoring: Prioritizes breaking news over older headlines.

Tail Risk Detection: Automatically defensive stance on extreme negative news.

Financial Lexicon: Adjusted to understand finance-specific terminology (e.g., "beats earnings", "downgraded").

🚦 3. Rule-Based Scoring Engine

Evaluates 9 independent market conditions (Trend, Divergence, Volume, Sentiment).

Market Regime Filter: Prevents "Strong Buy" signals in aggressive bear markets.

Divergence Detection: Identifies hidden RSI/Price divergences for early entry/exit.

🔄 4. Advanced Backtesting

Historical simulation of trading strategies with Dynamic Position Sizing.

Evaluates metrics including Total Return, Win Rate, and Max Drawdown.

Rule-based exit strategies to manage capital efficiently.

🚀 Quick Start

Get MarketLab running on your local machine in seconds.

# 1. Clone the repository
git clone [https://github.com/youcefbt-dz/python-finance-analyst.git](https://github.com/youcefbt-dz/python-finance-analyst.git)
cd python-finance-analyst

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Run Live Mode (Generates signals & PDF reports)
python main.py

# 4. Run Backtesting Mode (Validates historical strategies)
python main.py --mode backtest


🗺️ Roadmap & Future Enhancements

We are continuously building to make MarketLab a premier open-source FinTech tool:

[x] Technical indicators & Signal scoring system

[x] NLP News Sentiment Analysis (VADER)

[x] PDF institutional report generation

[x] Algorithmic Backtesting engine

[ ] Portfolio Optimization (Markowitz Efficient Frontier)

[ ] VaR (Value at Risk) & Sortino Ratio integration

[ ] Upgrade NLP to FinBERT for deep context analysis

[ ] Web-based UI using Streamlit

🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Please refer to the CONTRIBUTING.md file for formatting and PR guidelines.

<div align="center">





<i>Built with precision by <b>Youcef Boutemedjet</b> — Finance Student & Quantitative Researcher.</i>
</div>
</div>
