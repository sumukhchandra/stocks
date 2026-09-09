---
title: NSE Stocks Dashboard
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---
# 📈 NSE Automated Algorithmic Trading & AI Prediction System

An end-to-end quantitative trading, simulation, and machine learning framework designed specifically for Indian National Stock Exchange (NSE) intraday equities with compounding, real-world Indian tax computation (STT, GST, Stamp Duty, SEBI turnover fees, 25% STCG), and Zerodha brokerage modeling.

---

## 🏛️ Project Architecture (Strict 7-Folder Design)

The project adheres to a strict, modular 7-folder structure:

```
stocks/
├── frontend/        # Interactive Streamlit Web App & Dashboards
│   └── app.py       # Multi-page dashboard (Live Market, Auto-Trader, Backtests, AI Agent)
├── backend/         # Core Trading Engine & Risk Management
│   ├── nse_auto_trader.py       # Live & paper-trading execution loop
│   ├── execution/               # Dynamic regime router & order handlers
│   └── risk/                    # IndiaTaxEngine (Zerodha costs & 25% STCG tax)
├── database/        # Relations, Rules, Logics & SQLite State Persistence
│   ├── relations.py             # Stock sector classifications, weights & beta relations
│   ├── rules.py                 # Market hours (9:15-15:30), circuit breakers & trade viability
│   ├── schema.py                # Normalized schema (accounts, trades, positions, signals)
│   ├── db_manager.py            # SQLite manager for trading_system.db
│   └── stocks_config.py         # Config constants & stock universe definitions
├── ml_models/       # Machine Learning & AI Prediction Engine
│   ├── features/                # Feature engineering (RSI, MACD, Bollinger, Nifty Index)
│   ├── labels/                  # Triple-barrier labeling & forward return calculation
│   └── models/                  # Ensemble classifiers (CatBoost, XGBoost, LightGBM, RF) + Regressor
├── simulations/     # Historical & Walk-Forward Simulation Labs
│   ├── run_batch_simulation.py  # Chronological multi-session simulation (Aug 22 - Sep 7)
│   ├── stock_simulator.py       # Intraday candle-by-candle simulation engine
│   └── prediction_lab.py        # Out-of-sample prediction vs reality analysis
├── data/            # Market Data Ingestion & Protected Parquet Caches
│   ├── fetch_stock_data.py      # Multi-stock & Nifty 50 5-minute candle builder
│   └── processed/               # Master labeled parquet dataset (60d history)
└── agent/           # Local Autonomous AI Trader Agent
    ├── diagnostics.py           # Automated problem hunter & integrity auditor
    ├── account_manager.py       # Capital management, drawdown & sector risk tracker
    ├── data_sorter.py           # Temporary cache cleaner (safeguards master data)
    ├── strategy_advisor.py      # Quantitative metrics (Sharpe, Win Rate, Thresholds)
    └── trader_agent.py          # Master CLI and autonomous agent runner
```

---

## 🚀 Quick Start & Usage

### 1. Run the AI Autonomous Agent
The built-in AI agent audits your system, detects problems, and manages portfolio risks locally without third-party API dependencies:

```powershell
# Run full diagnostic audit
python agent/trader_agent.py --action diagnose

# Check portfolio health, risk score & sector allocation
python agent/trader_agent.py --action account

# Optimize data storage by clearing disposable cache
python agent/trader_agent.py --action clean

# Evaluate strategy metrics (Sharpe ratio, win rate, thresholds)
python agent/trader_agent.py --action optimize
```

### 2. Launch the Web Dashboard
```powershell
streamlit run frontend/app.py --server.port 8501
```
Open `http://localhost:8501` to view live signals, execute paper trades, monitor capital compounding, and interact with the AI Agent.

### 3. Run Historical Simulations
```powershell
# Run chronological multi-day batch simulation (Aug 22 to Sep 7)
python simulations/run_batch_simulation.py

# Run standalone stock simulator
python simulations/stock_simulator.py --start-date 2026-08-25 --end-date 2026-09-07
```

### 4. Fetch Market Data & Retrain Models
```powershell
# Ingest 60 days of 5-minute candles for active stocks + Nifty 50
python data/fetch_stock_data.py --period 60d --interval 5m

# Retrain ML ensemble (XGBoost, CatBoost, LightGBM, Random Forest)
python ml_models/models/train_ensemble.py
```

---

## ⚖️ Indian Taxation & Zerodha Brokerage Engine
Every simulated and live trade strictly accounts for:
- **Brokerage**: ₹20 flat or 0.03% (whichever is lower) per executed order
- **STT (Securities Transaction Tax)**: 0.025% on delivery / intraday sell turnover
- **Exchange Turnover Charges (NSE)**: 0.00345%
- **GST**: 18% on (Brokerage + Exchange Charges + SEBI Fees)
- **Stamp Duty**: 0.003% on buy turnover
- **Short-Term Capital Gains (STCG)**: 25% flat tax on net gains
- **Trade Viability Gating**: Orders are only executed if predicted return yields positive net profit after all the above frictions.