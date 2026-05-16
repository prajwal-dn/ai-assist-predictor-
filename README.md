# Forex Predictor AI

A machine learning model that predicts short-term Forex direction (UP / DOWN)
using XGBoost trained on technical indicators.

---

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your settings
Edit `config.yaml`:
- Change `pair` to your currency pair (e.g. `GBPUSD=X`, `USDJPY=X`)
- Adjust `forecast_horizon` (how many hours ahead to predict)
- Adjust `confidence_threshold` (0.65 = only trade when 65%+ confident)

### 3. Run the full pipeline (first time)
```bash
python src/data_loader.py        # Step 1: download historical data
python src/feature_engineer.py   # Step 2: add technical indicators
python src/labeller.py           # Step 3: add UP/DOWN labels
python src/train_model.py        # Step 4: train the model
python src/backtest.py           # Step 5: see historical performance
```

### 4. Get a live signal
```bash
python src/predict.py
```

### 5. Retrain monthly (keeps model fresh)
```bash
python src/retrain.py
```

---

## Project Structure

```
forex-predictor/
├── config.yaml              ← all settings in one place
├── requirements.txt         ← python libraries
│
├── data/
│   ├── raw/                 ← downloaded price CSVs
│   ├── processed/           ← CSVs with indicators added
│   └── labelled/            ← train.csv and test.csv with UP/DOWN labels
│
├── src/
│   ├── data_loader.py       ← downloads OHLCV data from Yahoo Finance
│   ├── feature_engineer.py  ← adds RSI, MA, ATR, Bollinger Bands, etc.
│   ├── labeller.py          ← adds UP/DOWN target column, splits train/test
│   ├── train_model.py       ← trains XGBoost, evaluates, saves model
│   ├── predict.py           ← loads model, outputs live BUY/SELL/FLAT signal
│   ├── backtest.py          ← replays signals on test data, shows P&L
│   └── retrain.py           ← automates the full pipeline end-to-end
│
├── models/
│   ├── eurusd_xgboost.pkl   ← trained model (created after training)
│   ├── eurusd_scaler.pkl    ← data normaliser (created after training)
│   ├── eurusd_feature_cols.pkl
│   ├── eurusd_trades.csv    ← backtest trade log
│   ├── eurusd_backtest.png  ← backtest chart
│   └── retrain_log.json     ← history of retrains
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── 02_model_training.ipynb
```

---

## Understanding the Output

```
╔══════════════════════════════════════════╗
║   FOREX PREDICTOR — EURUSD=X            ║
╠══════════════════════════════════════════╣
║  Time      : 2024-03-15 14:00           ║
║  Close     : 1.08542                    ║
║  Horizon   : 4h ahead                  ║
╠══════════════════════════════════════════╣
║  SIGNAL    :   >>>  BUY  <<<           ║
║  Prob UP   : 71.2%                     ║
║  Prob DOWN : 28.8%                     ║
║  Confidence: 71.2%                     ║
╠══════════════════════════════════════════╣
║  Stop Loss : 1.08312                   ║
║  Take Profit: 1.08982                  ║
║  Max Risk  : 1.0% of account           ║
╚══════════════════════════════════════════╝
```

- **BUY** = model predicts price will be higher in N hours
- **SELL** = model predicts price will be lower in N hours
- **FLAT** = model is not confident enough — do not trade

---

## Important Warnings

1. **Past performance does not guarantee future results.**
2. Always paper trade (test with fake money) before risking real capital.
3. Never risk more than 1–2% of your account on any single trade.
4. The model cannot predict news events. Check economic calendar before trading.
5. Retrain the model every 4–8 weeks to keep it current.

---

## Supported Currency Pairs (Yahoo Finance tickers)

| Pair    | Ticker      |
|---------|-------------|
| EUR/USD | `EURUSD=X`  |
| GBP/USD | `GBPUSD=X`  |
| USD/JPY | `USDJPY=X`  |
| USD/CHF | `USDCHF=X`  |
| AUD/USD | `AUDUSD=X`  |
| USD/CAD | `USDCAD=X`  |
| NZD/USD | `NZDUSD=X`  |
