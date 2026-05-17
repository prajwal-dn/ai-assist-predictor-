# Forex Predictor PRO 📈

<div align="center">
  <p><strong>An advanced, AI-powered trading workstation for the Foreign Exchange market.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.14-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/XGBoost-2.0-orange.svg" alt="XGBoost">
    <img src="https://img.shields.io/badge/Scikit--Learn-1.4-yellow.svg" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/UI-Glassmorphism-brightgreen.svg" alt="UI">
  </p>
</div>

---

## 🚀 Overview

**Forex Predictor PRO** is a professional-grade quantitative trading tool. It leverages state-of-the-art machine learning algorithms and technical analysis to predict short-term price movements in Forex markets. 

By automating the complex analysis of historical data, technical indicators, and multi-timeframe trends, it provides traders with clear, actionable signals (**BUY**, **SELL**, or **FLAT**) alongside calculated risk management protocols.

## ✨ Key Features

### 🧠 Intelligent AI Engine
*   **XGBoost Classifier:** Utilizes Gradient Boosting to identify complex, non-linear patterns in market volatility and momentum.
*   **Hyperparameter Auto-Tuning:** Features built-in `RandomizedSearchCV` to automatically discover the absolute optimal mathematical configuration for any specific currency pair.
*   **Multi-Timeframe Confluence:** The engine cross-references the primary trading timeframe (e.g., 4H) against the macro trend (Daily timeframe) to aggressively filter out false breakouts ("fakeouts").

### 📊 Advanced Quantitative Pipeline
*   **Automated Data Ingestion:** Seamlessly fetches years of high-fidelity OHLCV data via the Yahoo Finance API.
*   **Extensive Feature Engineering:** Calculates over 20 unique technical data points per candle, including:
    *   *Momentum:* RSI, MACD, Stochastic Oscillator
    *   *Trend:* Short & Long Moving Averages (SMA/EMA)
    *   *Volatility:* Bollinger Bands, Average True Range (ATR)
    *   *Price Action:* Candle body percentage, wick length, and volume analysis.

### 💻 Professional Trading Dashboard
*   **Live Interactive Charting:** Embedded real-time TradingView widget for simultaneous price action analysis.
*   **Premium Glassmorphism UI:** A sleek, dark-mode aesthetic with live status indicators and dynamic probability gauges.
*   **Automated Risk Management:** Calculates dynamic **Stop Loss** and **Take Profit** levels based on current market volatility (ATR), enforcing strict risk discipline.

---

## 🛠️ Installation & Setup

### 1. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Configure Your Strategy
Edit `config.yaml` to define your trading parameters:
*   `pair`: The currency pair to trade (e.g., `EURUSD=X`).
*   `timeframe`: Candle duration (e.g., `1h`, `4h`).
*   `confidence_threshold`: Minimum probability required to trigger a trade (e.g., `0.58` = 58%).

### 3. Initialize the AI (First Run)
Run the automated pipeline to download data, engineer features, and train the optimized model:
```bash
python src/retrain.py
```

### 4. Launch the Dashboard
Start the local web server to access the trading interface:
```bash
python app.py
```
*Then open [http://localhost:5000](http://localhost:5000) in your browser.*

---

## 💡 How to Use the Signals

When a prediction is run, the model returns one of three outcomes:

*   🟢 **BUY:** The model detects a high-probability bullish setup. Look to enter a long position using the provided Stop Loss.
*   🔴 **SELL:** The model detects a high-probability bearish setup. Look to enter a short position.
*   🟡 **FLAT:** The market lacks a clear statistical edge or the confidence is below your threshold. Capital preservation is prioritized; do not trade.

---

## ⚠️ Disclaimer
**This software is for educational and research purposes only.** Foreign exchange trading carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results. The authors and contributors are not responsible for any financial losses incurred while using this tool. Always practice strict risk management and test strategies on a demo account before risking real capital.
