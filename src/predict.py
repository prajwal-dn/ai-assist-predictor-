"""
predict.py
----------
Loads the saved model and produces a trading signal for the LATEST
available candle. Outputs: BUY, SELL, or FLAT (not confident enough).

Usage:
    python src/predict.py
"""

import os
import yaml
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Import our own modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineer import add_features


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_artifacts(model_dir: str, pair: str):
    model_path  = os.path.join(model_dir, f"{pair}_xgboost.pkl")
    scaler_path = os.path.join(model_dir, f"{pair}_scaler.pkl")
    cols_path   = os.path.join(model_dir, f"{pair}_feature_cols.pkl")

    for p in [model_path, scaler_path, cols_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model artifact not found: {p}\n"
                "Run train_model.py first."
            )

    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    feature_cols = joblib.load(cols_path)

    print(f"[INFO] Loaded model from {model_path}")
    return model, scaler, feature_cols


import requests

def fetch_recent_data(pair: str, timeframe: str, cfg: dict) -> pd.DataFrame:
    """Fetch candles. Uses TwelveData if API key is provided, else falls back to yfinance."""
    print(f"[INFO] Fetching {timeframe} data for {pair}...")
    
    api_key = cfg.get("twelvedata_api_key", "")
    
    if api_key:
        # Use TwelveData (Professional Permanent Fix)
        # Convert Yahoo pair (EURUSD=X) to TwelveData format (EUR/USD)
        td_pair = pair.replace("=X", "")
        if len(td_pair) == 6: td_pair = f"{td_pair[:3]}/{td_pair[3:]}"
        
        url = f"https://api.twelvedata.com/time_series?symbol={td_pair}&interval={timeframe}&outputsize=200&apikey={api_key}"
        res = requests.get(url).json()
        
        if "values" not in res:
            raise ValueError(f"TwelveData API Error: {res.get('message', 'Unknown error')}")
            
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # TwelveData returns newest to oldest; we need oldest to newest
        df = df.iloc[::-1].reset_index(drop=True)
        return df

    # Fallback to yfinance (Free but heavily rate-limited by Hugging Face IPs)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    df = yf.download(
        tickers=pair, period="60d", interval=timeframe,
        auto_adjust=True, progress=False, session=session
    )
    if df.empty: raise ValueError(f"No {timeframe} data returned from Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index.name = "datetime"; df.reset_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def predict_signal(df: pd.DataFrame, df_daily: pd.DataFrame, model, scaler, feature_cols: list,
                   threshold: float) -> dict:
    """Run feature engineering and predict on the LAST row."""
    df_feat = add_features(df, load_config(), df_daily)
    if df_feat.empty: raise ValueError("Not enough data for features.")

    last_row   = df_feat.iloc[[-1]]
    last_time  = last_row["datetime"].values[0]
    last_close = last_row["close"].values[0]
    last_atr   = last_row["atr"].values[0]

    X = last_row[feature_cols].values
    X = scaler.transform(X)

    proba_up   = model.predict_proba(X)[0][1]
    proba_down = 1 - proba_up

    if proba_up >= threshold: signal = "BUY"
    elif proba_down >= threshold: signal = "SELL"
    else: signal = "FLAT"

    return {
        "signal": signal, "prob_up": proba_up, "prob_down": proba_down,
        "confidence": max(proba_up, proba_down), "timestamp": str(last_time),
        "close": last_close, "atr": last_atr,
    }


def print_signal(result: dict, cfg: dict):
    """Pretty-print the prediction and risk levels."""
    atr_mult  = cfg["risk"]["atr_stop_multiplier"]
    risk_pct  = cfg["risk"]["max_risk_per_trade"] * 100
    pair      = cfg["pair"]
    horizon   = cfg["forecast_horizon"]

    stop_size = result["atr"] * atr_mult

    print("\n╔══════════════════════════════════════════╗")
    print(f"║   FOREX PREDICTOR — {pair:<21}║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Time      : {result['timestamp'][:16]:<28}║")
    print(f"║  Close     : {result['close']:.5f}{'':<26}║")
    print(f"║  Horizon   : {horizon}h ahead{'':<30}║")
    print("╠══════════════════════════════════════════╣")

    signal_display = f"  >>>  {result['signal']}  <<<"
    print(f"║  SIGNAL    :{signal_display:<30}║")
    print(f"║  Prob UP   : {result['prob_up']:.1%}{'':<28}║")
    print(f"║  Prob DOWN : {result['prob_down']:.1%}{'':<28}║")
    print(f"║  Confidence: {result['confidence']:.1%}{'':<28}║")
    print("╠══════════════════════════════════════════╣")

    if result["signal"] != "FLAT":
        if result["signal"] == "BUY":
            sl = result["close"] - stop_size
            tp = result["close"] + stop_size * 2
        else:
            sl = result["close"] + stop_size
            tp = result["close"] - stop_size * 2

        print(f"║  Stop Loss : {sl:.5f}{'':<26}║")
        print(f"║  Take Profit: {tp:.5f}{'':<25}║")
        print(f"║  Max Risk  : {risk_pct}% of account{'':<19}║")
    else:
        print(f"║  No trade — confidence below threshold  ║")

    print("╚══════════════════════════════════════════╝\n")


def main():
    cfg = load_config()
    pair = cfg["pair"]
    pair_key = pair.replace("=X", "").lower()
    timeframe = cfg["timeframe"]

    model, scaler, feature_cols = load_model_artifacts(
        cfg["paths"]["models"], pair_key
    )

    # Fetch primary timeframe
    df = fetch_recent_data(pair, timeframe, cfg)
    
    # Fetch Daily trend context
    df_daily = None
    if timeframe != "1d":
        try:
            df_daily = fetch_recent_data(pair, "1d", cfg)
        except Exception as e:
            print(f"[WARNING] Could not fetch daily trend: {e}")

    result = predict_signal(df, df_daily, model, scaler, feature_cols, cfg["confidence_threshold"])

    print_signal(result, cfg)
    return result


if __name__ == "__main__":
    main()
