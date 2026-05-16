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


def fetch_recent_data(pair: str, timeframe: str, cfg: dict) -> pd.DataFrame:
    """
    Fetch the most recent candles needed to compute all indicators.
    We need at least `ma_long` + some buffer candles.
    """
    ma_long   = cfg["features"]["ma_long"]
    n_candles = max(200, ma_long * 3)   # enough for all lookback periods

    print(f"[INFO] Fetching last {n_candles} candles for {pair} …")
    df = yf.download(
        tickers  = pair,
        period   = "60d",
        interval = timeframe,
        auto_adjust = True,
        progress = False,
    )

    if df.empty:
        raise ValueError("No live data returned. Check your internet connection.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "datetime"
    df.reset_index(inplace=True)
    df = df[["datetime", "Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    df.dropna(subset=["close"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[INFO] Fetched {len(df):,} candles. Latest: {df['datetime'].iloc[-1]}")
    return df


def predict_signal(df: pd.DataFrame, model, scaler, feature_cols: list,
                   threshold: float) -> dict:
    """
    Run feature engineering on recent data and predict on the LAST row.
    Returns a dict with signal, probability, and ATR for stop sizing.
    """
    # Temporarily suppress the "dropped N rows" print for clean output
    df_feat = add_features(df, load_config())

    if df_feat.empty:
        raise ValueError("Not enough data to compute all features.")

    # Take the last row (most recent candle)
    last_row   = df_feat.iloc[[-1]]
    last_time  = last_row["datetime"].values[0]
    last_close = last_row["close"].values[0]
    last_atr   = last_row["atr"].values[0]

    # Check all required features are present
    missing = [c for c in feature_cols if c not in last_row.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = last_row[feature_cols].values
    X = scaler.transform(X)

    proba_up   = model.predict_proba(X)[0][1]
    proba_down = 1 - proba_up

    if proba_up >= threshold:
        signal = "BUY"
    elif proba_down >= threshold:
        signal = "SELL"
    else:
        signal = "FLAT"

    return {
        "signal":      signal,
        "prob_up":     proba_up,
        "prob_down":   proba_down,
        "confidence":  max(proba_up, proba_down),
        "timestamp":   str(last_time),
        "close":       last_close,
        "atr":         last_atr,
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
    cfg       = load_config()
    pair      = cfg["pair"]
    pair_key  = pair.replace("=X", "").lower()
    timeframe = cfg["timeframe"]
    threshold = cfg["confidence_threshold"]

    model, scaler, feature_cols = load_model_artifacts(
        cfg["paths"]["models"], pair_key
    )

    df = fetch_recent_data(pair, timeframe, cfg)

    result = predict_signal(df, model, scaler, feature_cols, threshold)

    print_signal(result, cfg)
    return result


if __name__ == "__main__":
    main()
