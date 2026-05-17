"""
feature_engineer.py
--------------------
Reads raw OHLCV data and adds technical indicator columns
(RSI, moving averages, ATR, Bollinger Bands, etc.).

Usage:
    python src/feature_engineer.py
"""

import os
import yaml
import pandas as pd
import numpy as np
import ta


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_features(df: pd.DataFrame, cfg: dict, df_daily: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add all technical indicator features to the dataframe.
    """
    f = cfg["features"]
    df = df.copy()

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Parent Timeframe (Daily) Trend ---
    if df_daily is not None:
        df_daily = df_daily.copy()
        df_daily["daily_sma"] = ta.trend.SMAIndicator(df_daily["close"], window=50).sma_indicator()
        df_daily["daily_rsi"] = ta.momentum.RSIIndicator(df_daily["close"], window=14).rsi()
        
        # We only need the date part to merge daily data into intraday data
        df_daily["date_key"] = pd.to_datetime(df_daily["datetime"]).dt.date
        df["date_key"]       = pd.to_datetime(df["datetime"]).dt.date
        
        # Merge columns
        df = df.merge(df_daily[["date_key", "daily_sma", "daily_rsi"]], on="date_key", how="left")
        
        # Signal: Is intraday price above/below Daily SMA?
        df["above_daily_sma"] = np.where(df["close"] > df["daily_sma"], 1, -1)
        df.drop(columns=["date_key"], inplace=True)

    if len(df) < f["ma_long"] + 10:
        raise ValueError(f"Not enough rows ({len(df)}) for indicators.")

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ... [rest of the existing indicators] ...
    # (I'll keep the core logic but ensure the return is clean)
    # [Note: The model will now have 'daily_sma', 'daily_rsi', 'above_daily_sma' as features]

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close, window=f["rsi_period"]).rsi()

    # Moving averages
    df["ma_short"] = ta.trend.SMAIndicator(close, window=f["ma_short"]).sma_indicator()
    df["ma_long"]  = ta.trend.SMAIndicator(close, window=f["ma_long"]).sma_indicator()
    df["ma_cross"] = np.where(df["ma_short"] > df["ma_long"], 1, -1)

    df["ema_short"] = ta.trend.EMAIndicator(close, window=f["ma_short"]).ema_indicator()
    df["ema_long"]  = ta.trend.EMAIndicator(close, window=f["ma_long"]).ema_indicator()

    # MACD
    macd_obj        = ta.trend.MACD(close)
    df["macd"]      = macd_obj.macd()
    df["macd_sig"]  = macd_obj.macd_signal()
    df["macd_diff"] = macd_obj.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=f["bb_period"], window_dev=f["bb_std"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (close + 1e-10)
    df["bb_pct"]   = bb.bollinger_pband()

    # ATR
    df["atr"]     = ta.volatility.AverageTrueRange(high, low, close, window=f["atr_period"]).average_true_range()
    df["atr_pct"] = df["atr"] / (close + 1e-10)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Volume
    df["vol_ma"]    = vol.rolling(window=20).mean()
    df["vol_ratio"] = vol / (df["vol_ma"] + 1e-10)

    # Candle shape
    df["body"]       = (df["close"] - df["open"]).abs()
    df["body_pct"]   = df["body"] / (df["high"] - df["low"] + 1e-10)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_dir"] = np.where(df["close"] >= df["open"], 1, -1)

    # Time features
    dt = pd.to_datetime(df["datetime"])
    if dt.dt.tz is not None: dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    df["datetime"] = dt
    df["hour"] = dt.dt.hour
    df["dow"]  = dt.dt.dayofweek

    # Lag features
    for lag in [1, 2, 3]:
        df[f"close_lag{lag}"] = close.shift(lag)
        df[f"rsi_lag{lag}"]   = df["rsi"].shift(lag)

    initial_rows = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Features added. Rows: {len(df):,}")
    return df


def main():
    cfg       = load_config()
    pair      = cfg["pair"].replace("=X", "").lower()
    timeframe = cfg["timeframe"]
    raw_dir   = cfg["paths"]["raw_data"]
    proc_dir  = cfg["paths"]["processed_data"]

    raw_file = os.path.join(raw_dir, f"{pair}_{timeframe}.csv")
    df = pd.read_csv(raw_file, parse_dates=["datetime"])

    # Try to load daily data for trend context
    df_daily = None
    daily_file = os.path.join(raw_dir, f"{pair}_1d.csv")
    if timeframe != "1d" and os.path.exists(daily_file):
        print(f"[INFO] Merging Daily context from {daily_file}")
        df_daily = pd.read_csv(daily_file, parse_dates=["datetime"])

    df = add_features(df, cfg, df_daily)

    os.makedirs(proc_dir, exist_ok=True)
    out_file = os.path.join(proc_dir, f"{pair}_{timeframe}_features.csv")
    df.to_csv(out_file, index=False)
    print(f"[INFO] Saved processed data -> {out_file}")

    print("[DONE] feature_engineer complete.")


if __name__ == "__main__":
    main()
