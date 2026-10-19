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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def load_config(path: str = None) -> dict:
    path = project_path(path or "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: {path}")
    return cfg


def add_features(
    df: pd.DataFrame, cfg: dict, df_daily: pd.DataFrame = None,
    daily_context_mode: str = None
) -> pd.DataFrame:
    """Engineer indicators using valid, ordered candles.

    'completed' uses only the prior completed daily bar for intraday context.
    'legacy' preserves existing trained models' old same-day feature contract
    until they are retrained with schema version 2.
    """
    features_cfg = cfg["features"]
    mode = daily_context_mode or features_cfg.get("daily_context_mode", "completed")
    if mode not in ("legacy", "completed"):
        raise ValueError("daily_context_mode must be 'legacy' or 'completed'.")
    required = ["datetime", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"OHLCV input is missing columns: {missing}")
    df = df.copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    df.dropna(subset=["datetime", "open", "high", "low", "close"], inplace=True)
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid OHLCV rows remain for feature engineering.")

    if df_daily is not None:
        df_daily = df_daily.copy()
        if "datetime" not in df_daily.columns or "close" not in df_daily.columns:
            raise ValueError("Daily context requires datetime and close columns.")
        df_daily["datetime"] = pd.to_datetime(
            df_daily["datetime"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        df_daily["close"] = pd.to_numeric(df_daily["close"], errors="coerce")
        df_daily.dropna(subset=["datetime", "close"], inplace=True)
        df_daily = df_daily.sort_values("datetime").drop_duplicates("datetime", keep="last")
        if df_daily.empty:
            raise ValueError("Daily context contains no valid rows.")
        df_daily["daily_sma"] = ta.trend.SMAIndicator(df_daily["close"], window=50).sma_indicator()
        df_daily["daily_rsi"] = ta.momentum.RSIIndicator(df_daily["close"], window=14).rsi()
        if mode == "completed":
            # Daily rows are timestamped at the start of their day. Their
            # indicators become usable after the next UTC midnight. An exact
            # next-calendar-day join fails after weekends/market holidays, so
            # use the latest daily indicator whose completion time is <= row time.
            df_daily["available_at"] = df_daily["datetime"] + pd.Timedelta(days=1)
            left = df.sort_values("datetime").reset_index(drop=True)
            right = df_daily[["available_at", "daily_sma", "daily_rsi"]].sort_values("available_at")
            df = pd.merge_asof(
                left, right, left_on="datetime", right_on="available_at",
                direction="backward", allow_exact_matches=True
            )
            df["above_daily_sma"] = np.where(
                df["daily_sma"].notna(),
                np.where(df["close"] > df["daily_sma"], 1, -1),
                np.nan
            )
            df.drop(columns=["available_at"], inplace=True)
        else:
            # Preserve same-date joins for legacy model artifacts only.
            df_daily["date_key"] = df_daily["datetime"].dt.date
            df["date_key"] = df["datetime"].dt.date
            df = df.merge(
                df_daily[["date_key", "daily_sma", "daily_rsi"]],
                on="date_key", how="left", validate="many_to_one"
            )
            df["above_daily_sma"] = np.where(df["close"] > df["daily_sma"], 1, -1)
            df.drop(columns=["date_key"], inplace=True)

    if len(df) < features_cfg["ma_long"] + 10:
        raise ValueError(f"Not enough rows ({len(df)}) for indicators.")
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    df["rsi"] = ta.momentum.RSIIndicator(close, window=features_cfg["rsi_period"]).rsi()
    df["ma_short"] = ta.trend.SMAIndicator(close, window=features_cfg["ma_short"]).sma_indicator()
    df["ma_long"] = ta.trend.SMAIndicator(close, window=features_cfg["ma_long"]).sma_indicator()
    df["ma_cross"] = np.where(df["ma_short"] > df["ma_long"], 1, -1)
    df["ema_short"] = ta.trend.EMAIndicator(close, window=features_cfg["ma_short"]).ema_indicator()
    df["ema_long"] = ta.trend.EMAIndicator(close, window=features_cfg["ma_long"]).ema_indicator()

    macd_obj = ta.trend.MACD(close)
    df["macd"] = macd_obj.macd()
    df["macd_sig"] = macd_obj.macd_signal()
    df["macd_diff"] = macd_obj.macd_diff()

    bb = ta.volatility.BollingerBands(
        close, window=features_cfg["bb_period"], window_dev=features_cfg["bb_std"]
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (close + 1e-10)
    df["bb_pct"] = bb.bollinger_pband()

    df["atr"] = ta.volatility.AverageTrueRange(
        high, low, close, window=features_cfg["atr_period"]
    ).average_true_range()
    df["atr_pct"] = df["atr"] / (close + 1e-10)
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["vol_ma"] = vol.rolling(window=20).mean()
    df["vol_ratio"] = vol / (df["vol_ma"] + 1e-10)

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_pct"] = df["body"] / (candle_range + 1e-10)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_dir"] = np.where(df["close"] >= df["open"], 1, -1)

    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    df["datetime"] = dt
    df["hour"], df["dow"] = dt.dt.hour, dt.dt.dayofweek
    for lag in [1, 2, 3]:
        df[f"close_lag{lag}"] = close.shift(lag)
        df[f"rsi_lag{lag}"] = df["rsi"].shift(lag)

    df["feature_schema_version"] = 1 if mode == "legacy" else 2
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise ValueError("No rows remain after removing incomplete/non-finite features.")
    print(f"[INFO] Features added. Rows: {len(df):,} (schema v{int(df['feature_schema_version'].iloc[0])})")
    return df


def main():
    cfg       = load_config()
    pair      = cfg["pair"].replace("=X", "").lower()
    timeframe = cfg["timeframe"]
    raw_dir   = project_path(cfg["paths"]["raw_data"])
    proc_dir  = project_path(cfg["paths"]["processed_data"])

    raw_file = os.path.join(raw_dir, f"{pair}_{timeframe}.csv")
    df = pd.read_csv(raw_file, parse_dates=["datetime"])

    # Try to load daily data for trend context
    df_daily = None
    daily_file = os.path.join(raw_dir, f"{pair}_1d.csv")
    if timeframe != "1d" and os.path.exists(daily_file):
        print(f"[INFO] Merging Daily context from {daily_file}")
        df_daily = pd.read_csv(daily_file, parse_dates=["datetime"])

    df = add_features(df, cfg, df_daily, daily_context_mode=cfg["features"].get("daily_context_mode", "completed"))

    os.makedirs(proc_dir, exist_ok=True)
    out_file = os.path.join(proc_dir, f"{pair}_{timeframe}_features.csv")
    df.to_csv(out_file, index=False)
    print(f"[INFO] Saved processed data -> {out_file}")

    print("[DONE] feature_engineer complete.")


if __name__ == "__main__":
    main()
