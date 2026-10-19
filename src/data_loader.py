"""
data_loader.py
--------------
Downloads historical OHLCV (Open, High, Low, Close, Volume) data
for a Forex pair from Yahoo Finance and saves it as a CSV.

Usage:
    python src/data_loader.py
"""

import os
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

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


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.set_index("datetime").sort_index()
    rule = "4h"
    bars = indexed.resample(rule, origin="start_day", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    })
    counts = indexed["close"].resample(rule, origin="start_day", closed="left", label="left").count()
    return bars.loc[counts >= 4].reset_index()


def download_data(pair: str, years: int, timeframe: str) -> pd.DataFrame:
    """Download normalized OHLCV; 4h bars are aggregated from complete hourly buckets."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    download_interval = "1h" if timeframe == "4h" else timeframe
    if download_interval == "1h":
        max_start = end_date - timedelta(days=729)
        if start_date < max_start:
            print("[INFO] Hourly data limited to two years on Yahoo Finance. Adjusting start date.")
            start_date = max_start
    print(f"[INFO] Downloading {pair} | {timeframe} | {start_date.date()} to {end_date.date()}")
    df = yf.download(
        tickers=pair, start=start_date, end=end_date, interval=download_interval,
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {pair}. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.index.name = "datetime"
    df.reset_index(inplace=True)
    if "datetime" not in df.columns:
        for candidate in ("date", "timestamp"):
            if candidate in df.columns:
                df.rename(columns={candidate: "datetime"}, inplace=True)
                break
    needed = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns {missing}. Available: {list(df.columns)}")
    df = df[needed].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    valid = df["datetime"].notna() & df[["open", "high", "low", "close"]].notna().all(axis=1)
    valid &= (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= df["high"] >= df[["open", "close"]].max(axis=1)
    valid &= df["low"] <= df[["open", "close"]].min(axis=1)
    df = df.loc[valid].sort_values("datetime").drop_duplicates("datetime", keep="last")
    df.reset_index(drop=True, inplace=True)
    if timeframe == "4h":
        df = _resample_4h(df)
    if df.empty:
        raise ValueError(f"No valid candles remained for {pair} at {timeframe}.")
    print(f"[INFO] Downloaded {len(df):,} normalized candles.")
    return df


def save_data(df: pd.DataFrame, output_dir: str, pair: str, timeframe: str) -> str:
    output_dir = project_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{pair.replace('=X','').lower()}_{timeframe}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[INFO] Saved raw data -> {filepath}")
    return filepath


def main():
    cfg        = load_config()
    pair       = cfg["pair"]
    years      = cfg["history_years"]
    timeframe  = cfg["timeframe"]
    output_dir = cfg["paths"]["raw_data"]

    # Download primary timeframe
    df = download_data(pair, years, timeframe)
    save_data(df, output_dir, pair, timeframe)

    # Download DAILY timeframe for trend confirmation (if not already 1d)
    if timeframe != "1d":
        print(f"\n[INFO] Downloading Daily (1d) data for trend context...")
        df_daily = download_data(pair, years, "1d")
        save_data(df_daily, output_dir, pair, "1d")

    print("[DONE] data_loader complete.")


if __name__ == "__main__":
    main()
