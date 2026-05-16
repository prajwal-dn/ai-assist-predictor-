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


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_data(pair: str, years: int, timeframe: str) -> pd.DataFrame:
    """
    Download OHLCV candles for `pair` going back `years` years.
    Yahoo Finance supports 1h data only for the last 730 days.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    if timeframe == "1h":
        max_start = end_date - timedelta(days=729)
        if start_date < max_start:
            print("[INFO] 1h data limited to 2 years on Yahoo Finance. Adjusting start date.")
            start_date = max_start

    print(f"[INFO] Downloading {pair} | {timeframe} | {start_date.date()} to {end_date.date()}")

    df = yf.download(
        tickers=pair,
        start=start_date,
        end=end_date,
        interval=timeframe,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,   # yfinance 0.2.38+ flat columns
    )

    if df.empty:
        raise ValueError(f"No data returned for {pair}. Check the ticker symbol.")

    # Flatten MultiIndex columns just in case (older yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Lowercase all column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Move the DatetimeIndex into a regular column called 'datetime'
    df.index.name = "datetime"
    df.reset_index(inplace=True)

    # yfinance sometimes names it 'date' instead of 'datetime'
    if "datetime" not in df.columns:
        for candidate in ["date", "timestamp"]:
            if candidate in df.columns:
                df.rename(columns={candidate: "datetime"}, inplace=True)
                break

    print(f"[DEBUG] Columns after download: {list(df.columns)}")

    # Ensure we have all required columns
    needed = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Available: {list(df.columns)}\n"
            "Try updating yfinance: pip install --upgrade yfinance"
        )

    df = df[needed].copy()

    # Remove any non-numeric rows (yfinance 0.2+ sometimes injects a 'Ticker' row)
    df = df[pd.to_numeric(df["close"], errors="coerce").notna()].copy()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[INFO] Downloaded {len(df):,} candles.")
    return df


def save_data(df: pd.DataFrame, output_dir: str, pair: str, timeframe: str) -> str:
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

    df = download_data(pair, years, timeframe)
    save_data(df, output_dir, pair, timeframe)
    print("[DONE] data_loader complete.")


if __name__ == "__main__":
    main()
