"""
labeller.py
-----------
Reads feature-engineered data and adds a binary target column:
  1 = price is HIGHER than current close after N hours  (BUY signal)
  0 = price is LOWER or equal after N hours             (SELL signal)

Also splits data into train and test sets and saves them separately.

Usage:
    python src/labeller.py
"""

import os
import yaml
import pandas as pd
import numpy as np
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


def add_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Add target column: 1 if close price rises over the next `horizon` candles,
    0 otherwise. Rows where the future is unknown (last N rows) are dropped.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("forecast_horizon must be a positive integer number of candles.")
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime", keep="last")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) <= horizon:
        raise ValueError(f"Need more than {horizon} valid rows to create labels.")

    # Future close price (N candles ahead)
    df["future_close"] = df["close"].shift(-horizon)

    # Label: 1 = up, 0 = down/flat
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    # Drop rows where we don't know the future yet
    df.dropna(subset=["future_close"], inplace=True)
    df.drop(columns=["future_close"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    up_pct = df["target"].mean() * 100
    print(f"[INFO] Labels added. UP: {up_pct:.1f}%  |  DOWN: {100 - up_pct:.1f}%")
    return df


def split_and_save(df: pd.DataFrame, output_dir: str, pair: str,
                   timeframe: str, test_size: float, random_seed: int, gap: int = 0):
    """
    Split chronologically (NOT random) to avoid data leakage.
    Earlier rows = train, later rows = test.
    """
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not isinstance(gap, int) or gap < 0:
        raise ValueError("gap must be a non-negative integer.")
    split_idx = int(len(df) * (1 - test_size))
    train_end = split_idx - gap
    if train_end < 1 or split_idx >= len(df):
        raise ValueError("Not enough labelled rows for the requested test_size and purge gap.")
    # Purge the final horizon rows from training so their future-price labels
    # cannot overlap the beginning of the holdout set.
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[split_idx:].copy()

    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, f"{pair}_{timeframe}_train.csv")
    test_file  = os.path.join(output_dir, f"{pair}_{timeframe}_test.csv")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"[INFO] Train set: {len(train_df):,} rows -> {train_file}")
    print(f"[INFO] Test  set: {len(test_df):,} rows -> {test_file}")
    return train_df, test_df


def main():
    cfg       = load_config()
    pair      = cfg["pair"].replace("=X", "").lower()
    timeframe = cfg["timeframe"]
    horizon   = cfg["forecast_horizon"]
    proc_dir  = project_path(cfg["paths"]["processed_data"])
    label_dir = project_path(cfg["paths"]["labelled_data"])

    features_file = os.path.join(proc_dir, f"{pair}_{timeframe}_features.csv")
    if not os.path.exists(features_file):
        raise FileNotFoundError(
            f"Features file not found at {features_file}. "
            "Run feature_engineer.py first."
        )

    print(f"[INFO] Loading features from {features_file}")
    df = pd.read_csv(features_file, parse_dates=["datetime"])
    print(f"[INFO] Loaded {len(df):,} rows.")

    if len(df) == 0:
        raise ValueError(
            f"features.csv is empty! Re-run feature_engineer.py and check for errors."
        )

    df = add_labels(df, horizon)

    split_and_save(
        df,
        output_dir  = label_dir,
        pair        = pair,
        timeframe   = timeframe,
        test_size   = cfg["test_size"],
        random_seed = cfg["random_seed"],
        gap         = int(horizon),
    )
    print("[DONE] labeller complete.")


if __name__ == "__main__":
    main()
