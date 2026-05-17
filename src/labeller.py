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
from sklearn.model_selection import train_test_split


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Add target column: 1 if close price rises over the next `horizon` candles,
    0 otherwise. Rows where the future is unknown (last N rows) are dropped.
    """
    df = df.copy()

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
                   timeframe: str, test_size: float, random_seed: int):
    """
    Split chronologically (NOT random) to avoid data leakage.
    Earlier rows = train, later rows = test.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

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
    proc_dir  = cfg["paths"]["processed_data"]
    label_dir = cfg["paths"]["labelled_data"]

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
    )
    print("[DONE] labeller complete.")


if __name__ == "__main__":
    main()
