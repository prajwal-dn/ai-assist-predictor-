"""
predict.py
----------
Loads the saved model and produces a trading signal for the LATEST
available candle. Outputs: BUY, SELL, or FLAT (not confident enough).

Usage:
    python src/predict.py
"""

import os
import json
import math
import re
import yaml
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Import our own modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineer import add_features


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(BASE_DIR, "config.yaml")
    elif not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid application configuration at {path}")
    pair_override = os.environ.get("ASSET_PREDICTOR_PAIR")
    if pair_override:
        pair_override = pair_override.strip().upper()
        if len(pair_override) == 6 and "-" not in pair_override and not pair_override.endswith("=X"):
            pair_override += "=X"
        if not re.fullmatch(r"[A-Z0-9-]{3,12}(?:=X)?", pair_override):
            raise ValueError("ASSET_PREDICTOR_PAIR is not a valid market symbol.")
        cfg["pair"] = pair_override
    if str(cfg.get("timeframe", "")).lower() not in ("1h", "4h", "1d"):
        raise ValueError("timeframe must be one of: 1h, 4h, 1d.")
    if isinstance(cfg.get("forecast_horizon"), bool) or not isinstance(cfg.get("forecast_horizon"), int) or cfg["forecast_horizon"] < 1:
        raise ValueError("forecast_horizon must be a positive integer number of candles.")
    threshold = float(cfg.get("confidence_threshold", 0.0))
    if not math.isfinite(threshold) or not 0.5 < threshold < 1:
        raise ValueError("confidence_threshold must be greater than 0.5 and less than 1.")
    return cfg


def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(256).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def load_model_artifacts(model_dir: str, pair: str, timeframe: str = None):
    model_path = os.path.join(model_dir, f"{pair}_xgboost.pkl")
    scaler_path = os.path.join(model_dir, f"{pair}_scaler.pkl")
    cols_path = os.path.join(model_dir, f"{pair}_feature_cols.pkl")
    metadata_path = os.path.join(model_dir, f"{pair}_metadata.json")

    for path in (model_path, scaler_path, cols_path):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f"Missing or empty model artifact: {path}")
        if _is_lfs_pointer(path):
            raise ValueError(f"{path} is a Git LFS pointer, not the model binary. Fetch the real LFS object.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(cols_path)
    if not isinstance(feature_cols, (list, tuple)) or not feature_cols:
        raise ValueError("Feature-column artifact must be a non-empty list.")
    feature_cols = list(feature_cols)
    if any(not isinstance(col, str) or not col for col in feature_cols):
        raise ValueError("Feature-column artifact contains an invalid column name.")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("Feature-column artifact contains duplicate column names.")
    for name, artifact in (("scaler", scaler), ("model", model)):
        n_features = getattr(artifact, "n_features_in_", None)
        if n_features is not None and int(n_features) != len(feature_cols):
            raise ValueError(f"{name} expects {n_features} features but the artifact lists {len(feature_cols)}.")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Model artifact does not implement predict_proba().")

    # Existing model triplets have no metadata. Preserve their legacy feature
    # contract until retraining writes a versioned sidecar.
    metadata = {
        "feature_schema_version": 1, "daily_context_mode": "legacy",
        "timeframe": None, "legacy_artifact": True,
    }
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            loaded_metadata = json.load(f)
        if not isinstance(loaded_metadata, dict):
            raise ValueError(f"Invalid model metadata: {metadata_path}")
        metadata.update(loaded_metadata)
        metadata["legacy_artifact"] = False
        saved_columns = metadata.get("feature_columns")
        if saved_columns is not None and list(saved_columns) != feature_cols:
            raise ValueError("Model metadata feature_columns do not match the feature-column artifact.")
        if timeframe and metadata.get("timeframe") and metadata["timeframe"] != timeframe:
            raise ValueError(
                f"Model was trained for timeframe {metadata['timeframe']!r}, "
                f"but config requests {timeframe!r}. Retrain this pair for the configured timeframe."
            )
        if metadata.get("daily_context_mode") not in ("legacy", "completed"):
            raise ValueError("Model metadata specifies an unsupported daily_context_mode.")
    print(f"[INFO] Loaded model from {model_path}")
    return model, scaler, feature_cols, metadata


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider candles to sorted UTC-naive OHLCV rows."""
    if df is None or df.empty:
        raise ValueError("Market-data provider returned no candles.")
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    lower_columns = [str(c).strip().lower() for c in df.columns]
    if "datetime" not in lower_columns and isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "datetime"
        df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "datetime" not in df.columns:
        for candidate in ("date", "timestamp", "index"):
            if candidate in df.columns:
                df.rename(columns={candidate: "datetime"}, inplace=True)
                break
    if "volume" not in df.columns:
        df["volume"] = 0
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Market data is missing columns: {missing}")
    df = df[required].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    valid = df[["open", "high", "low", "close"]].apply(
        lambda column: np.isfinite(column.to_numpy(dtype=float))
    ).all(axis=1)
    valid &= df["datetime"].notna()
    valid &= (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= df["high"] >= df[["open", "close"]].max(axis=1)
    valid &= df["low"] <= df[["open", "close"]].min(axis=1)
    df = df.loc[valid].sort_values("datetime").drop_duplicates("datetime", keep="last")
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise ValueError("No valid OHLCV candles remained after data validation.")
    return df


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Build completed 4-hour candles from hourly Yahoo Finance candles."""
    indexed = df.set_index("datetime").sort_index()
    rule = "4h"
    bars = indexed.resample(rule, origin="start_day", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    })
    counts = indexed["close"].resample(rule, origin="start_day", closed="left", label="left").count()
    bars = bars.loc[counts >= 4].reset_index()
    return _normalize_ohlcv(bars)



def _drop_incomplete_candles(
    df: pd.DataFrame, timeframe: str, now: pd.Timestamp = None
) -> pd.DataFrame:
    """Exclude the currently forming bar so live inference only sees completed candles."""
    duration_hours = {"1h": 1, "4h": 4, "1d": 24}
    if timeframe not in duration_hours:
        raise ValueError(f"Unsupported prediction timeframe: {timeframe}")
    if df is None or df.empty:
        raise ValueError("No candles available to filter.")
    if now is None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)
    cutoff = now - pd.Timedelta(hours=duration_hours[timeframe])
    completed = df.loc[pd.to_datetime(df["datetime"], errors="coerce") <= cutoff].copy()
    completed.reset_index(drop=True, inplace=True)
    if completed.empty:
        raise ValueError(f"No completed {timeframe} candles are available yet.")
    return completed


def fetch_yfinance_data(pair: str, timeframe: str) -> pd.DataFrame:
    """Fetch enough history for the requested timeframe and rolling features."""
    print(f"[WARNING] Falling back to yfinance for {timeframe} data.")
    download_interval = "1h" if timeframe == "4h" else timeframe
    period = "1y" if timeframe == "1d" else "60d"
    df = yf.download(
        tickers=pair, period=period, interval=download_interval,
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    df = _normalize_ohlcv(df)
    if timeframe == "4h":
        df = _resample_4h(df)
    return _drop_incomplete_candles(df, timeframe)


def fetch_recent_data(pair: str, timeframe: str, cfg: dict) -> pd.DataFrame:
    """Fetch validated UTC candles from TwelveData, falling back to Yahoo safely."""
    print(f"[INFO] Fetching {timeframe} data for {pair}...")
    api_key = os.environ.get("TWELVEDATA_API_KEY") or cfg.get("twelvedata_api_key", "")
    if not api_key:
        print("[WARNING] TWELVEDATA_API_KEY not set.")
        return fetch_yfinance_data(pair, timeframe)

    td_pair = pair.replace("=X", "").replace("-", "/")
    if "/" not in td_pair and len(td_pair) == 6:
        td_pair = f"{td_pair[:3]}/{td_pair[3:]}"
    td_interval_map = {"1d": "1day", "1wk": "1week", "1mo": "1month"}
    params = {
        "symbol": td_pair, "interval": td_interval_map.get(timeframe, timeframe),
        "outputsize": 200, "apikey": api_key, "timezone": "UTC",
    }
    try:
        response = requests.get(
            "https://api.twelvedata.com/time_series", params=params, timeout=(5, 15)
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "values" not in payload:
            message = payload.get("message", "Malformed TwelveData response") if isinstance(payload, dict) else "Malformed TwelveData response"
            print(f"[WARNING] TwelveData returned no candles ({message}); falling back to yfinance.")
            return fetch_yfinance_data(pair, timeframe)
        candles = _normalize_ohlcv(pd.DataFrame(payload["values"]))
        return _drop_incomplete_candles(candles, timeframe)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"[WARNING] TwelveData request failed ({type(exc).__name__}); falling back to yfinance.")
        return fetch_yfinance_data(pair, timeframe)


def predict_signal(
    df: pd.DataFrame, df_daily: pd.DataFrame, model, scaler, feature_cols: list,
    threshold: float, daily_context_mode: str = "legacy"
) -> dict:
    """Engineer valid candles, then predict on the last feature row."""
    if not math.isfinite(float(threshold)) or not 0.5 < float(threshold) < 1.0:
        raise ValueError("confidence_threshold must be greater than 0.5 and less than 1.")
    cfg = load_config()
    df_feat = add_features(df, cfg, df_daily, daily_context_mode=daily_context_mode)
    if df_feat.empty:
        raise ValueError("Not enough valid data to calculate model features.")
    missing_features = [col for col in feature_cols if col not in df_feat.columns]
    if missing_features:
        raise ValueError(
            "Live features do not match the trained model. "
            f"Missing: {missing_features}. Check data-provider history and model metadata."
        )

    last_row = df_feat.iloc[[-1]]
    last_time = last_row["datetime"].iloc[0]
    last_close = float(last_row["close"].iloc[0])
    last_atr = float(last_row["atr"].iloc[0])
    X = last_row[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if X.shape != (1, len(feature_cols)) or not np.isfinite(X).all():
        raise ValueError("The latest candle contains missing or non-finite model features.")
    if not math.isfinite(last_close) or last_close <= 0:
        raise ValueError("The latest close price is invalid.")
    if not math.isfinite(last_atr) or last_atr <= 0:
        raise ValueError("ATR is zero or invalid; risk levels cannot be calculated.")

    expected_features = getattr(scaler, "n_features_in_", None)
    if expected_features is not None and int(expected_features) != X.shape[1]:
        raise ValueError(f"Scaler expects {expected_features} features, received {X.shape[1]}.")
    X_scaled = scaler.transform(X)
    probabilities = np.asarray(model.predict_proba(X_scaled), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    up_indices = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1 or len(up_indices) != 1:
        raise ValueError("Model must return one probability row with a class labelled 1 (UP).")
    proba_up = float(probabilities[0, int(up_indices[0])])
    if not math.isfinite(proba_up) or not 0.0 <= proba_up <= 1.0:
        raise ValueError("Model returned an invalid UP probability.")
    proba_down = 1.0 - proba_up
    if proba_up >= threshold:
        signal = "BUY"
    elif proba_down >= threshold:
        signal = "SELL"
    else:
        signal = "FLAT"
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

    print("\n==============================================")
    print(f"ASSET PREDICTOR - {pair}")
    print("----------------------------------------------")
    print(f"Time      : {result['timestamp'][:16]}")
    print(f"Close     : {result['close']:.5f}")
    timeframe = cfg.get("timeframe", "1h")
    if timeframe == "1h":
        horizon_label = f"{horizon} hour(s)"
    elif timeframe == "4h":
        horizon_label = f"{horizon * 4} hour(s)"
    elif timeframe == "1d":
        horizon_label = f"{horizon} day(s)"
    else:
        horizon_label = f"{horizon} {timeframe} candle(s)"
    print(f"Horizon   : {horizon_label} ahead")
    print("----------------------------------------------")

    print(f"SIGNAL    : >>> {result['signal']} <<<")
    print(f"Prob UP   : {result['prob_up']:.1%}")
    print(f"Prob DOWN : {result['prob_down']:.1%}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("----------------------------------------------")

    if result["signal"] != "FLAT":
        if result["signal"] == "BUY":
            sl = result["close"] - stop_size
            tp = result["close"] + stop_size * 2
        else:
            sl = result["close"] + stop_size
            tp = result["close"] - stop_size * 2

        print(f"Stop Loss : {sl:.5f}")
        print(f"Take Profit: {tp:.5f}")
        print(f"Max Risk  : {risk_pct}% of account")
    else:
        print("No trade - confidence below threshold")

    print("==============================================\n")

def main():
    cfg = load_config()
    pair = cfg["pair"]
    pair_key = pair.replace("=X", "").lower()
    timeframe = cfg["timeframe"]
    model_dir = cfg["paths"]["models"]
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(BASE_DIR, model_dir)

    model, scaler, feature_cols, metadata = load_model_artifacts(
        model_dir, pair_key, timeframe=timeframe
    )
    df = fetch_recent_data(pair, timeframe, cfg)
    df_daily = None
    if timeframe != "1d":
        try:
            df_daily = fetch_recent_data(pair, "1d", cfg)
        except Exception as exc:
            print(f"[WARNING] Could not fetch daily trend ({type(exc).__name__}).")

    result = predict_signal(
        df, df_daily, model, scaler, feature_cols, cfg["confidence_threshold"],
        daily_context_mode=metadata.get("daily_context_mode", "legacy")
    )
    print_signal(result, cfg)
    return result

if __name__ == "__main__":
    main()

