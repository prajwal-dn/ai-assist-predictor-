"""
retrain.py
----------
Runs the full pipeline end-to-end:
    1. Download fresh data
    2. Engineer features
    3. Add labels
    4. Train model
    5. Evaluate (warn if accuracy drops)
    6. Save new model

Run this monthly (or schedule it with cron / Task Scheduler).

Usage:
    python src/retrain.py
"""

import os
import sys
import yaml
import json
import joblib
import subprocess
from datetime import datetime, timezone
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rooted_path(path: str) -> str:
    """Resolve configured relative paths from the project root, not caller CWD."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(BASE_DIR, "config.yaml")
    elif not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_step(script: str, step_name: str):
    """Run a pipeline script and raise on failure."""
    print(f"\n{'='*50}")
    print(f"  STEP: {step_name}")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed. Check output above.")
    print(f"[OK] {step_name} complete.")


def check_accuracy_drop(cfg: dict, pair: str, min_accuracy: float = 0.52):
    """Evaluate the replacement model against the chronological held-out test set."""
    model_dir = rooted_path(cfg["paths"]["models"])
    label_dir = rooted_path(cfg["paths"]["labelled_data"])
    test_file = os.path.join(label_dir, f"{pair}_{cfg['timeframe']}_test.csv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Cannot validate retrained model: held-out test file missing: {test_file}")

    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score

    model_path = os.path.join(model_dir, f"{pair}_xgboost.pkl")
    scaler_path = os.path.join(model_dir, f"{pair}_scaler.pkl")
    cols_path = os.path.join(model_dir, f"{pair}_feature_cols.pkl")
    model, scaler, feature_cols = (
        joblib.load(model_path), joblib.load(scaler_path), joblib.load(cols_path)
    )
    test_df = pd.read_csv(test_file, parse_dates=["datetime"])
    if test_df.empty:
        raise ValueError("Held-out test dataset is empty.")
    missing = [column for column in feature_cols if column not in test_df.columns]
    if missing:
        raise ValueError(f"Held-out dataset is missing model features: {missing}")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test_df["target"], errors="coerce").to_numpy()
    if not np.isfinite(X_test).all() or not np.isfinite(y_test).all():
        raise ValueError("Held-out test data contains non-finite values.")
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    print(f"\n[INFO] Post-retrain held-out accuracy: {acc:.3f}")
    if acc < min_accuracy:
        print(
            f"\n[WARNING] Accuracy {acc:.3f} is below minimum threshold {min_accuracy}.\n"
            "    This is a warning, not proof the model is unprofitable.\n"
            "    Review out-of-sample metrics before publishing the artifact."
        )
    else:
        print(f"[OK] Accuracy {acc:.3f} is above the alert threshold {min_accuracy}.")


def save_retrain_log(cfg: dict, pair: str):
    """Append a retrain audit entry with an atomic file replacement."""
    model_dir = rooted_path(cfg["paths"]["models"])
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, "retrain_log.json")
    log = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        if not isinstance(log, list):
            raise ValueError(f"Retrain log must be a JSON list: {log_path}")
    log.append({
        "retrained_at": datetime.now(timezone.utc).isoformat(),
        "pair": pair,
        "timeframe": cfg["timeframe"],
    })
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=model_dir,
            prefix=".retrain-log-", suffix=".tmp", delete=False
        ) as f:
            temp_path = f.name
            json.dump(log, f, indent=2)
        os.replace(temp_path, log_path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    print(f"[INFO] Retrain log updated -> {log_path}")


def main():
    cfg      = load_config()
    pair_key = cfg["pair"].replace("=X", "").lower()

    src_dir = os.path.join(os.path.dirname(__file__))

    print(f"\n--- Starting retrain pipeline for {cfg['pair']} at {datetime.utcnow():%Y-%m-%d %H:%M} UTC ---")

    run_step(os.path.join(src_dir, "data_loader.py"),     "1/4 Download fresh data")
    run_step(os.path.join(src_dir, "feature_engineer.py"),"2/4 Engineer features")
    run_step(os.path.join(src_dir, "labeller.py"),        "3/4 Add labels")
    run_step(os.path.join(src_dir, "train_model.py"),     "4/4 Train model")

    check_accuracy_drop(cfg, pair_key)
    save_retrain_log(cfg, pair_key)

    print(f"\n--- Retrain pipeline complete at {datetime.utcnow():%Y-%m-%d %H:%M} UTC ---")
    print("[DONE] retrain complete.")


if __name__ == "__main__":
    main()
