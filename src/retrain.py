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
from datetime import datetime


def load_config(path: str = "config.yaml") -> dict:
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
    )
    if result.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed. Check output above.")
    print(f"[OK] {step_name} complete.")


def check_accuracy_drop(cfg: dict, pair: str, min_accuracy: float = 0.52):
    """
    After retraining, check the model still meets minimum accuracy.
    If not, print a loud warning (don't delete old model automatically).
    """
    model_dir  = cfg["paths"]["models"]
    test_file  = os.path.join(
        cfg["paths"]["labelled_data"],
        f"{pair}_{cfg['timeframe']}_test.csv"
    )

    if not os.path.exists(test_file):
        print("[WARN] Cannot check accuracy — test file missing.")
        return

    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score

    model        = joblib.load(os.path.join(model_dir, f"{pair}_xgboost.pkl"))
    scaler       = joblib.load(os.path.join(model_dir, f"{pair}_scaler.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, f"{pair}_feature_cols.pkl"))

    test_df = pd.read_csv(test_file, parse_dates=["datetime"])
    X_test  = scaler.transform(test_df[feature_cols].values)
    y_test  = test_df["target"].values
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)

    print(f"\n[INFO] Post-retrain accuracy: {acc:.3f}")

    if acc < min_accuracy:
        print(
            f"\n[WARNING] Accuracy {acc:.3f} is below minimum threshold {min_accuracy}.\n"
            "    Market conditions may have changed significantly.\n"
            "    Consider reviewing features or pausing live signals until investigated."
        )
    else:
        print(f"[OK] Accuracy {acc:.3f} is acceptable (≥ {min_accuracy}).")


def save_retrain_log(cfg: dict, pair: str):
    """Append a timestamp to the retrain log."""
    log_path = os.path.join(cfg["paths"]["models"], "retrain_log.json")

    log = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)

    log.append({
        "retrained_at": datetime.utcnow().isoformat(),
        "pair":         pair,
        "timeframe":    cfg["timeframe"],
    })

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

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
