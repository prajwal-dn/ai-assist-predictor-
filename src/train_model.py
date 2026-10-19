"""
train_model.py
--------------
Loads labelled train/test CSVs, trains an XGBoost classifier,
evaluates it on the test set, and saves the model + scaler.

Usage:
    python src/train_model.py
"""

import os
import json
from datetime import datetime, timezone
import yaml
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ── Columns that are NOT features ─────────────────────────────────────────────
NON_FEATURE_COLS = ["datetime", "target", "open", "high", "low", "close", "volume", "feature_schema_version"]


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def load_data(label_dir: str, pair: str, timeframe: str):
    train_file = os.path.join(label_dir, f"{pair}_{timeframe}_train.csv")
    test_file  = os.path.join(label_dir, f"{pair}_{timeframe}_test.csv")

    for f in [train_file, test_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}. Run labeller.py first.")

    train_df = pd.read_csv(train_file, parse_dates=["datetime"])
    test_df  = pd.read_csv(test_file,  parse_dates=["datetime"])
    if train_df.empty or test_df.empty:
        raise ValueError("Training and test CSV files must both contain rows.")
    print(f"[INFO] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df


def train(train_df: pd.DataFrame, cfg: dict):
    feature_cols = get_feature_columns(train_df)
    if not feature_cols:
        raise ValueError("Training data contains no model feature columns.")
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_train = pd.to_numeric(train_df["target"], errors="coerce").to_numpy()
    if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Training data contains non-finite feature or target values.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data must contain both UP and DOWN target classes.")
    horizon = max(0, int(cfg.get("forecast_horizon", 1)))
    if len(X_train) < max(40, 4 * (horizon + 2)):
        raise ValueError("Not enough rows for three time-series CV folds with the forecast-horizon purge gap.")

    param_dist = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__gamma": [0, 0.1, 0.2],
    }
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            random_state=cfg["random_seed"],
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )),
    ])
    # Fit the scaler inside each CV fold and never validate on earlier data
    # using a model trained on future time periods.
    cv = TimeSeriesSplit(n_splits=3, gap=horizon)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,
        cv=cv,
        scoring="accuracy",
        verbose=1,
        random_state=cfg["random_seed"],
        error_score="raise",
    )
    print(f"[INFO] Time-series tuning on {len(X_train):,} rows (purge gap={horizon}).")
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"[INFO] Best Parameters: {search.best_params_}")
    print("[INFO] Training complete with time-series cross-validation.")
    return best_pipeline.named_steps["model"], best_pipeline.named_steps["scaler"], feature_cols


def evaluate(model, scaler, test_df: pd.DataFrame, feature_cols: list,
             threshold: float) -> dict:
    """Evaluate the saved model contract on a chronological, unseen holdout."""
    if not 0.5 < float(threshold) < 1.0:
        raise ValueError("confidence_threshold must be greater than 0.5 and less than 1.")
    if test_df.empty:
        raise ValueError("Test dataset is empty.")
    missing = [column for column in feature_cols if column not in test_df.columns]
    if missing:
        raise ValueError(f"Test dataset is missing model features: {missing}")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(X_test).all():
        raise ValueError("Test features contain non-finite values.")
    y_test = pd.to_numeric(test_df["target"], errors="coerce").to_numpy()
    if not np.isfinite(y_test).all() or not np.isin(y_test, [0, 1]).all():
        raise ValueError("Test target must contain only finite binary labels 0/1.")

    X_test = scaler.transform(X_test)
    probs = np.asarray(model.predict_proba(X_test), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    up_idx = np.flatnonzero(classes == 1)
    if probs.ndim != 2 or probs.shape[0] != len(test_df) or len(up_idx) != 1:
        raise ValueError("Model probabilities must contain exactly one class labelled 1 (UP).")
    proba_up = probs[:, int(up_idx[0])]
    if not np.isfinite(proba_up).all() or ((proba_up < 0) | (proba_up > 1)).any():
        raise ValueError("Model returned invalid probabilities.")

    y_pred = np.where(proba_up >= threshold, 1,
             np.where(proba_up <= (1 - threshold), 0, -1))
    confident_mask = y_pred != -1
    confident_pct = float(confident_mask.mean() * 100)
    y_pred_confident = y_pred[confident_mask]
    y_true_confident = y_test[confident_mask]

    metrics = {
        "all_accuracy": float(accuracy_score(y_test, (proba_up >= 0.5).astype(int))),
        "confident_pct": confident_pct,
        "confident_accuracy": float(accuracy_score(y_true_confident, y_pred_confident))
                              if confident_mask.sum() else 0.0,
        "precision": float(precision_score(y_true_confident, y_pred_confident, zero_division=0)),
        "recall": float(recall_score(y_true_confident, y_pred_confident, zero_division=0)),
        "f1": float(f1_score(y_true_confident, y_pred_confident, zero_division=0)),
    }
    print("\n--- Evaluation Results ---")
    print(f"  All-row accuracy (50% threshold) : {metrics['all_accuracy']:.3f}")
    print(f"  Rows above confidence threshold  : {confident_pct:.1f}%")
    print(f"  Accuracy on confident rows       : {metrics['confident_accuracy']:.3f}")
    print(f"  Precision                        : {metrics['precision']:.3f}")
    print(f"  Recall                           : {metrics['recall']:.3f}")
    print(f"  F1 Score                         : {metrics['f1']:.3f}")
    print("----------------------------------\n")
    print(classification_report(
        y_true_confident, y_pred_confident, labels=[0, 1],
        target_names=["DOWN", "UP"], zero_division=0
    ))
    return metrics


def save_model(model, scaler, feature_cols: list, model_dir: str, pair: str, metadata: dict = None):
    os.makedirs(model_dir, exist_ok=True)
    model_path  = os.path.join(model_dir, f"{pair}_xgboost.pkl")
    scaler_path = os.path.join(model_dir, f"{pair}_scaler.pkl")
    cols_path   = os.path.join(model_dir, f"{pair}_feature_cols.pkl")

    # Replace artifacts atomically so an interrupted retrain cannot leave partial files.
    for artifact, path in ((model, model_path), (scaler, scaler_path), (list(feature_cols), cols_path)):
        temporary_path = path + ".tmp"
        try:
            joblib.dump(artifact, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    if metadata is not None:
        metadata_path = os.path.join(model_dir, f"{pair}_metadata.json")
        payload = dict(metadata)
        payload["feature_columns"] = list(feature_cols)
        payload["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary_path = metadata_path + ".tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(temporary_path, metadata_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    print(f"[INFO] Model  saved -> {model_path}")
    print(f"[INFO] Scaler saved -> {scaler_path}")
    print(f"[INFO] Cols   saved -> {cols_path}")


def main():
    cfg       = load_config()
    pair      = cfg["pair"].replace("=X", "").lower()
    timeframe = cfg["timeframe"]

    train_df, test_df = load_data(
        label_dir  = cfg["paths"]["labelled_data"],
        pair       = pair,
        timeframe  = timeframe,
    )

    model, scaler, feature_cols = train(train_df, cfg)

    evaluate(
        model        = model,
        scaler       = scaler,
        test_df      = test_df,
        feature_cols = feature_cols,
        threshold    = cfg["confidence_threshold"],
    )

    save_model(
        model        = model,
        scaler       = scaler,
        feature_cols = feature_cols,
        model_dir    = cfg["paths"]["models"],
        pair         = pair,
        metadata     = {
            "model_metadata_version": 1,
            "feature_schema_version": 1 if cfg["features"].get("daily_context_mode", "completed") == "legacy" else 2,
            "daily_context_mode": cfg["features"].get("daily_context_mode", "completed"),
            "timeframe": timeframe,
            "pair": cfg["pair"],
        },
    )

    print("[DONE] train_model complete.")


if __name__ == "__main__":
    main()
