"""
train_model.py
--------------
Loads labelled train/test CSVs, trains an XGBoost classifier,
evaluates it on the test set, and saves the model + scaler.

Usage:
    python src/train_model.py
"""

import os
import yaml
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ── Columns that are NOT features ─────────────────────────────────────────────
NON_FEATURE_COLS = ["datetime", "target", "open", "high", "low", "close", "volume"]


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

    print(f"[INFO] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df


def train(train_df: pd.DataFrame, cfg: dict):
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values

    # Scale features (zero mean, unit variance)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Hyperparameter search space
    param_dist = {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 5, 7, 9],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma":            [0, 0.1, 0.2],
    }

    from sklearn.model_selection import RandomizedSearchCV

    base_model = XGBClassifier(
        random_state      = cfg["random_seed"],
        eval_metric       = "logloss",
        use_label_encoder = False,
        n_jobs            = -1
    )

    print(f"[INFO] Starting hyperparameter tuning (Search CV) on {len(X_train):,} samples...")
    search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_dist, 
        n_iter=15, 
        cv=3, 
        scoring="accuracy", 
        verbose=1,
        random_state=cfg["random_seed"]
    )
    
    search.fit(X_train, y_train)
    
    model = search.best_estimator_
    print(f"[INFO] Best Parameters: {search.best_params_}")
    print("[INFO] Training complete with optimized parameters.")

    return model, scaler, feature_cols


def evaluate(model, scaler, test_df: pd.DataFrame, feature_cols: list,
             threshold: float) -> dict:
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values
    X_test = scaler.transform(X_test)

    # Raw probabilities
    proba  = model.predict_proba(X_test)[:, 1]

    # Apply confidence threshold — predict only when confident enough
    y_pred = np.where(proba >= threshold, 1,
             np.where(proba <= (1 - threshold), 0, -1))

    # Rows where model has enough confidence
    confident_mask = y_pred != -1
    confident_pct  = confident_mask.mean() * 100

    y_pred_confident = y_pred[confident_mask]
    y_true_confident = y_test[confident_mask]

    metrics = {
        "all_accuracy":         accuracy_score(y_test, (proba >= 0.5).astype(int)),
        "confident_pct":        confident_pct,
        "confident_accuracy":   accuracy_score(y_true_confident, y_pred_confident)
                                if confident_mask.sum() > 0 else 0,
        "precision":            precision_score(y_true_confident, y_pred_confident,
                                               zero_division=0),
        "recall":               recall_score(y_true_confident, y_pred_confident,
                                            zero_division=0),
        "f1":                   f1_score(y_true_confident, y_pred_confident,
                                        zero_division=0),
    }

    print("\n--- Evaluation Results ---")
    print(f"  All-row accuracy (50% threshold) : {metrics['all_accuracy']:.3f}")
    print(f"  Rows above confidence threshold  : {confident_pct:.1f}%")
    print(f"  Accuracy on confident rows       : {metrics['confident_accuracy']:.3f}")
    print(f"  Precision                        : {metrics['precision']:.3f}")
    print(f"  Recall                           : {metrics['recall']:.3f}")
    print(f"  F1 Score                         : {metrics['f1']:.3f}")
    print("----------------------------------\n")
    print(classification_report(y_true_confident, y_pred_confident,
                                 target_names=["DOWN", "UP"], zero_division=0))

    return metrics


def save_model(model, scaler, feature_cols: list, model_dir: str, pair: str):
    os.makedirs(model_dir, exist_ok=True)
    model_path  = os.path.join(model_dir, f"{pair}_xgboost.pkl")
    scaler_path = os.path.join(model_dir, f"{pair}_scaler.pkl")
    cols_path   = os.path.join(model_dir, f"{pair}_feature_cols.pkl")

    joblib.dump(model,        model_path)
    joblib.dump(scaler,       scaler_path)
    joblib.dump(feature_cols, cols_path)

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
    )

    print("[DONE] train_model complete.")


if __name__ == "__main__":
    main()
