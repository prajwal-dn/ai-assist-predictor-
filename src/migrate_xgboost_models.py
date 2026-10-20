"""One-time, verified conversion of legacy XGBoost joblib snapshots to native JSON.

This script never submits orders or changes the legacy .pkl files. It will only
write a native JSON model after verifying prediction equivalence in the same
runtime; an existing JSON model is kept unless it matches the legacy snapshot.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import xgboost
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DEFAULT_PAIRS = ("sol-usd", "btc-usd", "eth-usd")
MAX_PROBABILITY_DELTA = 1e-6


def _assert_equivalent(reference, candidate, scaler, feature_cols, pair: str) -> float:
    classes_reference = np.asarray(getattr(reference, "classes_", []))
    classes_candidate = np.asarray(getattr(candidate, "classes_", []))
    if not np.array_equal(classes_reference, classes_candidate):
        raise RuntimeError(f"{pair}: classes changed during native model conversion.")
    n_features = len(feature_cols)
    if n_features < 1:
        raise RuntimeError(f"{pair}: no feature columns were saved.")

    # Build a repeatable probe from the persisted scaler's training distribution.
    mean = np.asarray(getattr(scaler, "mean_", np.zeros(n_features)), dtype=float)
    scale = np.asarray(getattr(scaler, "scale_", np.ones(n_features)), dtype=float)
    if mean.shape != (n_features,) or scale.shape != (n_features,):
        raise RuntimeError(f"{pair}: scaler dimensions do not match feature columns.")
    rng = np.random.default_rng(20261020)
    standardized = np.vstack([
        np.zeros(n_features),
        np.ones(n_features),
        -np.ones(n_features),
        rng.normal(0.0, 1.0, size=(16, n_features)),
    ])
    raw_features = mean + standardized * scale
    scaled = np.asarray(scaler.transform(raw_features), dtype=float)
    if scaled.shape != (19, n_features) or not np.isfinite(scaled).all():
        raise RuntimeError(f"{pair}: probe feature matrix is invalid.")

    expected = np.asarray(reference.predict_proba(scaled), dtype=float)
    observed = np.asarray(candidate.predict_proba(scaled), dtype=float)
    if expected.shape != observed.shape or expected.ndim != 2 or expected.shape[0] != len(scaled):
        raise RuntimeError(f"{pair}: probability output shape changed during conversion.")
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        raise RuntimeError(f"{pair}: model returned non-finite probabilities during conversion.")
    delta = float(np.max(np.abs(expected - observed)))
    if delta > MAX_PROBABILITY_DELTA:
        raise RuntimeError(
            f"{pair}: native model differs from legacy probabilities "
            f"(max delta={delta:.9g}, limit={MAX_PROBABILITY_DELTA})."
        )
    return delta


def _load_legacy(model_path: Path, pair: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(model_path)
    if caught:
        warning_types = ", ".join(sorted({item.category.__name__ for item in caught}))
        print(
            f"[INFO] {pair}: legacy snapshot load emitted warning category/categories: "
            f"{warning_types}; checking native export equivalence before saving."
        )
    if not isinstance(model, XGBClassifier):
        raise TypeError(f"{pair}: expected an XGBClassifier in {model_path}, got {type(model).__name__}.")
    return model


def migrate_pair(pair: str) -> Path:
    if pair not in DEFAULT_PAIRS:
        raise ValueError(f"Unsupported migration pair: {pair}")
    model_path = MODEL_DIR / f"{pair}_xgboost.pkl"
    scaler_path = MODEL_DIR / f"{pair}_scaler.pkl"
    columns_path = MODEL_DIR / f"{pair}_feature_cols.pkl"
    native_path = MODEL_DIR / f"{pair}_xgboost.json"
    for path in (model_path, scaler_path, columns_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{pair}: missing or empty legacy artifact: {path}")
        with path.open("rb") as handle:
            if handle.read(256).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{pair}: {path} is a Git LFS pointer. Check out with LFS enabled.")

    legacy_model = _load_legacy(model_path, pair)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(columns_path)
    if not isinstance(feature_cols, (list, tuple)) or not feature_cols:
        raise ValueError(f"{pair}: invalid feature-column artifact.")
    feature_cols = list(feature_cols)
    if getattr(scaler, "n_features_in_", len(feature_cols)) != len(feature_cols):
        raise ValueError(f"{pair}: scaler and feature-column sizes differ.")

    fd, temp_name = tempfile.mkstemp(prefix=f".{pair}-", suffix=".json", dir=MODEL_DIR)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        legacy_model.save_model(str(temp_path))
        candidate = XGBClassifier()
        candidate.load_model(str(temp_path))
        delta = _assert_equivalent(legacy_model, candidate, scaler, feature_cols, pair)

        if native_path.exists():
            with native_path.open("rb") as handle:
                if handle.read(256).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError(f"{pair}: existing native model is an unresolved Git LFS pointer.")
            existing = XGBClassifier()
            existing.load_model(str(native_path))
            existing_delta = _assert_equivalent(legacy_model, existing, scaler, feature_cols, pair)
            print(
                f"[OK] {pair}: existing JSON artifact verified; "
                f"legacy-vs-existing max probability delta={existing_delta:.3g}."
            )
        else:
            os.replace(temp_path, native_path)
            temp_path = None
            print(
                f"[OK] {pair}: wrote {native_path.name}; "
                f"legacy-vs-JSON max probability delta={delta:.3g}."
            )
        return native_path
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert legacy Asset Predictor XGBoost snapshots into native JSON with prediction-equivalence checks."
    )
    parser.add_argument(
        "--pairs", default=",".join(DEFAULT_PAIRS),
        help="Comma-separated subset of sol-usd,btc-usd,eth-usd (default: all three).",
    )
    args = parser.parse_args()
    pairs = list(dict.fromkeys(part.strip().lower() for part in args.pairs.split(",") if part.strip()))
    if not pairs:
        parser.error("At least one pair must be selected.")
    print(f"XGBoost runtime: {xgboost.__version__}")
    for pair in pairs:
        migrate_pair(pair)
    print("All requested native JSON models passed prediction-equivalence checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
