# Model and backend audit

This audit hardens correctness and failure reporting. It does not prove profitability or guarantee that every stored model is accurate.

## Fixed in code

- Prediction-console parsing preserves already-percent probabilities, including small values such as 0.9%, and rejects incomplete, non-finite, inconsistent or failed model reports.
- Prediction cache keys include pair, input timeframe, horizon and confidence threshold. Failed/incomplete predictions are not cached as successful results.
- Model artifact checks reject empty files and Git LFS pointer placeholders; inference validates feature shape, finite inputs, class-1 probability, prices and ATR.
- Market data is normalized, sorted and deduplicated, malformed OHLC candles are dropped, remote calls have timeouts, daily Yahoo fallback fetches a long enough window for the 50-day context indicator, and 4-hour Yahoo data is aggregated from complete hourly candles.
- Newly trained schema-v2 models use previous completed daily indicators for intraday context, removing current-day lookahead. Existing model triplets lacking metadata continue to use schema-v1 legacy features until retrained.
- Chronological split purges forecast-horizon overlap; hyperparameter CV is time-series ordered, purged by the horizon, and scales within each fold.
- Training writes model artifacts atomically and records feature columns/schema/timeframe metadata. Post-retrain evaluation now applies its saved scaler.
- Backtest simulations now test ATR stops/targets against subsequent candle high/low, take the stop first when a candle hits both levels (conservative), handle gaps through stops, and prevent overlapping trades.
- Batch retraining restores the original config.yaml even if a model fails and exits non-zero when any pair fails.
- Regression tests and a CI workflow cover core parser, feature, label-split and backtest behavior.

## Required to finish model-quality remediation

The currently committed 15 model binaries were not retrained during this code change. Legacy model artifacts intentionally retain their original feature logic for compatibility. After the regression checks pass, retrain each supported pair with the fixed pipeline by running python train_all.py in an environment with all data-provider/model dependencies and sufficient compute. Inspect the new test metrics and only then publish replacement .pkl files plus *_metadata.json sidecars. The batch script preserves the original selected pair in config; training still requires market-data access and can fail per pair.

## Important limitations of the backtest

Backtest P&L is denominated in the quote currency of the pair (e.g. GBP for EUR/GBP, USD for SOL/USD); it is not converted to USD for every pair. Spread, commissions, financing and most slippage are not modeled. Use the results as diagnostics, not a claim of live trading performance.
