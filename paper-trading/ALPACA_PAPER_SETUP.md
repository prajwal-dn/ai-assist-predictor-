# Alpaca crypto paper trading setup

This integration submits orders only to Alpaca's fixed paper API host, https://paper-api.alpaca.markets. It uses the repository's saved Asset Predictor models for SOL-USD, BTC-USD, and ETH-USD. The model's hourly signal data comes from the existing validated market-data pipeline; account, positions, tradable-asset checks, order submission, and order-status reconciliation use Alpaca.

## 1. Create paper credentials

1. Create or open an Alpaca Paper Only account at https://app.alpaca.markets/.
2. Generate API keys from the paper account. Do not use or paste live-account credentials.
3. In this GitHub repository, open Settings → Secrets and variables → Actions → New repository secret.
4. Add these two secrets:
   - ALPACA_PAPER_API_KEY
   - ALPACA_PAPER_API_SECRET

Never commit these values, add them to the public log branch, or send them in chat. The script does not accept an API base URL from the environment, configuration, or command line.

Alpaca availability and supported tradable symbols can vary by account eligibility and may change. Each run checks Alpaca's authenticated crypto asset list and skips pairs that are not active and tradable.

## 2. One-time XGBoost serialization migration

Before validating crypto models for paper trading, open **Actions → Migrate XGBoost models to native JSON → Run workflow** on branch `main), and choose `ALL_CRYPTO`.

The workflow loads the existing SOL/BTC/ETH pickle snapshots, writes native JSON models, reloads each JSON file, and compares class labels plus probabilities on a deterministic 19-row probe. It commits the new JSON artifacts to `main) only if the comparisons pass within a maximum absolute probability difference of 1e-6. The old pickle files are kept as rollback/fallback artifacts. This migration does not retrain models and does not submit orders.

The one-time conversion may still report that the legacy pickle emitted a warning when loaded; the key acceptance check is that the freshly loaded native JSON artifacts pass without the legacy-snapshot warning in the subsequent validation run. If the conversion fails or predictions differ, it stops without committing the converted files.

## 3. Validate first — no orders submitted

Open Actions → Alpaca crypto paper trading → Run workflow:

- Branch: main
- Pair: ALL (or one of SOL/USD, BTC/USD, ETH/USD)
- Mode: validate

Validation checks the paper account, supported crypto assets, market-data freshness, saved model/scaler artifacts, signals, and existing broker positions/orders. It does not submit new orders. Inspect the run report/artifact to confirm these checks succeed for each requested pair.

If a pair fails to load its Git LFS model artifact or its available market data, don't enable order submission for that pair until corrected. The model artifacts are existing models; this integration does not train or certify them.

## 4. Submit simulated paper orders

Only after validation succeeds, manually run the workflow again with Mode set to paper_trade. This submits simulated market orders on the paper API only.

Safety constraints in this initial integration:

- Long-only spot strategy: a BUY signal can open a position; a SELL signal never opens a short.
- Default maximum entry notional is USD 25 per pair; the code enforces an absolute USD 50 cap per entry.
- One managed position per pair. Unmanaged existing holdings, open orders, quantity mismatches, and ambiguous order states fail closed rather than being automatically changed.
- Client order IDs are stable to reduce duplicate submissions after retries.
- The strategy tracks ATR-based stop/target triggers using completed hourly candles, then submits a paper market sell. These are software-managed exits, not exchange-hosted stop orders; if a workflow is delayed or unavailable, an exit can be delayed. Review results and open positions in the Alpaca paper dashboard.
- Estimated taker fees are included separately in the trade report at 25 basis points per side. They are an estimate, not a guarantee of the broker's actual fee treatment.

The account's simulated balance is controlled by Alpaca's paper account settings; it is separate from the original simulator's 10,000-unit virtual account.

## 5. Optional hourly automatic orders

The workflow runs hourly in validation-only mode by default. Scheduled runs must not place orders unless you explicitly opt in.

After successful manual paper-order testing, go to Settings → Secrets and variables → Actions → Variables and create:

- ALPACA_PAPER_AUTO_TRADE = true — enables hourly paper order submission; leave unset or false to keep scheduled runs validation-only.
- ALPACA_PAPER_MAX_NOTIONAL_USD — optional per-entry notional, default 25, hard maximum 50.

Do not enable scheduled orders until you have reviewed a successful manual paper-order run. Setting the variable only enables the paper endpoint; the code is still hard-locked to paper trading.

## Logs and limits

The workflow stores alpaca-state.json, alpaca-orders.jsonl, and alpaca-runs.jsonl on the paper-trading-log branch and uploads 90-day artifacts. This repository and the log branch are public, so order events, model signals, and estimated P&L are public. API credentials are not logged.

Paper fills are simulations and do not reproduce every market condition (latency, liquidity, or all fee effects). A green regression suite verifies code behavior, not trading profitability. A successful credentialed validation and a successful simulated broker order are separate milestones.
