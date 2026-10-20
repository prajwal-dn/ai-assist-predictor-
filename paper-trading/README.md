# Paper trading (no real orders)

This workflow observes real market data and produces hypothetical entries, exits, P&L, and trade logs using the model artifacts in this repository. It does not connect to a broker, submit orders, or risk money.

## How it works

- The Hourly paper trading workflow runs at 7 minutes past each hour after it is enabled on the default branch. Use Actions → Hourly paper trading → Run workflow for an immediate test on a selected branch and pair.
- The initial run does not backfill old trades. It records the latest completed candle, generates a signal from that completed candle, and—if a later 1-hour candle is available—uses the most recently observed price of that still-forming candle as the hypothetical entry. The entry timestamp is the workflow's actual execution time; it does not assume a fill at the candle's historical open.
- Open positions are monitored using later completed OHLC candles. The candle already in progress at entry is excluded from stop/target checks because its high/low may include price action from before the simulated entry. If a later candle touches both stop and target, the simulator assumes the stop happened first. Stop gaps fill at the candle open; target gaps fill conservatively at target.
- The configured horizon controls maximum holding duration in completed 1-hour candles. Only 1-hour execution is enabled initially; unsupported timeframes fail closed rather than inventing a fill.
- Each pair has its own virtual account, denominated in the pair's quote currency. Results across different quote currencies are not combined or converted to USD.
- Estimated round-trip spread, commission and slippage are subtracted from P&L. These values are configurable in config.yaml and must be replaced with realistic estimates for your intended venue.

## Output files

The workflow persists these files on the paper-trading-log branch:

- paper-trading/state.json — virtual equity, open position, and last processed candle.
- paper-trading/trades.jsonl — closed hypothetical trades, gross P&L, estimated costs, net P&L, and exit reason.
- paper-trading/runs.jsonl — one summary per workflow run.
- paper-trading/events.jsonl — position management events for newly completed candles.

Each run also uploads a downloadable GitHub Actions artifact with 90-day retention. The repository is public, so the log branch and committed reports are public too. Never put API keys, tokens, or other secrets in those files.

## Before relying on results

- Add TWELVEDATA_API_KEY under Settings → Secrets and variables → Actions → New repository secret to use TwelveData; otherwise the runner falls back to Yahoo Finance, subject to availability and rate limits.
- The workflow checks out model artifacts with Git LFS enabled.
- Existing model artifacts without metadata use their legacy feature contract. This workflow observes the current saved model; it does not retrain or certify accuracy.
- Keep this in paper-trading mode. Review drawdown, trade counts, costs, missed/failed runs, and out-of-sample performance over many market periods before making any live-trading decisions.
- The workflow must be merged into the repository's default branch before GitHub enables workflow_dispatch or the hourly schedule. After merging, run Actions → Hourly paper trading → Run workflow to begin the first cycle.
