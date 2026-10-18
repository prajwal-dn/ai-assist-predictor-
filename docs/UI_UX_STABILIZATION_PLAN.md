# AI Asset Predictor PRO — Stabilization & UI/UX Plan

## Goal
Ship a clean, polished dashboard that shows only capabilities which are actually available. Prefer a truthful, resilient core flow over placeholder cards, overstated "live" claims, and controls that silently fail.

## Repository audit (static review)
- App: Python stdlib HTTP server in `app.py`; serves `dashboard.html` and `history.html`.
- Prediction: `src/predict.py` loads serialized model artifacts and fetches candles from TwelveData or Yahoo Finance.
- UI: plain HTML/CSS/JS; dashboard calls `/api/config`, `/api/set_pair`, `/api/predict`, `/api/models`.
- History: JSON persistence in `src/db.py`, with optional Hugging Face Dataset synchronization when `HF_TOKEN` exists.
- Chart: TradingView iframe.
- Critical artifact issue: tracked `.pkl` files reviewed here resolve to Git LFS pointer text, not serialized model bytes. File-existence checks alone can incorrectly mark these models as usable. The live prediction flow should validate artifacts and report an actionable error.
- Status issue: dashboard's "LIVE AI AGENT" pill is static and may imply realtime health even before data/model requests succeed.
- UI issue: no theme toggle; responsive behavior is limited and large chart height can crowd smaller screens.
- Controls: auto-refresh is set to one hour by default; selecting a market writes to shared `config.yaml`, which can cause cross-user/shared deployment state issues. Use cautiously in hosted environments.

## Keep (when health checks pass)
1. Prediction flow (model load -> data fetch -> feature validation -> probability/signal result).
2. TradingView chart as a contextual market chart; label separately from prediction data.
3. Market pair selector, manual Run Prediction, confidence/probability display, ATR-based stop/take-profit levels.
4. Local run history and optional HF sync, clearly distinguished.
5. Existing training/retraining scripts as developer tooling, not main user-facing navigation.

## Hide or mark unavailable until verified
- Any model pair whose artifacts are only LFS pointers, missing, or fail model/scaler/feature-column loading.
- "Live" status claims unless actual server, model, and data-provider readiness are reported.
- Dataset "online" language when only local JSON storage is available; show "Local history" / "Cloud sync enabled" accurately.
- Backtest performance claims unless backed by a completed backtest with explicit test window, costs/slippage, and metrics.

## Implementation phases
### P0 — Correctness (do first)
- Detect Git LFS pointer files before presenting a model as trained.
- Make API responses distinguish ready, unavailable, and error states with actionable messages.
- Fix prediction parser/format risks; preferably return structured JSON from prediction code instead of parsing human-readable stdout.
- Validate pair input against the known trained-pair allowlist.
- Do not modify the shared market config on every request in multi-user deployments; move selection to request-scoped input or clearly document single-user behavior.
- Add timeouts and error handling for data provider requests.
- Ensure failed prediction runs aren't recorded as valid signals.

### P1 — UI/UX polish
- Establish a consistent responsive layout, max-width content container, spacing scale, and accessible contrast.
- More precise status copy: "Service ready", "Model unavailable", "Waiting for prediction", "Data provider unavailable".
- Clear page hierarchy: header, market controls, primary signal, risk levels, market chart, probabilities, and recent runs.
- Reduce the chart height on tablet/mobile and add a stacked mobile layout.
- Add clear loading, empty, stale-data, and error states; disable Run only during request or when selected model is unavailable.
- Add accessible focus states, semantic labels, aria-live updates, and reduced-motion support.
- Add a light/dark toggle only if persisted safely and tested; dark can remain default.

### P2 — History UX
- Preserve current table/filter if endpoints work.
- Clearly state when history is local vs cloud-synced.
- Improve empty/error states and refresh feedback.
- Escape server-returned values before using them in HTML templates.
- Either implement real pagination or remove the decorative pagination area.

### P3 — Verification
- Unit tests for LFS pointer detection, model readiness, pair normalization, prediction parsing, and no-record-on-failure.
- Smoke test `/`, `/history`, `/api/config`, `/api/models`, `/api/db_status`, and prediction failure behavior.
- Manually verify desktop/tablet/mobile, keyboard-only use, light/dark if retained, and stale-network errors.
- Document setup, data sources, secret configuration, limitations, and financial-risk disclaimer.

## Acceptance criteria
- Every visible control has an observable, working result.
- A missing/invalid model never produces a zero-valued success-looking signal.
- Status labels never claim live connectivity without a real health check.
- Local history works without a cloud token; cloud sync status is explicit.
- Failed external requests provide a short actionable message.
- The main workflow is usable at 360px mobile width and with keyboard navigation.
- All tests/smoke checks pass before merging.
