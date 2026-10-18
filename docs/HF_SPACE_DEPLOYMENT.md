# Hugging Face Spaces deployment and smoke-test checklist

## Deployment target

This project is a single Python service with a Dockerfile. Keep the full backend, XGBoost dependencies, model artifacts, dashboard HTML, and history logic together in the Hugging Face Space. The Dockerfile exposes port 7860 and `app.py` binds to `PORT` (default 7860), matching the Space metadata in the README.

## Confirm the canonical Space first

The repository currently contains two different app subdomain references:
- `https://praj-9035-ai-asset-predictor.hf.space` (GitHub repository homepage metadata)
- `https://praj-9035-ai-forex-predictor.hf.space` (README live-deployment link)

Open the Space in Hugging Face and copy the actual public app URL from the Space page before treating either URL as canonical. After confirming it, update the README link and GitHub repository homepage to match. Do not infer that a URL is live just because it is present in documentation.

## Verify the deployed Space

After the Space build reaches **Running**, open these URLs using the confirmed base URL:

1. `/` — the Asset Predictor dashboard should load.
2. `/health` — returns JSON with app liveness, model artifact readiness, history storage, and whether cloud sync is configured.
3. `/api/config` — returns the active market, timeframe, forecast horizon, chart symbol, and model readiness.
4. `/api/models` — returns the model pairs whose model, scaler, and feature-column files are present and not Git LFS pointer placeholders.
5. `/api/db_status` — reports local history availability and whether Hugging Face cloud sync is configured.
6. `/history` — the history dashboard should show local-history or cloud-sync status and a sensible empty state if there are no runs.

Expected `/health` shape:

```json
{
  "status": "ok",
  "service": "ai-asset-predictor",
  "model_ready": true,
  "model_status": "Model artifacts detected",
  "history_storage_available": true,
  "history_storage": "local-json",
  "cloud_sync_configured": false
}
```

The values above are illustrative; use the actual JSON from the running Space. `status: degraded` or `model_ready: false` means the service process is responding but the configured market's model files are not ready. A successful HTTP response alone is not proof that predictions work.

## Prediction smoke test

On the dashboard, choose a listed verified market and select **Run Prediction**. Confirm a BUY, SELL, or FLAT result appears with non-zero/meaningful price fields, timestamp, probability percentages, and volatility-based stop/take-profit fields where applicable. If prediction fails, record the exact message and inspect Space **Logs**. Check that the deployed container pulled the real Git LFS objects for `*_xgboost.pkl`, `*_scaler.pkl`, and `*_feature_cols.pkl` instead of only their pointer text.

Also verify that a successful prediction appears at `/history`. Without `HF_TOKEN`, the app should still store history locally, but a container restart/rebuild may reset that local file. Configure `HF_TOKEN` only through Space **Settings → Variables and secrets**; never commit it.

## Vercel

The GitHub repository includes a `vercel.json` setting that disables automatic Vercel Git deployments because Hugging Face Spaces is the intended host for the full Dockerized ML app. This setting does not modify or disable the Hugging Face Space. It does not erase historical failed Vercel statuses from earlier commits. To fully stop Vercel activity, the owner can also open Vercel → project `ai-assist-predictor` → **Settings → Git** and choose **Disconnect** under the connected Git repository. This disconnects only that Vercel project's Git integration; it does not delete the GitHub repository or Hugging Face Space.

## Deployment limits and caveats

- Do not remove XGBoost, scikit-learn, numpy/pandas, or serialized model artifacts just to make this app fit a Vercel serverless bundle. The complete Docker runtime belongs on the Space.
- The embedded TradingView chart is market context; it is separate from the Python data-provider response used for prediction.
- A configured Hugging Face token means cloud sync is enabled/configured, not that each upload succeeded. Check Space logs if sync attempts fail.
- Treat all prediction outputs as research estimates, not guaranteed future returns or financial advice.
