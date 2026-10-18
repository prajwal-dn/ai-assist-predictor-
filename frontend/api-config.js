/*
 * Vercel static frontend -> Railway Python API
 *
 * Set this to the public HTTPS domain shown under Railway > backend service
 * > Settings > Networking, for example:
 * window.ASSET_PREDICTOR_API_BASE_URL = "https://your-service.up.railway.app";
 *
 * Keep the URL origin only: do not append /api or a trailing slash.
 */
window.ASSET_PREDICTOR_API_BASE_URL = "";

window.assetPredictorApiUrl = function (path) {
  const base = String(window.ASSET_PREDICTOR_API_BASE_URL || "").trim().replace(/\/+$/, "");
  if (!base) {
    throw new Error(
      "Railway API URL is not configured. Set ASSET_PREDICTOR_API_BASE_URL in frontend/api-config.js and redeploy the Vercel frontend."
    );
  }
  if (!/^https:\/\//i.test(base)) {
    throw new Error("The Railway API URL must start with https://.");
  }
  const suffix = String(path || "");
  return base + (suffix.startsWith("/") ? suffix : "/" + suffix);
};
