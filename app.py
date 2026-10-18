"""
app.py
------
Simple web server that serves the dashboard and runs predict.py on demand.
Prediction history is stored in local JSON with optional Hugging Face Dataset sync.

Usage:
    python app.py
Then open: http://localhost:7860
"""

import os, sys, json, subprocess, time, yaml, threading, re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add src/ to path so we can import db module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from db import save_prediction_run, get_recent_runs, get_run_stats, test_connection

PORT = int(os.environ.get("PORT", 7860))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_TTL_SECONDS = int(os.environ.get("PREDICTION_CACHE_SECONDS", 60))
_prediction_cache = {}   # keyed by pair+timeframe
_train_status = {"running": False, "log": "", "success": None}
_train_lock = threading.Lock()
_db_available = False


def is_git_lfs_pointer(path: str) -> bool:
    """Return True when a model artifact is only a Git LFS pointer placeholder."""
    try:
        with open(path, "rb") as f:
            header = f.read(256)
        return header.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def artifact_status(pair: str, model_dir: str) -> dict:
    """Validate all three model artifacts without deserializing them."""
    paths = {
        "model": os.path.join(model_dir, f"{pair}_xgboost.pkl"),
        "scaler": os.path.join(model_dir, f"{pair}_scaler.pkl"),
        "features": os.path.join(model_dir, f"{pair}_feature_cols.pkl"),
    }
    missing = [kind for kind, path in paths.items() if not os.path.isfile(path)]
    pointers = [kind for kind, path in paths.items() if os.path.isfile(path) and is_git_lfs_pointer(path)]
    if missing:
        return {"ready": False, "reason": "Missing model artifacts: " + ", ".join(missing)}
    if pointers:
        return {
            "ready": False,
            "reason": "Git LFS placeholder files detected for: " + ", ".join(pointers)
                + ". Pull/download the actual model artifacts or retrain the selected pair."
        }
    return {"ready": True, "reason": ""}


def load_app_config():
    with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pair = cfg["pair"]
    pair_code = pair.replace("=X", "").upper()
    timeframe = cfg["timeframe"]
    horizon = cfg["forecast_horizon"]
    interval_map = {"1h": "60", "4h": "240", "1d": "D"}

    pair_key = pair.replace("=X", "").lower()
    model_dir = cfg["paths"]["models"]
    model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(BASE_DIR, model_dir)
    status = artifact_status(pair_key, model_dir)
    model_exists = status["ready"]

    # Determine TradingView symbol based on asset class (crypto contains hyphen)
    tv_symbol = pair_code.replace("-", "")
    if "-" in pair:
        tradingview_symbol = f"COINBASE:{tv_symbol}"
    elif tv_symbol in ("INRUSD", "USDINR"):
        tradingview_symbol = f"FX_IDC:{tv_symbol}"
    else:
        tradingview_symbol = f"FX:{tv_symbol}"

    return {
        "pair": pair,
        "pair_code": pair_code,
        "timeframe": timeframe,
        "forecast_horizon": horizon,
        "tradingview_symbol": tradingview_symbol,
        "tradingview_interval": interval_map.get(timeframe, "60"),
        "model_exists": model_exists,
        "model_status": status["reason"] if not model_exists else "Model artifacts detected",
    }


def save_pair_to_config(new_pair: str):
    """Update the pair in config.yaml."""
    cfg_path = os.path.join(BASE_DIR, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(cfg_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("pair:"):
                f.write(f'pair: "{new_pair}"\n')
            else:
                f.write(line)


def run_predict():
    """Run predict.py and return structured result."""
    cfg = load_app_config()
    cache_key = f"{cfg['pair']}_{cfg['timeframe']}"
    now = time.time()

    cached = _prediction_cache.get(cache_key)
    if cached and now - cached["time"] < CACHE_TTL_SECONDS:
        return cached["data"]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore"
    # load_app_config() intentionally returns a public/runtime view and does not
    # include the raw YAML "paths" section. Resolve artifacts from the app root
    # instead of indexing cfg["paths"] (which caused KeyError and broke /api/predict).
    model_dir = os.path.join(BASE_DIR, "models")
    cfg_status = artifact_status(
        cfg["pair"].replace("=X", "").lower(),
        model_dir
    )
    if not cfg_status["ready"]:
        return {
            "signal": "UNKNOWN", "prob_up": 0, "prob_down": 0,
            "confidence": 0, "close": 0, "timestamp": "",
            "stop_loss": None, "take_profit": None,
            "raw": cfg_status["reason"], "error": True
        }

    try:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "src", "predict.py")],
            capture_output=True, text=True, encoding="utf-8",
            cwd=BASE_DIR, env=env,
            timeout=max(10, int(os.environ.get("PREDICTION_TIMEOUT_SECONDS", "45")))
        )
    except subprocess.TimeoutExpired:
        return {
            "signal": "UNKNOWN", "prob_up": 0, "prob_down": 0,
            "confidence": 0, "close": 0, "timestamp": "",
            "stop_loss": None, "take_profit": None,
            "raw": "Prediction timed out. Check the market-data provider and retry.",
            "error": True
        }
    except OSError as exc:
        return {
            "signal": "UNKNOWN", "prob_up": 0, "prob_down": 0,
            "confidence": 0, "close": 0, "timestamp": "",
            "stop_loss": None, "take_profit": None,
            "raw": f"Could not start prediction process: {exc}",
            "error": True
        }

    stdout = result.stdout if result.stdout is not None else ""
    stderr = result.stderr if result.stderr is not None else ""
    output = stdout + stderr
    lines  = output.splitlines()

    data = {
        "signal":     "UNKNOWN",
        "prob_up":    0,
        "prob_down":  0,
        "confidence": 0,
        "close":      0,
        "timestamp":  "",
        "stop_loss":  None,
        "take_profit":None,
        "raw":        output,
        "error":      result.returncode != 0,
    }

    for line in lines:
        l = line.strip().replace("║", "").strip()
        if "SIGNAL" in l:
            if "BUY"  in l: data["signal"] = "BUY"
            elif "SELL" in l: data["signal"] = "SELL"
            elif "FLAT" in l: data["signal"] = "FLAT"
        elif "Prob UP" in l:
            try: data["prob_up"] = float(l.split(":")[-1].strip().replace("%","")) * (1 if "%" not in l else 1)
            except: pass
        elif "Prob DOWN" in l:
            try: data["prob_down"] = float(l.split(":")[-1].strip().replace("%",""))
            except: pass
        elif "Confidence" in l and "threshold" not in l.lower():
            try: data["confidence"] = float(l.split(":")[-1].strip().replace("%",""))
            except: pass
        elif "Close" in l:
            try: data["close"] = float(l.split(":")[-1].strip())
            except: pass
        elif "Time" in l:
            try: data["timestamp"] = l.split(":", 1)[-1].strip()
            except: pass
        elif "Stop Loss" in l:
            try: data["stop_loss"] = float(l.split(":")[-1].strip())
            except: pass
        elif "Take Profit" in l:
            try: data["take_profit"] = float(l.split(":")[-1].strip())
            except: pass

    # An exited process or missing signal is not a valid prediction.
    if result.returncode != 0 or data["signal"] == "UNKNOWN":
        data["error"] = True
        if not data["raw"].strip():
            data["raw"] = "Prediction failed without diagnostic output."
        return data

    # predict.py prints probabilities as fractions (e.g. 0.63 -> 63%).
    if 0.0 <= data["prob_up"] <= 1.0:
        data["prob_up"] *= 100
        data["prob_down"] *= 100
        data["confidence"] *= 100

    if not data["error"]:
        _prediction_cache[cache_key] = {"time": now, "data": data}

    # Persist valid prediction runs to local JSON with optional HF Dataset sync
    if _db_available:
        try:
            run_id = save_prediction_run(data, cfg["pair"], cfg["timeframe"], cfg["forecast_horizon"])
            data["run_id"] = run_id
        except Exception as e:
            print(f"[DB WARNING] Could not save run: {e}")

    return data


def load_trained_pairs():
    with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_dir = cfg.get("paths", {}).get("models", "models/")
    model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(BASE_DIR, model_dir)
    pairs = []
    unavailable = {}
    if os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if not file.endswith("_xgboost.pkl"):
                continue
            pair = file.replace("_xgboost.pkl", "")
            status = artifact_status(pair, model_dir)
            if status["ready"]:
                pairs.append(pair)
            else:
                unavailable[pair] = status["reason"]
    return {"trained_pairs": sorted(pairs), "unavailable_pairs": unavailable}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence request logs

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(BASE_DIR, "dashboard.html")
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())

        elif path == "/history":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(BASE_DIR, "history.html")
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())

        elif path == "/health":
            # Liveness check for Hugging Face Spaces / Docker deployments.
            # A live process can still be degraded if model artifacts are unavailable.
            cfg = load_app_config()
            model_ready = bool(cfg["model_exists"])
            self.send_json({
                "status": "ok" if model_ready else "degraded",
                "service": "ai-asset-predictor",
                "model_ready": model_ready,
                "model_status": cfg["model_status"],
                "history_storage_available": _db_available,
                "history_storage": "local-json",
                "cloud_sync_configured": bool(os.environ.get("HF_TOKEN")),
            })

        elif path == "/api/predict":
            try:
                prediction = run_predict()
                self.send_json(prediction, 200 if not prediction.get("error") else 503)
            except Exception as exc:
                # Do not let prediction exceptions drop the socket and surface as
                # a generic Railway 502. Return a diagnosable JSON response instead.
                print(f"[PREDICTION ERROR] {type(exc).__name__}: {exc}", flush=True)
                self.send_json({
                    "error": True,
                    "signal": "UNKNOWN",
                    "prob_up": 0,
                    "prob_down": 0,
                    "confidence": 0,
                    "close": 0,
                    "timestamp": "",
                    "stop_loss": None,
                    "take_profit": None,
                    "raw": f"Prediction failed: {type(exc).__name__}: {exc}",
                }, 500)

        elif path == "/api/config":
            self.send_json(load_app_config())

        elif path == "/api/models":
            self.send_json(load_trained_pairs())

        elif path == "/api/history":
            if not _db_available:
                self.send_json({"error": "History storage is unavailable. Check the application logs."}, 503)
                return
            pair_filter = qs.get("pair", [None])[0]
            try:
                limit = max(1, min(int(qs.get("limit", ["50"])[0]), 200))
            except (TypeError, ValueError):
                self.send_json({"error": "Invalid history limit. Use a number from 1 to 200."}, 400)
                return
            runs = get_recent_runs(limit=limit, pair_filter=pair_filter)
            self.send_json({"runs": runs})

        elif path == "/api/stats":
            if not _db_available:
                self.send_json({"error": "Database not connected"}, 503)
                return
            self.send_json(get_run_stats())

        elif path == "/api/db_status":
            cloud_sync_enabled = bool(os.environ.get("HF_TOKEN"))
            self.send_json({
                "connected": _db_available,
                "storage": "local-json",
                "cloud_sync_enabled": cloud_sync_enabled,
                "message": (
                    "Local history is available; Hugging Face sync is configured (remote sync attempts may still fail)."
                    if _db_available and cloud_sync_enabled else
                    "Local history is available; cloud sync is not configured."
                    if _db_available else
                    "History storage is unavailable."
                )
            })

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/set_pair":
            new_pair = payload.get("pair", "").strip()
            if not new_pair:
                self.send_json({"error": "No pair provided"}, 400)
                return
            # Normalise: "EURUSD" -> "EURUSD=X", but keep "BTC-USD" or symbols that already have "=X"
            if "-" not in new_pair and not new_pair.endswith("=X") and len(new_pair) == 6:
                new_pair = new_pair + "=X"
            if not re.fullmatch(r"[A-Za-z0-9-]{3,12}(?:=X)?", new_pair):
                self.send_json({"error": "Invalid market symbol."}, 400)
                return
            pair_key = new_pair.replace("=X", "").lower()
            cfg_file = yaml.safe_load(open(os.path.join(BASE_DIR, "config.yaml"), encoding="utf-8"))
            model_dir = cfg_file["paths"]["models"]
            model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(BASE_DIR, model_dir)
            status = artifact_status(pair_key, model_dir)
            if not status["ready"]:
                self.send_json({"error": status["reason"], "model_exists": False}, 409)
                return
            save_pair_to_config(new_pair)
            cfg = load_app_config()
            self.send_json({"ok": True, "config": cfg})

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    # Test Database (HF Datasets/Local File) connection at startup
    try:
        _db_available = test_connection()
        if _db_available:
            print(" ✅ History Tracking enabled — runs will be saved locally & synced to HF Hub")
        else:
            print(" ⚠️  Tracking unavailable — runs will NOT be saved")
    except Exception as e:
        print(f" ⚠️  Database initialization failed: {e}")
        _db_available = False

    print(f"\n Asset Predictor Dashboard", flush=True)
    print(f" Listening on 0.0.0.0:{PORT}", flush=True)
    print(f" Run History route: /history", flush=True)
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f" HTTP server listening on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
