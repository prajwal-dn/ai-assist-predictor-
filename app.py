"""
app.py
------
Simple web server that serves the dashboard and runs predict.py on demand.
Prediction history is stored in local JSON with optional Hugging Face Dataset sync.

Usage:
    python app.py
Then open: http://localhost:7860
"""

import os, sys, json, subprocess, time, yaml, threading, re, math, tempfile
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
    missing = [kind for kind, path in paths.items() if not os.path.isfile(path) or os.path.getsize(path) == 0]
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

    replaced = False
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(cfg_path),
                                         prefix=".config-", suffix=".tmp", delete=False) as f:
            temporary_path = f.name
            for line in lines:
                if line.strip().startswith("pair:"):
                    f.write(f'pair: "{new_pair}"\n')
                    replaced = True
                else:
                    f.write(line)
        if not replaced:
            raise ValueError("config.yaml does not contain a pair setting.")
        os.replace(temporary_path, cfg_path)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)



def _parse_output_number(line: str, label: str):
    """Parse one labelled numeric line and retain whether it used percent units."""
    pattern = rf"^\s*{re.escape(label)}\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(%)?\s*$"
    match = re.match(pattern, line, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1)), bool(match.group(2))
    except (TypeError, ValueError):
        return None


def parse_prediction_output(output: str, returncode: int = 0) -> dict:
    """Parse predict.py's report without guessing percent units."""
    data = {
        "signal": "UNKNOWN", "prob_up": 0.0, "prob_down": 0.0,
        "confidence": 0.0, "close": 0.0, "timestamp": "",
        "stop_loss": None, "take_profit": None, "raw": output or "",
        "error": returncode != 0,
    }
    seen = set()
    numeric_fields = (
        ("Prob UP", "prob_up", True), ("Prob DOWN", "prob_down", True),
        ("Confidence", "confidence", True), ("Close", "close", False),
        ("Stop Loss", "stop_loss", False), ("Take Profit", "take_profit", False),
    )
    for line in (output or "").splitlines():
        signal_match = re.match(
            r"^\s*SIGNAL\s*:\s*(?:>{1,3}\s*)?(BUY|SELL|FLAT)\b", line, re.IGNORECASE
        )
        if signal_match:
            data["signal"] = signal_match.group(1).upper()
            seen.add("signal")
            continue
        time_match = re.match(r"^\s*Time\s*:\s*(.*?)\s*$", line, re.IGNORECASE)
        if time_match:
            data["timestamp"] = time_match.group(1)
            if data["timestamp"]:
                seen.add("timestamp")
            continue
        for label, key, is_probability in numeric_fields:
            parsed = _parse_output_number(line, label)
            if parsed is None:
                continue
            value, had_percent = parsed
            if is_probability and not had_percent and 0.0 <= value <= 1.0:
                value *= 100.0
            data[key] = value
            seen.add(key)
            break

    required = {"signal", "prob_up", "prob_down", "confidence", "close", "timestamp"}
    if data["signal"] == "UNKNOWN" or not required.issubset(seen):
        data["error"] = True
    finite_keys = ["prob_up", "prob_down", "confidence", "close"]
    finite_keys.extend(key for key in ("stop_loss", "take_profit") if data[key] is not None)
    if any(not math.isfinite(float(data[key])) for key in finite_keys):
        data["error"] = True
    if not (0.0 <= data["prob_up"] <= 100.0 and
            0.0 <= data["prob_down"] <= 100.0 and
            0.0 <= data["confidence"] <= 100.0 and data["close"] > 0.0):
        data["error"] = True
    if seen.issuperset({"prob_up", "prob_down"}) and abs(
        data["prob_up"] + data["prob_down"] - 100.0
    ) > 1.0:
        data["error"] = True
    for key in ("stop_loss", "take_profit"):
        if data[key] is not None and data[key] <= 0:
            data["error"] = True
    if data["error"] and not data["raw"].strip():
        data["raw"] = "Prediction failed or returned an incomplete/invalid report."
    return data


def run_predict():
    """Run predict.py and return structured result."""
    cfg = load_app_config()
    cache_key = f"{cfg['pair']}|{cfg['timeframe']}|{cfg['forecast_horizon']}|{cfg.get('confidence_threshold', '')}"
    now = time.time()

    cached = _prediction_cache.get(cache_key)
    if cached and now - cached["time"] < CACHE_TTL_SECONDS:
        return dict(cached["data"])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore"
    # load_app_config() intentionally returns a public/runtime view and does not
    # include the raw YAML "paths" section. Resolve artifacts from the app root
    # instead of indexing cfg["paths"] (which caused KeyError and broke /api/predict).
    with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    model_dir = raw_cfg.get("paths", {}).get("models", "models/")
    model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(BASE_DIR, model_dir)
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
    data = parse_prediction_output(output, result.returncode)

    # A failed/incomplete prediction must never be cached or persisted as valid.
    if data["error"]:
        if result.returncode != 0:
            print(f"[PREDICTION ERROR] predict.py exited with code {result.returncode}", flush=True)
        return data

    cacheable = True
    if _db_available:
        try:
            run_id = save_prediction_run(data, cfg["pair"], cfg["timeframe"], cfg["forecast_horizon"])
            data["run_id"] = run_id
        except Exception as e:
            cacheable = False
            print(f"[DB WARNING] Could not save run: {e}", flush=True)

    # Cache only after persistence was attempted, so a failed history write can retry.
    if cacheable:
        _prediction_cache[cache_key] = {"time": time.time(), "data": dict(data)}
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
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self.send_json({"error": "Invalid Content-Length."}, 400)
            return
        if length < 0 or length > 4096:
            self.send_json({"error": "Request body is too large."}, 413)
            return
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "Request body must be valid JSON."}, 400)
            return
        if not isinstance(payload, dict):
            self.send_json({"error": "Request body must be a JSON object."}, 400)
            return

        if path == "/api/set_pair":
            raw_pair = payload.get("pair", "")
            if not isinstance(raw_pair, str):
                self.send_json({"error": "Pair must be a string."}, 400)
                return
            new_pair = raw_pair.strip().upper()
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
