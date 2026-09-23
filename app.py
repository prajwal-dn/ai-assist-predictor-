"""
app.py
------
Simple web server that serves the dashboard and runs predict.py on demand.
All prediction runs are persisted to MariaDB.

Usage:
    python app.py
Then open: http://localhost:7860
"""

import os, sys, json, subprocess, time, yaml, threading
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
    model_exists = os.path.exists(os.path.join(model_dir, f"{pair_key}_xgboost.pkl"))

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
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "src", "predict.py")],
        capture_output=True, text=True, encoding="utf-8", cwd=BASE_DIR, env=env
    )
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

    # Convert raw probabilities (0-1) to percentages if needed
    if data["prob_up"] <= 1.0 and data["prob_up"] > 0:
        data["prob_up"]    *= 100
        data["prob_down"]  *= 100
        data["confidence"] *= 100

    if not data["error"]:
        _prediction_cache[cache_key] = {"time": now, "data": data}

    # Persist run to MariaDB
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
    pairs = []
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith("_xgboost.pkl"):
                pairs.append(file.replace("_xgboost.pkl", ""))
    return {"trained_pairs": pairs}


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

        elif path == "/api/predict":
            self.send_json(run_predict())

        elif path == "/api/config":
            self.send_json(load_app_config())

        elif path == "/api/models":
            self.send_json(load_trained_pairs())

        elif path == "/api/history":
            if not _db_available:
                self.send_json({"error": "Database not connected"}, 503)
                return
            pair_filter = qs.get("pair", [None])[0]
            limit = int(qs.get("limit", [50])[0])
            runs = get_recent_runs(limit=limit, pair_filter=pair_filter)
            self.send_json({"runs": runs})

        elif path == "/api/stats":
            if not _db_available:
                self.send_json({"error": "Database not connected"}, 503)
                return
            self.send_json(get_run_stats())

        elif path == "/api/db_status":
            self.send_json({"connected": _db_available})

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
            save_pair_to_config(new_pair)
            cfg = load_app_config()
            self.send_json({"ok": True, "config": cfg})

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    # Test MariaDB connection at startup
    try:
        _db_available = test_connection()
        if _db_available:
            print(" ✅ MariaDB connected — all runs will be persisted")
        else:
            print(" ⚠️  MariaDB unavailable — runs will NOT be saved")
    except Exception as e:
        print(f" ⚠️  MariaDB connection failed: {e}")
        _db_available = False

    print(f"\n Asset Predictor Dashboard")
    print(f" Open this in your browser: http://localhost:{PORT}")
    print(f" Run History page:          http://localhost:{PORT}/history\n")
    server = HTTPServer(("", PORT), Handler)
    server.serve_forever()
