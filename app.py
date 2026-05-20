"""
app.py
------
Simple web server that serves the dashboard and runs predict.py on demand.

Usage:
    python app.py
Then open: http://localhost:7860
"""

import os, sys, json, subprocess, time, yaml, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

PORT = int(os.environ.get("PORT", 7860))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_TTL_SECONDS = int(os.environ.get("PREDICTION_CACHE_SECONDS", 60))
_prediction_cache = {}   # keyed by pair+timeframe
_train_status = {"running": False, "log": "", "success": None}
_train_lock = threading.Lock()


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

    return {
        "pair": pair,
        "pair_code": pair_code,
        "timeframe": timeframe,
        "forecast_horizon": horizon,
        "tradingview_symbol": f"FX:{pair_code}",
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

    return data


def _run_train_background(pair: str):
    """Run the full retrain pipeline in a background thread."""
    global _train_status
    with _train_lock:
        _train_status = {"running": True, "log": f"Starting pipeline for {pair}...\n", "success": None}

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore"

    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "src", "retrain.py")],
        capture_output=True, text=True, encoding="utf-8", cwd=BASE_DIR, env=env
    )
    output = (result.stdout or "") + (result.stderr or "")
    success = result.returncode == 0

    with _train_lock:
        _train_status = {"running": False, "log": output, "success": success}

    # Clear prediction cache for this pair so next predict uses fresh model
    pair_key = pair.replace("=X", "").lower()
    for key in list(_prediction_cache.keys()):
        if key.startswith(pair_key):
            del _prediction_cache[key]


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

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(BASE_DIR, "dashboard.html")
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())

        elif path == "/api/predict":
            self.send_json(run_predict())

        elif path == "/api/config":
            self.send_json(load_app_config())

        elif path == "/api/train_status":
            with _train_lock:
                self.send_json(dict(_train_status))

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
            # Normalise: "EURUSD" → "EURUSD=X", already "EURUSD=X" stays
            if not new_pair.endswith("=X"):
                new_pair = new_pair + "=X"
            save_pair_to_config(new_pair)
            cfg = load_app_config()
            self.send_json({"ok": True, "config": cfg})

        elif path == "/api/train":
            with _train_lock:
                if _train_status.get("running"):
                    self.send_json({"error": "Training already in progress"}, 409)
                    return
            cfg = load_app_config()
            t = threading.Thread(target=_run_train_background, args=(cfg["pair"],), daemon=True)
            t.start()
            self.send_json({"ok": True, "message": f"Training started for {cfg['pair']}"})

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print(f"\n Forex Predictor Dashboard")
    print(f" Open this in your browser: http://localhost:{PORT}\n")
    server = HTTPServer(("", PORT), Handler)
    server.serve_forever()
