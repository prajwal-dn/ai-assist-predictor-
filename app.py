"""
app.py
------
Simple web server that serves the dashboard and runs predict.py on demand.

Usage:
    python app.py
Then open: http://localhost:5000
"""

import os, sys, json, subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_predict():
    """Run predict.py and return structured result."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "src", "predict.py")],
        capture_output=True, text=True, encoding="utf-8", cwd=BASE_DIR, env=env
    )
    # Ensure we have strings even if capture failed for some reason
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
        l = line.strip().replace("║","").strip()
        if "SIGNAL" in l:
            if "BUY"  in l: data["signal"] = "BUY"
            elif "SELL" in l: data["signal"] = "SELL"
            elif "FLAT" in l: data["signal"] = "FLAT"
        elif "Prob UP" in l:
            try: data["prob_up"] = float(l.split(":")[-1].strip().replace("%",""))
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

    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence request logs

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html_path = os.path.join(BASE_DIR, "dashboard.html")
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())

        elif path == "/api/predict":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = run_predict()
            self.wfile.write(json.dumps(data).encode())

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print(f"\n Forex Predictor Dashboard")
    print(f" Open this in your browser: http://localhost:{PORT}\n")
    server = HTTPServer(("", PORT), Handler)
    server.serve_forever()
