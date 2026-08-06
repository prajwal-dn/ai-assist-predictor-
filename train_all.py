import os
import subprocess
import sys

PAIRS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "AUDUSD=X",
    "USDCAD=X",
    "NZDUSD=X",
    "EURGBP=X",
    "EURJPY=X",
    "GBPJPY=X",
    "INRUSD=X",
    "USDINR=X",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD"
]

def save_pair_to_config(new_pair: str):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(cfg_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("pair:"):
                f.write(f'pair: "{new_pair}"\n')
            else:
                f.write(line)

def main():
    print("Starting batch training of all models...")
    for pair in PAIRS:
        print(f"\\n--- Training {pair} ---")
        save_pair_to_config(pair)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            [sys.executable, os.path.join("src", "retrain.py")], 
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"SUCCESS: {pair} model trained successfully.")
        else:
            print(f"FAILED: {pair} model training failed.")
            print(result.stderr)
            
if __name__ == "__main__":
    main()
