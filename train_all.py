"""Batch retrain every configured market model and restore config.yaml afterward."""
import os
import subprocess
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "INRUSD=X", "USDINR=X", "BTC-USD", "ETH-USD", "SOL-USD",
]


def _atomic_write(path: str, text: str) -> None:
    """Write config atomically in its own directory."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=os.path.dirname(path),
            prefix=".config-", suffix=".tmp", delete=False
        ) as handle:
            temp_path = handle.name
            handle.write(text)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def save_pair_to_config(new_pair: str) -> None:
    if new_pair not in PAIRS:
        raise ValueError(f"Unsupported pair requested: {new_pair}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        original = handle.read()
    lines = original.splitlines(keepends=True)
    updated = []
    replaced = False
    for line in lines:
        if line.strip().startswith("pair:"):
            newline = "\n" if line.endswith("\n") else ""
            updated.append(f'pair: "{new_pair}"{newline}')
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        raise ValueError(f"No pair setting found in {CONFIG_PATH}")
    _atomic_write(CONFIG_PATH, "".join(updated))


def main() -> int:
    if not os.path.isfile(CONFIG_PATH):
        print(f"ERROR: config file not found: {CONFIG_PATH}", file=sys.stderr)
        return 2

    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        original_config = handle.read()

    failures = []
    try:
        print(f"Starting batch training for {len(PAIRS)} market pairs.", flush=True)
        for index, pair in enumerate(PAIRS, start=1):
            print(f"\n--- [{index}/{len(PAIRS)}] Training {pair} ---", flush=True)
            try:
                save_pair_to_config(pair)
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                result = subprocess.run(
                    [sys.executable, os.path.join(BASE_DIR, "src", "retrain.py")],
                    env=env, cwd=BASE_DIR, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                if result.stdout:
                    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
                if result.returncode != 0:
                    if result.stderr:
                        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
                    failures.append((pair, result.returncode))
                    print(f"FAILED: {pair} (exit {result.returncode})", flush=True)
                else:
                    print(f"SUCCESS: {pair}", flush=True)
                    if result.stderr.strip():
                        print(result.stderr, file=sys.stderr)
            except Exception as exc:
                failures.append((pair, type(exc).__name__))
                print(f"FAILED: {pair}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        # Never leave the app pointed at the final pair from this loop if any
        # training command failed, was interrupted, or completed successfully.
        _atomic_write(CONFIG_PATH, original_config)
        print("Restored original config.yaml pair setting.", flush=True)

    if failures:
        print(f"\nBatch retraining failed for {len(failures)} pair(s):", file=sys.stderr)
        for pair, reason in failures:
            print(f" - {pair}: {reason}", file=sys.stderr)
        return 1

    print(f"\nAll {len(PAIRS)} pair models retrained successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
