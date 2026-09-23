"""
db.py
-----
Hugging Face Dataset integration for persisting prediction runs.

Stores every prediction run with full metadata into a JSON file,
which is automatically synced to a Hugging Face Dataset repository.
"""

import os
import yaml
import json
from datetime import datetime

try:
    from huggingface_hub import HfApi, hf_hub_download
    _hf_available = True
except ImportError:
    _hf_available = False

_cache = None


def _load_app_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_api():
    if not _hf_available:
        return None
    token = os.environ.get("HF_TOKEN")
    if token:
        return HfApi(token=token)
    return None


def _get_repo_id():
    cfg = _load_app_config()
    return os.environ.get("HF_DATASET_REPO", cfg.get("hf_dataset_repo", "praj-9035/ai-asset-predictor-history"))


def _get_local_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history.json")


def _load_data():
    global _cache
    if _cache is not None:
        return _cache

    repo_id = _get_repo_id()
    api = _get_api()
    local_path = _get_local_path()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if api and repo_id:
        try:
            print(f"[DB] Attempting to fetch history from dataset: {repo_id}")
            hf_hub_download(
                repo_id=repo_id, 
                filename="history.json", 
                repo_type="dataset", 
                local_dir=os.path.dirname(local_path), 
                token=api.token
            )
        except Exception as e:
            print(f"[DB] Could not fetch from HF hub (might not exist yet): {e}")

    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = []
    else:
        _cache = []

    return _cache


def _save_data():
    global _cache
    data = _cache if _cache is not None else []
    local_path = _get_local_path()

    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    repo_id = _get_repo_id()
    api = _get_api()
    if api and repo_id:
        try:
            # Create dataset if it doesn't exist
            try:
                api.dataset_info(repo_id)
            except Exception:
                print(f"[DB] Creating new dataset repo: {repo_id}")
                api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

            print(f"[DB] Pushing updated history to dataset: {repo_id}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo="history.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update prediction history ({len(data)} runs)"
            )
        except Exception as e:
            print(f"[DB WARNING] Failed to upload to HF dataset: {e}")


def test_connection() -> bool:
    """Test if we can initialize the local file system (and optionally HF token)."""
    try:
        _load_data()
        token = os.environ.get("HF_TOKEN")
        if not token:
            print(" ⚠️  HF_TOKEN not set. Running in LOCAL ONLY mode (Run history won't sync to cloud).")
        return True
    except Exception as e:
        print(f"[DB ERROR] Initialization failed: {e}")
        return False


def save_prediction_run(prediction: dict, pair: str, timeframe: str, horizon: int) -> int:
    """
    Save a prediction run to the local JSON and push to HF Dataset if available.
    Returns the auto-generated run ID.
    """
    data_list = _load_data()
    
    # Generate auto-increment ID
    new_id = 1 if len(data_list) == 0 else data_list[-1].get("id", 0) + 1
    
    new_run = {
        "id": new_id,
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pair": pair,
        "timeframe": timeframe,
        "forecast_horizon": horizon,
        "trade_signal": prediction.get("signal", "UNKNOWN"),
        "confidence": round(prediction.get("confidence", 0), 2),
        "prob_up": round(prediction.get("prob_up", 0), 2),
        "prob_down": round(prediction.get("prob_down", 0), 2),
        "close_price": round(prediction.get("close", 0), 5) if prediction.get("close") else 0,
        "stop_loss": round(prediction.get("stop_loss", 0), 5) if prediction.get("stop_loss") else None,
        "take_profit": round(prediction.get("take_profit", 0), 5) if prediction.get("take_profit") else None,
        "candle_timestamp": str(prediction.get("timestamp", "")),
        "is_error": 1 if prediction.get("error", False) else 0,
    }
    
    data_list.append(new_run)
    _save_data()
    return new_id


def get_recent_runs(limit: int = 50, pair_filter: str = None) -> list:
    """Fetch the most recent prediction runs."""
    data_list = _load_data()
    
    # Filter by pair if provided
    filtered = data_list
    if pair_filter:
        filtered = [r for r in data_list if r.get("pair") == pair_filter]
        
    # Sort descending by run_timestamp (newest first)
    filtered = sorted(filtered, key=lambda x: x.get("run_timestamp", ""), reverse=True)
    
    return filtered[:limit]


def get_run_stats() -> dict:
    """Get aggregate statistics across all runs."""
    data_list = _load_data()
    
    valid_runs = [r for r in data_list if r.get("is_error") == 0]
    total = len(valid_runs)
    
    if total == 0:
        return {
            "total_runs": 0,
            "buy_count": 0,
            "sell_count": 0,
            "flat_count": 0,
            "avg_confidence": 0,
            "pairs_analyzed": 0,
            "first_run": None,
            "last_run": None
        }
        
    buy_count = sum(1 for r in valid_runs if r.get("trade_signal") == "BUY")
    sell_count = sum(1 for r in valid_runs if r.get("trade_signal") == "SELL")
    flat_count = sum(1 for r in valid_runs if r.get("trade_signal") == "FLAT")
    
    total_conf = sum(r.get("confidence", 0) for r in valid_runs)
    avg_conf = round(total_conf / total, 2)
    
    pairs = set(r.get("pair") for r in valid_runs)
    
    times = [r.get("run_timestamp") for r in valid_runs if r.get("run_timestamp")]
    times.sort()
    
    first_run = times[0] if times else None
    last_run = times[-1] if times else None
    
    return {
        "total_runs": total,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "flat_count": flat_count,
        "avg_confidence": avg_conf,
        "pairs_analyzed": len(pairs),
        "first_run": first_run,
        "last_run": last_run
    }
