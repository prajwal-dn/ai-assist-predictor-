"""
db.py
-----
MariaDB integration for persisting prediction runs.

Stores every prediction run with full metadata including:
  - Date/time of run
  - Pair, timeframe, forecast horizon
  - Signal, confidence, probabilities
  - Close price, stop loss, take profit
  - Whether the run had an error
"""

import os
import yaml
import mysql.connector
from mysql.connector import pooling
from datetime import datetime

_pool = None


def _load_db_config() -> dict:
    """Load database config from config.yaml or environment variables."""
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    db_cfg = cfg.get("database", {})
    return {
        "host": os.environ.get("DB_HOST", db_cfg.get("host", "localhost")),
        "port": int(os.environ.get("DB_PORT", db_cfg.get("port", 3306))),
        "user": os.environ.get("DB_USER", db_cfg.get("user", "root")),
        "password": os.environ.get("DB_PASSWORD", db_cfg.get("password", "praj")),
        "database": os.environ.get("DB_NAME", db_cfg.get("name", "forex_predictor")),
    }


def _get_pool():
    """Get or create a connection pool."""
    global _pool
    if _pool is None:
        db_cfg = _load_db_config()
        _pool = pooling.MySQLConnectionPool(
            pool_name="forex_pool",
            pool_size=3,
            host=db_cfg["host"],
            port=db_cfg["port"],
            user=db_cfg["user"],
            password=db_cfg["password"],
            database=db_cfg["database"],
            autocommit=True,
        )
    return _pool


def get_connection():
    """Get a connection from the pool."""
    return _get_pool().get_connection()


def save_prediction_run(prediction: dict, pair: str, timeframe: str, horizon: int) -> int:
    """
    Save a prediction run to the database.
    
    Returns the auto-generated run ID.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_runs
                (pair, timeframe, forecast_horizon, trade_signal, confidence,
                 prob_up, prob_down, close_price, stop_loss, take_profit,
                 candle_timestamp, is_error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                pair,
                timeframe,
                horizon,
                prediction.get("signal", "UNKNOWN"),
                round(prediction.get("confidence", 0), 2),
                round(prediction.get("prob_up", 0), 2),
                round(prediction.get("prob_down", 0), 2),
                prediction.get("close", 0),
                prediction.get("stop_loss"),
                prediction.get("take_profit"),
                prediction.get("timestamp", ""),
                1 if prediction.get("error", False) else 0,
            ),
        )
        run_id = cursor.lastrowid
        cursor.close()
        return run_id
    finally:
        conn.close()


def get_recent_runs(limit: int = 50, pair_filter: str = None) -> list:
    """
    Fetch the most recent prediction runs.
    
    Args:
        limit: max number of rows to return
        pair_filter: if set, only return runs for this pair
        
    Returns:
        List of dicts, newest first.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        if pair_filter:
            cursor.execute(
                """
                SELECT id, run_timestamp, pair, timeframe, forecast_horizon,
                       trade_signal, confidence, prob_up, prob_down,
                       close_price, stop_loss, take_profit, candle_timestamp, is_error
                FROM prediction_runs
                WHERE pair = %s
                ORDER BY run_timestamp DESC
                LIMIT %s
                """,
                (pair_filter, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, run_timestamp, pair, timeframe, forecast_horizon,
                       trade_signal, confidence, prob_up, prob_down,
                       close_price, stop_loss, take_profit, candle_timestamp, is_error
                FROM prediction_runs
                ORDER BY run_timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        cursor.close()

        # Serialise datetime and Decimal for JSON
        for row in rows:
            if isinstance(row.get("run_timestamp"), datetime):
                row["run_timestamp"] = row["run_timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            for key in ("confidence", "prob_up", "prob_down", "close_price", "stop_loss", "take_profit"):
                if row.get(key) is not None:
                    row[key] = float(row[key])
        return rows
    finally:
        conn.close()


def get_run_stats() -> dict:
    """Get aggregate statistics across all runs."""
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_runs,
                SUM(CASE WHEN trade_signal = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN trade_signal = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                SUM(CASE WHEN trade_signal = 'FLAT' THEN 1 ELSE 0 END) as flat_count,
                ROUND(AVG(confidence), 2) as avg_confidence,
                COUNT(DISTINCT pair) as pairs_analyzed,
                MIN(run_timestamp) as first_run,
                MAX(run_timestamp) as last_run
            FROM prediction_runs
            WHERE is_error = 0
            """
        )
        stats = cursor.fetchone()
        cursor.close()

        for key in ("avg_confidence",):
            if stats.get(key) is not None:
                stats[key] = float(stats[key])
        for key in ("first_run", "last_run"):
            if isinstance(stats.get(key), datetime):
                stats[key] = stats[key].strftime("%Y-%m-%d %H:%M:%S")

        return stats
    finally:
        conn.close()


def test_connection() -> bool:
    """Test if the database connection works."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return True
    except Exception:
        return False
