"""Hourly paper-trading runner. Never submits real broker orders."""
import argparse
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = BASE_DIR / "paper-trading"
STATE_PATH = PAPER_DIR / "state.json"
TRADES_PATH = PAPER_DIR / "trades.jsonl"
RUNS_PATH = PAPER_DIR / "runs.jsonl"
EVENTS_PATH = PAPER_DIR / "events.jsonl"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineer import add_features
from predict import fetch_recent_data, load_config, load_model_artifacts


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_pair(pair: str) -> str:
    if not isinstance(pair, str):
        raise ValueError("Pair must be a string.")
    value = pair.strip().upper()
    if len(value) == 6 and value.isalpha():
        value += "=X"
    if not re.fullmatch(r"[A-Z0-9-]{3,12}(?:=X)?", value):
        raise ValueError(f"Invalid market pair: {pair!r}")
    return value


def model_key(pair: str) -> str:
    return normalize_pair(pair).replace("=X", "").lower()


def quote_currency(pair: str) -> str:
    code = normalize_pair(pair).replace("=X", "")
    if "-" in code:
        return code.split("-", 1)[1]
    if len(code) == 6:
        return code[3:]
    return "QUOTE"


def _timestamp(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result


def _iso(value) -> str:
    return _timestamp(value).isoformat()


def _as_float(value, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}-", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_row = json.loads(json.dumps(row, allow_nan=False, default=str))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_state(path: Path = STATE_PATH) -> dict:
    if not path.exists():
        return {"schema_version": 1, "updated_at_utc": None, "pairs": {}}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if (not isinstance(state, dict) or state.get("schema_version") != 1
            or not isinstance(state.get("pairs"), dict)):
        raise ValueError(f"Invalid paper-trading state file: {path}")
    return state


def initial_pair_state(starting_balance: float) -> dict:
    starting_balance = _as_float(starting_balance, "starting_balance")
    if starting_balance <= 0:
        raise ValueError("starting_balance must be positive.")
    return {
        "equity": starting_balance,
        "starting_balance": starting_balance,
        "realized_pnl_quote": 0.0,
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
        "position": None,
        "last_processed_candle": None,
    }


def _bar_values(bar: pd.Series) -> dict:
    values = {name: _as_float(bar[name], name) for name in ("open", "high", "low", "close")}
    if any(values[name] <= 0 for name in ("open", "high", "low", "close")):
        raise ValueError("OHLC prices must be positive.")
    if values["high"] < max(values["open"], values["close"]) or values["low"] > min(values["open"], values["close"]):
        raise ValueError("OHLC candle is internally inconsistent.")
    values["datetime"] = _iso(bar["datetime"])
    return values


def position_exit(position: dict, bar: pd.Series, horizon: int):
    """Check a completed candle; when stop and target both hit, assume stop first."""
    values = _bar_values(bar)
    op, hi, lo, close = values["open"], values["high"], values["low"], values["close"]
    direction = 1 if position["direction"] == "BUY" else -1
    stop, target = float(position["stop_loss"]), float(position["take_profit"])
    position["bars_held"] = int(position.get("bars_held", 0)) + 1

    if direction == 1:
        gap_stop, gap_target = op <= stop, op >= target
        hit_stop, hit_target = lo <= stop, hi >= target
    else:
        gap_stop, gap_target = op >= stop, op <= target
        hit_stop, hit_target = hi >= stop, lo <= target

    if gap_stop:
        return op, "STOP_LOSS_GAP"
    if gap_target:
        return target, "TAKE_PROFIT_GAP"
    if hit_stop and hit_target:
        return stop, "STOP_LOSS_AMBIGUOUS"
    if hit_stop:
        return stop, "STOP_LOSS"
    if hit_target:
        return target, "TAKE_PROFIT"
    if position["bars_held"] >= horizon:
        return close, "HORIZON"
    return None


def close_position(pair: str, pair_state: dict, bar: pd.Series,
                   exit_price: float, exit_reason: str, paper_cfg: dict) -> dict:
    position = pair_state["position"]
    direction = 1 if position["direction"] == "BUY" else -1
    exit_price = _as_float(exit_price, "exit_price")
    quantity = float(position["quantity"])
    gross_pnl = direction * (exit_price - float(position["entry_price"])) * quantity
    average_notional = quantity * (float(position["entry_price"]) + exit_price) / 2.0
    spread_bps = _as_float(paper_cfg.get("spread_bps", 5.0), "spread_bps")
    commission_bps = _as_float(paper_cfg.get("commission_bps", 0.0), "commission_bps")
    slippage_bps = _as_float(paper_cfg.get("slippage_bps", 2.0), "slippage_bps")
    if min(spread_bps, commission_bps, slippage_bps) < 0:
        raise ValueError("Paper-trading cost settings cannot be negative.")
    estimated_cost = average_notional * (spread_bps + commission_bps + slippage_bps) / 10_000.0
    raw_net_pnl = gross_pnl - estimated_cost
    equity_before = float(pair_state["equity"])
    net_pnl = max(-equity_before, raw_net_pnl)
    if raw_net_pnl < -equity_before:
        exit_reason = "ACCOUNT_BLOWN"
    pair_state["equity"] = max(0.0, equity_before + net_pnl)
    pair_state["realized_pnl_quote"] = float(pair_state["realized_pnl_quote"]) + net_pnl
    pair_state["closed_trades"] = int(pair_state["closed_trades"]) + 1
    pair_state["wins"] = int(pair_state["wins"]) + int(net_pnl > 0)
    pair_state["losses"] = int(pair_state["losses"]) + int(net_pnl <= 0)
    trade = {
        "trade_id": f"{pair}:{position['signal_time']}:{position['entry_time']}",
        "pair": pair,
        "quote_currency": quote_currency(pair),
        "direction": position["direction"],
        "signal_time": position["signal_time"],
        "entry_time": position["entry_time"],
        "exit_time": _iso(bar["datetime"]),
        "entry_price": float(position["entry_price"]),
        "exit_price": exit_price,
        "stop_loss": float(position["stop_loss"]),
        "take_profit": float(position["take_profit"]),
        "exit_reason": exit_reason,
        "bars_held": int(position["bars_held"]),
        "horizon": int(position["horizon"]),
        "confidence": float(position["confidence"]),
        "prob_up": float(position["prob_up"]),
        "prob_down": float(position["prob_down"]),
        "quantity": quantity,
        "risk_amount_quote": float(position["risk_amount_quote"]),
        "gross_pnl_quote": gross_pnl,
        "estimated_cost_quote": estimated_cost,
        "net_pnl_quote": net_pnl,
        "equity_before": equity_before,
        "equity_after": float(pair_state["equity"]),
        "cost_model_round_trip_bps": {
            "spread": spread_bps, "commission": commission_bps, "slippage": slippage_bps,
        },
        "simulation_only": True,
    }
    pair_state["position"] = None
    return trade


def open_position(pair: str, pair_state: dict, bar: pd.Series,
                  signal: dict, cfg: dict) -> dict:
    entry = _as_float(bar["open"], "entry_open")
    atr = _as_float(signal["atr"], "signal_atr")
    atr_multiplier = _as_float(cfg["risk"]["atr_stop_multiplier"], "atr_stop_multiplier")
    risk_fraction = _as_float(cfg["risk"]["max_risk_per_trade"], "max_risk_per_trade")
    equity = _as_float(pair_state["equity"], "equity")
    if atr <= 0 or atr_multiplier <= 0:
        raise ValueError("ATR and ATR stop multiplier must be positive.")
    if not 0 < risk_fraction <= 0.05:
        raise ValueError("Paper-trading risk per trade must be in (0, 5%].")
    if equity <= 0:
        raise ValueError("Paper account has no remaining equity.")
    stop_size = atr * atr_multiplier
    quantity = equity * risk_fraction / stop_size
    is_long = signal["signal"] == "BUY"
    pair_state["position"] = {
        "direction": signal["signal"],
        "signal_time": signal["timestamp"],
        "signal_candle_start": signal.get("candle_start", signal["timestamp"]),
        "entry_time": _iso(bar["datetime"]),
        "entry_price": entry,
        "stop_loss": entry - stop_size if is_long else entry + stop_size,
        "take_profit": entry + 2 * stop_size if is_long else entry - 2 * stop_size,
        "atr_at_signal": atr,
        "confidence": float(signal["confidence"]),
        "prob_up": float(signal["prob_up"]),
        "prob_down": float(signal["prob_down"]),
        "quantity": quantity,
        "risk_amount_quote": equity * risk_fraction,
        "bars_held": 0,
        "horizon": int(cfg["forecast_horizon"]),
    }
    return {
        "type": "OPENED", "pair": pair, "direction": signal["signal"],
        "signal_time": signal["timestamp"], "entry_time": _iso(bar["datetime"]),
        "entry_price": entry, "stop_loss": pair_state["position"]["stop_loss"],
        "take_profit": pair_state["position"]["take_profit"],
        "confidence": float(signal["confidence"]), "quantity": quantity,
        "simulation_only": True,
    }


def process_completed_candle(pair: str, pair_state: dict, bar: pd.Series,
                             paper_cfg: dict) -> list:
    """Manage an existing position. Never invent entries for bars already gone."""
    if pair_state.get("position") is None:
        return []
    position = pair_state["position"]
    exit_signal = position_exit(position, bar, int(position["horizon"]))
    if exit_signal is None:
        return [{
            "type": "POSITION_HELD", "pair": pair,
            "candle_time": _iso(bar["datetime"]), "bars_held": int(position["bars_held"]),
        }]
    exit_price, reason = exit_signal
    trade = close_position(pair, pair_state, bar, exit_price, reason, paper_cfg)
    return [{"type": "CLOSED", **trade}]


def infer_feature_signals(features: pd.DataFrame, model, scaler,
                          feature_cols: list, threshold: float) -> list:
    missing = [column for column in feature_cols if column not in features.columns]
    if missing:
        raise ValueError(f"Live feature schema is missing columns: {missing}")
    matrix = features[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Model input contains non-finite features.")
    transformed = scaler.transform(matrix)
    probabilities = np.asarray(model.predict_proba(transformed), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    up_columns = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(features) or len(up_columns) != 1:
        raise ValueError("Model must return one class-1 (UP) probability per feature row.")
    up = probabilities[:, int(up_columns[0])]
    if not np.isfinite(up).all() or ((up < 0) | (up > 1)).any():
        raise ValueError("Model returned invalid probabilities.")
    down = 1 - up
    if not 0.5 < float(threshold) < 1:
        raise ValueError("confidence_threshold must be greater than 0.5 and less than 1.")
    signals = []
    for index, row in features.reset_index(drop=True).iterrows():
        prob_up = float(up[index])
        prob_down = float(down[index])
        signal = "BUY" if prob_up >= threshold else "SELL" if prob_down >= threshold else "FLAT"
        signals.append({
            "timestamp": _iso(_timestamp(row["datetime"]) + pd.Timedelta(hours=1)),
            "candle_start": _iso(row["datetime"]),
            "signal": signal,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "confidence": max(prob_up, prob_down),
            "atr": _as_float(row["atr"], "atr"),
        })
    return signals


def _prepare_market(pair: str, cfg: dict, model_dir: Path):
    if cfg["timeframe"] != "1h":
        raise ValueError("Automated paper execution is currently restricted to 1h candles.")
    model, scaler, feature_cols, metadata = load_model_artifacts(
        str(model_dir), model_key(pair), timeframe=cfg["timeframe"]
    )
    completed = fetch_recent_data(pair, "1h", cfg, drop_incomplete=True)
    forming = fetch_recent_data(pair, "1h", cfg, drop_incomplete=False)
    if completed.empty:
        raise ValueError("Market-data provider returned no completed candles.")
    completed_latest = _timestamp(completed["datetime"].iloc[-1])
    forming = forming.loc[pd.to_datetime(forming["datetime"]) > completed_latest].copy()
    forming.reset_index(drop=True, inplace=True)
    daily_mode = metadata.get("daily_context_mode", "legacy")
    daily = fetch_recent_data(pair, "1d", cfg, drop_incomplete=(daily_mode != "legacy"))
    features = add_features(completed, cfg, daily, daily_context_mode=daily_mode)
    if features.empty:
        raise ValueError("Insufficient completed candles for model feature calculation.")
    if _timestamp(features["datetime"].iloc[-1]) != completed_latest:
        raise ValueError(
            "Latest usable feature row is older than the latest market candle. "
            "Refusing to generate a stale paper-trading signal."
        )
    return model, scaler, feature_cols, completed, forming, features


def run_pair(pair_arg: str, state: dict, state_path: Path = STATE_PATH) -> dict:
    pair = normalize_pair(pair_arg)
    os.environ["ASSET_PREDICTOR_PAIR"] = pair
    cfg = load_config()
    if cfg.get("pair") != pair:
        raise ValueError("Configured pair does not match requested paper-trading pair.")
    paper_cfg = cfg.get("paper_trading", {})
    if paper_cfg.get("enabled", True) is not True:
        raise RuntimeError("Paper trading is disabled in config.yaml.")
    model_dir = Path(cfg["paths"]["models"])
    if not model_dir.is_absolute():
        model_dir = BASE_DIR / model_dir
    model, scaler, feature_cols, completed, forming, features = _prepare_market(pair, cfg, model_dir)
    pair_state = state["pairs"].setdefault(
        pair, initial_pair_state(paper_cfg.get("starting_balance", 10_000))
    )
    latest_closed = _iso(completed["datetime"].iloc[-1])
    previous_last = pair_state.get("last_processed_candle")
    completed_times = pd.to_datetime(completed["datetime"], errors="coerce")

    if previous_last is None:
        # Do not create fake historical fills during first initialization.
        pair_state["last_processed_candle"] = latest_closed
        new_completed = pd.DataFrame()
    else:
        previous_ts = _timestamp(previous_last)
        earliest = _timestamp(completed["datetime"].iloc[0])
        latest = _timestamp(completed["datetime"].iloc[-1])
        if previous_ts < earliest:
            raise RuntimeError(
                "Paper state is older than the provider's retained history. "
                "Refusing to replay old bars; inspect/reset paper-trading state."
            )
        if previous_ts > latest:
            raise RuntimeError("Paper state timestamp is ahead of latest market data.")
        new_completed = completed.loc[completed_times > previous_ts].copy()
        new_completed.reset_index(drop=True, inplace=True)
        pair_state["last_processed_candle"] = latest_closed

    actions, events, closed_trades = [], [], []
    for _, candle in new_completed.iterrows():
        candle_actions = process_completed_candle(pair, pair_state, candle, paper_cfg)
        actions.extend(candle_actions)
        closed_trades.extend(item for item in candle_actions if item.get("type") == "CLOSED")
        events.append({
            "event_time_utc": utc_now(), "pair": pair,
            "completed_candle_time": _iso(candle["datetime"]),
            "status": "POSITION_UPDATED" if candle_actions else "NO_OPEN_POSITION",
            "actions": candle_actions, "simulation_only": True,
        })

    latest_signal = infer_feature_signals(
        features, model, scaler, feature_cols, float(cfg["confidence_threshold"])
    )[-1]
    forming_bar = forming.iloc[-1] if not forming.empty else None
    if forming_bar is not None and _timestamp(forming_bar["datetime"]) <= _timestamp(latest_signal["candle_start"]):
        forming_bar = None

    if pair_state.get("position") is None and forming_bar is not None:
        if latest_signal["signal"] in ("BUY", "SELL"):
            actions.append(open_position(pair, pair_state, forming_bar, latest_signal, cfg))
        else:
            actions.append({
                "type": "NO_ENTRY", "pair": pair,
                "signal_time": latest_signal["timestamp"], "signal": "FLAT",
                "confidence": latest_signal["confidence"],
                "reason": "Confidence threshold not reached.", "simulation_only": True,
            })
    elif pair_state.get("position") is not None:
        actions.append({
            "type": "POSITION_ALREADY_OPEN", "pair": pair,
            "position": pair_state["position"], "simulation_only": True,
        })
    else:
        actions.append({
            "type": "NO_FORMING_CANDLE", "pair": pair,
            "signal_time": latest_signal["timestamp"], "signal": latest_signal["signal"],
            "reason": "No later forming 1h candle is available for a non-lookahead entry.",
            "simulation_only": True,
        })

    state["updated_at_utc"] = utc_now()
    state["pairs"][pair] = pair_state
    _atomic_json(state_path, state)
    for trade in closed_trades:
        _append_jsonl(TRADES_PATH, trade)
    for event in events:
        _append_jsonl(EVENTS_PATH, event)
    summary = {
        "run_time_utc": utc_now(), "pair": pair, "timeframe": cfg["timeframe"],
        "latest_completed_candle": latest_closed, "last_signal": latest_signal,
        "completed_candles_processed": len(new_completed), "new_actions": actions,
        "equity_quote": float(pair_state["equity"]), "quote_currency": quote_currency(pair),
        "realized_pnl_quote": float(pair_state["realized_pnl_quote"]),
        "closed_trades": int(pair_state["closed_trades"]),
        "wins": int(pair_state["wins"]), "losses": int(pair_state["losses"]),
        "open_position": pair_state.get("position"), "simulation_only": True,
    }
    _append_jsonl(RUNS_PATH, summary)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one non-executing paper-trading cycle.")
    parser.add_argument("--pair", default=None, help="Trained pair, e.g. SOL-USD or EURUSD=X.")
    args = parser.parse_args()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRADES_PATH, RUNS_PATH, EVENTS_PATH):
        path.touch(exist_ok=True)
    try:
        base_cfg = load_config()
        pair = normalize_pair(args.pair or base_cfg["pair"])
        state = load_state()
        # Persist a clean baseline before contacting the provider/model so the
        # workflow can still record diagnostics if the first prediction fails.
        if not STATE_PATH.exists():
            _atomic_json(STATE_PATH, state)
        run_pair(pair, state)
    except Exception as exc:
        error_record = {
            "run_time_utc": utc_now(), "pair": args.pair, "status": "ERROR",
            "error_type": type(exc).__name__, "error": str(exc), "simulation_only": True,
        }
        _append_jsonl(RUNS_PATH, error_record)
        print(json.dumps(error_record, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
