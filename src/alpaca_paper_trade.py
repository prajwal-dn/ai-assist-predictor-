"""Alpaca-backed crypto paper trading. The trading endpoint is hard-locked to paper."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = BASE_DIR / "paper-trading"
STATE_PATH = PAPER_DIR / "alpaca-state.json"
RUNS_PATH = PAPER_DIR / "alpaca-runs.jsonl"
ORDERS_PATH = PAPER_DIR / "alpaca-orders.jsonl"

# Deliberately constant: never take an API endpoint from a secret, variable, or CLI.
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALLOWED_PAIRS = {"SOL-USD": "SOL/USD", "BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD"}
DEFAULT_MAX_NOTIONAL_USD = 25.0
ABSOLUTE_MAX_NOTIONAL_USD = 50.0
ESTIMATED_TAKER_FEE_BPS = 25.0
ATR_STOP_MULTIPLIER = 1.5
REWARD_RISK_RATIO = 2.0
TERMINAL_ORDER_STATUSES = {"filled", "canceled", "expired", "rejected"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_pair(pair: str) -> str:
    if not isinstance(pair, str):
        raise ValueError("Pair must be a string.")
    value = pair.strip().upper().replace("/", "-")
    if value in ALLOWED_PAIRS:
        return value
    if value in {"SOLUSD", "BTCUSD", "ETHUSD"}:
        return value[:-3] + "-USD"
    raise ValueError("Only SOL/USD, BTC/USD, and ETH/USD are allowed.")


def broker_symbol(pair: str) -> str:
    return ALLOWED_PAIRS[normalize_pair(pair)]


def _timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _iso(value) -> str:
    return _timestamp(value).isoformat()


def _finite_positive(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive.")
    return number


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}-", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True, allow_nan=False, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_state(path: Path = STATE_PATH) -> dict:
    if not path.exists():
        return {"schema_version": 1, "updated_at_utc": None, "pairs": {}}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or state.get("schema_version") != 1 or not isinstance(state.get("pairs"), dict):
        raise ValueError(f"Invalid Alpaca paper state file: {path}")
    return state


def client_order_id(pair: str, side: str, candle_time: str, reason: str) -> str:
    """Stable idempotency key for a single pair/decision/side."""
    pair = normalize_pair(pair)
    symbol = broker_symbol(pair).replace("/", "").lower()
    digest = hashlib.sha256(f"{pair}|{side}|{candle_time}|{reason}".encode("utf-8")).hexdigest()[:20]
    return f"ap-{symbol}-{side[0]}-{digest}"[:48]


def find_exit_trigger(position: dict, candles: pd.DataFrame) -> dict | None:
    """Return a stop/target trigger; actual exit price will be the later market fill."""
    if candles is None or candles.empty:
        return None
    entry_candle = position.get("entry_candle_start")
    last_checked = position.get("last_checked_candle")
    stop, target = float(position["stop_loss"]), float(position["take_profit"])
    for _, candle in candles.sort_values("datetime").iterrows():
        ts = _iso(candle["datetime"])
        if entry_candle and _timestamp(ts) <= _timestamp(entry_candle):
            continue
        if last_checked and _timestamp(ts) <= _timestamp(last_checked):
            continue
        op, high, low = float(candle["open"]), float(candle["high"]), float(candle["low"])
        if op <= stop:
            return {"reason": "STOP_LOSS_GAP", "candle_time": ts}
        if op >= target:
            return {"reason": "TAKE_PROFIT_GAP", "candle_time": ts}
        hit_stop, hit_target = low <= stop, high >= target
        if hit_stop and hit_target:
            return {"reason": "STOP_LOSS_AMBIGUOUS", "candle_time": ts}
        if hit_stop:
            return {"reason": "STOP_LOSS", "candle_time": ts}
        if hit_target:
            return {"reason": "TAKE_PROFIT", "candle_time": ts}
    return None


class AlpacaPaperClient:
    """REST client with a non-overridable paper-trading host."""

    def __init__(self, key: str, secret: str, session=None):
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Missing ALPACA_PAPER_API_KEY GitHub Actions secret.")
        if not isinstance(secret, str) or not secret.strip():
            raise ValueError("Missing ALPACA_PAPER_API_SECRET GitHub Actions secret.")
        self._key, self._secret = key.strip(), secret.strip()
        self._session = session or requests.Session()

    def request(self, method: str, path: str, params: dict | None = None,
                body: dict | None = None, allow_404: bool = False):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Invalid Alpaca API path.")
        response = self._session.request(
            method, ALPACA_PAPER_BASE_URL + path,
            headers={
                "APCA-API-KEY-ID": self._key,
                "APCA-API-SECRET-KEY": self._secret,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            params=params, json=body, timeout=(5, 15),
        )
        if response.status_code == 404 and allow_404:
            return None
        try:
            payload = response.json() if response.content else {}
        except ValueError:
            payload = {}
        if response.status_code >= 400:
            message = payload.get("message", "request failed") if isinstance(payload, dict) else "request failed"
            raise RuntimeError(f"Alpaca paper API {method} {path} failed ({response.status_code}): {message}")
        return payload

    def validate_account(self) -> dict:
        account = self.request("GET", "/v2/account")
        if not isinstance(account, dict):
            raise RuntimeError("Alpaca paper account response was invalid.")
        if account.get("status") != "ACTIVE":
            raise RuntimeError(f"Alpaca paper account is not active (status={account.get('status')!r}).")
        if account.get("trading_blocked") is True:
            raise RuntimeError("Alpaca paper account reports trading_blocked=true.")
        return account

    def tradable_crypto_symbols(self) -> set[str]:
        assets = self.request("GET", "/v2/assets", params={"asset_class": "crypto"})
        if not isinstance(assets, list):
            raise RuntimeError("Alpaca crypto-asset response was invalid.")
        return {
            str(item.get("symbol", "")).upper().replace("/", "")
            for item in assets if isinstance(item, dict)
            and item.get("status") == "active" and item.get("tradable") is True
        }

    def positions(self) -> list[dict]:
        items = self.request("GET", "/v2/positions")
        if not isinstance(items, list):
            raise RuntimeError("Alpaca positions response was invalid.")
        return items

    def open_orders(self) -> list[dict]:
        items = self.request("GET", "/v2/orders", params={"status": "open", "limit": 500, "direction": "desc"})
        if not isinstance(items, list):
            raise RuntimeError("Alpaca open-orders response was invalid.")
        return items

    def get_order(self, order_id: str) -> dict:
        result = self.request("GET", f"/v2/orders/{quote(order_id, safe='')}")
        if not isinstance(result, dict):
            raise RuntimeError("Alpaca order response was invalid.")
        return result

    def get_order_by_client_id(self, order_id: str) -> dict | None:
        result = self.request(
            "GET", f"/v2/orders:by_client_order_id/{quote(order_id, safe='')}", allow_404=True
        )
        if result is not None and not isinstance(result, dict):
            raise RuntimeError("Alpaca client-order lookup response was invalid.")
        return result

    def submit_market_order(self, pair: str, side: str, *, client_id: str,
                            qty: str | None = None, notional: str | None = None) -> dict:
        if ALPACA_PAPER_BASE_URL != "https://paper-api.alpaca.markets":
            raise RuntimeError("Paper trading host guard failed; refusing order submission.")
        if side not in {"buy", "sell"}:
            raise ValueError("Only buy/sell sides are allowed.")
        if (qty is None) == (notional is None):
            raise ValueError("Specify exactly one of qty or notional.")
        if side == "buy" and notional is None:
            raise ValueError("Buy entries must use a USD notional.")
        if side == "sell" and qty is None:
            raise ValueError("Sell exits must use the held crypto quantity.")
        body = {
            "symbol": broker_symbol(pair), "side": side, "type": "market",
            "time_in_force": "gtc", "client_order_id": client_id,
        }
        if qty is not None:
            body["qty"] = str(qty)
        if notional is not None:
            body["notional"] = str(notional)
        result = self.request("POST", "/v2/orders", body=body)
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("Alpaca did not return a valid order identifier.")
        return result


def _order_qty(order: dict) -> float:
    try:
        return float(order.get("filled_qty") or 0)
    except (TypeError, ValueError):
        return 0.0


def _order_fill_price(order: dict) -> float:
    price = order.get("filled_avg_price")
    if price in (None, ""):
        raise RuntimeError("Order is marked filled but has no average fill price.")
    return _finite_positive(price, "filled_avg_price")


def _make_managed_position(pair: str, pending: dict, order: dict) -> dict:
    qty, price = _order_qty(order), _order_fill_price(order)
    if qty <= 0:
        raise RuntimeError("Filled buy order returned a non-positive filled quantity.")
    atr = _finite_positive(pending["atr"], "signal ATR")
    risk_distance = atr * ATR_STOP_MULTIPLIER
    return {
        "pair": pair, "symbol": broker_symbol(pair), "qty": qty, "entry_price": price,
        "entry_order_id": order["id"], "entry_client_order_id": pending["client_order_id"],
        "entry_time": str(order.get("filled_at") or utc_now()),
        "entry_signal_time": pending["signal_time"],
        "entry_candle_start": pending["entry_candle_start"],
        "last_checked_candle": pending["entry_candle_start"],
        "atr_at_entry": atr, "stop_loss": price - risk_distance,
        "take_profit": price + REWARD_RISK_RATIO * risk_distance,
        "estimated_entry_fee_usd": qty * price * ESTIMATED_TAKER_FEE_BPS / 10_000.0,
        "simulation_only": True,
    }


def _finalize_pending(pair: str, pair_state: dict, order: dict) -> dict:
    pending = pair_state.get("pending_order")
    if not pending:
        return {"status": "NO_PENDING_ORDER", "pair": pair}
    status, side = str(order.get("status", "")).lower(), pending["side"]
    if status not in TERMINAL_ORDER_STATUSES:
        pair_state["pending_order"] = {
            **pending, "order_id": order.get("id", pending.get("order_id")), "last_status": status
        }
        return {"status": "ORDER_PENDING", "pair": pair,
                "client_order_id": pending["client_order_id"], "order_status": status}
    if status != "filled":
        pair_state["pending_order"] = None
        return {"status": "ORDER_NOT_FILLED", "pair": pair,
                "client_order_id": pending["client_order_id"], "order_status": status,
                "reason": order.get("reject_reason")}

    if side == "buy":
        pair_state["position"] = _make_managed_position(pair, pending, order)
        pair_state["pending_order"] = None
        return {"status": "BUY_FILLED", "pair": pair, "symbol": broker_symbol(pair),
                "qty": pair_state["position"]["qty"], "filled_avg_price": pair_state["position"]["entry_price"],
                "order_id": order["id"], "simulation_only": True}

    position = pair_state.get("position")
    if not isinstance(position, dict):
        raise RuntimeError("Sell filled but managed position state is missing; inspect the broker paper account.")
    exit_qty, exit_price = _order_qty(order), _order_fill_price(order)
    if exit_qty <= 0:
        raise RuntimeError("Filled sell order returned a non-positive filled quantity.")
    if not math.isclose(exit_qty, float(position["qty"]), rel_tol=1e-4, abs_tol=1e-8):
        raise RuntimeError("Sell fill quantity differs from the managed position. Refusing to guess remaining exposure.")
    gross_pnl = (exit_price - float(position["entry_price"])) * exit_qty
    exit_fee = exit_qty * exit_price * ESTIMATED_TAKER_FEE_BPS / 10_000.0
    trade = {
        "pair": pair, "symbol": broker_symbol(pair), "side": "long",
        "entry_order_id": position["entry_order_id"], "exit_order_id": order["id"],
        "entry_time": position["entry_time"], "exit_time": order.get("filled_at") or utc_now(),
        "entry_price": float(position["entry_price"]), "exit_price": exit_price, "qty": exit_qty,
        "gross_pnl_usd": gross_pnl,
        "estimated_fees_usd": float(position.get("estimated_entry_fee_usd", 0.0)) + exit_fee,
        "estimated_taker_fee_bps_per_side": ESTIMATED_TAKER_FEE_BPS,
        "exit_reason": pending.get("exit_reason", "MODEL_SELL_SIGNAL"), "simulation_only": True,
    }
    trade["estimated_net_pnl_usd"] = trade["gross_pnl_usd"] - trade["estimated_fees_usd"]
    _append_jsonl(ORDERS_PATH, {"event": "TRADE_CLOSED", **trade})
    pair_state["position"], pair_state["pending_order"] = None, None
    return {"status": "SELL_FILLED", "pair": pair, "symbol": broker_symbol(pair),
            "qty": exit_qty, "filled_avg_price": exit_price, "trade": trade, "simulation_only": True}


def _persist_state(state: dict, state_path: Path) -> None:
    _atomic_json(state_path, {"schema_version": 1, "updated_at_utc": utc_now(), "pairs": state["pairs"]})


def _submit_or_resume(pair: str, pair_state: dict, state: dict, client: AlpacaPaperClient,
                      pending: dict, execute: bool, state_path: Path) -> dict:
    if not execute:
        return {"status": "WOULD_SUBMIT_" + pending["side"].upper(), "pair": pair,
                "symbol": broker_symbol(pair), "reason": pending.get("exit_reason", "MODEL_SIGNAL"),
                "notional_usd": pending.get("notional_usd"), "simulation_only": True}
    # Persist intent before POST. A failed/ambiguous response is reconciled by stable client_order_id.
    pair_state["pending_order"] = pending
    _persist_state(state, state_path)
    existing = client.get_order_by_client_id(pending["client_order_id"])
    if existing is not None:
        order = existing
    elif pending["side"] == "buy":
        order = client.submit_market_order(
            pair, "buy", client_id=pending["client_order_id"],
            notional=f"{float(pending['notional_usd']):.2f}",
        )
    else:
        order = client.submit_market_order(
            pair, "sell", client_id=pending["client_order_id"],
            qty=format(Decimal(str(pending["qty"])).normalize(), "f"),
        )
    pair_state["pending_order"] = {
        **pending, "order_id": order.get("id"), "last_status": order.get("status", "unknown")
    }
    _persist_state(state, state_path)
    result = order
    for _ in range(6):
        if str(result.get("status", "")).lower() in TERMINAL_ORDER_STATUSES:
            break
        if not result.get("id"):
            break
        time.sleep(2)
        result = client.get_order(result["id"])
    return _finalize_pending(pair, pair_state, result)


def _current_hour_start() -> str:
    return pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None).isoformat()


def _get_model_signal(pair: str) -> tuple[dict, pd.DataFrame]:
    # Reuse the project's validated model + market-candle pipeline.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paper_trade import _prepare_market, infer_feature_signals
    from predict import load_config

    os.environ["ASSET_PREDICTOR_PAIR"] = pair
    cfg = load_config()
    if cfg.get("timeframe") != "1h":
        raise RuntimeError("Alpaca automated paper trading is restricted to 1h models.")
    model_dir = Path(cfg["paths"]["models"])
    if not model_dir.is_absolute():
        model_dir = BASE_DIR / model_dir
    model, scaler, cols, completed, _forming, features = _prepare_market(pair, cfg, model_dir)
    latest = _timestamp(completed["datetime"].iloc[-1])
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if now - (latest + pd.Timedelta(hours=1)) > pd.Timedelta(minutes=120):
        raise RuntimeError("Latest completed market candle is stale; refusing to trade.")
    signal = infer_feature_signals(features, model, scaler, cols, float(cfg["confidence_threshold"]))[-1]
    signal["atr"] = _finite_positive(signal["atr"], "signal ATR")
    signal["candle_start"] = _iso(signal["candle_start"])
    signal["timestamp"] = _iso(signal["timestamp"])
    return signal, completed


def _match_position(positions: list[dict], pair: str) -> dict | None:
    expected = broker_symbol(pair).replace("/", "")
    for position in positions:
        actual = str(position.get("symbol", "")).upper().replace("/", "")
        if actual == expected and float(position.get("qty", 0) or 0) > 0:
            return position
    return None


def _has_open_order(open_orders: list[dict], pair: str) -> dict | None:
    expected = broker_symbol(pair).replace("/", "")
    for order in open_orders:
        if str(order.get("symbol", "")).upper().replace("/", "") == expected:
            return order
    return None


def _pair_state(state: dict, pair: str) -> dict:
    return state["pairs"].setdefault(pair, {
        "position": None, "pending_order": None, "last_processed_candle": None,
    })


def run_pair(pair_arg: str, state: dict, client: AlpacaPaperClient, available_symbols: set[str],
             max_notional_usd: float, execute: bool, positions: list[dict],
             open_orders: list[dict], state_path: Path) -> dict:
    pair, symbol = normalize_pair(pair_arg), broker_symbol(pair_arg)
    pair_state = _pair_state(state, pair)
    if symbol.replace("/", "") not in available_symbols:
        return {"status": "ASSET_NOT_TRADABLE", "pair": pair, "symbol": symbol}

    pending = pair_state.get("pending_order")
    if pending:
        found = client.get_order_by_client_id(pending["client_order_id"])
        if found is None:
            return {"status": "PENDING_ORDER_NOT_FOUND_MANUAL_REVIEW", "pair": pair,
                    "client_order_id": pending["client_order_id"]}
        return _finalize_pending(pair, pair_state, found)

    external_order = _has_open_order(open_orders, pair)
    if external_order:
        return {"status": "OPEN_ORDER_PRESENT", "pair": pair, "order_id": external_order.get("id"),
                "order_status": external_order.get("status")}

    signal, completed = _get_model_signal(pair)
    managed = pair_state.get("position")
    actual = _match_position(positions, pair)
    if managed is None and actual is not None:
        return {"status": "UNMANAGED_POSITION_PRESENT", "pair": pair, "symbol": symbol,
                "message": "Existing account holding is not managed by this strategy; no order submitted."}
    if managed is not None and actual is None:
        pair_state["position"] = None
        pair_state["last_processed_candle"] = signal["candle_start"]
        return {"status": "POSITION_MISSING_RECONCILED", "pair": pair,
                "message": "Managed state existed but Alpaca has no matching holding. Cleared state; no new order this cycle."}
    if managed is not None and actual is not None:
        actual_qty = float(actual.get("qty", 0) or 0)
        if not math.isclose(actual_qty, float(managed["qty"]), rel_tol=1e-4, abs_tol=1e-8):
            return {"status": "POSITION_QTY_MISMATCH", "pair": pair, "managed_qty": managed["qty"],
                    "broker_qty": actual_qty, "message": "Manual review required; no order submitted."}
        cursor = managed.get("last_checked_candle") or managed.get("entry_candle_start")
        unseen = completed
        if cursor:
            unseen = completed.loc[pd.to_datetime(completed["datetime"], errors="coerce") > _timestamp(cursor)].copy()
        trigger = find_exit_trigger(managed, unseen)
        reason = trigger["reason"] if trigger else None
        if reason is None and signal["signal"] == "SELL":
            reason = "MODEL_SELL_SIGNAL"
        if reason:
            exit_pending = {
                "side": "sell",
                "client_order_id": client_order_id(pair, "sell", signal["candle_start"], reason),
                "qty": actual_qty, "exit_reason": reason, "signal_time": signal["candle_start"],
            }
            if not execute:
                return {"status": "WOULD_EXIT", "pair": pair, "symbol": symbol,
                        "reason": reason, "signal": signal["signal"], "simulation_only": True}
            return _submit_or_resume(pair, pair_state, state, client, exit_pending, True, state_path)
        if execute and not unseen.empty:
            managed["last_checked_candle"] = _iso(unseen["datetime"].iloc[-1])
            pair_state["last_processed_candle"] = managed["last_checked_candle"]
        return {"status": "HOLD", "pair": pair, "symbol": symbol, "signal": signal["signal"],
                "confidence": signal["confidence"], "entry_price": managed["entry_price"],
                "stop_loss": managed["stop_loss"], "take_profit": managed["take_profit"],
                "qty": managed["qty"], "simulation_only": True}

    if signal["signal"] != "BUY":
        return {"status": "NO_ENTRY", "pair": pair, "symbol": symbol, "signal": signal["signal"],
                "confidence": signal["confidence"], "reason": "Long-only strategy does not open shorts.",
                "simulation_only": True}
    if not math.isfinite(float(max_notional_usd)) or not 0 < max_notional_usd <= ABSOLUTE_MAX_NOTIONAL_USD:
        raise ValueError(f"Per-entry paper notional must be greater than zero and no more than {ABSOLUTE_MAX_NOTIONAL_USD:.2f} USD.")
    entry_pending = {
        "side": "buy",
        "client_order_id": client_order_id(pair, "buy", signal["candle_start"], "MODEL_BUY"),
        "notional_usd": float(max_notional_usd), "signal_time": signal["candle_start"],
        "atr": signal["atr"], "entry_candle_start": _current_hour_start(),
    }
    if not execute:
        return {"status": "WOULD_BUY", "pair": pair, "symbol": symbol,
                "notional_usd": float(max_notional_usd), "signal": signal["signal"],
                "confidence": signal["confidence"], "signal_time": signal["candle_start"],
                "simulation_only": True}
    pair_state["pending_order"] = entry_pending
    return _submit_or_resume(pair, pair_state, state, client, entry_pending, True, state_path)


def run_once(pairs: list[str], *, execute: bool, max_notional_usd: float,
             state_path: Path = STATE_PATH, key: str | None = None, secret: str | None = None) -> dict:
    state = load_state(state_path)
    client = AlpacaPaperClient(
        key if key is not None else os.environ.get("ALPACA_PAPER_API_KEY", ""),
        secret if secret is not None else os.environ.get("ALPACA_PAPER_API_SECRET", ""),
    )
    account = client.validate_account()
    available, positions, open_orders = (
        client.tradable_crypto_symbols(), client.positions(), client.open_orders()
    )
    results = []
    for pair_arg in pairs:
        pair = normalize_pair(pair_arg)
        try:
            result = run_pair(pair, state, client, available, max_notional_usd,
                              execute, positions, open_orders, state_path)
        except Exception as exc:
            result = {"status": "ERROR", "pair": pair, "error_type": type(exc).__name__,
                      "error": str(exc), "simulation_only": True}
        result["run_time_utc"] = utc_now()
        results.append(result)
        _append_jsonl(ORDERS_PATH, result)
        _persist_state(state, state_path)
    summary = {
        "run_time_utc": utc_now(),
        "mode": "PAPER_ORDER_SUBMISSION_ENABLED" if execute else "VALIDATION_ONLY",
        "account_status": account.get("status"), "paper_endpoint": ALPACA_PAPER_BASE_URL,
        "pairs": results, "simulation_only": True,
    }
    _append_jsonl(RUNS_PATH, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if any(item.get("status") == "ERROR" for item in results):
        raise RuntimeError("One or more crypto pairs failed; inspect the run report.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Use saved Asset Predictor models with Alpaca paper trading only.")
    parser.add_argument("--pairs", default="ALL", help="ALL or comma-separated SOL/USD,BTC/USD,ETH/USD")
    parser.add_argument("--execute", action="store_true", help="Submit simulated orders to the hard-coded paper endpoint.")
    parser.add_argument("--max-notional-usd", default=None,
                        help="Per-entry USD notional (default 25; absolute max 50).")
    args = parser.parse_args()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for path in (RUNS_PATH, ORDERS_PATH):
        path.touch(exist_ok=True)
    try:
        raw = args.pairs.strip().upper()
        pairs = list(ALLOWED_PAIRS) if raw == "ALL" else list(dict.fromkeys(
            normalize_pair(item) for item in raw.split(",") if item.strip()
        ))
        if not pairs:
            raise ValueError("Select at least one allowed crypto pair.")
        notional = float(args.max_notional_usd if args.max_notional_usd is not None
                         else os.environ.get("ALPACA_PAPER_MAX_NOTIONAL_USD", DEFAULT_MAX_NOTIONAL_USD))
        _finite_positive(notional, "max_notional_usd")
        if notional > ABSOLUTE_MAX_NOTIONAL_USD:
            raise ValueError(f"Per-entry paper notional may not exceed {ABSOLUTE_MAX_NOTIONAL_USD:.2f} USD.")
        run_once(pairs, execute=args.execute, max_notional_usd=notional)
        return 0
    except Exception as exc:
        failure = {"run_time_utc": utc_now(), "status": "ERROR", "error_type": type(exc).__name__,
                   "error": str(exc), "paper_endpoint": ALPACA_PAPER_BASE_URL, "simulation_only": True}
        _append_jsonl(RUNS_PATH, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
