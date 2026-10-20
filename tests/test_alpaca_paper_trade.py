import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src.alpaca_paper_trade import (
    ALPACA_PAPER_BASE_URL,
    AlpacaPaperClient,
    _make_managed_position,
    _pair_state,
    broker_symbol,
    client_order_id,
    find_exit_trigger,
    load_state,
    normalize_pair,
    run_pair,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def bar(time, op, high, low, close):
    return {"datetime": pd.Timestamp(time), "open": op, "high": high, "low": low, "close": close}


class AlpacaCryptoPaperTests(unittest.TestCase):
    def test_only_the_three_requested_crypto_pairs_are_allowed(self):
        self.assertEqual(normalize_pair("SOL/USD"), "SOL-USD")
        self.assertEqual(normalize_pair("btc-usd"), "BTC-USD")
        self.assertEqual(normalize_pair("ETHUSD"), "ETH-USD")
        self.assertEqual(broker_symbol("SOL-USD"), "SOL/USD")
        for invalid in ("DOGE/USD", "EURUSD=X", "AAPL", "", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_pair(invalid)

    def test_client_fails_closed_without_paper_secrets(self):
        with self.assertRaisesRegex(ValueError, "ALPACA_PAPER_API_KEY"):
            AlpacaPaperClient("", "secret")
        with self.assertRaisesRegex(ValueError, "ALPACA_PAPER_API_SECRET"):
            AlpacaPaperClient("key", "")

    def test_market_order_is_hard_locked_to_paper_host(self):
        session = FakeSession(FakeResponse({"id": "order-123", "status": "new"}))
        client = AlpacaPaperClient("paper-key", "paper-secret", session=session)
        result = client.submit_market_order(
            "BTC-USD", "buy", client_id="ap-btcusd-b-123", notional="25.00"
        )
        self.assertEqual(result["id"], "order-123")
        args, kwargs = session.calls[0]
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], ALPACA_PAPER_BASE_URL + "/v2/orders")
        self.assertNotIn("api.alpaca.markets/v2/orders", args[1].replace("paper-api.", ""))
        self.assertEqual(kwargs["json"]["symbol"], "BTC/USD")
        self.assertEqual(kwargs["json"]["notional"], "25.00")
        self.assertEqual(kwargs["json"]["side"], "buy")
        self.assertEqual(kwargs["json"]["type"], "market")
        self.assertEqual(kwargs["json"]["time_in_force"], "gtc")
        self.assertIn("APCA-API-KEY-ID", kwargs["headers"])

    def test_sell_orders_require_quantity_and_buy_can_use_notional(self):
        session = FakeSession(FakeResponse({"id": "order-123", "status": "new"}))
        client = AlpacaPaperClient("key", "secret", session=session)
        with self.assertRaises(ValueError):
            client.submit_market_order("SOL-USD", "sell", client_id="id", notional="25")
        with self.assertRaises(ValueError):
            client.submit_market_order("SOL-USD", "buy", client_id="id")
        client.submit_market_order("SOL-USD", "sell", client_id="id", qty="0.25")
        self.assertEqual(session.calls[0][1]["json"]["qty"], "0.25")
        self.assertNotIn("notional", session.calls[0][1]["json"])

    def test_client_order_id_is_stable_and_side_specific(self):
        a = client_order_id("SOL-USD", "buy", "2026-10-20T12:00:00", "MODEL_BUY")
        b = client_order_id("SOL/USD", "buy", "2026-10-20T12:00:00", "MODEL_BUY")
        c = client_order_id("SOL-USD", "sell", "2026-10-20T12:00:00", "MODEL_BUY")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertLessEqual(len(a), 48)

    def test_exit_trigger_skips_candle_already_in_progress_at_entry(self):
        position = {
            "entry_candle_start": "2026-10-20T10:00:00",
            "last_checked_candle": "2026-10-20T10:00:00",
            "stop_loss": 90, "take_profit": 120,
        }
        candles = pd.DataFrame([
            bar("2026-10-20T10:00:00", 100, 125, 85, 105),
            bar("2026-10-20T11:00:00", 105, 115, 95, 110),
        ])
        self.assertIsNone(find_exit_trigger(position, candles))

    def test_stop_wins_when_completed_candle_hits_stop_and_target(self):
        position = {
            "entry_candle_start": "2026-10-20T10:00:00",
            "stop_loss": 90, "take_profit": 120,
        }
        candles = pd.DataFrame([
            bar("2026-10-20T11:00:00", 100, 125, 85, 105),
        ])
        self.assertEqual(
            find_exit_trigger(position, candles),
            {"reason": "STOP_LOSS_AMBIGUOUS", "candle_time": "2026-10-20T11:00:00"},
        )

    def test_stop_gap_triggers_a_market_exit_reason(self):
        position = {
            "entry_candle_start": "2026-10-20T10:00:00",
            "stop_loss": 90, "take_profit": 120,
        }
        candles = pd.DataFrame([bar("2026-10-20T11:00:00", 85, 95, 80, 88)])
        self.assertEqual(find_exit_trigger(position, candles)["reason"], "STOP_LOSS_GAP")

    def test_buy_fill_creates_managed_long_position_with_conservative_levels(self):
        pending = {
            "client_order_id": "test-id", "atr": 2.0,
            "signal_time": "2026-10-20T10:00:00",
            "entry_candle_start": "2026-10-20T11:00:00",
        }
        order = {
            "id": "entry-order", "status": "filled", "filled_qty": "0.5",
            "filled_avg_price": "100", "filled_at": "2026-10-20T11:07:00Z",
        }
        position = _make_managed_position("SOL-USD", pending, order)
        self.assertEqual(position["qty"], 0.5)
        self.assertEqual(position["entry_price"], 100)
        self.assertEqual(position["stop_loss"], 97)
        self.assertEqual(position["take_profit"], 106)
        self.assertEqual(position["symbol"], "SOL/USD")
        self.assertTrue(position["simulation_only"])

    def test_unmanaged_broker_position_is_never_automatically_sold(self):
        state = {"schema_version": 1, "updated_at_utc": None, "pairs": {}}
        positions = [{"symbol": "BTC/USD", "qty": "0.01", "avg_entry_price": "50000"}]
        fake_client = Mock()
        fake_client.get_order_by_client_id.side_effect = AssertionError("no pending order expected")
        fake_signal = {
            "signal": "SELL", "confidence": 0.8, "atr": 100,
            "candle_start": "2026-10-20T10:00:00", "timestamp": "2026-10-20T11:00:00",
        }
        candles = pd.DataFrame([bar("2026-10-20T10:00:00", 50000, 50100, 49900, 50050)])
        with patch("src.alpaca_paper_trade._get_model_signal", return_value=(fake_signal, candles)):
            result = run_pair(
                "BTC-USD", state, fake_client, {"BTCUSD"}, 25, True,
                positions, [], Path(tempfile.gettempdir()) / "unused-alpaca-state.json"
            )
        self.assertEqual(result["status"], "UNMANAGED_POSITION_PRESENT")
        fake_client.submit_market_order.assert_not_called()

    def test_state_schema_loads_from_a_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            expected = {"schema_version": 1, "updated_at_utc": None, "pairs": {}}
            path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(load_state(path), expected)


if __name__ == "__main__":
    unittest.main()
