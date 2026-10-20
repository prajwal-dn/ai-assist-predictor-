import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.paper_trade import (
    _atomic_json, close_position, initial_pair_state, load_state,
    normalize_pair, open_position, position_exit, process_completed_candle,
    quote_currency,
)


def test_config():
    return {
        "forecast_horizon": 2,
        "risk": {"max_risk_per_trade": 0.01, "atr_stop_multiplier": 1.5},
        "paper_trading": {
            "starting_balance": 10000, "spread_bps": 0,
            "commission_bps": 0, "slippage_bps": 0,
        },
    }


def bar(time, op, high, low, close):
    return pd.Series({
        "datetime": pd.Timestamp(time),
        "open": float(op), "high": float(high), "low": float(low), "close": float(close),
    })


def buy_signal(time="2025-01-01T01:00:00", atr=2.0):
    return {
        "timestamp": time, "candle_start": "2025-01-01T00:00:00",
        "signal": "BUY", "prob_up": 0.8, "prob_down": 0.2,
        "confidence": 0.8, "atr": atr,
    }


class PaperTradingUnitTests(unittest.TestCase):
    def test_pair_normalization_and_quote_currency(self):
        self.assertEqual(normalize_pair("EURUSD"), "EURUSD=X")
        self.assertEqual(normalize_pair("sol-usd"), "SOL-USD")
        self.assertEqual(quote_currency("EURUSD=X"), "USD")
        self.assertEqual(quote_currency("SOL-USD"), "USD")

    def test_opening_uses_next_bar_open_and_atr_risk(self):
        state = initial_pair_state(10000)
        action = open_position(
            "SOL-USD", state, bar("2025-01-01T02:00:00", 98, 101, 97, 100),
            buy_signal(), test_config(), entered_at="2025-01-01T02:07:00Z"
        )
        self.assertEqual(action["type"], "OPENED")
        # The sampled live price is the current candle close/last price, not
        # the candle open from seven minutes earlier.
        self.assertEqual(action["entry_price"], 100)
        self.assertEqual(state["position"]["entry_time"], "2025-01-01T02:07:00Z")
        self.assertEqual(state["position"]["stop_loss"], 97)
        self.assertEqual(state["position"]["take_profit"], 106)
        self.assertAlmostEqual(state["position"]["risk_amount_quote"], 100)
        self.assertEqual(state["equity"], 10000)

    def test_stop_loss_closes_position_and_updates_equity(self):
        cfg = test_config()
        state = initial_pair_state(10000)
        open_position("SOL-USD", state, bar("2025-01-01T01:00:00", 100, 101, 99, 100), buy_signal(), cfg)
        first = position_exit(state["position"], bar("2025-01-01T01:00:00", 100, 101, 99, 100), 2)
        self.assertIsNone(first)
        result = position_exit(state["position"], bar("2025-01-01T02:00:00", 98, 99, 96, 97), 2)
        self.assertEqual(result, (97.0, "STOP_LOSS"))
        trade = close_position("SOL-USD", state, bar("2025-01-01T02:00:00", 98, 99, 96, 97), *result, cfg["paper_trading"])
        self.assertAlmostEqual(trade["net_pnl_quote"], -100)
        self.assertAlmostEqual(state["equity"], 9900)
        self.assertIsNone(state["position"])

    def test_estimated_costs_are_deducted(self):
        cfg = test_config()
        cfg["paper_trading"]["spread_bps"] = 10
        state = initial_pair_state(10000)
        open_position("SOL-USD", state, bar("2025-01-01T01:00:00", 100, 101, 99, 100), buy_signal(), cfg)
        trade = close_position(
            "SOL-USD", state, bar("2025-01-01T02:00:00", 100, 110, 99, 108),
            106, "TAKE_PROFIT", cfg["paper_trading"]
        )
        self.assertLess(trade["net_pnl_quote"], trade["gross_pnl_quote"])
        self.assertGreater(trade["estimated_cost_quote"], 0)

    def test_if_stop_and_target_hit_same_candle_stop_wins(self):
        state = initial_pair_state(10000)
        open_position("SOL-USD", state, bar("2025-01-01T01:00:00", 100, 101, 99, 100), buy_signal(), test_config())
        result = position_exit(state["position"], bar("2025-01-01T02:00:00", 100, 107, 96, 100), 2)
        self.assertEqual(result, (97.0, "STOP_LOSS_AMBIGUOUS"))

    def test_horizon_exits_at_close(self):
        state = initial_pair_state(10000)
        open_position("SOL-USD", state, bar("2025-01-01T01:00:00", 100, 101, 99, 100), buy_signal(), test_config())
        self.assertIsNone(position_exit(state["position"], bar("2025-01-01T01:00:00", 100, 101, 99, 100), 2))
        self.assertEqual(position_exit(state["position"], bar("2025-01-01T02:00:00", 100, 101, 99, 100), 2), (100.0, "HORIZON"))

    def test_no_position_means_no_exit_actions(self):
        state = initial_pair_state(10000)
        self.assertEqual(process_completed_candle("SOL-USD", state, bar("2025-01-01T02:00:00", 100, 101, 99, 100), test_config()["paper_trading"]), [])


    def test_partial_entry_candle_is_not_used_for_exit(self):
        cfg = test_config()
        state = initial_pair_state(10000)
        open_position(
            "SOL-USD", state, bar("2025-01-01T01:00:00", 100, 101, 99, 100),
            buy_signal(), cfg, entered_at="2025-01-01T01:07:00Z"
        )
        # These extremes happened in a candle already in progress at entry.
        actions = process_completed_candle(
            "SOL-USD", state, bar("2025-01-01T01:00:00", 100, 110, 90, 100),
            cfg["paper_trading"]
        )
        self.assertEqual(actions, [])
        self.assertIsNotNone(state["position"])

    def test_state_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            expected = {"schema_version": 1, "updated_at_utc": None, "pairs": {}}
            _atomic_json(path, expected)
            self.assertEqual(load_state(path), expected)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), expected)


if __name__ == "__main__":
    unittest.main()
