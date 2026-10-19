import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.feature_engineer import add_features
from src.labeller import add_labels, split_and_save
from src.backtest import simulate_trades, quote_currency
from src.predict import _drop_incomplete_candles, fetch_recent_data, load_model_artifacts
import src.db as db_module


def feature_config():
    return {"features": {
        "rsi_period": 14, "ma_short": 10, "ma_long": 50,
        "atr_period": 14, "bb_period": 20, "bb_std": 2,
        "daily_context_mode": "completed",
    }}



class HistoryPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.history_path = Path(self.tempdir.name) / "history.json"
        self.previous_cache = db_module._cache
        db_module._cache = None
        self.path_patcher = patch.object(db_module, "_get_local_path", return_value=str(self.history_path))
        self.repo_patcher = patch.object(db_module, "_get_repo_id", return_value=None)
        self.api_patcher = patch.object(db_module, "_get_api", return_value=None)
        self.path_patcher.start()
        self.repo_patcher.start()
        self.api_patcher.start()

    def tearDown(self):
        self.path_patcher.stop()
        self.repo_patcher.stop()
        self.api_patcher.stop()
        db_module._cache = self.previous_cache
        self.tempdir.cleanup()

    def test_history_file_is_atomically_written_as_json(self):
        db_module._cache = [{"id": 1, "trade_signal": "BUY"}]
        db_module._save_data()
        self.assertEqual(__import__("json").loads(self.history_path.read_text(encoding="utf-8")), db_module._cache)
        self.assertFalse(list(Path(self.tempdir.name).glob(".history-*.tmp")))

    def test_corrupt_history_is_preserved_for_recovery(self):
        self.history_path.write_text("{corrupt json", encoding="utf-8")
        self.assertEqual(db_module._load_data(), [])
        backups = list(Path(self.tempdir.name).glob("history.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), "{corrupt json")
        self.assertFalse(self.history_path.exists())


class FeatureEngineeringTests(unittest.TestCase):
    def test_completed_daily_context_uses_previous_day_only(self):
        dates = pd.date_range("2025-01-01", periods=180, freq="D")
        close = np.arange(1.0, 181.0)
        daily = pd.DataFrame({"datetime": dates, "close": close})
        rows = pd.DataFrame({
            "datetime": dates, "open": close - 0.2, "high": close + 1.0,
            "low": close - 1.0, "close": close, "volume": np.arange(1000.0, 1180.0),
        })
        features = add_features(rows, feature_config(), daily, daily_context_mode="completed")
        match = features.loc[features["datetime"] == pd.Timestamp("2025-04-11")]
        self.assertEqual(len(match), 1)
        self.assertAlmostEqual(float(match.iloc[0]["daily_sma"]), float(np.mean(close[50:100])))
        self.assertEqual(int(match.iloc[0]["feature_schema_version"]), 2)

    def test_latest_intraday_day_uses_last_completed_daily_context(self):
        daily_dates = pd.date_range("2025-01-01", periods=180, freq="D")
        daily_close = np.arange(1.0, 181.0)
        daily = pd.DataFrame({"datetime": daily_dates, "close": daily_close})

        # Intraday data includes the next date while the daily provider only
        # includes the last completed bar. That new date must still get context.
        row_dates = pd.date_range("2025-01-01", periods=181, freq="D")
        row_close = np.arange(1.0, 182.0)
        rows = pd.DataFrame({
            "datetime": row_dates, "open": row_close - 0.2,
            "high": row_close + 1, "low": row_close - 1,
            "close": row_close, "volume": np.ones(len(row_close)),
        })
        features = add_features(rows, feature_config(), daily, daily_context_mode="completed")
        newest = features.iloc[-1]
        self.assertEqual(newest["datetime"], row_dates[-1])
        self.assertTrue(np.isfinite(float(newest["daily_sma"])))
        self.assertAlmostEqual(float(newest["daily_sma"]), float(np.mean(daily_close[130:180])))

    def test_legacy_daily_context_keeps_schema_v1(self):
        dates = pd.date_range("2025-01-01", periods=180, freq="D")
        close = np.arange(1.0, 181.0)
        daily = pd.DataFrame({"datetime": dates, "close": close})
        rows = pd.DataFrame({
            "datetime": dates, "open": close - .2, "high": close + 1,
            "low": close - 1, "close": close, "volume": np.ones(len(close)),
        })
        features = add_features(rows, feature_config(), daily, daily_context_mode="legacy")
        self.assertEqual(int(features.iloc[-1]["feature_schema_version"]), 1)



class MarketDataTests(unittest.TestCase):
    def test_incomplete_last_candle_is_removed(self):
        frame = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"], utc=True).tz_localize(None),
            "open": [10.0, 11.0], "high": [11.0, 12.0],
            "low": [9.0, 10.0], "close": [10.5, 11.5], "volume": [100.0, 100.0],
        })
        completed = _drop_incomplete_candles(
            frame, "1h", now=pd.Timestamp("2025-01-01T01:30:00Z")
        )
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed.iloc[0]["datetime"], pd.Timestamp("2025-01-01T00:00:00"))

    @patch("src.predict.requests.get")
    def test_twelvedata_timeout_and_chronological_normalization(self, mocked_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"values": [
            {"datetime": "2025-01-01 02:00:00", "open": "2", "high": "3", "low": "1", "close": "2.5"},
            {"datetime": "2025-01-01 01:00:00", "open": "1", "high": "2", "low": ".5", "close": "1.5"},
        ]}
        mocked_get.return_value = response
        result = fetch_recent_data("EURUSD=X", "1h", {"twelvedata_api_key": "test-key"})
        self.assertEqual(len(result), 2)
        self.assertLess(result.iloc[0]["datetime"], result.iloc[1]["datetime"])
        self.assertEqual(mocked_get.call_args.kwargs["timeout"], (5, 15))
        self.assertNotIn("test-key", str(mocked_get.call_args.args))

    def test_lfs_pointer_is_not_loaded_as_a_model_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("eurusd_xgboost.pkl", "eurusd_scaler.pkl", "eurusd_feature_cols.pkl"):
                Path(directory, name).write_text(
                    "version https://git-lfs.github.com/spec/v1\\noid sha256:deadbeef\\nsize 12\\n",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                load_model_artifacts(directory, "eurusd", timeframe="1h")

class LabellingTests(unittest.TestCase):
    def test_add_labels_removes_unknown_future_rows(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=4, freq="h"),
            "close": [10.0, 11.0, 12.0, 9.0],
        })
        labelled = add_labels(frame, horizon=1)
        self.assertEqual(len(labelled), 3)
        self.assertEqual(labelled["target"].tolist(), [1, 1, 0])

    def test_split_purges_forecast_horizon_from_training_tail(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=100, freq="h"),
            "target": np.tile([0, 1], 50),
        })
        with tempfile.TemporaryDirectory() as directory:
            train, test = split_and_save(frame, directory, "test", "1h", .2, 42, gap=4)
            self.assertEqual((len(train), len(test)), (76, 20))
            self.assertEqual(train["datetime"].max(), frame.iloc[75]["datetime"])
            self.assertEqual(test["datetime"].min(), frame.iloc[80]["datetime"])
            self.assertTrue((Path(directory) / "test_1h_train.csv").is_file())


class BacktestTests(unittest.TestCase):
    def test_stop_loss_is_applied_and_position_does_not_overlap(self):
        candles = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=4, freq="h"),
            "open": [100, 101, 100, 100], "high": [101, 102, 103, 101],
            "low": [99, 98, 99.5, 99], "close": [100, 98.5, 102, 100],
            "atr": [1, 1, 1, 1], "signal": [1, 0, -1, 0],
            "confidence": [.8, .6, .7, .5],
        })
        trades = simulate_trades(
            candles, {"risk": {"atr_stop_multiplier": 1, "max_risk_per_trade": .01}},
            horizon=2, starting_balance=10_000,
        )
        self.assertEqual(len(trades), 1)
        first = trades.iloc[0]
        self.assertIn(first["exit_reason"], {"STOP_LOSS", "STOP_LOSS_GAP", "STOP_LOSS_AMBIGUOUS"})
        self.assertAlmostEqual(float(first["entry_price"]), 101.0)
        self.assertEqual(first["entry_time"], pd.Timestamp("2025-01-01 01:00:00"))
        self.assertAlmostEqual(float(first["exit_price"]), 100.0)
        self.assertAlmostEqual(float(first["pnl_quote"]), -100.0)
        self.assertEqual(first["direction"], "BUY")

    def test_drawdown_includes_initial_account_balance(self):
        from src.backtest import compute_max_drawdown
        pnl = pd.Series([-1000.0, -500.0, 200.0])
        self.assertAlmostEqual(compute_max_drawdown(pnl, starting_balance=10_000.0), 10.0)

    def test_quote_currency_is_reported_correctly(self):
        self.assertEqual(quote_currency("EURUSD=X"), "USD")
        self.assertEqual(quote_currency("EURGBP=X"), "GBP")
        self.assertEqual(quote_currency("BTC-USD"), "USD")


if __name__ == "__main__":
    unittest.main()
