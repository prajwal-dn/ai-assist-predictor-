import unittest
from pathlib import Path

from app import parse_prediction_output, load_app_config, BASE_DIR, format_forecast_horizon


class PredictionOutputParserTests(unittest.TestCase):
    def report(self, prob_up, prob_down, confidence, suffix="%"):
        return "\n".join([
            "Time      : 2026-10-18T19:00", "Close     : 101.13000",
            "SIGNAL    : >>> SELL <<<", f"Prob UP   : {prob_up}{suffix}",
            f"Prob DOWN : {prob_down}{suffix}", f"Confidence: {confidence}{suffix}",
            "Stop Loss : 102.33923", "Take Profit: 98.71154",
        ])

    def test_small_percentage_is_not_scaled_twice(self):
        parsed = parse_prediction_output(self.report("0.9", "99.1", "99.1"), 0)
        self.assertFalse(parsed["error"])
        self.assertAlmostEqual(parsed["prob_up"], 0.9)
        self.assertAlmostEqual(parsed["prob_down"], 99.1)
        self.assertAlmostEqual(parsed["confidence"], 99.1)

    def test_fraction_without_percent_is_supported(self):
        parsed = parse_prediction_output(self.report("0.009", "0.991", "0.991", suffix=""), 0)
        self.assertFalse(parsed["error"])
        self.assertAlmostEqual(parsed["prob_up"], 0.9)
        self.assertAlmostEqual(parsed["prob_down"], 99.1)

    def test_nonzero_process_exit_is_an_error(self):
        self.assertTrue(parse_prediction_output(self.report("31.5", "68.5", "68.5"), 1)["error"])

    def test_missing_signal_is_rejected(self):
        output = "\n".join([
            "Time : 2026-10-18T19:00", "Close : 101.13",
            "Prob UP : 31.5%", "Prob DOWN : 68.5%", "Confidence: 68.5%"
        ])
        self.assertTrue(parse_prediction_output(output, 0)["error"])

    def test_probabilities_that_do_not_sum_to_one_are_rejected(self):
        self.assertTrue(parse_prediction_output(self.report("31.5", "80.0", "80.0"), 0)["error"])

    def test_directional_signal_requires_valid_stop_and_target(self):
        output = "\n".join([
            "Time : 2026-10-18T19:00", "Close : 101.13", "SIGNAL : BUY",
            "Prob UP : 70%", "Prob DOWN : 30%", "Confidence: 70%",
        ])
        self.assertTrue(parse_prediction_output(output, 0)["error"])

    def test_confidence_must_match_the_larger_probability(self):
        output = self.report("31.5", "68.5", "98.0")
        self.assertTrue(parse_prediction_output(output, 0)["error"])

    def test_buy_stop_and_target_must_bracket_close(self):
        output = "\n".join([
            "Time : 2026-10-18T19:00", "Close : 101.13", "SIGNAL : BUY",
            "Prob UP : 70%", "Prob DOWN : 30%", "Confidence: 70%",
            "Stop Loss : 102.00", "Take Profit: 98.00",
        ])
        self.assertTrue(parse_prediction_output(output, 0)["error"])

    def test_request_scoped_pair_does_not_mutate_config(self):
        config_path = Path(BASE_DIR) / "config.yaml"
        original = config_path.read_text(encoding="utf-8")
        cfg = load_app_config("EURUSD")
        after = config_path.read_text(encoding="utf-8")
        self.assertEqual(cfg["pair"], "EURUSD=X")
        self.assertEqual(cfg["confidence_threshold"], 0.55)
        self.assertEqual(after, original)

    def test_invalid_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            load_app_config("../config")

    def test_forecast_horizon_respects_candle_timeframe(self):
        self.assertEqual(format_forecast_horizon("1h", 4), "4 hours")
        self.assertEqual(format_forecast_horizon("4h", 4), "16 hours")
        self.assertEqual(format_forecast_horizon("1d", 4), "4 days")
        self.assertEqual(format_forecast_horizon("1d", 1), "1 day")



if __name__ == "__main__":
    unittest.main()
