import unittest

from app import parse_prediction_output


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


if __name__ == "__main__":
    unittest.main()
