import unittest

import pandas as pd

from src.experiments.training_windows import training_bounds, windowed_train
from src.models.temporal_validation import expanding_date_splits, purged_train


class TrainingWindowTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2015-01-01", "2025-12-31")
        self.data = pd.DataFrame([(t, d, dates[i+1], i % 2) for t in ["AAPL", "MSFT"]
            for i, d in enumerate(dates[:-1])], columns=["ticker", "Date", "target_end", "target"])

    def test_all_is_existing_purge(self):
        start = pd.Timestamp("2023-10-16")
        pd.testing.assert_frame_equal(windowed_train(self.data, start), purged_train(self.data, start))

    def test_calendar_window_and_purge(self):
        for years in [3, 5]:
            start = pd.Timestamp("2024-02-29")
            fit = windowed_train(self.data, start, years)
            self.assertGreaterEqual(fit.Date.min(), start - pd.DateOffset(years=years))
            self.assertLess(fit.target_end.max(), start)
            self.assertEqual(fit.groupby("ticker").size().nunique(), 1)

    def test_each_inner_fold_has_own_window(self):
        outer = windowed_train(self.data, "2025-01-01")
        for _, dates in expanding_date_splits(outer, 3):
            valid = outer[outer.Date.isin(dates)]
            fit = windowed_train(outer, dates.min(), 3)
            record = training_bounds(fit, valid, 3)
            self.assertLess(record["train_target_end"], record["valid_start"])
            self.assertGreaterEqual(record["train_start"], dates.min() - pd.DateOffset(years=3))
            past = outer[outer.Date < dates.min()]
            pd.testing.assert_frame_equal(fit, windowed_train(past, dates.min(), 3))

    def test_invalid_and_empty_windows_rejected(self):
        with self.assertRaises(ValueError):
            windowed_train(self.data, "2024-01-01", 1)
        with self.assertRaises(ValueError):
            windowed_train(self.data, "2010-01-01", 3)


if __name__ == "__main__":
    unittest.main()
