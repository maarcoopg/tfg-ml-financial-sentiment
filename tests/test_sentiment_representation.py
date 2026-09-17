import unittest

import numpy as np
import pandas as pd

from src.experiments.sentiment_representation import EXTRA, enriched_panel, relative_panel
from src.experiments.run_ticker_aware import ticker_features
from tests.test_ticker_aware import panel


class RepresentationTests(unittest.TestCase):
    def dataset(self):
        data = panel()
        data["has_news"] = True
        data["news_count"] = 2
        data["sentiment_mean"] = 0.2
        return data

    def test_relative_schema_reproduces_previous(self):
        data = self.dataset()
        a, schema = relative_panel(data)
        b, old = ticker_features(data)
        self.assertEqual(schema, old["relative_pooled"])
        pd.testing.assert_frame_equal(a[schema["lag_1"]], b[schema["lag_1"]])

    def test_disagreement_and_intensity_keep_opposite_news(self):
        news = pd.DataFrame({"ticker": ["AAPL", "AAPL"], "trading_date": [pd.Timestamp("2024-01-02")] * 2,
                             "sentiment_score": [-0.8, 0.8]})
        frame, schema = enriched_panel(self.dataset(), news)
        row = frame[(frame.ticker == "AAPL") & (frame.Date == "2024-01-02")].iloc[0]
        self.assertEqual(row.news_sentiment_std, 0.8)
        self.assertEqual(row.news_abs_sentiment, 0.8)
        self.assertEqual([len(schema[v]) for v in ["base", "hybrid", "lag_1"]], [9, 27, 34])
        self.assertFalse(any(c in schema["base"] for c in EXTRA))

    def test_surprise_reference_excludes_current_day(self):
        data = self.dataset()
        day = sorted(data.Date.unique())[10]
        data.loc[(data.ticker == "AAPL") & (data.Date == day), "sentiment_mean"] = 0.7
        frame, _ = enriched_panel(data, pd.DataFrame())
        row = frame[(frame.ticker == "AAPL") & (frame.Date == day)].iloc[0]
        self.assertAlmostEqual(row.sentiment_surprise_20, 0.5)
        self.assertEqual(row.sentiment_history_ready, 1)

    def test_future_other_company_and_labels_cannot_change_past(self):
        data = self.dataset()
        a, schemas = enriched_panel(data, pd.DataFrame())
        data["target"] = 1 - data.target
        data.loc[(data.ticker != "AAPL") | (data.Date >= "2024-02-01"), "sentiment_mean"] = -0.9
        b, _ = enriched_panel(data, pd.DataFrame())
        keep = (a.ticker == "AAPL") & (a.Date < "2024-02-01")
        pd.testing.assert_frame_equal(a.loc[keep, schemas["lag_1"]], b.loc[keep, schemas["lag_1"]])
        for _, company in a.groupby("ticker"):
            for column in EXTRA:
                np.testing.assert_array_equal(company[column + "_lag_1"].iloc[1:], company[column].iloc[:-1])


if __name__ == "__main__":
    unittest.main()
