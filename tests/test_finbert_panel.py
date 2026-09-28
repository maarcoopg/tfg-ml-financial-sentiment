import unittest

import numpy as np
import pandas as pd

from src.experiments.finbert_panel import build_paired_panels, news_eligibility, replace_sentiment
from src.nlp.sentiment_data import digest
from tests.test_ticker_aware import panel


def news():
    return pd.DataFrame({"ticker": ["AAPL", "MSFT"],
        "published_at": ["2024-01-02 10:00:00Z"] * 2,
        "trading_date": pd.to_datetime(["2024-01-02"] * 2),
        "title": ["Apple profits rise", "Microsoft profits fall"],
        "summary": ["Apple forecasts gains in 2026.", "Microsoft cuts its forecast."],
        "text": ["Apple profits rise. Apple forecasts gains in 2026.", "Microsoft profits fall."],
        "url": ["https://example.com/news/1", "https://example.com/news/2"],
        "sentiment_score": [0.1, -0.1], "sentiment_label": ["positive", "negative"],
        "relevance_score": [0.8, 0.7], "is_non_trading_day": [False, False]})


def scores(frame):
    return pd.DataFrame({"text_sha256": frame.text_sha256,
        "positive": 0.1, "negative": 0.7, "neutral": 0.2, "score": -0.6, "label": "negative"})


class FinbertPanelTests(unittest.TestCase):
    def test_fixed_eligibility_and_forecast_not_future_leak_by_itself(self):
        original = news()
        self.assertTrue(news_eligibility(original).eligible.all())
        self.assertFalse(news_eligibility(original.assign(published_at="2020-12-01")).eligible.any())
        self.assertFalse(news_eligibility(original.assign(title="Market update", summary="No company")).eligible.any())
        self.assertFalse(news_eligibility(original.assign(text=" ")).eligible.any())
        for title in ["FIAX ETF Stock Price & Overview", "IWY ETF Holdings List",
                      "TSLL ETF Stock Price & Overview", "RONB ETF Holdings List"]:
            self.assertFalse(news_eligibility(original.assign(title=title)).eligible.any())
        self.assertFalse(news_eligibility(original.assign(url="https://example.com/quote/AAPL")).eligible.any())
        self.assertTrue(news_eligibility(original.assign(title="Apple stock price falls after results")).eligible.all())

    def test_identifiers_use_exact_text_and_company(self):
        original = news_eligibility(news())
        changed = news_eligibility(news().assign(text=lambda x: x.text + " "))
        self.assertTrue((original.text_sha256 != changed.text_sha256).all())
        self.assertEqual(original.text_sha256.iloc[0], digest(news().text.iloc[0]))
        extended = pd.concat([news(), news().assign(published_at="2025-01-01 10:00:00Z")], ignore_index=True)
        pd.testing.assert_frame_equal(original, news_eligibility(extended).iloc[:2])

    def test_score_validation_and_original_preserved(self):
        original = news_eligibility(news())
        table = scores(original)
        converted = replace_sentiment(original, table)
        np.testing.assert_allclose(converted.sentiment_score, -0.6)
        self.assertEqual(original.sentiment_score.tolist(), [0.1, -0.1])
        for invalid in [table.iloc[:1], pd.concat([table, table]), table.drop(columns="text_sha256"),
                        table.assign(positive=np.nan), table.assign(positive=1.1), table.assign(label="positive")]:
            with self.assertRaises(ValueError):
                replace_sentiment(original, invalid)
        for invalid in [table.assign(score=0), table.assign(neutral=0.1)]:
            with self.assertRaises(AssertionError):
                replace_sentiment(original, invalid)

    def test_paired_panels_zero_days_and_isolated_lag(self):
        original = news_eligibility(news())
        panels, schema, _ = build_paired_panels(panel(), original, original, scores(original))
        self.assertEqual([len(schema[v]) for v in ["base", "hybrid", "lag_1"]], [9, 22, 24])
        fb = panels["finbert_matched"]
        apple = fb[fb.ticker == "AAPL"]
        self.assertAlmostEqual(apple.sentiment_mean.iloc[0], -0.6)
        self.assertEqual(apple.sentiment_mean.iloc[1], 0)
        self.assertAlmostEqual(apple.sentiment_mean_lag_1.iloc[1], -0.6)
        self.assertTrue((fb.loc[fb.ticker == "NVDA", "sentiment_mean_lag_1"] == 0).all())
        self.assertEqual(len(fb), len(panel()))


if __name__ == "__main__":
    unittest.main()
