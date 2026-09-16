import unittest

import pandas as pd

from src.experiments.news_quality import deduplicate_news, normalized_title, rebuild_sentiment
from tests.test_ticker_aware import panel


class NewsQualityTests(unittest.TestCase):
    def news(self):
        return pd.DataFrame({"ticker": ["AAPL"] * 4 + ["MSFT"],
            "published_at": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 11:00", "2024-01-03 12:00",
                                           "2024-01-02 12:00", "2024-01-02 10:00"], utc=True),
            "title": ["Profit rises 10%", " PROFIT  rises 10% ", "Profit rises 10%", "Profit rises 20%", "Profit rises 10%"]})

    def test_conservative_and_company_specific(self):
        cleaned, links = deduplicate_news(self.news())
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(links.record_id.tolist(), [1])
        self.assertEqual(links.representative_id.tolist(), [0])
        self.assertNotEqual(normalized_title("Profit does not rise"), normalized_title("Profit does rise"))
        self.assertNotEqual(normalized_title("-10%"), normalized_title("10%"))

    def test_future_records_never_change_past(self):
        original = self.news()
        past = original[original.published_at < pd.Timestamp("2024-01-03", tz="UTC")]
        a, _ = deduplicate_news(past)
        b, _ = deduplicate_news(original)
        b = b[b.published_at < pd.Timestamp("2024-01-03", tz="UTC")]
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_missing_title_rejected(self):
        with self.assertRaises(ValueError):
            deduplicate_news(self.news().assign(title=" "))

    def test_empty_corpus_preserves_financial_panel(self):
        original = panel()
        rebuilt = rebuild_sentiment(original, pd.DataFrame())
        self.assertEqual(len(original), len(rebuilt))
        self.assertTrue((rebuilt.news_count == 0).all())
        self.assertFalse(rebuilt.has_news.any())
        keys = ["ticker", "Date", "target", "target_end", "daily_return"]
        pd.testing.assert_frame_equal(original[keys].sort_values(["Date", "ticker"]).reset_index(drop=True), rebuilt[keys])


if __name__ == "__main__":
    unittest.main()
