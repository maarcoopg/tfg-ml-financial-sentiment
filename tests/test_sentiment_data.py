import unittest

import pandas as pd

from src.nlp.sentiment_data import (annotation_templates, canonical_url, company_context,
                                    prepare_news, sample_news, validated_annotations)


def corpus():
    return pd.DataFrame([
        dict(ticker=ticker, title=f"{ticker} event {year} {j}", summary="The company issued a report.",
             text=f"{ticker} event {year} {j}. The company issued a report.",
             published_at=f"{year}-01-02", url=f"https://example.test/{ticker}/{year}/{j}")
        for ticker in ["AAPL", "MSFT", "NVDA", "TSLA"] for year in [2020, 2024] for j in range(5)])


class SentimentDataTests(unittest.TestCase):
    def test_transitive_duplicates_are_grouped_across_tickers(self):
        raw = corpus()
        raw.loc[1, "url"] = raw.loc[0, "url"]
        raw.loc[10, "title"] = raw.loc[1, "title"]
        data = prepare_news(raw)
        self.assertEqual(data.loc[[0, 1, 10], "group_id"].nunique(), 1)
        self.assertTrue(data.loc[0, "multi_company"])

    def test_cross_boundary_groups_are_excluded(self):
        raw = corpus()
        raw.loc[5, "title"] = raw.loc[0, "title"]
        data = prepare_news(raw)
        sample = sample_news(data, per_cell=2)
        self.assertFalse(sample.cross_boundary.any())
        self.assertTrue(data.loc[[0, 5], "cross_boundary"].all())
        self.assertEqual(sample.group_id.nunique(), len(sample))

    def test_sampling_is_stable_under_row_permutation(self):
        a = sample_news(prepare_news(corpus()), per_cell=2)
        b = sample_news(prepare_news(corpus().sample(frac=1, random_state=99)), per_cell=2)
        self.assertEqual(a.news_id.tolist(), b.news_id.tolist())

    def test_scores_do_not_affect_sample(self):
        raw = corpus()
        a = sample_news(prepare_news(raw), per_cell=2)
        raw["sentiment_score"] = range(len(raw))
        b = sample_news(prepare_news(raw), per_cell=2)
        self.assertEqual(a.news_id.tolist(), b.news_id.tolist())

    def test_context_is_company_specific_or_absent(self):
        summary = "Apple raised its outlook. Tesla lowered its outlook. Demand was mixed."
        self.assertEqual(company_context("Market update", summary, "AAPL"), "Apple raised its outlook.")
        self.assertEqual(company_context("Market update", summary, "MSFT"), "")

    def test_url_query_is_preserved(self):
        self.assertEqual(canonical_url("https://Example.test/news?id=1#x"), "https://example.test/news?id=1")
        self.assertNotEqual(canonical_url("https://example.test/news?id=1"), canonical_url("https://example.test/news?id=2"))

    def test_templates_hide_all_model_labels(self):
        raw = corpus()
        raw["sentiment_score"] = 0.8
        first, second = annotation_templates(sample_news(prepare_news(raw), per_cell=4))
        self.assertNotIn("sentiment_score", first)
        self.assertTrue(first.label.eq("").all())
        self.assertEqual(len(second), 8)

    def test_missing_labels_prevent_metrics(self):
        sample = sample_news(prepare_news(corpus()), per_cell=2)
        first, _ = annotation_templates(sample)
        with self.assertRaises(ValueError):
            validated_annotations(sample, first)

    def test_validated_metadata_cannot_be_reassigned(self):
        sample = sample_news(prepare_news(corpus()), per_cell=2)
        first, _ = annotation_templates(sample)
        first = first.assign(label="neutral", language="en", annotator_id="test_fixture", reviewed_at="2026-09-25")
        self.assertEqual(len(validated_annotations(sample, first)), len(sample))
        first.loc[0, "partition"] = "wrong_partition"
        with self.assertRaises(ValueError):
            validated_annotations(sample, first)

    def test_insufficient_sample_fails_loudly(self):
        with self.assertRaises(ValueError):
            sample_news(prepare_news(corpus()), per_cell=6)

    def test_invalid_dates_fail_instead_of_silent_partitioning(self):
        raw = corpus()
        raw.loc[0, "published_at"] = "bad date"
        with self.assertRaises(ValueError):
            prepare_news(raw)


if __name__ == "__main__":
    unittest.main()
