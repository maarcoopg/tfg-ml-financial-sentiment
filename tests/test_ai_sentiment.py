import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.experiments.artifacts import sha256
from src.nlp.score_ai_sentiment import compare_systems, load_ai_review, markdown_review
from src.nlp.sentiment_data import prepare_news, sample_news, annotation_templates
from tests.test_sentiment_data import corpus


class AIReviewTests(unittest.TestCase):
    def setUp(self):
        self.sample = sample_news(prepare_news(corpus()), per_cell=2).assign(sentiment_label="positive")
        blank, _ = annotation_templates(self.sample)
        self.labels = blank.assign(label="positive", language="en", annotation_kind="ai",
            reason="Synthetic test fixture, not a real annotation", annotator_id="fixture_ai", reviewed_at="2026-09-28")
        self.pred = pd.DataFrame([dict(news_id=i, variant=v, label="positive")
            for i in self.sample.news_id for v in ["headline", "full_text", "company_context"]])

    def test_comparison_excludes_insufficient_and_reports_abstentions(self):
        self.labels.loc[0, "label"] = "insufficient"
        self.pred.loc[(self.pred.news_id == self.sample.news_id.iloc[1]) &
                      (self.pred.variant == "company_context"), "label"] = "abstain"
        result = compare_systems(self.sample, self.labels, self.pred)
        rows = [r for r in result["results"] if r["partition"] == "development" and r["ticker"] == "all"
                and r["panel"] == "all_eligible"]
        full = next(r for r in rows if r["system"] == "full_text")
        context = next(r for r in rows if r["system"] == "company_context")
        self.assertEqual(full["scored"], 7)
        self.assertEqual(context["scored"], 6)
        self.assertEqual(full["agreement"], 1)
        self.assertTrue(result["not_human_accuracy"])

    def test_unknown_provider_label_is_rejected(self):
        self.sample.loc[0, "sentiment_label"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Alpha Vantage"):
            compare_systems(self.sample, self.labels, self.pred)

    def test_review_binds_rows_to_source_hash_and_requires_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.labels.drop(columns="annotation_kind").to_csv(root / "source.csv", index=False)
            provenance = dict(annotation_kind="ai", source="source.csv", source_sha256=sha256(root / "source.csv"),
                rows=len(self.labels), annotator_id="fixture_ai", reviewed_at="2026-09-28")
            (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            review = pd.DataFrame(dict(row=range(1, len(self.labels) + 1), label="positive", reason="fixture"))
            review.to_csv(root / "review.tsv", sep="\t", index=False)
            annotations, _ = load_ai_review(root, root)
            self.assertTrue(annotations.annotation_kind.eq("ai").all())
            review.iloc[::-1].to_csv(root / "review.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "frozen order"):
                load_ai_review(root, root)
            (root / "source.csv").write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source differs"):
                load_ai_review(root, root)

    def test_readable_review_identifies_nonhuman_reference(self):
        result = markdown_review(self.sample, self.labels, self.pred)
        self.assertIn("no por una persona", result)
        self.assertIn(self.sample.news_id.iloc[0], result)


if __name__ == "__main__":
    unittest.main()
