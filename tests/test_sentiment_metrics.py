import unittest
import pandas as pd

from src.nlp.sentiment_data import prepare_news, sample_news, annotation_templates
from src.nlp.sentiment_metrics import human_metrics, annotator_agreement
from tests.test_sentiment_data import corpus


class HumanMetricsTests(unittest.TestCase):
    def setUp(self):
        self.sample = sample_news(prepare_news(corpus()), per_cell=2)
        self.blank, _ = annotation_templates(self.sample)
        # These are synthetic software-test fixtures, never news annotations.
        self.labels = self.blank.assign(label="neutral", language="en", annotator_id="fixture_a", reviewed_at="2026-09-25")
        self.pred = pd.DataFrame([dict(news_id=i, variant=v, label="neutral")
            for i in self.sample.news_id for v in ["headline", "full_text", "company_context"]])

    def test_blank_template_cannot_produce_quality_metrics(self):
        with self.assertRaises(ValueError):
            human_metrics(self.sample, self.blank, self.pred)

    def test_abstention_reduces_coverage_not_silently_accuracy(self):
        self.pred.loc[(self.pred.variant == "company_context") &
                      (self.pred.news_id == self.sample.news_id.iloc[0]), "label"] = "abstain"
        result = human_metrics(self.sample, self.labels, self.pred)
        row = next(x for x in result["metrics"] if x["variant"] == "company_context" and
                   x["panel"] == "all_available" and x["ticker"] == "all")
        self.assertEqual(row["scored"], len(self.sample) - 1)
        self.assertLess(row["coverage"], 1)

    def test_missing_prediction_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            human_metrics(self.sample, self.labels, self.pred.iloc[1:])

    def test_double_annotation_must_be_independent(self):
        with self.assertRaises(ValueError):
            annotator_agreement(self.sample, self.labels, self.labels.iloc[:3])
        other = self.labels.iloc[:3].assign(annotator_id="fixture_b")
        result = annotator_agreement(self.sample, self.labels, other)
        self.assertEqual(result["agreement"], 1)
        self.assertIsNone(result["kappa"])
