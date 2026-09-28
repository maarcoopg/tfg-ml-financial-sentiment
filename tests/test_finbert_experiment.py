from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments.finbert_panel import build_paired_panels, news_eligibility
from src.experiments.run_finbert import fit_panels, paired_predictions, summarize
from tests.test_finbert_panel import news, scores
from tests.test_ticker_aware import panel


class FixedEstimator:
    def set_params(self, **params):
        return self

    def fit(self, features, target):
        self.probability = target.mean()
        return self

    def predict_proba(self, features):
        p = np.full(len(features), self.probability)
        return np.column_stack([1 - p, p])


class FinbertExperimentTests(unittest.TestCase):
    def test_outer_budget_purging_baseline_reuse_and_complete_summary(self):
        aligned = news_eligibility(news())
        panels, schema, _ = build_paired_panels(panel(), aligned, aligned, scores(aligned))
        def choose(train, columns, model, configs, metric, splits):
            self.assertEqual((metric, splits, len(configs)), ("macro_auc", 3, 2))
            self.assertNotIn("target", columns)
            return {"years": None, "params": {}}, [], 0.5
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch("src.experiments.run_finbert.build_model", side_effect=lambda name: FixedEstimator()) as build:
                with patch("src.experiments.run_finbert.choose_candidate", side_effect=choose) as selection:
                    with patch("src.experiments.run_finbert.artifact_dir", return_value=output / "models"):
                        with redirect_stdout(io.StringIO()):
                            predictions = fit_panels(panels, schema, output, outer_start="2024-01-30")
            self.assertEqual(selection.call_count, 63)
            self.assertEqual(build.call_count, 66)
            bounds = pd.read_csv(output / "tuning/selected_params.csv")
            self.assertTrue((pd.to_datetime(bounds.train_target_end) < pd.to_datetime(bounds.valid_start)).all())
            paired = paired_predictions(predictions)
            base = predictions.query("approach == 'financial' and model_name != 'dummy'")
            for variant in ["hybrid", "lag_1"]:
                copied = paired[(paired.approach == "financial") & (paired.dataset_type == variant)]
                np.testing.assert_array_equal(copied.probability, base.probability)
            summarize(predictions, output, 100)
            macro = pd.read_csv(output / "metrics/macro_metrics.csv")
            intervals = pd.read_csv(output / "metrics/paired_intervals.csv")
            self.assertEqual(len(macro), 22)
            self.assertEqual(len(intervals), 150)
            self.assertEqual(len(intervals[intervals.ticker == "macro"]), 30)


if __name__ == "__main__":
    unittest.main()
