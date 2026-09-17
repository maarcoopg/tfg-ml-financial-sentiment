import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments.round_selection import ROUND_MODELS, auc_scores, candidates, choose_candidate
from tests.test_ticker_aware import panel


class SelectionTests(unittest.TestCase):
    def test_expanded_candidates_reproducible_and_include_references(self):
        for model in ROUND_MODELS:
            base, expanded = candidates(model), candidates(model, expanded=True)
            self.assertEqual(expanded, candidates(model, expanded=True))
            self.assertEqual(len(expanded), 6 if model == "logistic_regression" else 20)
            self.assertTrue(all(c in expanded for c in base))
            self.assertEqual({c["years"] for c in expanded}, {None, 3, 5})

    def test_macro_does_not_compare_cross_company_scores(self):
        frame = pd.DataFrame({"ticker": ["A", "A", "B", "B"], "target": [0, 1, 0, 1]})
        scores = auc_scores(frame, [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(scores["macro_auc"], 1)
        self.assertEqual(scores["pooled_auc"], 0.75)
        with self.assertRaises(ValueError):
            auc_scores(frame.iloc[[0, 2, 3]], [0.1, 0.8, 0.9])

    def test_selection_uses_requested_score_and_same_folds(self):
        data = panel()
        choices = candidates("logistic_regression")
        class Fake:
            def set_params(self, **params):
                self.c = params["model__C"]
                return self
            def fit(self, x, y):
                return self
            def predict_proba(self, x):
                p = np.full(len(x), 0.2 if self.c == 0.1 else 0.8)
                return np.c_[1-p, p]
        def score(frame, prob):
            return {"pooled_auc": float(prob[0]), "macro_auc": float(1-prob[0])}
        with patch("src.experiments.round_selection.build_model", side_effect=lambda _: Fake()), \
             patch("src.experiments.round_selection.auc_scores", side_effect=score):
            pooled, rows, _ = choose_candidate(data, ["daily_return"], "logistic_regression", choices)
            macro, _, _ = choose_candidate(data, ["daily_return"], "logistic_regression", choices, "macro_auc")
        self.assertEqual(pooled, choices[1])
        self.assertEqual(macro, choices[0])
        log = pd.DataFrame(rows)
        self.assertTrue((log.train_target_end < log.valid_start).all())
        self.assertTrue(log.groupby("inner_fold").valid_start.nunique().eq(1).all())


if __name__ == "__main__":
    unittest.main()
