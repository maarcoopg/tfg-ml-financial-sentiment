import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.split_train_test import FINANCIAL_FEATURES, SENTIMENT_FEATURES
from src.experiments.panel_uncertainty import block_weights, panel_intervals, weighted_auc_samples
from src.experiments.run_ticker_aware import checked_reference, ticker_features
from src.experiments.run_review import GRIDS, ROOT
from src.models.temporal_validation import paired_auc_interval


def panel():
    dates = pd.bdate_range("2024-01-02", periods=61)
    data = pd.DataFrame([(t, d, dates[i + 1], i % 2) for t in ["AAPL", "MSFT", "NVDA", "TSLA"]
                         for i, d in enumerate(dates[:-1])],
                        columns=["ticker", "Date", "target_end", "target"])
    for column in FINANCIAL_FEATURES + SENTIMENT_FEATURES:
        data[column] = np.arange(len(data), dtype=float) + 10
    data["sentiment_weighted"] = 0.1
    return data


class FeatureTests(unittest.TestCase):
    def test_boolean_news_flag_matches_real_dataset(self):
        data = panel()
        data["has_news"] = data.index % 2 == 0
        frame, schemas = ticker_features(data)
        self.assertTrue(np.isfinite(frame[schemas["relative_interactions"]["hybrid"]].to_numpy(dtype=float)).all())

    def test_schema_identity_interactions_and_no_news_in_base(self):
        frame, schemas = ticker_features(panel())
        self.assertEqual([len(schemas["relative_pooled"][v]) for v in ["base", "hybrid", "lag_1"]], [9, 22, 24])
        self.assertEqual([len(schemas["raw_ticker"][v]) for v in ["base", "hybrid", "lag_1"]], [17, 30, 32])
        self.assertEqual(len(schemas["relative_interactions"]["hybrid"]), 91)
        self.assertEqual(frame.filter(regex="^company_[A-Z]+$").sum(axis=1).unique().tolist(), [0.0, 1.0])
        self.assertFalse(any("news" in c or "sentiment" in c for s in schemas.values() for c in s["base"]))
        name = "company_MSFT__daily_return"
        np.testing.assert_array_equal(frame[name], frame.daily_return * frame.ticker.eq("MSFT"))
        self.assertTrue((frame.loc[frame.ticker == "AAPL", name] == 0).all())

    def test_features_do_not_use_labels_other_company_or_future(self):
        original = panel()
        frame, schemas = ticker_features(original)
        changed = original.copy()
        changed["target"] = 1 - changed.target
        same, _ = ticker_features(changed)
        columns = sorted({c for s in schemas.values() for cols in s.values() for c in cols})
        pd.testing.assert_frame_equal(frame[columns], same[columns])
        boundary = pd.Timestamp("2024-02-01")
        changed = original.copy()
        changed.loc[(changed.ticker == "MSFT") | (changed.Date >= boundary), FINANCIAL_FEATURES + SENTIMENT_FEATURES] *= 3
        altered, _ = ticker_features(changed)
        keep = (frame.ticker == "AAPL") & (frame.Date < boundary)
        pd.testing.assert_frame_equal(frame.loc[keep, columns], altered.loc[keep, columns])
        for ticker in frame.ticker.unique():
            company = frame[frame.ticker == ticker]
            self.assertEqual(company.news_count_relative_lag_1.iloc[0], 0)
            np.testing.assert_array_equal(company.news_count_relative_lag_1.iloc[1:], company.news_count_relative.iloc[:-1])

    def test_relative_price_features_are_scale_invariant(self):
        original = panel()
        changed = original.copy()
        for col in ["Adj Close", "Close", "Open", "High", "Low", "sma_5", "sma_20", "sma_50", "MACD", "MACD_signal", "Volume"]:
            changed.loc[changed.ticker == "MSFT", col] *= 100
        a, schemas = ticker_features(original)
        b, _ = ticker_features(changed)
        np.testing.assert_allclose(a[schemas["relative_pooled"]["base"]], b[schemas["relative_pooled"]["base"]], atol=1e-12)
        with self.assertRaises(ValueError):
            ticker_features(original.assign(ticker="SPY"))


class BootstrapTests(unittest.TestCase):
    def test_macro_interval_uses_shared_dates_and_matches_sklearn(self):
        frame = panel().assign(dataset_type="base", model_name="logistic_regression", outer_fold=1)
        rng = np.random.default_rng(9)
        a = frame.assign(approach="new", probability=rng.random(len(frame)))
        b = frame.assign(approach="old", probability=rng.random(len(frame)))
        result = panel_intervals(pd.concat([a, b]), [("new", "old")], repeats=100)
        weights = block_weights(60, repeats=100)
        expected = []
        for w in weights:
            expected.append(np.mean([roc_auc_score(x.target, x.probability, sample_weight=w)
                - roc_auc_score(y.target, y.probability, sample_weight=w)
                for (_, x), (_, y) in zip(a.groupby("ticker"), b.groupby("ticker"))]))
        macro = result[result.ticker == "macro"].iloc[0]
        np.testing.assert_allclose([macro.delta_low, macro.delta_high], np.quantile(expected, [0.025, 0.975]), atol=1e-14)

    def test_weighted_auc_matches_sklearn_with_ties_and_missing_classes(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        scores = np.array([0.1, 0.6, 0.6, 0.6, 0.9, 0.9])
        rng = np.random.default_rng(42)
        weights = rng.integers(0, 6, size=(120, 6))
        weights[0] = 0
        actual = weighted_auc_samples(y, scores, weights)
        for i, w in enumerate(weights):
            if w[y == 0].sum() and w[y == 1].sum():
                self.assertAlmostEqual(actual[i], roc_auc_score(y, scores, sample_weight=w), places=14)
            else:
                self.assertTrue(np.isnan(actual[i]))
        with self.assertRaises(ValueError):
            weighted_auc_samples(y, scores, -weights)

    def test_matches_existing_paired_block_bootstrap(self):
        frame = panel().query("ticker == 'AAPL'")
        rng = np.random.default_rng(1)
        a = frame.assign(probability=rng.random(len(frame)))
        b = frame.assign(probability=rng.random(len(frame)))
        old = paired_auc_interval(a, b, block=20, repeats=100)
        weights = block_weights(len(a), repeats=100)
        x = weighted_auc_samples(a.target, a.probability, weights)
        y = weighted_auc_samples(b.target, b.probability, weights)
        np.testing.assert_allclose(np.quantile(x, [0.025, 0.975]), [old["auc_low"], old["auc_high"]], atol=1e-14)
        np.testing.assert_allclose(np.quantile(x-y, [0.025, 0.975]), [old["delta_low"], old["delta_high"]], atol=1e-14)

    def test_macro_and_pairing(self):
        frame = panel().assign(dataset_type="base", model_name="logistic_regression", outer_fold=1)
        frame["probability"] = np.random.default_rng(4).random(len(frame))
        a, b = frame.assign(approach="new"), frame.assign(approach="old")
        result = panel_intervals(pd.concat([a, b]), [("new", "old")], repeats=100)
        self.assertEqual(set(result.ticker), {"AAPL", "MSFT", "NVDA", "TSLA", "macro"})
        self.assertTrue((result[["auc_delta", "delta_low", "delta_high"]] == 0).all().all())
        self.assertAlmostEqual(result.loc[result.ticker == "macro", "roc_auc"].item(),
                               result.loc[result.ticker != "macro", "roc_auc"].mean())
        with self.assertRaises(ValueError):
            panel_intervals(pd.concat([a, b.iloc[1:]]), [("new", "old")], repeats=100)
        with self.assertRaises(ValueError):
            block_weights(10, 99)


class ReferenceTests(unittest.TestCase):
    def test_rejects_changed_protocol_versions_and_inputs(self):
        config = dict(outer_start="2023-10-16", outer_splits=3, inner_splits=3,
                      source_timezone="UTC", parameter_grids=GRIDS, threshold=0.5)
        previous = dict(status="complete", config=config, versions={"numpy": "same"},
                        input_sha256={"input.csv": "old"})
        args = SimpleNamespace(**config)
        path = Path("unused-reference")
        with patch.object(Path, "read_text", return_value=json.dumps(previous)):
            with self.assertRaisesRegex(ValueError, "versiones"):
                checked_reference(path, args, {"versions": {}}, [], Path("unused-dataset"))
            with patch("src.experiments.run_ticker_aware.sha256", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "Datos distintos"):
                    checked_reference(path, args, {"versions": previous["versions"]}, [ROOT / "input.csv"], Path("unused-dataset"))
            args.outer_splits = 2
            with self.assertRaisesRegex(ValueError, "Protocolo"):
                checked_reference(path, args, {"versions": previous["versions"]}, [], Path("unused-dataset"))


if __name__ == "__main__":
    unittest.main()
