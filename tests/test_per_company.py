import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.aggregate_daily_sentiment import aggregate_daily_sentiment
from src.data.split_train_test import FINANCIAL_FEATURES, SENTIMENT_FEATURES
from src.experiments.run_per_company import company_data, verify_pairing
from src.experiments.run_review import nested_predictions


def panel():
    dates = pd.bdate_range("2024-01-02", periods=61)
    df = pd.DataFrame([(t, d, dates[i + 1], i % 2) for t in ["AAPL", "MSFT"]
                       for i, d in enumerate(dates[:-1])],
                      columns=["ticker", "Date", "target_end", "target"])
    for column in FINANCIAL_FEATURES + SENTIMENT_FEATURES:
        df[column] = np.arange(len(df), dtype=float) + 1
    df["sentiment_weighted"] = 0.1
    return df


class CompanyTests(unittest.TestCase):
    def test_other_company_cannot_change_features_or_news(self):
        df = panel()
        news = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "sentiment_score": [0.2, -0.8]})
        before, before_news, variants = company_data(df, news, "AAPL")
        changed, changed_news = df.copy(), news.copy()
        changed.loc[changed.ticker == "MSFT", FINANCIAL_FEATURES + SENTIMENT_FEATURES] = -100
        changed_news.loc[changed_news.ticker == "MSFT", "sentiment_score"] = 1
        after, after_news, _ = company_data(changed, changed_news, "AAPL")
        pd.testing.assert_frame_equal(before, after)
        pd.testing.assert_frame_equal(before_news, after_news)
        self.assertEqual([len(variants[v]) for v in variants], [14, 27, 29])
        self.assertEqual(before.sentiment_mean_lag_1.iloc[0], 0)
        self.assertEqual(before.sentiment_mean_lag_1.iloc[1], before.sentiment_mean.iloc[0])
        with self.assertRaises(ValueError):
            company_data(df, news, "SPY")

    def test_shared_article_retains_company_specific_score(self):
        news = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "title": ["same article"] * 2,
            "trading_date": pd.to_datetime(["2024-01-02"] * 2),
            "sentiment_score": [0.8, -0.4], "sentiment_label": ["positive", "negative"],
            "is_non_trading_day": [False, False]})
        result = aggregate_daily_sentiment(news).set_index("ticker")
        self.assertEqual(result.loc["AAPL", "sentiment_mean"], 0.8)
        self.assertEqual(result.loc["MSFT", "sentiment_mean"], -0.4)
        self.assertTrue((result.news_count == 1).all())

    def test_fitting_is_isolated_purged_and_saved_separately(self):
        df = panel()
        received, results = [], []
        def choose(train, columns, name, splits):
            received.append(train.copy())
            return {}, [], 0.5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ticker in ["AAPL", "MSFT"]:
                frame = df[df.ticker == ticker].copy()
                with patch("src.experiments.run_review.choose_params", side_effect=choose):
                    result = nested_predictions(frame, {"base": ["daily_return"]},
                        ["logistic_regression"], root / "reports" / ticker,
                        "2024-02-13", 3, 3, model_dir=root / "models" / ticker)
                results.append(result)
                self.assertEqual(len(list((root / "models" / ticker).glob("*.joblib"))), 6)
                self.assertEqual(set(result.ticker), {ticker})
            for index, train in enumerate(received):
                self.assertEqual(train.ticker.nunique(), 1)
                result = results[index // 3]
                self.assertLess(train.target_end.max(), result.loc[result.outer_fold == index % 3 + 1, "Date"].min())
            for a, b in zip(results[0].groupby("outer_fold"), results[1].groupby("outer_fold")):
                self.assertEqual(a[1].Date.tolist(), b[1].Date.tolist())

    def test_pairing_rejects_missing_dates_targets_folds_and_invalid_scores(self):
        df = panel().assign(dataset_type="base", model_name="dummy", outer_fold=1,
                            probability=0.5, prediction=1)
        verify_pairing(df, df.sample(frac=1, random_state=42))
        for column, value in [("target", 1), ("outer_fold", 2), ("probability", 1.2), ("prediction", 0)]:
            altered = df.copy()
            altered.loc[0, column] = value
            with self.assertRaises(ValueError):
                verify_pairing(df, altered)
        with self.assertRaises(ValueError):
            verify_pairing(df, df.iloc[1:])
        with self.assertRaises(ValueError):
            verify_pairing(df, pd.concat([df, df.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
