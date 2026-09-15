import tempfile
import json
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from src.data import download_financial_news as downloader
from src.data import split_train_test as splitter
from src.data import create_lagged_sentiment_dataset as lagged
from src.data.point_in_time import align_at_close, trading_schedule
from src.data.split_train_test import FINANCIAL_FEATURES, SENTIMENT_FEATURES
from src.experiments.ablations import feature_variants
from src.experiments.artifacts import create_run
from src.experiments.run_review import nested_predictions
from src.models.evaluation import evaluate_binary_classification
from src.models.temporal_validation import expanding_date_splits, paired_auc_interval, purged_train, validate_panel
from src.models.train_models import build_model


def panel(n=60):
    dates = pd.bdate_range("2024-01-02", periods=n + 1)
    df = pd.DataFrame([(ticker, d, dates[i + 1], i % 2) for i, d in enumerate(dates[:-1])
                       for ticker in ["AAPL", "MSFT"]], columns=["ticker", "Date", "target_end", "target"])
    for column in FINANCIAL_FEATURES + SENTIMENT_FEATURES:
        df[column] = np.arange(len(df), dtype=float) + 10
    df["sentiment_weighted"] = 0.1
    return df


class CalendarTests(unittest.TestCase):
    def test_early_close_and_holiday(self):
        schedule = trading_schedule("2024-07-03", "2024-07-08")
        news = pd.DataFrame({"published_at": ["2024-07-03 16:59:00", "2024-07-03 17:01:00", "2024-07-04 10:00:00"], "ticker": "AAPL"})
        aligned, missing = align_at_close(news, schedule)
        self.assertTrue(missing.empty)
        self.assertEqual(aligned.trading_date.dt.strftime("%Y-%m-%d").tolist(), ["2024-07-03", "2024-07-05", "2024-07-05"])

    def test_regular_close_weekend_and_exact_boundary(self):
        schedule = trading_schedule("2024-07-05", "2024-07-08")
        news = pd.DataFrame({"published_at": ["2024-07-05 20:00:00", "2024-07-05 20:00:01", "2024-07-06 12:00:00"], "ticker": "AAPL"})
        aligned, _ = align_at_close(news, schedule)
        self.assertEqual(aligned.trading_date.dt.day.tolist(), [5, 8, 8])
        self.assertTrue((aligned.published_at <= aligned.market_close).all())

    def test_dst_changes_utc_close(self):
        schedule = trading_schedule("2024-03-08", "2024-03-11")
        self.assertEqual(schedule.market_close.dt.hour.tolist(), [21, 20])

    def test_unknown_dates_and_invalid_timestamp(self):
        schedule = trading_schedule("2024-07-05", "2024-07-05")
        aligned, missing = align_at_close(pd.DataFrame({"published_at": ["2024-07-06 10:00:00"]}), schedule)
        self.assertTrue(aligned.empty)
        self.assertEqual(len(missing), 1)
        with self.assertRaises(ValueError):
            align_at_close(pd.DataFrame({"published_at": [None]}), schedule)

    def test_explicit_local_timezone_matches_utc(self):
        schedule = trading_schedule("2024-07-05", "2024-07-08")
        a, _ = align_at_close(pd.DataFrame({"published_at": ["2024-07-05 16:01:00"]}), schedule, "America/New_York")
        b, _ = align_at_close(pd.DataFrame({"published_at": ["2024-07-05 20:01:00+00:00"]}), schedule)
        self.assertEqual(a.trading_date.iloc[0], b.trading_date.iloc[0])


class TemporalTests(unittest.TestCase):
    def test_standard_and_lagged_splits_match(self):
        df = panel()
        a, b, boundary = splitter.split_by_time(df)
        x, y, other_boundary = lagged.split_by_time(df)
        self.assertTrue(a.equals(x))
        self.assertTrue(b.equals(y))
        self.assertEqual(boundary, other_boundary)
        self.assertLess(a.target_end.max(), boundary)

    def test_regenerating_split_keeps_lagged_feature_schema(self):
        train, test, _ = splitter.split_by_time(panel())
        with tempfile.TemporaryDirectory() as directory:
            paths = {name: Path(directory) / value.name for name, value in vars(splitter).items()
                     if isinstance(value, Path) and name not in ["PROJECT_ROOT", "FINAL_DATASET_FILE"]}
            paths["OUTPUT_DIR"] = Path(directory)
            with patch.multiple(splitter, **paths):
                splitter.save_splits(train, test)
            config = json.loads(paths["FEATURE_CONFIG_FILE"].read_text(encoding="utf-8"))
            self.assertEqual(len(config["lagged_hybrid_features"]), 92)
            self.assertTrue(config["purged_datasets"]["base"])

    def test_all_folds_purge_label_horizon(self):
        df = panel()
        for train_dates, valid_dates in expanding_date_splits(df):
            train = df[df.Date.isin(train_dates)]
            self.assertLess(train.target_end.max(), valid_dates.min())
            self.assertEqual(train.groupby("Date").size().unique().tolist(), [2])

    def test_multi_session_label_is_purged(self):
        df = panel()
        df["target_end"] += pd.Timedelta(days=5)
        boundary = df.Date.iloc[70]
        train = purged_train(df, boundary)
        self.assertTrue((train.target_end < boundary).all())

    def test_legacy_fallback_omits_boundary(self):
        df = panel().drop(columns="target_end")
        boundary = sorted(df.Date.unique())[30]
        train = purged_train(df, boundary)
        self.assertEqual(train.Date.nunique(), 29)

    def test_duplicate_and_short_panel_rejected(self):
        with self.assertRaises(ValueError):
            validate_panel(pd.concat([panel(), panel().iloc[:1]]))
        with self.assertRaises(ValueError):
            expanding_date_splits(panel(3))

    def test_lags_do_not_cross_tickers_or_use_future(self):
        df = panel()
        original, _ = feature_variants(df)
        altered = df.copy()
        boundary = sorted(df.Date.unique())[30]
        altered.loc[altered.Date >= boundary, "sentiment_mean"] = 10000
        changed, _ = feature_variants(altered)
        columns = ["sentiment_mean_lag_1", "sentiment_mean_rolling_5"]
        self.assertTrue(original.loc[original.Date < boundary, columns].equals(changed.loc[changed.Date < boundary, columns]))
        for ticker in ["AAPL", "MSFT"]:
            rows = original[original.ticker == ticker]
            self.assertEqual(rows.sentiment_mean_lag_1.iloc[0], 0)
            self.assertEqual(rows.sentiment_mean_lag_1.iloc[1], rows.sentiment_mean.iloc[0])

    def test_outer_tuning_receives_only_purged_past(self):
        df = panel(40)
        received = []
        def choose(train, columns, name, splits):
            received.append(train.copy())
            return {}, [], 0.5
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.experiments.run_review.choose_params", side_effect=choose), patch(
                "src.experiments.run_review.artifact_dir", side_effect=lambda output, kind: output / kind
            ):
                result = nested_predictions(df, {"base": ["daily_return"]}, ["logistic_regression"], Path(directory), str(df.Date.iloc[40].date()), 2, 2)
        for i, train in enumerate(received, start=1):
            self.assertLess(train.target_end.max(), result.loc[result.outer_fold == i, "Date"].min())


class EvaluationTests(unittest.TestCase):
    def test_majority_baseline_is_not_high_balanced_performance(self):
        true = pd.Series([1] * 60 + [0] * 40)
        scores = evaluate_binary_classification(true, pd.Series([1] * 100), pd.Series([0.6] * 100))
        self.assertEqual(scores["balanced_accuracy"], 0.5)
        self.assertEqual(scores["roc_auc"], 0.5)
        self.assertEqual(scores["mcc"], 0)

    def test_paired_bootstrap_identity(self):
        df = panel().assign(probability=np.tile([0.2, 0.7, 0.6, 0.4], 30))
        result = paired_auc_interval(df, df.sample(frac=1, random_state=1), block=5, repeats=100)
        self.assertEqual(result["delta_low"], 0)
        self.assertEqual(result["delta_high"], 0)

    def test_pairing_mismatch_is_rejected(self):
        df = panel().assign(probability=0.5)
        other = df.copy()
        other.loc[0, "target"] = 1 - other.loc[0, "target"]
        with self.assertRaises(ValueError):
            paired_auc_interval(df, other, repeats=100)

    def test_lightgbm_sampling_is_enabled(self):
        model = build_model("lightgbm").named_steps["model"]
        self.assertGreater(model.subsample_freq, 0)


class DownloadTests(unittest.TestCase):
    def test_empty_chunk_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(downloader, "CHUNKS_DIR", Path(directory)):
                period = downloader.DateRange(date(2020, 1, 1), date(2020, 1, 2))
                path = downloader.save_chunk("AAPL", period, [])
                self.assertEqual(len(pd.read_csv(path)), 0)
                with patch.object(downloader, "request_news") as request:
                    self.assertEqual(downloader.download_range("AAPL", period, "unused"), [path])
                    request.assert_not_called()
                self.assertTrue(downloader.combine_chunks([path]).empty)

    def test_saturated_day_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(downloader, "CHUNKS_DIR", Path(directory)), patch.object(downloader, "request_news", return_value={"feed": [{}] * 1000}), patch.object(downloader.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "saturado"):
                    downloader.download_range("AAPL", downloader.DateRange(date(2020, 1, 1), date(2020, 1, 1)), "unused")
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_transient_failure_retries(self):
        bad, good = Mock(status_code=503), Mock(status_code=200)
        good.json.return_value = {"feed": []}
        with patch.object(downloader.requests, "get", side_effect=[bad, good]) as get, patch.object(downloader.time, "sleep"):
            result = downloader.request_news("AAPL", downloader.DateRange(date(2020, 1, 1), date(2020, 1, 2)), "unused")
        self.assertEqual(result, {"feed": []})
        self.assertEqual(get.call_count, 2)

    def test_malformed_response_does_not_become_empty_data(self):
        response = Mock(status_code=200)
        response.json.return_value = {}
        with patch.object(downloader.requests, "get", return_value=response):
            with self.assertRaises(RuntimeError):
                downloader.request_news("AAPL", downloader.DateRange(date(2020, 1, 1), date(2020, 1, 2)), "unused")


class ArtifactTests(unittest.TestCase):
    def test_run_cannot_overwrite_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                create_run(Path(directory), Path(directory), {})


if __name__ == "__main__":
    unittest.main()
