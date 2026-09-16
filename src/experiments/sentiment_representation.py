from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.split_train_test import SENTIMENT_FEATURES
from src.experiments.ablations import feature_variants


EXTRA = ["news_sentiment_std", "news_abs_sentiment", "sentiment_surprise_20",
         "sentiment_history_ready", "news_log_surprise_20"]


def relative_panel(dataset):
    frame, previous = feature_variants(dataset)
    frame = frame.sort_values(["ticker", "Date"]).reset_index(drop=True)
    frame["news_count_relative_lag_1"] = frame.groupby("ticker").news_count_relative.shift(1).fillna(0)
    relative = previous["relative_features"]
    schemas = {"base": [c for c in relative if c not in SENTIMENT_FEATURES and c != "news_count_relative"],
               "hybrid": relative,
               "lag_1": relative + ["sentiment_mean_lag_1", "news_count_relative_lag_1"]}
    return frame.sort_values(["Date", "ticker"]).reset_index(drop=True), schemas


def enriched_panel(dataset, news):
    frame, schemas = relative_panel(dataset)
    frame = frame.sort_values(["ticker", "Date"]).reset_index(drop=True)
    if news.empty:
        frame["news_sentiment_std"] = 0.0
        frame["news_abs_sentiment"] = 0.0
    else:
        values = news.assign(absolute=news.sentiment_score.abs()).groupby(["ticker", "trading_date"]).agg(
            news_sentiment_std=("sentiment_score", lambda x: x.std(ddof=0)),
            news_abs_sentiment=("absolute", "mean")).reset_index().rename(columns={"trading_date": "Date"})
        values["Date"] = pd.to_datetime(values.Date)
        frame = frame.merge(values, on=["ticker", "Date"], how="left", validate="one_to_one")
        frame[["news_sentiment_std", "news_abs_sentiment"]] = frame[["news_sentiment_std", "news_abs_sentiment"]].fillna(0)
    # Historical tone ignores sessions without recorded news; the current day
    # is excluded from every reference distribution, including volume surprise.
    frame["observed_tone"] = frame.sentiment_mean.where(frame.has_news)
    historical = frame.groupby("ticker").observed_tone.transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    frame["sentiment_history_ready"] = historical.notna().astype(float)
    frame["sentiment_surprise_20"] = (frame.sentiment_mean - historical).where(frame.has_news, 0).fillna(0)
    frame["log_news"] = np.log1p(frame.news_count)
    average = frame.groupby("ticker").log_news.transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    deviation = frame.groupby("ticker").log_news.transform(lambda x: x.shift(1).rolling(20, min_periods=5).std(ddof=0))
    frame["news_log_surprise_20"] = ((frame.log_news - average) / deviation.replace(0, np.nan)).clip(-5, 5).fillna(0)
    lagged = []
    for column in EXTRA:
        name = column + "_lag_1"
        frame[name] = frame.groupby("ticker")[column].shift(1).fillna(0)
        lagged.append(name)
    schemas["hybrid"] = schemas["hybrid"] + EXTRA
    schemas["lag_1"] = schemas["lag_1"] + EXTRA + lagged
    frame = frame.drop(columns=["observed_tone", "log_news"])
    columns = sorted(set(c for cols in schemas.values() for c in cols))
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Variables enriquecidas no finitas")
    return frame.sort_values(["Date", "ticker"]).reset_index(drop=True), schemas
