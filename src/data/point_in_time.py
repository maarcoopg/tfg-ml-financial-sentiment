from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.data.aggregate_daily_sentiment import aggregate_daily_sentiment
from src.data.handle_days_without_news import fill_days_without_news
from src.data.split_train_test import FINANCIAL_FEATURES, SENTIMENT_FEATURES, MODELING_TICKERS
from src.models.temporal_validation import validate_panel


def trading_schedule(start, end) -> pd.DataFrame:
    schedule = mcal.get_calendar("NASDAQ").schedule(start_date=start, end_date=end)
    schedule.index = pd.to_datetime(schedule.index).tz_localize(None)
    schedule.index.name = "trading_date"
    return schedule.reset_index()[["trading_date", "market_close"]]


def align_at_close(news: pd.DataFrame, schedule: pd.DataFrame,
                   source_timezone: str = "UTC") -> tuple[pd.DataFrame, pd.DataFrame]:
    news = news.copy()
    published = pd.to_datetime(news["published_at"], errors="raise")
    if published.isna().any():
        raise ValueError("Noticias sin timestamp")
    if published.dt.tz is None:
        published = published.dt.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
    news["published_at"] = published.dt.tz_convert("UTC")
    news["published_date"] = news["published_at"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    schedule = schedule.copy().sort_values("market_close")
    schedule["market_close"] = pd.to_datetime(schedule["market_close"], utc=True)
    if schedule["market_close"].isna().any() or schedule["trading_date"].duplicated().any():
        raise ValueError("Calendario invalido")
    result = pd.merge_asof(news.sort_values("published_at"), schedule,
                           left_on="published_at", right_on="market_close", direction="forward")
    result["is_non_trading_day"] = ~result["published_date"].isin(schedule["trading_date"])
    result["assigned_later_date"] = result["trading_date"] > result["published_date"]
    aligned = result[result["market_close"].notna()].copy()
    unaligned = result[result["market_close"].isna()].copy()
    if (aligned["published_at"] > aligned["market_close"]).any():
        raise ValueError("Noticia posterior al instante de decision")
    return aligned, unaligned


def build_dataset(root: Path, output: Path, source_timezone: str = "UTC"):
    news_file = root / "data/processed/news/financial_news_with_sentiment.csv"
    news = pd.read_csv(news_file)
    news = news[news["ticker"].isin(MODELING_TICKERS)].copy()
    for column, lower, upper in [("sentiment_score", -1, 1), ("relevance_score", 0, 1)]:
        news[column] = pd.to_numeric(news[column], errors="raise")
        if not news[column].between(lower, upper).all():
            raise ValueError(f"{column} fuera de rango o nulo")
    if not news["sentiment_label"].isin(["positive", "negative", "neutral"]).all():
        raise ValueError("Etiqueta de sentimiento desconocida")
    frames = []
    inputs = [news_file]
    for ticker in MODELING_TICKERS:
        price_file = root / f"data/processed/prices/{ticker}_with_target.csv"
        raw_file = root / f"data/raw/prices/{ticker}.csv"
        inputs.extend([price_file, raw_file])
        prices = pd.read_csv(price_file, parse_dates=["Date"])
        raw = pd.read_csv(raw_file, parse_dates=["Date"]).sort_values("Date")
        raw["target_end"] = raw["Date"].shift(-1)
        raw["expected_target"] = (raw["Adj Close"].shift(-1) > raw["Adj Close"]).astype(int)
        prices = prices.merge(raw[["Date", "target_end", "expected_target"]], on="Date", validate="one_to_one")
        if (prices["target"] != prices["expected_target"]).any():
            raise ValueError(f"Objetivos inconsistentes: {ticker}")
        prices["ticker"] = ticker
        frames.append(prices[["ticker", "Date", "target_end", "target"] + FINANCIAL_FEATURES])
    financial = pd.concat(frames, ignore_index=True).sort_values(["Date", "ticker"])
    validate_panel(financial)
    schedule = trading_schedule(financial["Date"].min(), financial["target_end"].max())
    if not financial["Date"].isin(schedule["trading_date"]).all():
        raise ValueError("Fechas de precios fuera del calendario NASDAQ")
    aligned, unaligned = align_at_close(news, schedule, source_timezone)
    daily = aggregate_daily_sentiment(aligned)
    daily["trading_date"] = pd.to_datetime(daily["trading_date"])
    aligned["weighted_score"] = aligned["sentiment_score"] * aligned["relevance_score"]
    weighted = aligned.groupby(["ticker", "trading_date"]).agg(
        weighted_sum=("weighted_score", "sum"), relevance_sum=("relevance_score", "sum"))
    weighted["sentiment_weighted"] = weighted["weighted_sum"].div(weighted["relevance_sum"].replace(0, np.nan)).fillna(0)
    daily = daily.merge(weighted[["sentiment_weighted"]], on=["ticker", "trading_date"], validate="one_to_one")
    merged = financial.merge(daily, left_on=["ticker", "Date"], right_on=["ticker", "trading_date"], how="left", validate="one_to_one")
    merged = fill_days_without_news(merged.drop(columns="trading_date"))
    merged["sentiment_weighted"] = merged["sentiment_weighted"].fillna(0)
    validate_panel(merged)
    if not np.isfinite(merged[FINANCIAL_FEATURES + SENTIMENT_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Variables no finitas")
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "dataset.csv", index=False)
    schedule.to_csv(output / "trading_schedule.csv", index=False)
    aligned[["ticker", "published_at", "trading_date", "market_close", "assigned_later_date"]].to_csv(output / "news_alignment.csv", index=False)
    unaligned.to_csv(output / "unaligned_news.csv", index=False)
    return merged, aligned, inputs
