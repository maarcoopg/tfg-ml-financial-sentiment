from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from src.data.aggregate_daily_sentiment import aggregate_daily_sentiment
from src.data.handle_days_without_news import fill_days_without_news
from src.data.split_train_test import FINANCIAL_FEATURES, SENTIMENT_FEATURES


def normalized_title(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Titulo ausente")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def deduplicate_news(news, hours=24):
    """Keep the first known identical normalized title per company and time window.

    Do not remove numbers, punctuation or negations, and do not infer semantic
    equivalence. Later records never change an earlier keep/remove decision.
    """
    if hours <= 0:
        raise ValueError("Ventana de deduplicacion no positiva")
    frame = news.copy().reset_index(drop=True)
    frame["record_id"] = np.arange(len(frame))
    frame["published_at"] = pd.to_datetime(frame.published_at, utc=True, errors="raise")
    if frame[["ticker", "published_at"]].isna().any().any():
        raise ValueError("Noticia sin empresa o fecha")
    frame["normalized_title"] = frame.title.map(normalized_title)
    frame = frame.sort_values(["published_at", "record_id"], kind="stable")
    first, kept, links = {}, [], []
    for row in frame.itertuples():
        key = (row.ticker, row.normalized_title)
        previous = first.get(key)
        remove = previous is not None and row.published_at - previous[0] <= pd.Timedelta(hours=hours)
        kept.append(not remove)
        if remove:
            links.append({"record_id": row.record_id, "representative_id": previous[1],
                          "ticker": row.ticker, "published_at": row.published_at})
        else:
            first[key] = (row.published_at, row.record_id)
    columns = ["record_id", "representative_id", "ticker", "published_at"]
    return frame.loc[kept].drop(columns=["record_id", "normalized_title"]), pd.DataFrame(links, columns=columns)


def rebuild_sentiment(dataset, news):
    """Reaggregate only news; preserve every financial row and target."""
    financial = dataset[["ticker", "Date", "target_end", "target"] + FINANCIAL_FEATURES].copy()
    if news.empty:
        for column in SENTIMENT_FEATURES + ["sentiment_weighted"]:
            financial[column] = False if column == "has_news" else 0.0
        return financial.sort_values(["Date", "ticker"]).reset_index(drop=True)
    daily = aggregate_daily_sentiment(news)
    daily["trading_date"] = pd.to_datetime(daily.trading_date)
    weighted = news.assign(weighted_score=news.sentiment_score * news.relevance_score).groupby(
        ["ticker", "trading_date"]).agg(total=("weighted_score", "sum"), weight=("relevance_score", "sum"))
    weighted["sentiment_weighted"] = weighted.total.div(weighted.weight.replace(0, np.nan)).fillna(0)
    daily = daily.merge(weighted[["sentiment_weighted"]], on=["ticker", "trading_date"], validate="one_to_one")
    result = financial.merge(daily, left_on=["ticker", "Date"], right_on=["ticker", "trading_date"],
                             how="left", validate="one_to_one").drop(columns="trading_date")
    result = fill_days_without_news(result)
    result["sentiment_weighted"] = result.sentiment_weighted.fillna(0)
    return result.sort_values(["Date", "ticker"]).reset_index(drop=True)


def quality_audit(dataset, original, cleaned, links, output):
    output.mkdir(parents=True, exist_ok=True)
    tables, source_tables = [], []
    calendar = dataset.assign(month=dataset.Date.dt.to_period("M").astype(str)).groupby(
        ["ticker", "month"]).size().rename("sessions")
    for name, records in [("original", original), ("deduplicated", cleaned)]:
        records = records.merge(dataset[["ticker", "Date"]], left_on=["ticker", "trading_date"],
                                right_on=["ticker", "Date"], how="inner", validate="many_to_one")
        rows = records.assign(month=pd.to_datetime(records.trading_date).dt.to_period("M").astype(str))
        counts = rows.groupby(["ticker", "month"]).agg(news=("title", "size"),
            recorded_days=("trading_date", "nunique"), sources=("source", "nunique"),
            mean_relevance=("relevance_score", "mean"))
        table = calendar.to_frame().join(counts).fillna(0)
        table["news_per_session"] = table.news / table.sessions
        table["recorded_day_ratio"] = table.recorded_days / table.sessions
        tables.append(table.reset_index().assign(approach=name))
        sources = rows.groupby(["ticker", "month", "source"], dropna=False).size().rename("news").reset_index()
        sources["share"] = sources.news / sources.groupby(["ticker", "month"]).news.transform("sum")
        source_tables.append(sources.assign(approach=name))
    pd.concat(tables).to_csv(output / "monthly_coverage.csv", index=False)
    pd.concat(source_tables).to_csv(output / "monthly_sources.csv", index=False)
    removed = links.assign(year=pd.to_datetime(links.published_at).dt.year).groupby(
        ["ticker", "year"]).size().rename("removed_records").reset_index()
    removed.to_csv(output / "removed_by_year.csv", index=False)
    return {"original_records": len(original), "retained_records": len(cleaned), "removed_records": len(links),
            "policy": "first identical NFKC/casefold/whitespace-normalized title per ticker within 24 hours",
            "coverage_verified": False, "semantic_duplicates_removed": False}
