"""Fixed news eligibility and strictly paired sentiment panels for issue 51."""

import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from src.experiments.news_quality import rebuild_sentiment
from src.experiments.sentiment_representation import relative_panel
from src.nlp.sentiment_data import LABELS, company_context, digest


START = "2021-01-01"
DYNAMIC_TITLE = re.compile(
    r"\b(?:stock|share|etf|fund) price\s*(?:history|&|[|:]|$)|"
    r"\b(?:holdings list|interactive stock chart|stock quote|stock price and chart|stock fund price and chart)\b|"
    r"\betf\b.*\b(?:holdings|dividend yield)\b|\bcompany profile\s*(?:&|$)", re.I)


def identify_news(news):
    result = news.copy().fillna("")
    result["published_at"] = pd.to_datetime(result.published_at, utc=True, errors="raise")
    result["news_id"] = [digest("\x1f".join([r.ticker, r.published_at.isoformat(), r.url, r.title, r.text]))
                         for r in result.itertuples()]
    result["text_sha256"] = result.text.map(digest)
    return result


def news_eligibility(news):
    frame = identify_news(news)
    frame["explicit_company"] = [bool(company_context(r.title, r.summary, r.ticker)) for r in frame.itertuples()]
    frame["dynamic_page"] = [bool(DYNAMIC_TITLE.search(r.title)) or
        urlsplit(r.url).path.lower().startswith(("/quote/", "/symbols/", "/etfs/")) for r in frame.itertuples()]
    frame["post_checkpoint"] = frame.published_at >= pd.Timestamp(START, tz="UTC")
    frame["eligible"] = frame.explicit_company & ~frame.dynamic_page & frame.post_checkpoint & frame.text.str.strip().ne("")
    return frame


def replace_sentiment(news, scores):
    required = {"text_sha256", "positive", "negative", "neutral", "score", "label"}
    if not required <= set(scores):
        raise ValueError("Incomplete score schema")
    if scores.text_sha256.duplicated().any():
        raise ValueError("Duplicate text scores")
    merged = news.merge(scores[list(required)], on="text_sha256", how="left", validate="many_to_one")
    probabilities = merged[list(LABELS)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Missing or invalid probabilities")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    if not merged.label.isin(LABELS).all():
        raise ValueError("Unknown FinBERT labels")
    np.testing.assert_allclose(merged.score, merged.positive - merged.negative, atol=1e-7)
    expected = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    if not (expected == merged.label).all():
        raise ValueError("Label does not match probability maximum")
    merged["sentiment_score"] = merged.score
    merged["sentiment_label"] = merged.label
    return merged


def build_paired_panels(dataset, all_news, matched_news, scores):
    finbert = replace_sentiment(matched_news, scores)
    panels, schemas = {}, None
    for name, news in [("alpha_all", all_news), ("alpha_matched", matched_news), ("finbert_matched", finbert)]:
        frame, schema = relative_panel(rebuild_sentiment(dataset, news))
        frame = frame[frame.Date >= START].reset_index(drop=True)
        if schemas is not None and schema != schemas:
            raise ValueError("Mismatched feature schemas")
        panels[name], schemas = frame, schema
    panels["financial"] = panels["alpha_all"].copy()
    key = ["ticker", "Date", "target", "target_end"]
    for frame in panels.values():
        pd.testing.assert_frame_equal(frame[key + schemas["base"]], panels["financial"][key + schemas["base"]])
    common = ["news_count", "news_count_relative", "news_count_relative_lag_1", "has_news", "non_trading_news_count"]
    pd.testing.assert_frame_equal(panels["alpha_matched"][common], panels["finbert_matched"][common])
    return panels, schemas, finbert
