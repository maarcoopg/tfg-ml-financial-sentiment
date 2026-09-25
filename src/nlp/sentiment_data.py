"""News grouping, blind sampling and explicit company-context extraction."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd


TICKERS = ("AAPL", "MSFT", "NVDA", "TSLA")
CUTOFF = "2023-10-16"
ALIASES = {"AAPL": r"\b(?:Apple|AAPL)\b", "MSFT": r"\b(?:Microsoft|MSFT)\b",
           "NVDA": r"\b(?:NVIDIA|NVDA)\b", "TSLA": r"\b(?:Tesla|TSLA)\b"}
LABELS = ("negative", "neutral", "positive")
ANNOTATION_LABELS = (*LABELS, "insufficient")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if not parts.netloc:
        return value.strip()
    # Keep queries: some publishers use a query parameter as the article ID.
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path,
                       parts.query, ""))


def company_context(title: str, summary: str, ticker: str) -> str:
    """Deterministic sentence selection, not a trained entity-sentiment model."""
    pattern = re.compile(ALIASES[ticker], flags=re.I)
    pieces = [title.strip(), *re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", summary.strip())]
    return " ".join(dict.fromkeys(piece for piece in pieces if piece and pattern.search(piece)))


def prepare_news(news: pd.DataFrame, cutoff: str = CUTOFF) -> pd.DataFrame:
    required = {"ticker", "title", "summary", "text", "url", "published_at"}
    if not required <= set(news):
        raise ValueError(f"Missing columns: {sorted(required - set(news))}")
    data = news.loc[news.ticker.isin(TICKERS)].copy().fillna("").reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data.published_at, utc=True, errors="raise")
    if data.timestamp.isna().any():
        raise ValueError("Missing news timestamps")
    keys = [[("url", canonical_url(row.url)), ("title", normalized(row.title)),
             ("text", normalized(row.text))] for row in data.itertuples()]
    # Connected components join transitive duplicates even across companies.
    parents = list(range(len(data)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row_keys in enumerate(keys):
        for key in row_keys:
            if key[1]:
                if key in seen:
                    parents[root(index)] = root(seen[key])
                else:
                    seen[key] = index
    members = {}
    for index, row_keys in enumerate(keys):
        members.setdefault(root(index), []).extend(f"{kind}:{value}" for kind, value in row_keys if value)
    group_names = {key: digest(min(values)) for key, values in members.items()}
    data["group_id"] = [group_names[root(i)] for i in range(len(data))]
    data["news_id"] = [digest("\x1f".join([r.ticker, r.timestamp.isoformat(), r.url, r.title, r.text]))
                       for r in data.itertuples()]
    data["multi_company"] = data.groupby("group_id").ticker.transform("nunique") > 1
    boundary = pd.Timestamp(cutoff, tz="UTC")
    data["partition"] = np.where(data.timestamp < boundary, "development", "evaluation")
    data["cross_boundary"] = data.groupby("group_id").partition.transform("nunique") > 1
    data["context"] = [company_context(r.title, r.summary, r.ticker) for r in data.itertuples()]
    # Diagnostic flags only; neither ASCII nor a stopword count certifies English.
    data["english_function_words"] = data.text.map(
        lambda text: len(re.findall(r"\b(?:the|and|of|to|in|for|with|is|on|its)\b", text, re.I)))
    data["language_review_flag"] = data.english_function_words < 2
    return data


def sample_news(data: pd.DataFrame, per_cell: int = 50, seed: int = 50) -> pd.DataFrame:
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    eligible = data[~data.cross_boundary & data.title.str.strip().ne("") & data.text.str.strip().ne("")]
    eligible = eligible.drop_duplicates("news_id").copy()
    eligible["priority"] = eligible.news_id.map(lambda value: digest(f"{seed}:{value}"))
    chosen, used = [], set()
    for partition in ["development", "evaluation"]:
        for ticker in TICKERS:
            candidates = eligible[(eligible.partition == partition) & (eligible.ticker == ticker)]
            candidates = candidates.sort_values(["priority", "news_id"])
            count = 0
            for row in candidates.itertuples():
                if row.group_id in used:
                    continue
                used.add(row.group_id)
                chosen.append(row.news_id)
                count += 1
                if count == per_cell:
                    break
            if count != per_cell:
                raise ValueError(f"Insufficient distinct groups for {partition}/{ticker}: {count}")
    return eligible.set_index("news_id").loc[chosen].reset_index().drop(columns="priority")


def annotation_templates(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["news_id", "group_id", "partition", "ticker", "published_at", "title", "summary"]
    first = sample[columns].copy()
    for column in ["language", "label", "reason", "annotator_id", "reviewed_at"]:
        first[column] = ""
    second = pd.concat([rows.sort_values("news_id").head(max(1, len(rows) // 4))
                        for _, rows in first.groupby(["partition", "ticker"])])
    return first.reset_index(drop=True), second.reset_index(drop=True)


def validated_annotations(sample: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    immutable = ["ticker", "partition", "group_id", "title", "summary"]
    required = {"news_id", "label", "language", "annotator_id", "reviewed_at", *immutable}
    if not required <= set(labels):
        raise ValueError("Annotation schema is incomplete")
    if labels.news_id.duplicated().any() or set(labels.news_id) != set(sample.news_id):
        raise ValueError("Annotations must match every requested ID exactly once")
    if not labels.label.isin(ANNOTATION_LABELS).all():
        raise ValueError("Human labels are missing or invalid; no quality metrics generated")
    if not labels.language.isin(["en", "other", "uncertain"]).all():
        raise ValueError("Human language review is missing")
    if any(labels[c].fillna("").str.strip().eq("").any() for c in ["annotator_id", "reviewed_at"]):
        raise ValueError("Human annotation provenance is required")
    dates = pd.to_datetime(labels.reviewed_at, errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("Invalid annotation dates")
    merged = sample[["news_id", *immutable]].merge(labels, on="news_id", validate="one_to_one",
                                                  suffixes=("", "_annotated"))
    if any(not merged[c].equals(merged[c + "_annotated"]) for c in immutable):
        raise ValueError("Immutable news metadata changed during annotation")
    return merged
