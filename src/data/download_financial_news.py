from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


API_URL = "https://www.alphavantage.co/query"
RAW_NEWS_DIR = Path("data/raw/news")
CHUNKS_DIR = RAW_NEWS_DIR / "chunks"
OUTPUT_FILE = RAW_NEWS_DIR / "financial_news_raw.csv"

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]
START_DATE = date(2015, 1, 1)
END_DATE = date(2025, 12, 31)

LIMIT = 1000
SPLIT_THRESHOLD = 950
REQUEST_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def to_alpha_vantage_params(self) -> tuple[str, str]:
        return (
            f"{self.start:%Y%m%d}T0000",
            f"{self.end:%Y%m%d}T2359",
        )

    def to_file_fragment(self) -> str:
        return f"{self.start:%Y%m%d}_{self.end:%Y%m%d}"


def get_api_key() -> str:
    load_dotenv(Path(".env"))

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        raise RuntimeError("Falta ALPHA_VANTAGE_API_KEY en el archivo .env")

    return api_key


def yearly_ranges(start: date, end: date) -> list[DateRange]:
    ranges = []
    current = start

    while current <= end:
        range_end = min(date(current.year, 12, 31), end)
        ranges.append(DateRange(current, range_end))
        current = range_end + timedelta(days=1)

    return ranges


def split_range(date_range: DateRange) -> list[DateRange]:
    if date_range.days <= 1:
        return [date_range]

    midpoint = date_range.start + timedelta(days=(date_range.days // 2) - 1)

    return [
        DateRange(date_range.start, midpoint),
        DateRange(midpoint + timedelta(days=1), date_range.end),
    ]


def request_news(ticker: str, date_range: DateRange, api_key: str) -> dict:
    time_from, time_to = date_range.to_alpha_vantage_params()

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "sort": "EARLIEST",
        "limit": LIMIT,
        "apikey": api_key,
    }

    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "Note" in data:
        raise RuntimeError(data["Note"])

    if "Information" in data:
        raise RuntimeError(data["Information"])

    if "Error Message" in data:
        raise RuntimeError(data["Error Message"])

    return data


def get_ticker_sentiment(news_item: dict, ticker: str) -> dict:
    ticker_sentiment = news_item.get("ticker_sentiment", [])

    for item in ticker_sentiment:
        if item.get("ticker") == ticker:
            return item

    return {}


def flatten_news_item(news_item: dict, ticker: str) -> dict:
    sentiment = get_ticker_sentiment(news_item, ticker)

    return {
        "ticker": ticker,
        "title": news_item.get("title"),
        "summary": news_item.get("summary"),
        "url": news_item.get("url"),
        "source": news_item.get("source"),
        "source_domain": news_item.get("source_domain"),
        "time_published": news_item.get("time_published"),
        "banner_image": news_item.get("banner_image"),
        "overall_sentiment_score": news_item.get("overall_sentiment_score"),
        "overall_sentiment_label": news_item.get("overall_sentiment_label"),
        "relevance_score": sentiment.get("relevance_score"),
        "ticker_sentiment_score": sentiment.get("ticker_sentiment_score"),
        "ticker_sentiment_label": sentiment.get("ticker_sentiment_label"),
    }


def chunk_path(ticker: str, date_range: DateRange) -> Path:
    return CHUNKS_DIR / f"{ticker}_{date_range.to_file_fragment()}.csv"


def save_chunk(ticker: str, date_range: DateRange, feed: list[dict]) -> Path:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [flatten_news_item(item, ticker) for item in feed]
    df = pd.DataFrame(rows)

    output_path = chunk_path(ticker, date_range)
    df.to_csv(output_path, index=False)

    return output_path


def download_range(ticker: str, date_range: DateRange, api_key: str) -> list[Path]:
    output_path = chunk_path(ticker, date_range)

    if output_path.exists():
        existing_rows = len(pd.read_csv(output_path))

        if existing_rows >= SPLIT_THRESHOLD and date_range.days > 1:
            paths = []

            for child_range in split_range(date_range):
                paths.extend(download_range(ticker, child_range, api_key))

            return paths

        print(f"Ya existe: {output_path}")
        return [output_path]

    data = request_news(ticker, date_range, api_key)
    time.sleep(REQUEST_DELAY_SECONDS)

    feed = data.get("feed", [])
    item_count = len(feed)

    print(f"{ticker} {date_range.start} - {date_range.end}: {item_count} noticias")

    if item_count >= SPLIT_THRESHOLD and date_range.days > 31:
        paths = []

        for child_range in split_range(date_range):
            paths.extend(download_range(ticker, child_range, api_key))

        return paths

    if item_count >= SPLIT_THRESHOLD and date_range.days > 1:
        paths = []

        for child_range in split_range(date_range):
            paths.extend(download_range(ticker, child_range, api_key))

        return paths

    output_path = save_chunk(ticker, date_range, feed)

    return [output_path]


def combine_chunks(chunk_files: list[Path]) -> pd.DataFrame:
    dataframes = []

    for file_path in chunk_files:
        if file_path.exists() and file_path.stat().st_size > 0:
            dataframes.append(pd.read_csv(file_path))

    if not dataframes:
        return pd.DataFrame()

    df = pd.concat(dataframes, ignore_index=True)

    df["published_at"] = pd.to_datetime(
        df["time_published"],
        format="%Y%m%dT%H%M%S",
        errors="coerce",
    )
    df["published_date"] = df["published_at"].dt.date

    df = df.drop_duplicates(
        subset=["ticker", "url", "title", "published_at"],
        keep="first",
    )
    df = df.sort_values(["ticker", "published_at", "title"]).reset_index(drop=True)

    return df


def main():
    api_key = get_api_key()
    chunk_files = []

    for ticker in TICKERS:
        for date_range in yearly_ranges(START_DATE, END_DATE):
            chunk_files.extend(download_range(ticker, date_range, api_key))

    df = combine_chunks(chunk_files)

    RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(df)} noticias")
    print(df["ticker"].value_counts().sort_index())


if __name__ == "__main__":
    main()
