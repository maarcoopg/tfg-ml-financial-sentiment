import re
from pathlib import Path

import pandas as pd


RAW_NEWS_FILE = Path("data/raw/news/financial_news_raw.csv")
PROCESSED_NEWS_DIR = Path("data/processed/news")
OUTPUT_FILE = PROCESSED_NEWS_DIR / "financial_news_clean.csv"

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]


def load_raw_news() -> pd.DataFrame:
    df = pd.read_csv(RAW_NEWS_FILE)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    return df


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_text_column(df: pd.DataFrame) -> pd.Series:
    title = df["title"].map(clean_text)
    summary = df["summary"].map(clean_text)

    text = title
    text = text.mask(summary != "", title + ". " + summary)

    return text.map(clean_text)


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["ticker"].isin(TICKERS)]
    df = df.dropna(subset=["published_at", "title", "url"])

    df["title"] = df["title"].map(clean_text)
    df["summary"] = df["summary"].map(clean_text)
    df["text"] = build_text_column(df)

    df = df[df["text"] != ""]

    df = df.drop_duplicates(
        subset=["ticker", "url", "title", "published_at"],
        keep="first",
    )

    numeric_columns = [
        "overall_sentiment_score",
        "relevance_score",
        "ticker_sentiment_score",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    columns = [
        "ticker",
        "published_at",
        "published_date",
        "source",
        "source_domain",
        "url",
        "title",
        "summary",
        "text",
        "overall_sentiment_score",
        "overall_sentiment_label",
        "relevance_score",
        "ticker_sentiment_score",
        "ticker_sentiment_label",
    ]

    df = df[columns]
    df = df.sort_values(["ticker", "published_at", "title"]).reset_index(drop=True)

    return df


def save_clean_news(df: pd.DataFrame) -> None:
    PROCESSED_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(df)} noticias")


def main():
    df = load_raw_news()
    df = clean_news(df)

    save_clean_news(df)

    print(df["ticker"].value_counts().sort_index())
    print(f"Nulos en published_at: {df['published_at'].isna().sum()}")
    print(f"Textos vacios: {(df['text'] == '').sum()}")
    print(f"Duplicados: {df.duplicated(subset=['ticker', 'url', 'title', 'published_at']).sum()}")


if __name__ == "__main__":
    main()
