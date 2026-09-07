from pathlib import Path

import pandas as pd


CLEAN_NEWS_FILE = Path("data/processed/news/financial_news_clean.csv")
PROCESSED_NEWS_DIR = Path("data/processed/news")
OUTPUT_FILE = PROCESSED_NEWS_DIR / "financial_news_with_sentiment.csv"


def load_clean_news() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_NEWS_FILE)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df["ticker_sentiment_score"] = pd.to_numeric(
        df["ticker_sentiment_score"],
        errors="coerce",
    )

    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sentiment_score"] = df["ticker_sentiment_score"]
    df["sentiment_label_raw"] = df["ticker_sentiment_label"]
    df["sentiment_source"] = "alpha_vantage_ticker_sentiment"

    df = df.dropna(subset=["sentiment_score", "sentiment_label_raw"])
    df = df.sort_values(["ticker", "published_at", "title"]).reset_index(drop=True)

    return df


def save_news_with_sentiment(df: pd.DataFrame) -> None:
    PROCESSED_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(df)} noticias")


def main():
    df = load_clean_news()
    df = add_sentiment(df)

    save_news_with_sentiment(df)

    print(df["ticker"].value_counts().sort_index())
    print(df["sentiment_label_raw"].value_counts())
    print(f"Nulos en sentiment_score: {df['sentiment_score'].isna().sum()}")
    print(df[["ticker", "published_date", "title", "sentiment_score", "sentiment_label_raw"]].head())


if __name__ == "__main__":
    main()
