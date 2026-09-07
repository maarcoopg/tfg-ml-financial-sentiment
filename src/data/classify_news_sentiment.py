from pathlib import Path

import pandas as pd


NEWS_WITH_SENTIMENT_FILE = Path("data/processed/news/financial_news_with_sentiment.csv")
OUTPUT_FILE = Path("data/processed/news/financial_news_with_sentiment.csv")

POSITIVE_LABELS = {"Bullish", "Somewhat-Bullish"}
NEGATIVE_LABELS = {"Bearish", "Somewhat-Bearish"}
NEUTRAL_LABELS = {"Neutral"}


def load_news_with_sentiment() -> pd.DataFrame:
    df = pd.read_csv(NEWS_WITH_SENTIMENT_FILE)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    return df


def classify_sentiment_label(raw_label: str) -> str:
    if raw_label in POSITIVE_LABELS:
        return "positive"

    if raw_label in NEGATIVE_LABELS:
        return "negative"

    if raw_label in NEUTRAL_LABELS:
        return "neutral"

    return "neutral"


def add_sentiment_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sentiment_label"] = df["sentiment_label_raw"].map(classify_sentiment_label)

    df = df.sort_values(["ticker", "published_at", "title"]).reset_index(drop=True)

    return df


def save_classified_news(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(df)} noticias")


def main():
    df = load_news_with_sentiment()
    df = add_sentiment_class(df)

    save_classified_news(df)

    print(df["sentiment_label"].value_counts())
    print(pd.crosstab(df["sentiment_label_raw"], df["sentiment_label"]))
    print(f"Nulos en sentiment_label: {df['sentiment_label'].isna().sum()}")


if __name__ == "__main__":
    main()
