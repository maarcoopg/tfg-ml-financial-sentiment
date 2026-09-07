from pathlib import Path

import pandas as pd


NEWS_TRADING_DAYS_FILE = Path("data/processed/news/financial_news_trading_days.csv")
PROCESSED_NEWS_DIR = Path("data/processed/news")
OUTPUT_FILE = PROCESSED_NEWS_DIR / "daily_sentiment.csv"


def load_news() -> pd.DataFrame:
    df = pd.read_csv(NEWS_TRADING_DAYS_FILE)

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    return df


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    class_counts = (
        df.pivot_table(
            index=["ticker", "trading_date"],
            columns="sentiment_label",
            values="title",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for label in ["positive", "negative", "neutral"]:
        if label not in class_counts.columns:
            class_counts[label] = 0

    class_counts = class_counts.rename(
        columns={
            "positive": "positive_news_count",
            "negative": "negative_news_count",
            "neutral": "neutral_news_count",
        },
    )

    numeric_aggregation = (
        df.groupby(["ticker", "trading_date"])
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_median=("sentiment_score", "median"),
            sentiment_min=("sentiment_score", "min"),
            sentiment_max=("sentiment_score", "max"),
            news_count=("sentiment_score", "size"),
            non_trading_news_count=("is_non_trading_day", "sum"),
        )
        .reset_index()
    )

    daily_sentiment = numeric_aggregation.merge(
        class_counts,
        on=["ticker", "trading_date"],
        how="left",
    )

    for label in ["positive", "negative", "neutral"]:
        count_column = f"{label}_news_count"
        ratio_column = f"{label}_news_ratio"

        daily_sentiment[ratio_column] = (
            daily_sentiment[count_column] / daily_sentiment["news_count"]
        )

    daily_sentiment["trading_date"] = daily_sentiment["trading_date"].dt.date

    daily_sentiment = daily_sentiment.sort_values(["ticker", "trading_date"])
    daily_sentiment = daily_sentiment.reset_index(drop=True)

    return daily_sentiment


def save_daily_sentiment(df: pd.DataFrame) -> None:
    PROCESSED_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(df)} filas")


def main():
    df = load_news()
    daily_sentiment = aggregate_daily_sentiment(df)

    save_daily_sentiment(daily_sentiment)

    print(daily_sentiment.groupby("ticker")["trading_date"].count())
    print(f"Nulos en sentiment_mean: {daily_sentiment['sentiment_mean'].isna().sum()}")
    print(daily_sentiment.head())


if __name__ == "__main__":
    main()
