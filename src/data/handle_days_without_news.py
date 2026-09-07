from pathlib import Path

import pandas as pd


MERGED_FILE = Path("data/processed/hybrid/financial_sentiment_merged.csv")
OUTPUT_FILE = Path("data/processed/hybrid/financial_sentiment_filled.csv")

SENTIMENT_SCORE_COLUMNS = [
    "sentiment_mean",
    "sentiment_median",
    "sentiment_min",
    "sentiment_max",
]
SENTIMENT_COUNT_COLUMNS = [
    "news_count",
    "non_trading_news_count",
    "positive_news_count",
    "negative_news_count",
    "neutral_news_count",
]
SENTIMENT_RATIO_COLUMNS = [
    "positive_news_ratio",
    "negative_news_ratio",
    "neutral_news_ratio",
]


def load_merged_data() -> pd.DataFrame:
    df = pd.read_csv(MERGED_FILE)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def fill_days_without_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_news"] = df["news_count"].notna()

    for column in SENTIMENT_SCORE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    for column in SENTIMENT_COUNT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    for column in SENTIMENT_RATIO_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    return df


def save_filled_data(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(df)} filas")


def main():
    df = load_merged_data()
    filled = fill_days_without_news(df)

    save_filled_data(filled)

    print(f"Filas sin noticias: {(~filled['has_news']).sum()}")
    print(f"Nulos restantes en variables de sentimiento: {filled[SENTIMENT_SCORE_COLUMNS + SENTIMENT_COUNT_COLUMNS + SENTIMENT_RATIO_COLUMNS].isna().sum().sum()}")
    print(filled["has_news"].value_counts())


if __name__ == "__main__":
    main()
