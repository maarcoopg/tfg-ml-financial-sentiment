from pathlib import Path

import pandas as pd


PROCESSED_PRICES_DIR = Path("data/processed/prices")
DAILY_SENTIMENT_FILE = Path("data/processed/news/daily_sentiment.csv")
PROCESSED_HYBRID_DIR = Path("data/processed/hybrid")
OUTPUT_FILE = PROCESSED_HYBRID_DIR / "financial_sentiment_merged.csv"

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]


def load_financial_data() -> pd.DataFrame:
    dataframes = []

    for ticker in TICKERS:
        file_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"
        df = pd.read_csv(file_path)

        df["ticker"] = ticker
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        dataframes.append(df)

    financial_data = pd.concat(dataframes, ignore_index=True)
    financial_data = financial_data.sort_values(["ticker", "Date"]).reset_index(drop=True)

    return financial_data


def load_daily_sentiment() -> pd.DataFrame:
    df = pd.read_csv(DAILY_SENTIMENT_FILE)

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")

    return df


def merge_financial_sentiment(
    financial_data: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    merged = financial_data.merge(
        daily_sentiment,
        left_on=["ticker", "Date"],
        right_on=["ticker", "trading_date"],
        how="left",
    )

    merged = merged.drop(columns=["trading_date"])
    merged = merged.sort_values(["ticker", "Date"]).reset_index(drop=True)

    return merged


def save_merged_data(df: pd.DataFrame) -> None:
    PROCESSED_HYBRID_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(df)} filas")


def main():
    financial_data = load_financial_data()
    daily_sentiment = load_daily_sentiment()
    merged = merge_financial_sentiment(financial_data, daily_sentiment)

    save_merged_data(merged)

    print(merged["ticker"].value_counts().sort_index())
    print(f"Filas sin sentimiento: {merged['news_count'].isna().sum()}")
    print(f"Duplicados ticker-fecha: {merged.duplicated(['ticker', 'Date']).sum()}")


if __name__ == "__main__":
    main()
