from pathlib import Path

import pandas as pd


FILLED_HYBRID_FILE = Path("data/processed/hybrid/financial_sentiment_filled.csv")
OUTPUT_FILE = Path("data/processed/hybrid/final_hybrid_dataset.csv")

FINAL_COLUMNS = [
    "ticker",
    "Date",
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "daily_return",
    "sma_5",
    "sma_20",
    "sma_50",
    "RSI",
    "MACD",
    "MACD_signal",
    "volatility_20",
    "sentiment_mean",
    "sentiment_median",
    "sentiment_min",
    "sentiment_max",
    "news_count",
    "non_trading_news_count",
    "positive_news_count",
    "negative_news_count",
    "neutral_news_count",
    "positive_news_ratio",
    "negative_news_ratio",
    "neutral_news_ratio",
    "has_news",
    "target",
]


def load_filled_hybrid_data() -> pd.DataFrame:
    df = pd.read_csv(FILLED_HYBRID_FILE)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def create_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    final_dataset = df[FINAL_COLUMNS]
    final_dataset = final_dataset.dropna(subset=["Date", "target"])
    final_dataset = final_dataset.sort_values(["ticker", "Date"]).reset_index(drop=True)

    return final_dataset


def save_final_dataset(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(df)} filas")


def main():
    df = load_filled_hybrid_data()
    final_dataset = create_final_dataset(df)

    save_final_dataset(final_dataset)

    print(final_dataset["ticker"].value_counts().sort_index())
    print(f"Nulos totales: {final_dataset.isna().sum().sum()}")
    print(f"Duplicados ticker-fecha: {final_dataset.duplicated(['ticker', 'Date']).sum()}")
    print(final_dataset.head())


if __name__ == "__main__":
    main()
