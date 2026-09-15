from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.point_in_time import align_at_close, trading_schedule


NEWS_WITH_SENTIMENT_FILE = Path("data/processed/news/financial_news_with_sentiment.csv")
PROCESSED_PRICES_DIR = Path("data/processed/prices")
PROCESSED_NEWS_DIR = Path("data/processed/news")
OUTPUT_FILE = PROCESSED_NEWS_DIR / "financial_news_trading_days.csv"
UNALIGNED_FILE = PROCESSED_NEWS_DIR / "financial_news_unaligned.csv"

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]


def load_news() -> pd.DataFrame:
    df = pd.read_csv(NEWS_WITH_SENTIMENT_FILE)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    return df


def load_trading_dates(ticker: str) -> pd.DataFrame:
    file_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"
    df = pd.read_csv(file_path, usecols=["Date"])

    df["trading_date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["trading_date"])
    df = df.sort_values("trading_date").reset_index(drop=True)

    return df[["trading_date"]]


def align_ticker_news(news: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ticker_news = news[news["ticker"] == ticker].copy()
    trading_dates = load_trading_dates(ticker)

    schedule = trading_schedule(trading_dates["trading_date"].min(), trading_dates["trading_date"].max())
    aligned, unaligned = align_at_close(ticker_news, schedule)
    result = pd.concat([aligned, unaligned], ignore_index=True)
    result["original_published_date"] = result["published_date"]
    return result


def align_news_to_trading_days(news: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned_dataframes = []

    for ticker in TICKERS:
        aligned_dataframes.append(align_ticker_news(news, ticker))

    df = pd.concat(aligned_dataframes, ignore_index=True)

    unaligned = df[df["trading_date"].isna()].copy()
    aligned = df.dropna(subset=["trading_date"]).copy()

    aligned["trading_date"] = pd.to_datetime(aligned["trading_date"]).dt.date
    aligned["published_date"] = pd.to_datetime(aligned["published_date"]).dt.date
    aligned["original_published_date"] = pd.to_datetime(
        aligned["original_published_date"],
    ).dt.date

    if not unaligned.empty:
        unaligned["published_date"] = pd.to_datetime(unaligned["published_date"]).dt.date
        unaligned["original_published_date"] = pd.to_datetime(
            unaligned["original_published_date"],
        ).dt.date

    aligned = aligned.sort_values(["ticker", "trading_date", "published_at", "title"])
    aligned = aligned.reset_index(drop=True)

    unaligned = unaligned.sort_values(["ticker", "published_at", "title"])
    unaligned = unaligned.reset_index(drop=True)

    return aligned, unaligned


def save_outputs(aligned: pd.DataFrame, unaligned: pd.DataFrame) -> None:
    PROCESSED_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    aligned.to_csv(OUTPUT_FILE, index=False)
    unaligned.to_csv(UNALIGNED_FILE, index=False)

    print(f"Guardado: {OUTPUT_FILE} - {len(aligned)} noticias")
    print(f"Guardado: {UNALIGNED_FILE} - {len(unaligned)} noticias")


def main():
    news = load_news()
    aligned, unaligned = align_news_to_trading_days(news)

    save_outputs(aligned, unaligned)

    print(aligned["ticker"].value_counts().sort_index())
    print(f"Noticias reasignadas: {aligned['is_non_trading_day'].sum()}")
    print(f"Noticias sin dia bursatil posterior: {len(unaligned)}")


if __name__ == "__main__":
    main()
