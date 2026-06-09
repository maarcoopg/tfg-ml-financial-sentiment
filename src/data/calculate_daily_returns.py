from pathlib import Path
import pandas as pd


PROCESSED_PRICES_DIR = Path("data/processed/prices")

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]


def load_processed_price_data(ticker: str) -> pd.DataFrame:
    file_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["daily_return"] = df["Adj Close"].pct_change().fillna(0.0)

    return df


def save_processed_data(ticker: str, df: pd.DataFrame) -> None:
    output_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df.to_csv(output_path, index=False)

    print(f"Guardado: {output_path}")


def main():
    for ticker in TICKERS:
        df = load_processed_price_data(ticker)
        df = add_daily_returns(df)

        save_processed_data(ticker, df)

        print(f"{ticker}")
        print(df[["Date", "Adj Close", "daily_return"]].head())
        print(f"Nulos en daily_return: {df['daily_return'].isna().sum()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
