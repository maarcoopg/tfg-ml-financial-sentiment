from pathlib import Path
import pandas as pd


PROCESSED_PRICES_DIR = Path("data/processed/prices")

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]
MOVING_AVERAGE_WINDOWS = [5, 20, 50]


def load_processed_price_data(ticker: str) -> pd.DataFrame:
    file_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for window in MOVING_AVERAGE_WINDOWS:
        column_name = f"sma_{window}"
        df[column_name] = df["Adj Close"].rolling(window=window, min_periods=1).mean()

    return df


def save_processed_data(ticker: str, df: pd.DataFrame) -> None:
    output_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df.to_csv(output_path, index=False)

    print(f"Guardado: {output_path}")


def main():
    moving_average_columns = [f"sma_{window}" for window in MOVING_AVERAGE_WINDOWS]

    for ticker in TICKERS:
        df = load_processed_price_data(ticker)
        df = add_moving_averages(df)

        save_processed_data(ticker, df)

        print(f"{ticker}")
        print(df[["Date", "Adj Close", *moving_average_columns]].head())
        print(df[moving_average_columns].isna().sum())
        print("-" * 50)


if __name__ == "__main__":
    main()
