from pathlib import Path
import pandas as pd


RAW_PRICES_DIR = Path("data/raw/prices")
PROCESSED_PRICES_DIR = Path("data/processed/prices")

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]


def load_price_data(ticker: str) -> pd.DataFrame:
    file_path = RAW_PRICES_DIR / f"{ticker}.csv"

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["next_adj_close"] = df["Adj Close"].shift(-1)

    df["target"] = (df["next_adj_close"] > df["Adj Close"]).astype(int)

    df = df.dropna(subset=["next_adj_close"])

    return df


def save_processed_data(ticker: str, df: pd.DataFrame) -> None:
    PROCESSED_PRICES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df.to_csv(output_path, index=False)

    print(f"Guardado: {output_path}")


def main():
    for ticker in TICKERS:
        df = load_price_data(ticker)
        df = create_binary_target(df)

        save_processed_data(ticker, df)

        print(f"{ticker}")
        print(df[["Date", "Adj Close", "next_adj_close", "target"]].head())
        print(df["target"].value_counts(normalize=True))
        print("-" * 50)


if __name__ == "__main__":
    main()