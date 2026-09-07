from pathlib import Path
import pandas as pd


PROCESSED_PRICES_DIR = Path("data/processed/prices")

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]
VOLATILITY_WINDOW = 20


def load_processed_price_data(ticker: str) -> pd.DataFrame:
    file_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["daily_return"] = pd.to_numeric(df["daily_return"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def add_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
    """Calcula volatilidad como desviacion tipica movil de retornos diarios."""
    df = df.copy()

    df["volatility_20"] = (
        df["daily_return"]
        .rolling(window=window, min_periods=2)
        .std()
        .fillna(0.0)
    )

    return df


def save_processed_data(ticker: str, df: pd.DataFrame) -> None:
    output_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df.to_csv(output_path, index=False)

    print(f"Guardado: {output_path}")


def main():
    for ticker in TICKERS:
        df = load_processed_price_data(ticker)
        df = add_volatility(df)

        save_processed_data(ticker, df)

        print(f"{ticker}")
        print(df[["Date", "daily_return", "volatility_20"]].head())
        print(f"Nulos en volatility_20: {df['volatility_20'].isna().sum()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
