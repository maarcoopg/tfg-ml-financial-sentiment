from pathlib import Path
import pandas as pd


PROCESSED_PRICES_DIR = Path("data/processed/prices")

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9


def load_processed_price_data(ticker: str) -> pd.DataFrame:
    file_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula MACD con EMA de 12 y 26 sesiones y senal de 9 sesiones."""
    df = df.copy()

    fast_ema = df["Adj Close"].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
    slow_ema = df["Adj Close"].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()

    df["MACD"] = fast_ema - slow_ema
    df["MACD_signal"] = df["MACD"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()

    return df


def save_processed_data(ticker: str, df: pd.DataFrame) -> None:
    output_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df.to_csv(output_path, index=False)

    print(f"Guardado: {output_path}")


def main():
    macd_columns = ["MACD", "MACD_signal"]

    for ticker in TICKERS:
        df = load_processed_price_data(ticker)
        df = add_macd(df)

        save_processed_data(ticker, df)

        print(f"{ticker}")
        print(df[["Date", "Adj Close", *macd_columns]].head())
        print(df[macd_columns].isna().sum())
        print("-" * 50)


if __name__ == "__main__":
    main()
