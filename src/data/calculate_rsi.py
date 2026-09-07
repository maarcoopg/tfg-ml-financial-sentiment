from pathlib import Path
import pandas as pd


PROCESSED_PRICES_DIR = Path("data/processed/prices")

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]
RSI_PERIOD = 14


def load_processed_price_data(ticker: str) -> pd.DataFrame:
    file_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calcula el RSI de Wilder con un periodo por defecto de 14 sesiones."""
    df = df.copy()

    price_delta = df["Adj Close"].diff()
    gains = price_delta.clip(lower=0)
    losses = -price_delta.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50)
    rsi = rsi.mask((average_gain > 0) & (average_loss == 0), 100)
    rsi = rsi.fillna(50).clip(lower=0, upper=100)

    df["RSI"] = rsi

    return df


def save_processed_data(ticker: str, df: pd.DataFrame) -> None:
    output_path = PROCESSED_PRICES_DIR / f"{ticker}_with_target.csv"

    df.to_csv(output_path, index=False)

    print(f"Guardado: {output_path}")


def main():
    for ticker in TICKERS:
        df = load_processed_price_data(ticker)
        df = add_rsi(df)

        save_processed_data(ticker, df)

        print(f"{ticker}")
        print(df[["Date", "Adj Close", "RSI"]].head())
        print(f"Nulos en RSI: {df['RSI'].isna().sum()}")
        print(f"RSI fuera de rango: {(~df['RSI'].between(0, 100)).sum()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
