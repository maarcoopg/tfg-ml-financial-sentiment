import yfinance as yf
import pandas as pd
from pathlib import Path

tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]

start_date = "2015-01-01"
end_date = "2025-12-31"

output_dir = Path("data/raw/prices")
output_dir.mkdir(parents=True, exist_ok=True)

for ticker in tickers:
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )

    df = df.reset_index()
    df.to_csv(output_dir / f"{ticker}.csv", index=False)

    print(f"Guardado: {ticker}.csv - {len(df)} filas")