from pathlib import Path
import json

import pandas as pd


FINAL_DATASET_FILE = Path("data/processed/hybrid/final_hybrid_dataset.csv")
OUTPUT_DIR = Path("data/processed/model")

TRAIN_FILE = OUTPUT_DIR / "train.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"
X_TRAIN_FILE = OUTPUT_DIR / "X_train.csv"
X_TEST_FILE = OUTPUT_DIR / "X_test.csv"
Y_TRAIN_FILE = OUTPUT_DIR / "y_train.csv"
Y_TEST_FILE = OUTPUT_DIR / "y_test.csv"
BASE_TRAIN_FILE = OUTPUT_DIR / "base_train.csv"
BASE_TEST_FILE = OUTPUT_DIR / "base_test.csv"
HYBRID_TRAIN_FILE = OUTPUT_DIR / "hybrid_train.csv"
HYBRID_TEST_FILE = OUTPUT_DIR / "hybrid_test.csv"
X_BASE_TRAIN_FILE = OUTPUT_DIR / "X_base_train.csv"
X_BASE_TEST_FILE = OUTPUT_DIR / "X_base_test.csv"
X_HYBRID_TRAIN_FILE = OUTPUT_DIR / "X_hybrid_train.csv"
X_HYBRID_TEST_FILE = OUTPUT_DIR / "X_hybrid_test.csv"
FEATURE_CONFIG_FILE = OUTPUT_DIR / "feature_sets.json"

TRAIN_RATIO = 0.8
TARGET_COLUMN = "target"
ID_COLUMNS = ["ticker", "Date"]
MODELING_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]

FINANCIAL_FEATURES = [
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
]

SENTIMENT_FEATURES = [
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
]

HYBRID_FEATURES = FINANCIAL_FEATURES + SENTIMENT_FEATURES


def load_final_dataset() -> pd.DataFrame:
    df = pd.read_csv(FINAL_DATASET_FILE)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", TARGET_COLUMN])
    df = df[df["ticker"].isin(MODELING_TICKERS)]
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

    return df


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    sorted_dates = pd.Series(df["Date"].drop_duplicates().sort_values().to_list())
    split_index = int(len(sorted_dates) * TRAIN_RATIO)
    split_date = sorted_dates.iloc[split_index]

    train = df[df["Date"] < split_date].copy()
    test = df[df["Date"] >= split_date].copy()

    return train, test, split_date


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[HYBRID_FEATURES]
    y = df[ID_COLUMNS + [TARGET_COLUMN]]

    return x, y


def build_model_dataset(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return df[ID_COLUMNS + feature_columns + [TARGET_COLUMN]].copy()


def save_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = split_features_target(train)
    x_test, y_test = split_features_target(test)
    base_train = build_model_dataset(train, FINANCIAL_FEATURES)
    base_test = build_model_dataset(test, FINANCIAL_FEATURES)
    hybrid_train = build_model_dataset(train, HYBRID_FEATURES)
    hybrid_test = build_model_dataset(test, HYBRID_FEATURES)

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    x_train.to_csv(X_TRAIN_FILE, index=False)
    x_test.to_csv(X_TEST_FILE, index=False)
    y_train.to_csv(Y_TRAIN_FILE, index=False)
    y_test.to_csv(Y_TEST_FILE, index=False)
    base_train.to_csv(BASE_TRAIN_FILE, index=False)
    base_test.to_csv(BASE_TEST_FILE, index=False)
    hybrid_train.to_csv(HYBRID_TRAIN_FILE, index=False)
    hybrid_test.to_csv(HYBRID_TEST_FILE, index=False)
    train[FINANCIAL_FEATURES].to_csv(X_BASE_TRAIN_FILE, index=False)
    test[FINANCIAL_FEATURES].to_csv(X_BASE_TEST_FILE, index=False)
    train[HYBRID_FEATURES].to_csv(X_HYBRID_TRAIN_FILE, index=False)
    test[HYBRID_FEATURES].to_csv(X_HYBRID_TEST_FILE, index=False)
    feature_config = {
        "modeling_tickers": MODELING_TICKERS,
        "financial_features": FINANCIAL_FEATURES,
        "sentiment_features": SENTIMENT_FEATURES,
        "hybrid_features": HYBRID_FEATURES,
    }
    FEATURE_CONFIG_FILE.write_text(
        json.dumps(feature_config, indent=2),
        encoding="utf-8",
    )

    print(f"Guardado: {TRAIN_FILE} - {len(train)} filas")
    print(f"Guardado: {TEST_FILE} - {len(test)} filas")
    print(f"Guardado: {X_TRAIN_FILE} - {len(x_train)} filas")
    print(f"Guardado: {X_TEST_FILE} - {len(x_test)} filas")
    print(f"Guardado: {Y_TRAIN_FILE} - {len(y_train)} filas")
    print(f"Guardado: {Y_TEST_FILE} - {len(y_test)} filas")
    print(f"Guardado: {BASE_TRAIN_FILE} - {len(base_train)} filas")
    print(f"Guardado: {BASE_TEST_FILE} - {len(base_test)} filas")
    print(f"Guardado: {HYBRID_TRAIN_FILE} - {len(hybrid_train)} filas")
    print(f"Guardado: {HYBRID_TEST_FILE} - {len(hybrid_test)} filas")


def main():
    df = load_final_dataset()
    train, test, split_date = split_by_time(df)

    save_splits(train, test)

    print(f"Fecha de corte: {split_date.date()}")
    print(f"Max fecha train: {train['Date'].max().date()}")
    print(f"Min fecha test: {test['Date'].min().date()}")
    print(f"Filas train: {len(train)}")
    print(f"Filas test: {len(test)}")
    print(f"Columnas predictoras: {len(train.columns) - 1}")


if __name__ == "__main__":
    main()
