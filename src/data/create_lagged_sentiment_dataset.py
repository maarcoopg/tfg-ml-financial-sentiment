from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_validation import purged_train


FINAL_HYBRID_FILE = Path("data/processed/hybrid/final_hybrid_dataset.csv")
LAGGED_HYBRID_FILE = Path("data/processed/hybrid/final_hybrid_dataset_lagged.csv")
MODEL_DIR = Path("data/processed/model")
FEATURE_CONFIG_FILE = MODEL_DIR / "feature_sets.json"

LAGGED_TRAIN_FILE = MODEL_DIR / "lagged_hybrid_train.csv"
LAGGED_TEST_FILE = MODEL_DIR / "lagged_hybrid_test.csv"
X_LAGGED_TRAIN_FILE = MODEL_DIR / "X_lagged_hybrid_train.csv"
X_LAGGED_TEST_FILE = MODEL_DIR / "X_lagged_hybrid_test.csv"

TRAIN_RATIO = 0.8
ID_COLUMNS = ["ticker", "Date"]
TARGET_COLUMN = "target"
LAGS = [1, 2, 3]
ROLLING_WINDOWS = [3, 5]

LAG_SOURCE_COLUMNS = [
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


def load_feature_sets() -> dict[str, list[str]]:
    with FEATURE_CONFIG_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def load_final_hybrid_dataset() -> pd.DataFrame:
    df = pd.read_csv(FINAL_HYBRID_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", TARGET_COLUMN])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    return df


def add_lagged_sentiment_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    lagged_features = []

    for column in LAG_SOURCE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in LAG_SOURCE_COLUMNS:
        grouped = df.groupby("ticker", group_keys=False)[column]

        for lag in LAGS:
            feature_name = f"{column}_lag_{lag}"
            df[feature_name] = grouped.shift(lag)
            lagged_features.append(feature_name)

        for window in ROLLING_WINDOWS:
            feature_name = f"{column}_rolling_{window}"
            df[feature_name] = grouped.transform(
                lambda values: values.rolling(window=window, min_periods=1).mean()
            )
            lagged_features.append(feature_name)

    df[lagged_features] = df[lagged_features].fillna(0).astype(float)
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

    return df, lagged_features


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    sorted_dates = pd.Series(df["Date"].drop_duplicates().sort_values().to_list())
    split_index = int(len(sorted_dates) * TRAIN_RATIO)
    split_date = sorted_dates.iloc[split_index]

    train = purged_train(df, split_date)
    test = df[df["Date"] >= split_date].copy()

    return train, test, split_date


def update_feature_config(lagged_features: list[str]) -> list[str]:
    feature_sets = load_feature_sets()
    lagged_hybrid_features = feature_sets["hybrid_features"] + lagged_features
    feature_sets["lagged_sentiment_features"] = lagged_features
    feature_sets["lagged_hybrid_features"] = lagged_hybrid_features
    feature_sets.setdefault("purged_datasets", {})["lagged_hybrid"] = True

    FEATURE_CONFIG_FILE.write_text(
        json.dumps(feature_sets, indent=2),
        encoding="utf-8",
    )

    return lagged_hybrid_features


def save_outputs(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    LAGGED_HYBRID_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(LAGGED_HYBRID_FILE, index=False)
    train[ID_COLUMNS + feature_columns + [TARGET_COLUMN]].to_csv(
        LAGGED_TRAIN_FILE,
        index=False,
    )
    test[ID_COLUMNS + feature_columns + [TARGET_COLUMN]].to_csv(
        LAGGED_TEST_FILE,
        index=False,
    )
    train[feature_columns].to_csv(X_LAGGED_TRAIN_FILE, index=False)
    test[feature_columns].to_csv(X_LAGGED_TEST_FILE, index=False)

    print(f"Guardado: {LAGGED_HYBRID_FILE} - {len(df)} filas")
    print(f"Guardado: {LAGGED_TRAIN_FILE} - {len(train)} filas")
    print(f"Guardado: {LAGGED_TEST_FILE} - {len(test)} filas")
    print(f"Guardado: {X_LAGGED_TRAIN_FILE} - {len(train)} filas")
    print(f"Guardado: {X_LAGGED_TEST_FILE} - {len(test)} filas")


def main():
    df = load_final_hybrid_dataset()
    lagged_df, lagged_features = add_lagged_sentiment_features(df)
    lagged_hybrid_features = update_feature_config(lagged_features)
    train, test, split_date = split_by_time(lagged_df)

    save_outputs(lagged_df, train, test, lagged_hybrid_features)

    print(f"Fecha de corte: {split_date.date()}")
    print(f"Variables lagged nuevas: {len(lagged_features)}")
    print(f"Variables lagged híbridas totales: {len(lagged_hybrid_features)}")
    print(f"Nulos totales: {lagged_df.isna().sum().sum()}")
    print(lagged_df["ticker"].value_counts().sort_index())


if __name__ == "__main__":
    main()
