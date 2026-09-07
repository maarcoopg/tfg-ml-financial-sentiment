from pathlib import Path

import pandas as pd


FINAL_DATASET_FILE = Path("data/processed/hybrid/final_hybrid_dataset.csv")
OUTPUT_DIR = Path("data/processed/model")

TRAIN_FILE = OUTPUT_DIR / "train.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"
X_TRAIN_FILE = OUTPUT_DIR / "X_train.csv"
X_TEST_FILE = OUTPUT_DIR / "X_test.csv"
Y_TRAIN_FILE = OUTPUT_DIR / "y_train.csv"
Y_TEST_FILE = OUTPUT_DIR / "y_test.csv"

TRAIN_RATIO = 0.8
TARGET_COLUMN = "target"
ID_COLUMNS = ["ticker", "Date"]


def load_final_dataset() -> pd.DataFrame:
    df = pd.read_csv(FINAL_DATASET_FILE)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", TARGET_COLUMN])
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
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[ID_COLUMNS + [TARGET_COLUMN]]

    return x, y


def save_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = split_features_target(train)
    x_test, y_test = split_features_target(test)

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    x_train.to_csv(X_TRAIN_FILE, index=False)
    x_test.to_csv(X_TEST_FILE, index=False)
    y_train.to_csv(Y_TRAIN_FILE, index=False)
    y_test.to_csv(Y_TEST_FILE, index=False)

    print(f"Guardado: {TRAIN_FILE} - {len(train)} filas")
    print(f"Guardado: {TEST_FILE} - {len(test)} filas")
    print(f"Guardado: {X_TRAIN_FILE} - {len(x_train)} filas")
    print(f"Guardado: {X_TEST_FILE} - {len(x_test)} filas")
    print(f"Guardado: {Y_TRAIN_FILE} - {len(y_train)} filas")
    print(f"Guardado: {Y_TEST_FILE} - {len(y_test)} filas")


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
