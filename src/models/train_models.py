from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluation import evaluate_predictions, save_metrics


DATA_DIR = Path("data/processed/model")
FEATURE_CONFIG_FILE = DATA_DIR / "feature_sets.json"
MODEL_DIR = Path("models/trained")
PREDICTIONS_DIR = Path("reports/predictions")
METRICS_DIR = Path("reports/metrics")
METADATA_DIR = Path("reports/model_metadata")

ID_COLUMNS = ["ticker", "Date"]
TARGET_COLUMN = "target"
RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainResult:
    dataset_type: str
    model_name: str
    model_file: Path
    predictions_file: Path
    metrics_file: Path
    metadata_file: Path


def load_feature_sets() -> dict[str, list[str]]:
    with FEATURE_CONFIG_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def load_dataset(dataset_type: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if dataset_type not in {"base", "hybrid"}:
        raise ValueError("dataset_type debe ser 'base' o 'hybrid'")

    feature_sets = load_feature_sets()
    feature_key = f"{dataset_type}_features"
    if dataset_type == "base":
        feature_columns = feature_sets["financial_features"]
    else:
        feature_columns = feature_sets[feature_key]

    train = pd.read_csv(DATA_DIR / f"{dataset_type}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{dataset_type}_test.csv")

    return train, test, feature_columns


def build_model(model_name: str) -> Pipeline:
    if model_name == "dummy":
        estimator = DummyClassifier(strategy="most_frequent")
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    elif model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return Pipeline(steps)


def create_predictions(
    model: Pipeline,
    test: pd.DataFrame,
    feature_columns: list[str],
    dataset_type: str,
    model_name: str,
) -> pd.DataFrame:
    x_test = test[feature_columns]
    predictions = test[ID_COLUMNS + [TARGET_COLUMN]].copy()
    predictions["dataset_type"] = dataset_type
    predictions["model_name"] = model_name
    predictions["prediction"] = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        predictions["probability"] = model.predict_proba(x_test)[:, 1]

    return predictions


def train_one_model(dataset_type: str, model_name: str) -> TrainResult:
    train, test, feature_columns = load_dataset(dataset_type)
    x_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]

    model = build_model(model_name)
    model.fit(x_train, y_train)

    predictions = create_predictions(
        model,
        test,
        feature_columns,
        dataset_type,
        model_name,
    )
    metrics = evaluate_predictions(predictions, group_columns=["ticker"])
    metrics.insert(0, "model_name", model_name)
    metrics.insert(0, "dataset_type", dataset_type)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODEL_DIR / f"{dataset_type}_{model_name}.joblib"
    predictions_file = PREDICTIONS_DIR / f"{dataset_type}_{model_name}_predictions.csv"
    metrics_file = METRICS_DIR / f"{dataset_type}_{model_name}_metrics.csv"
    metadata_file = METADATA_DIR / f"{dataset_type}_{model_name}.json"

    joblib.dump(model, model_file)
    predictions.to_csv(predictions_file, index=False)
    save_metrics(metrics, metrics_file)

    metadata = {
        "dataset_type": dataset_type,
        "model_name": model_name,
        "feature_columns": feature_columns,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": train["Date"].min(),
        "train_end": train["Date"].max(),
        "test_start": test["Date"].min(),
        "test_end": test["Date"].max(),
        "random_state": RANDOM_STATE,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainResult(
        dataset_type=dataset_type,
        model_name=model_name,
        model_file=model_file,
        predictions_file=predictions_file,
        metrics_file=metrics_file,
        metadata_file=metadata_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena modelos financieros e híbridos.")
    parser.add_argument("--dataset", choices=["base", "hybrid"], required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy", "logistic_regression", "random_forest", "hist_gradient_boosting"],
        choices=["dummy", "logistic_regression", "random_forest", "hist_gradient_boosting"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model_name in args.models:
        result = train_one_model(args.dataset, model_name)
        print(
            f"Entrenado {result.dataset_type}/{result.model_name}: "
            f"{result.model_file}, {result.predictions_file}, {result.metrics_file}"
        )


if __name__ == "__main__":
    main()
