from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluation import evaluate_predictions, save_metrics
from src.models.temporal_validation import purged_train, validate_panel
from src.experiments.artifacts import artifact_dir, create_run, finish_run, write_json


DATA_DIR = Path("data/processed/model")
FEATURE_CONFIG_FILE = DATA_DIR / "feature_sets.json"
MODEL_DIR = Path("models/experiments/legacy-trained")
PREDICTIONS_DIR = Path("reports/historical/predictions")
METRICS_DIR = Path("reports/historical/metrics")
METADATA_DIR = Path("reports/historical/model_metadata")

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
    if dataset_type not in {"base", "hybrid", "lagged_hybrid"}:
        raise ValueError("dataset_type debe ser 'base', 'hybrid' o 'lagged_hybrid'")

    feature_sets = load_feature_sets()
    if dataset_type == "base":
        feature_columns = feature_sets["financial_features"]
    elif dataset_type == "lagged_hybrid":
        feature_columns = feature_sets["lagged_hybrid_features"]
    else:
        feature_columns = feature_sets["hybrid_features"]

    train = pd.read_csv(DATA_DIR / f"{dataset_type}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{dataset_type}_test.csv")

    validate_panel(train)
    validate_panel(test)
    combined = pd.concat([train, test], ignore_index=True)
    validate_panel(combined)
    if pd.to_datetime(train["Date"]).max() >= pd.to_datetime(test["Date"]).min():
        raise ValueError("Entrenamiento y test no estan separados temporalmente")
    if not feature_sets.get("purged_datasets", {}).get(dataset_type, False):
        train = purged_train(combined, pd.to_datetime(test["Date"]).min())

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
    elif model_name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
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


def train_one_model(dataset_type: str, model_name: str, output_root: Path | None = None) -> TrainResult:
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

    if output_root is None:
        output_root = Path("reports/experiments") / datetime.now(timezone.utc).strftime("train-%Y%m%dT%H%M%S%fZ")
    model_file = artifact_dir(output_root, "models") / f"{dataset_type}_{model_name}.joblib"
    predictions_file = output_root / "predictions" / f"{dataset_type}_{model_name}_predictions.csv"
    metrics_file = output_root / "metrics" / f"{dataset_type}_{model_name}_metrics.csv"
    metadata_file = output_root / "metadata" / f"{dataset_type}_{model_name}.json"
    for path in [model_file, predictions_file, metrics_file, metadata_file]:
        if path.exists():
            raise FileExistsError(f"La salida ya existe: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument("--dataset", nargs="+", choices=["base", "hybrid", "lagged_hybrid"], required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "dummy",
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
        ],
        choices=[
            "dummy",
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "xgboost",
            "lightgbm",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.run_dir or Path("reports/experiments") / datetime.now(timezone.utc).strftime("train-%Y%m%dT%H%M%S%fZ")
    run_dir, manifest = create_run(PROJECT_ROOT, output, vars(args))
    manifest["evaluation_status"] = "Legacy processed dataset; exploratory holdout, not nested evaluation."
    inputs = [PROJECT_ROOT / FEATURE_CONFIG_FILE]
    try:
        for dataset_type in args.dataset:
            inputs.extend(PROJECT_ROOT / DATA_DIR / f"{dataset_type}_{split}.csv" for split in ["train", "test"])
            for model_name in args.models:
                result = train_one_model(dataset_type, model_name, run_dir)
                print(
                    f"Entrenado {result.dataset_type}/{result.model_name}: "
                    f"{result.model_file}, {result.predictions_file}, {result.metrics_file}"
                )
        finish_run(run_dir, manifest, inputs, PROJECT_ROOT)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
