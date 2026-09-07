from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from itertools import product
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluation import evaluate_binary_classification, evaluate_predictions, save_metrics
from src.models.train_models import build_model, load_dataset


ID_COLUMNS = ["ticker", "Date"]
TARGET_COLUMN = "target"
N_SPLITS = 3
THRESHOLDS = np.round(np.arange(0.40, 0.61, 0.025), 3)

TUNED_MODEL_DIR = Path("models/tuned")
TUNED_PREDICTIONS_DIR = Path("reports/tuned_predictions")
TUNED_METRICS_DIR = Path("reports/tuned_metrics")
TUNING_DIR = Path("reports/tuning")

CV_RESULTS_FILE = TUNING_DIR / "temporal_cv_results.csv"
FOLD_RESULTS_FILE = TUNING_DIR / "temporal_cv_fold_results.csv"
BEST_PARAMS_FILE = TUNING_DIR / "best_temporal_cv_params.csv"
TUNED_GLOBAL_METRICS_FILE = TUNING_DIR / "tuned_global_metrics.csv"
TUNED_VS_INITIAL_FILE = TUNING_DIR / "tuned_vs_initial_global.csv"
TUNED_BASE_VS_HYBRID_FILE = TUNING_DIR / "tuned_base_vs_hybrid_global.csv"

INITIAL_GLOBAL_METRICS_FILE = Path("reports/metrics/global_model_metrics.csv")


PARAM_GRIDS = {
    "logistic_regression": [
        {"model__C": c, "model__class_weight": class_weight}
        for c, class_weight in product([0.01, 0.1, 1.0, 10.0, 100.0], [None, "balanced"])
    ],
    "random_forest": [
        {
            "model__n_estimators": 150,
            "model__max_depth": max_depth,
            "model__min_samples_leaf": min_samples_leaf,
            "model__max_features": max_features,
            "model__class_weight": class_weight,
        }
        for max_depth, min_samples_leaf, max_features, class_weight in product(
            [4, 8],
            [10, 30],
            ["sqrt", 0.7],
            [None, "balanced_subsample"],
        )
    ],
    "hist_gradient_boosting": [
        {
            "model__max_iter": max_iter,
            "model__learning_rate": learning_rate,
            "model__max_leaf_nodes": max_leaf_nodes,
            "model__l2_regularization": l2_regularization,
        }
        for max_iter, learning_rate, max_leaf_nodes, l2_regularization in product(
            [100, 200],
            [0.02, 0.05],
            [7, 15],
            [0.0, 0.1],
        )
    ],
    "xgboost": [
        {
            "model__n_estimators": n_estimators,
            "model__max_depth": max_depth,
            "model__learning_rate": learning_rate,
            "model__subsample": subsample,
            "model__colsample_bytree": colsample_bytree,
            "model__min_child_weight": min_child_weight,
        }
        for n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight
        in product(
            [100, 250],
            [2, 3],
            [0.03, 0.07],
            [0.8],
            [0.8],
            [1, 5],
        )
    ],
    "lightgbm": [
        {
            "model__n_estimators": n_estimators,
            "model__num_leaves": num_leaves,
            "model__learning_rate": learning_rate,
            "model__subsample": subsample,
            "model__colsample_bytree": colsample_bytree,
            "model__min_child_samples": min_child_samples,
        }
        for n_estimators, num_leaves, learning_rate, subsample, colsample_bytree, min_child_samples
        in product(
            [100, 250],
            [7, 15],
            [0.03, 0.07],
            [0.8],
            [0.8],
            [20, 50],
        )
    ],
}


def temporal_date_splits(
    train: pd.DataFrame,
    n_splits: int = N_SPLITS,
) -> list[tuple[pd.Series, pd.Series]]:
    dates = pd.Series(pd.to_datetime(train["Date"]).drop_duplicates().sort_values().to_list())
    fold_size = len(dates) // (n_splits + 1)
    if fold_size == 0:
        raise ValueError("No hay fechas suficientes para validación temporal")

    folds = []
    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        valid_start = train_end
        valid_end = fold_size * (fold + 2) if fold < n_splits - 1 else len(dates)
        train_dates = dates.iloc[:train_end]
        valid_dates = dates.iloc[valid_start:valid_end]
        folds.append((train_dates, valid_dates))

    return folds


def choose_threshold(y_true: pd.Series, probabilities: pd.Series) -> tuple[float, float]:
    candidates = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        candidates.append((score, -abs(threshold - 0.5), threshold))

    best_score, _, best_threshold = max(candidates)
    return float(best_threshold), float(best_score)


def metrics_with_prefix(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    prefix: str,
) -> dict[str, float | int | None]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = evaluate_binary_classification(y_true, predictions, probabilities)
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def params_to_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def evaluate_params(
    dataset_type: str,
    model_name: str,
    params: dict,
    train: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, list[dict]]:
    validation_predictions = []
    fold_rows = []

    for fold_number, (train_dates, valid_dates) in enumerate(temporal_date_splits(train), start=1):
        fold_train = train[pd.to_datetime(train["Date"]).isin(train_dates)]
        fold_valid = train[pd.to_datetime(train["Date"]).isin(valid_dates)]

        model = build_model(model_name)
        model.set_params(**params)
        model.fit(fold_train[feature_columns], fold_train[TARGET_COLUMN])

        probabilities = model.predict_proba(fold_valid[feature_columns])[:, 1]
        fold_prediction_frame = fold_valid[ID_COLUMNS + [TARGET_COLUMN]].copy()
        fold_prediction_frame["probability"] = probabilities
        fold_prediction_frame["fold"] = fold_number
        validation_predictions.append(fold_prediction_frame)

        fold_threshold, fold_f1 = choose_threshold(
            fold_prediction_frame[TARGET_COLUMN],
            fold_prediction_frame["probability"],
        )
        fold_metrics_optimized = metrics_with_prefix(
            fold_prediction_frame[TARGET_COLUMN],
            fold_prediction_frame["probability"],
            fold_threshold,
            "optimized",
        )
        fold_metrics_fixed = metrics_with_prefix(
            fold_prediction_frame[TARGET_COLUMN],
            fold_prediction_frame["probability"],
            0.5,
            "fixed_0_5",
        )
        fold_rows.append(
            {
                "dataset_type": dataset_type,
                "model_name": model_name,
                "params": params_to_json(params),
                "fold": fold_number,
                "train_start": train_dates.min().date().isoformat(),
                "train_end": train_dates.max().date().isoformat(),
                "valid_start": valid_dates.min().date().isoformat(),
                "valid_end": valid_dates.max().date().isoformat(),
                "threshold": fold_threshold,
                "threshold_f1": fold_f1,
                **fold_metrics_optimized,
                **fold_metrics_fixed,
            }
        )

    validation = pd.concat(validation_predictions, ignore_index=True)
    threshold, threshold_f1 = choose_threshold(
        validation[TARGET_COLUMN],
        validation["probability"],
    )
    optimized_metrics = evaluate_binary_classification(
        validation[TARGET_COLUMN],
        (validation["probability"] >= threshold).astype(int),
        validation["probability"],
    )
    fixed_metrics = evaluate_binary_classification(
        validation[TARGET_COLUMN],
        (validation["probability"] >= 0.5).astype(int),
        validation["probability"],
    )

    summary = {
        "dataset_type": dataset_type,
        "model_name": model_name,
        "params": params_to_json(params),
        "threshold": threshold,
        "threshold_f1": threshold_f1,
        "accuracy": optimized_metrics["accuracy"],
        "precision": optimized_metrics["precision"],
        "recall": optimized_metrics["recall"],
        "f1": optimized_metrics["f1"],
        "roc_auc": optimized_metrics["roc_auc"],
        "fixed_0_5_accuracy": fixed_metrics["accuracy"],
        "fixed_0_5_precision": fixed_metrics["precision"],
        "fixed_0_5_recall": fixed_metrics["recall"],
        "fixed_0_5_f1": fixed_metrics["f1"],
        "fixed_0_5_roc_auc": fixed_metrics["roc_auc"],
    }

    return summary, fold_rows


def select_best_config(results: pd.DataFrame, dataset_type: str, model_name: str) -> pd.Series:
    candidates = results[
        (results["dataset_type"] == dataset_type) & (results["model_name"] == model_name)
    ].copy()
    candidates = candidates.sort_values(
        ["roc_auc", "fixed_0_5_f1", "fixed_0_5_accuracy"],
        ascending=[False, False, False],
    )
    return candidates.iloc[0]


def train_final_tuned_model(
    dataset_type: str,
    model_name: str,
    best_config: pd.Series,
) -> pd.DataFrame:
    train, test, feature_columns = load_dataset(dataset_type)
    params = json.loads(best_config["params"])
    threshold = float(best_config["threshold"])

    model = build_model(model_name)
    model.set_params(**params)
    model.fit(train[feature_columns], train[TARGET_COLUMN])

    probabilities = model.predict_proba(test[feature_columns])[:, 1]
    TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, TUNED_MODEL_DIR / f"{dataset_type}_{model_name}.joblib")

    metric_frames = []
    for variant, variant_threshold in [
        ("tuned_fixed_0_5", 0.5),
        ("tuned_optimized_threshold", threshold),
    ]:
        predictions = test[ID_COLUMNS + [TARGET_COLUMN]].copy()
        predictions["dataset_type"] = dataset_type
        predictions["model_name"] = model_name
        predictions["model_variant"] = variant
        predictions["threshold"] = variant_threshold
        predictions["probability"] = probabilities
        predictions["prediction"] = (
            predictions["probability"] >= variant_threshold
        ).astype(int)

        metrics = evaluate_predictions(predictions, group_columns=["ticker"])
        metrics.insert(0, "model_variant", variant)
        metrics.insert(0, "model_name", model_name)
        metrics.insert(0, "dataset_type", dataset_type)
        metric_frames.append(metrics)

        predictions.to_csv(
            TUNED_PREDICTIONS_DIR / f"{dataset_type}_{model_name}_{variant}_predictions.csv",
            index=False,
        )
        save_metrics(
            metrics,
            TUNED_METRICS_DIR / f"{dataset_type}_{model_name}_{variant}_metrics.csv",
        )

    return pd.concat(metric_frames, ignore_index=True)


def compare_with_initial(tuned_global_metrics: pd.DataFrame) -> pd.DataFrame:
    initial = pd.read_csv(INITIAL_GLOBAL_METRICS_FILE)
    initial = initial[initial["model_name"].isin(PARAM_GRIDS.keys())]
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    comparison = initial.merge(
        tuned_global_metrics,
        on=["dataset_type", "model_name"],
        suffixes=("_initial", "_tuned"),
        validate="one_to_many",
    )
    for metric in metric_columns:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_tuned"] - comparison[f"{metric}_initial"]
        )

    return comparison


def compare_tuned_base_hybrid(tuned_global_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    index_columns = ["model_name", "model_variant"]
    base = tuned_global_metrics[tuned_global_metrics["dataset_type"] == "base"][
        index_columns + metric_columns
    ].copy()
    hybrid = tuned_global_metrics[tuned_global_metrics["dataset_type"] == "hybrid"][
        index_columns + metric_columns
    ].copy()

    comparison = base.merge(
        hybrid,
        on=index_columns,
        suffixes=("_base", "_hybrid"),
        validate="one_to_one",
    )
    for metric in metric_columns:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_hybrid"] - comparison[f"{metric}_base"]
        )

    return comparison


def run_tuning(datasets: list[str], models: list[str]) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    cv_rows = []
    fold_rows = []
    for dataset_type in datasets:
        train, _, feature_columns = load_dataset(dataset_type)
        for model_name in models:
            for params in PARAM_GRIDS[model_name]:
                summary, folds = evaluate_params(
                    dataset_type,
                    model_name,
                    params,
                    train,
                    feature_columns,
                )
                cv_rows.append(summary)
                fold_rows.extend(folds)
            current_results = pd.DataFrame(cv_rows)
            best_so_far = select_best_config(current_results, dataset_type, model_name)
            print(
                f"Mejor CV {dataset_type}/{model_name}: "
                f"roc_auc={best_so_far['roc_auc']:.4f} "
                f"f1_0.5={best_so_far['fixed_0_5_f1']:.4f} "
                f"threshold={best_so_far['threshold']:.3f}"
            )

    cv_results = pd.DataFrame(cv_rows)
    fold_results = pd.DataFrame(fold_rows)
    best_rows = [
        select_best_config(cv_results, dataset_type, model_name)
        for dataset_type in datasets
        for model_name in models
    ]
    best_params = pd.DataFrame(best_rows).reset_index(drop=True)

    tuned_metrics = [
        train_final_tuned_model(row["dataset_type"], row["model_name"], row)
        for _, row in best_params.iterrows()
    ]
    tuned_metrics = pd.concat(tuned_metrics, ignore_index=True)
    tuned_global_metrics = tuned_metrics[tuned_metrics["scope"] == "global"].copy()
    tuned_vs_initial = compare_with_initial(tuned_global_metrics)
    tuned_base_vs_hybrid = compare_tuned_base_hybrid(tuned_global_metrics)

    cv_results.to_csv(CV_RESULTS_FILE, index=False)
    fold_results.to_csv(FOLD_RESULTS_FILE, index=False)
    best_params.to_csv(BEST_PARAMS_FILE, index=False)
    tuned_global_metrics.to_csv(TUNED_GLOBAL_METRICS_FILE, index=False)
    tuned_vs_initial.to_csv(TUNED_VS_INITIAL_FILE, index=False)
    tuned_base_vs_hybrid.to_csv(TUNED_BASE_VS_HYBRID_FILE, index=False)

    print(f"Guardado: {CV_RESULTS_FILE} - {len(cv_results)} filas")
    print(f"Guardado: {FOLD_RESULTS_FILE} - {len(fold_results)} filas")
    print(f"Guardado: {BEST_PARAMS_FILE} - {len(best_params)} filas")
    print(f"Guardado: {TUNED_GLOBAL_METRICS_FILE} - {len(tuned_global_metrics)} filas")
    print(f"Guardado: {TUNED_VS_INITIAL_FILE} - {len(tuned_vs_initial)} filas")
    print(f"Guardado: {TUNED_BASE_VS_HYBRID_FILE} - {len(tuned_base_vs_hybrid)} filas")
    print(
        tuned_vs_initial[
            [
                "dataset_type",
                "model_name",
                "model_variant",
                "f1_initial",
                "f1_tuned",
                "f1_delta",
                "roc_auc_initial",
                "roc_auc_tuned",
                "roc_auc_delta",
            ]
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajusta hiperparámetros con validación cruzada temporal."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["base", "hybrid"],
        choices=["base", "hybrid"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "xgboost",
            "lightgbm",
        ],
        choices=list(PARAM_GRIDS.keys()),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tuning(args.datasets, args.models)


if __name__ == "__main__":
    main()
