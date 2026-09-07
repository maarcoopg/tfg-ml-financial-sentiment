from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluation import evaluate_predictions, save_metrics


PREDICTIONS_DIR = Path("reports/predictions")
METRICS_DIR = Path("reports/metrics")
SUMMARY_FILE = METRICS_DIR / "all_model_metrics.csv"
GLOBAL_SUMMARY_FILE = METRICS_DIR / "global_model_metrics.csv"


def load_prediction_files() -> list[Path]:
    prediction_files = sorted(PREDICTIONS_DIR.glob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No hay predicciones en {PREDICTIONS_DIR}")
    return prediction_files


def evaluate_prediction_file(prediction_file: Path) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_file)
    required_columns = {"dataset_type", "model_name", "ticker", "target", "prediction"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"{prediction_file} no contiene columnas requeridas: {sorted(missing_columns)}"
        )

    metrics = evaluate_predictions(predictions, group_columns=["ticker"])
    metrics.insert(0, "model_name", predictions["model_name"].iloc[0])
    metrics.insert(0, "dataset_type", predictions["dataset_type"].iloc[0])

    return metrics


def build_metrics_summary() -> pd.DataFrame:
    metric_frames = [
        evaluate_prediction_file(prediction_file)
        for prediction_file in load_prediction_files()
    ]
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics = metrics.sort_values(
        ["scope", "dataset_type", "model_name", "group"],
        na_position="first",
    ).reset_index(drop=True)

    return metrics


def main() -> None:
    metrics = build_metrics_summary()
    global_metrics = metrics[metrics["scope"] == "global"].copy()

    save_metrics(metrics, SUMMARY_FILE)
    save_metrics(global_metrics, GLOBAL_SUMMARY_FILE)

    print(f"Guardado: {SUMMARY_FILE} - {len(metrics)} filas")
    print(f"Guardado: {GLOBAL_SUMMARY_FILE} - {len(global_metrics)} filas")
    print(global_metrics[["dataset_type", "model_name", "accuracy", "f1", "roc_auc"]])


if __name__ == "__main__":
    main()
