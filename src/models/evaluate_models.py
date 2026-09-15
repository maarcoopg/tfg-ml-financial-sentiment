from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluation import evaluate_predictions, save_metrics
from src.experiments.artifacts import refresh_manifest


PREDICTIONS_DIR = Path("reports/historical/predictions")
METRICS_DIR = Path("reports/historical/metrics")
SUMMARY_FILE = METRICS_DIR / "all_model_metrics.csv"
GLOBAL_SUMMARY_FILE = METRICS_DIR / "global_model_metrics.csv"


def load_prediction_files(predictions_dir: Path = PREDICTIONS_DIR) -> list[Path]:
    prediction_files = sorted(predictions_dir.glob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No hay predicciones en {predictions_dir}")
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
    if "model_variant" in predictions:
        metrics.insert(0, "model_variant", predictions["model_variant"].iloc[0])

    return metrics


def build_metrics_summary(predictions_dir: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    metric_frames = [
        evaluate_prediction_file(prediction_file)
        for prediction_file in load_prediction_files(predictions_dir)
    ]
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics = metrics.sort_values(
        ["scope", "dataset_type", "model_name", "group"],
        na_position="first",
    ).reset_index(drop=True)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    predictions_dir = args.run_dir / "predictions" if args.run_dir else PREDICTIONS_DIR
    metrics = build_metrics_summary(predictions_dir)
    global_metrics = metrics[metrics["scope"] == "global"].copy()

    summary_file = args.run_dir / "metrics" / SUMMARY_FILE.name if args.run_dir else SUMMARY_FILE
    global_file = args.run_dir / "metrics" / GLOBAL_SUMMARY_FILE.name if args.run_dir else GLOBAL_SUMMARY_FILE
    save_metrics(metrics, summary_file)
    save_metrics(global_metrics, global_file)
    if args.run_dir:
        refresh_manifest(args.run_dir)

    print(f"Guardado: {summary_file} - {len(metrics)} filas")
    print(f"Guardado: {global_file} - {len(global_metrics)} filas")
    print(global_metrics[["dataset_type", "model_name", "accuracy", "f1", "roc_auc"]])


if __name__ == "__main__":
    main()
