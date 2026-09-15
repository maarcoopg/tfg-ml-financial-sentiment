from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.artifacts import refresh_manifest

METRICS_FILE = Path("reports/historical/metrics/all_model_metrics.csv")
OUTPUT_DIR = Path("reports/historical/comparison")
GLOBAL_COMPARISON_FILE = OUTPUT_DIR / "base_vs_hybrid_global.csv"
TICKER_COMPARISON_FILE = OUTPUT_DIR / "base_vs_hybrid_by_ticker.csv"

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_metrics(metrics_file: Path = METRICS_FILE) -> pd.DataFrame:
    if not metrics_file.exists():
        raise FileNotFoundError(f"No existe {metrics_file}. Ejecuta src/models/evaluate_models.py")
    return pd.read_csv(metrics_file)


def compare_scope(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = metrics[metrics["scope"] == scope].copy()
    index_columns = ["model_name"]
    if "model_variant" in scoped:
        index_columns.append("model_variant")
    if scope != "global":
        index_columns.append("group")

    base = scoped[scoped["dataset_type"] == "base"][
        index_columns + METRIC_COLUMNS
    ].copy()
    hybrid = scoped[scoped["dataset_type"] == "hybrid"][
        index_columns + METRIC_COLUMNS
    ].copy()

    comparison = base.merge(
        hybrid,
        on=index_columns,
        suffixes=("_base", "_hybrid"),
        validate="one_to_one",
    )

    for metric in METRIC_COLUMNS:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_hybrid"] - comparison[f"{metric}_base"]
        )

    return comparison.sort_values(index_columns).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    metrics = load_metrics(args.run_dir / "metrics" / METRICS_FILE.name if args.run_dir else METRICS_FILE)
    output_dir = args.run_dir / "comparison" if args.run_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    global_comparison = compare_scope(metrics, "global")
    ticker_comparison = compare_scope(metrics, "ticker")

    global_file = output_dir / GLOBAL_COMPARISON_FILE.name
    ticker_file = output_dir / TICKER_COMPARISON_FILE.name
    global_comparison.to_csv(global_file, index=False)
    ticker_comparison.to_csv(ticker_file, index=False)
    if args.run_dir:
        refresh_manifest(args.run_dir)

    print(f"Guardado: {global_file} - {len(global_comparison)} filas")
    print(f"Guardado: {ticker_file} - {len(ticker_comparison)} filas")
    print(
        global_comparison[
            [
                "model_name",
                "accuracy_base",
                "accuracy_hybrid",
                "accuracy_delta",
                "f1_base",
                "f1_hybrid",
                "f1_delta",
                "roc_auc_base",
                "roc_auc_hybrid",
                "roc_auc_delta",
            ]
        ]
    )


if __name__ == "__main__":
    main()
