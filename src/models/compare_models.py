from __future__ import annotations

from pathlib import Path

import pandas as pd


METRICS_FILE = Path("reports/metrics/all_model_metrics.csv")
OUTPUT_DIR = Path("reports/comparison")
GLOBAL_COMPARISON_FILE = OUTPUT_DIR / "base_vs_hybrid_global.csv"
TICKER_COMPARISON_FILE = OUTPUT_DIR / "base_vs_hybrid_by_ticker.csv"

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_metrics() -> pd.DataFrame:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"No existe {METRICS_FILE}. Ejecuta src/models/evaluate_models.py")
    return pd.read_csv(METRICS_FILE)


def compare_scope(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = metrics[metrics["scope"] == scope].copy()
    index_columns = ["model_name"]
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
    metrics = load_metrics()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_comparison = compare_scope(metrics, "global")
    ticker_comparison = compare_scope(metrics, "ticker")

    global_comparison.to_csv(GLOBAL_COMPARISON_FILE, index=False)
    ticker_comparison.to_csv(TICKER_COMPARISON_FILE, index=False)

    print(f"Guardado: {GLOBAL_COMPARISON_FILE} - {len(global_comparison)} filas")
    print(f"Guardado: {TICKER_COMPARISON_FILE} - {len(ticker_comparison)} filas")
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
