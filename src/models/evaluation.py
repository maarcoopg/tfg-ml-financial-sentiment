from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp",
]


def evaluate_binary_classification(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series | None = None,
) -> dict[str, float | int | None]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, float | int | None] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if y_proba is not None and y_true.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def evaluate_predictions(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    required_columns = {"target", "prediction"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

    y_proba = predictions["probability"] if "probability" in predictions.columns else None
    rows = [
        {
            "scope": "global",
            **evaluate_binary_classification(
                predictions["target"],
                predictions["prediction"],
                y_proba,
            ),
        }
    ]

    if group_columns:
        for group_column in group_columns:
            if group_column not in predictions.columns:
                raise ValueError(f"No existe la columna de agrupación: {group_column}")
            for group_value, group_df in predictions.groupby(group_column):
                group_proba = (
                    group_df["probability"] if "probability" in group_df.columns else None
                )
                rows.append(
                    {
                        "scope": group_column,
                        "group": group_value,
                        **evaluate_binary_classification(
                            group_df["target"],
                            group_df["prediction"],
                            group_proba,
                        ),
                    }
                )

    return pd.DataFrame(rows)


def save_metrics(metrics: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_file, index=False)
