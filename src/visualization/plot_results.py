from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS_FILE = Path("reports/metrics/global_model_metrics.csv")
COMPARISON_FILE = Path("reports/comparison/base_vs_hybrid_global.csv")
FEATURE_IMPORTANCE_FILE = Path("reports/feature_importance/top_10_feature_importance.csv")
OUTPUT_DIR = Path("reports/figures")


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 160


def save_current_figure(output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_file}")


def plot_global_metrics() -> None:
    metrics = pd.read_csv(METRICS_FILE)
    melted = metrics.melt(
        id_vars=["dataset_type", "model_name"],
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=melted,
        x="model_name",
        y="value",
        hue="dataset_type",
        errorbar=None,
    )
    ax.set_title("Métricas globales por modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Dataset")
    save_current_figure(OUTPUT_DIR / "global_metrics_by_model.png")

    grid = sns.catplot(
        data=melted,
        x="model_name",
        y="value",
        hue="dataset_type",
        col="metric",
        kind="bar",
        col_wrap=3,
        height=3.2,
        aspect=1.2,
        errorbar=None,
        sharey=True,
    )
    grid.set_axis_labels("Modelo", "Valor")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flatten():
        axis.tick_params(axis="x", rotation=35)
    grid.figure.suptitle("Comparación de métricas globales", y=1.03)
    grid.savefig(OUTPUT_DIR / "global_metrics_facets.png", bbox_inches="tight", dpi=160)
    plt.close(grid.figure)
    print(f"Guardado: {OUTPUT_DIR / 'global_metrics_facets.png'}")


def plot_base_hybrid_deltas() -> None:
    comparison = pd.read_csv(COMPARISON_FILE)
    delta_columns = ["accuracy_delta", "precision_delta", "recall_delta", "f1_delta", "roc_auc_delta"]
    melted = comparison.melt(
        id_vars=["model_name"],
        value_vars=delta_columns,
        var_name="metric",
        value_name="delta",
    )
    melted["metric"] = melted["metric"].str.replace("_delta", "", regex=False)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=melted,
        x="model_name",
        y="delta",
        hue="metric",
        errorbar=None,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Diferencia de rendimiento: híbrido menos base")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Delta")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Métrica")
    save_current_figure(OUTPUT_DIR / "base_vs_hybrid_metric_deltas.png")


def plot_feature_importance() -> None:
    importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    hybrid_importance = importance[importance["dataset_type"] == "hybrid"].copy()

    grid = sns.catplot(
        data=hybrid_importance,
        x="importance",
        y="feature",
        hue="feature_group",
        col="model_name",
        kind="bar",
        height=5,
        aspect=0.9,
        sharex=False,
        sharey=False,
        errorbar=None,
    )
    grid.set_axis_labels("Importancia", "Variable")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Top 10 variables en modelos híbridos", y=1.03)
    grid.savefig(OUTPUT_DIR / "hybrid_top_feature_importance.png", bbox_inches="tight", dpi=160)
    plt.close(grid.figure)
    print(f"Guardado: {OUTPUT_DIR / 'hybrid_top_feature_importance.png'}")


def main() -> None:
    set_style()
    plot_global_metrics()
    plot_base_hybrid_deltas()
    plot_feature_importance()


if __name__ == "__main__":
    main()
