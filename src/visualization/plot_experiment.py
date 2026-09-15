import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_experiment(run_dir: Path) -> None:
    metrics = pd.read_csv(run_dir / "metrics/global_metrics.csv")
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    for metric in ["roc_auc", "balanced_accuracy", "f1_macro", "mcc"]:
        pivot = metrics[metrics["model_name"] != "dummy"].pivot(index="dataset_type", columns="model_name", values=metric)
        fig, ax = plt.subplots(figsize=(9, max(3, len(pivot) * 0.45 + 1)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="vlag", center=0 if metric == "mcc" else 0.5, ax=ax)
        ax.set_title(f"{metric}: validacion temporal externa, historico exploratorio")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Variables")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{metric}.png", dpi=160)
        plt.close(fig)
    from src.experiments.artifacts import refresh_manifest
    refresh_manifest(run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    plot_experiment(parser.parse_args().run_dir)
