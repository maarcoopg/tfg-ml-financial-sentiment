"""Regenera figuras documentales desde artefactos existentes, sin entrenar."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIRST = ROOT / "reports/experiments/first-round-full-20260916"
ABLATION = ROOT / "reports/experiments/variable-ablation-full-20260917"
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
COLORS = ["#2864a0", "#b44848", "#23846e", "#84579b"]
inputs, outputs = {}, []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_csv(path, manifest, **kwargs):
    relative = path.relative_to(ROOT).as_posix()
    assert digest(path) == manifest["output_sha256"][relative], relative
    inputs[relative] = digest(path)
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white")
    plt.close(fig)
    outputs.append(name)


def main():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2})
    first = json.loads((FIRST / "manifest.json").read_text(encoding="utf-8"))
    ablation = json.loads((ABLATION / "manifest.json").read_text(encoding="utf-8"))
    assert first["status"] == ablation["status"] == "complete"
    for directory in [FIRST, ABLATION]:
        path = directory / "manifest.json"
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    data = checked_csv(ROOT / first["artifact_paths"]["data"] / "dataset.csv", first, parse_dates=["Date"])
    assert len(data) == 11056 and not data.duplicated(["ticker", "Date"]).any()
    for column, name, title, ylabel in [
        ("Adj Close", "precios.png", "Cierre ajustado normalizado", "Base 100; escala logarítmica"),
        ("volatility_20", "volatilidad.png", "Volatilidad móvil de 20 sesiones", "Desviación diaria (%)"),
        ("Volume", "volumen.png", "Volumen diario negociado", "Millones de acciones"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        fig.suptitle(title)
        for ticker, color, ax in zip(TICKERS, COLORS, axes.flat):
            part = data[data.ticker == ticker].sort_values("Date")
            values = part[column]
            if column == "Adj Close":
                values = values / values.iloc[0] * 100
                ax.set_yscale("log")
            elif column == "Volume":
                values = values / 1e6
            else:
                values = values * 100
            ax.plot(part.Date, values, color=color, linewidth=0.8)
            ax.set(title=ticker, ylabel=ylabel, xlabel="Año")
        save(fig, name)
    columns = ["daily_return", "RSI", "volatility_20", "sentiment_mean", "news_count", "has_news"]
    names = ["Retorno", "RSI", "Volatilidad", "Tono", "Noticias", "Presencia"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    for ticker, ax in zip(TICKERS, axes.flat):
        corr = data[data.ticker == ticker][columns].corr(method="spearman")
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(6), names, rotation=35, ha="right")
        ax.set_yticks(range(6), names)
        ax.set_title(ticker)
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr.iloc[i,j]) > 0.65 else "black")
    fig.colorbar(im, ax=axes, shrink=0.65, label="Correlación de Spearman")
    save(fig, "correlaciones.png")
    coverage = checked_csv(FIRST / "quality/monthly_coverage.csv", first)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", sharex=True)
    for ticker, color in zip(TICKERS, COLORS):
        for approach, style in [("original", "--"), ("deduplicated", "-")]:
            part = coverage[(coverage.ticker == ticker) & (coverage.approach == approach)].sort_values("month")
            assert len(part) > 0, approach
            label = ticker + (" original" if approach == "original" else " depurado")
            for ax, column in zip(axes, ["news_per_session", "recorded_day_ratio"]):
                ax.plot(pd.to_datetime(part.month), part[column], color=color, linestyle=style, linewidth=1, label=label)
    axes[0].set(ylabel="Noticias por sesión", title="Densidad y cobertura mensual por empresa")
    axes[1].set(ylabel="Fracción de sesiones con noticias", xlabel="Año", ylim=(-0.03, 1.03))
    axes[0].legend(fontsize=8, ncol=4)
    save(fig, "cobertura.png")
    predictions = checked_csv(ABLATION / "predictions/all_predictions.csv", ablation, parse_dates=["Date"])
    predictions = predictions[(predictions.model_name == "random_forest") & (predictions.dataset_type == "hybrid") &
                              predictions.approach.isin(["deduplicated", "add_news_abs_sentiment"])]
    metrics = checked_csv(ABLATION / "metrics/company_metrics.csv", ablation)
    labels = {"deduplicated": "Sin extras", "add_news_abs_sentiment": "+ intensidad absoluta"}
    diagnostic = []
    for kind in ["roc", "precision_recall"]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        for ticker, ax in zip(TICKERS, axes.flat):
            part = predictions[predictions.ticker == ticker]
            for approach, color in zip(labels, ["#2864a0", "#b44848"]):
                group = part[part.approach == approach].sort_values("Date")
                assert len(group) == 553 and not group.Date.duplicated().any()
                if kind == "roc":
                    x, y, _ = roc_curve(group.target, group.probability)
                    score = roc_auc_score(group.target, group.probability)
                    saved = metrics[(metrics.ticker == ticker) & (metrics.approach == approach) &
                                    (metrics.model_name == "random_forest") & (metrics.dataset_type == "hybrid")]
                    np.testing.assert_allclose(score, saved.roc_auc.iloc[0], atol=1e-14, rtol=0)
                    score_name = "AUC"
                else:
                    y, x, _ = precision_recall_curve(group.target, group.probability)
                    score = average_precision_score(group.target, group.probability)
                    score_name = "AP"
                ax.plot(x, y, color=color, linewidth=1.2, label=f"{labels[approach]} ({score_name} {score:.3f})")
                diagnostic.append(dict(ticker=ticker, approach=approach, metric=score_name, value=score))
            if kind == "roc":
                ax.plot([0, 1], [0, 1], "--", color="gray", label="Diagonal")
                ax.set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
            else:
                ax.axhline(part.target.mean(), linestyle="--", color="gray", label="Prevalencia")
                ax.set(xlabel="Sensibilidad", ylabel="Precisión")
            ax.set(title=ticker, xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(fontsize=8, loc="lower right" if kind == "roc" else "lower left")
        save(fig, kind + ".png")
    fig, axes = plt.subplots(4, 2, figsize=(8, 12), layout="constrained")
    for i, ticker in enumerate(TICKERS):
        for j, approach in enumerate(labels):
            group = predictions[(predictions.ticker == ticker) & (predictions.approach == approach)]
            matrix = confusion_matrix(group.target, group.prediction, labels=[0, 1])
            assert matrix.sum() == 553
            ax = axes[i, j]
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=553)
            ax.set(title=f"{ticker}: {labels[approach]}", xlabel="Clase predicha", ylabel="Clase real")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.grid(False)
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                            color="white" if matrix[row, col] > 280 else "black", fontsize=12)
    save(fig, "confusion.png")
    tuning = checked_csv(FIRST / "tuning/inner_cv.csv", first)
    tuning = tuning[(tuning.approach == "tuned") & (tuning.dataset_type == "hybrid")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 9), layout="constrained")
    for ax, model, label in zip(axes, ["logistic_regression", "random_forest", "hist_gradient_boosting"],
                                ["Regresión logística", "Bosque aleatorio", "Potenciación por histogramas"]):
        part = tuning[tuning.model_name == model]
        assert part.groupby(["outer_fold", "config_id"]).size().eq(3).all()
        table = part.groupby(["config_id", "outer_fold"]).selection_score.mean().unstack()
        assert len(table) == (6 if model == "logistic_regression" else 20)
        im = ax.imshow(table, cmap="viridis", aspect="auto", vmin=0.45, vmax=0.60)
        ax.set_xticks(range(3), [f"Bloque {b}" for b in table.columns])
        ax.set_yticks(range(len(table)), table.index)
        ax.set(title=label, ylabel="Identificador de candidato")
        ax.grid(False)
        for row in range(len(table)):
            for col in range(3):
                ax.text(col, row, f"{table.iloc[row,col]:.3f}", ha="center", va="center", fontsize=8,
                        color="black" if table.iloc[row,col] > 0.55 else "white")
    fig.colorbar(im, ax=axes, shrink=0.5, label="AUC macro interno medio")
    save(fig, "candidatos.png")
    inputs[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__))
    provenance = {"purpose": "Figuras documentales; sin entrenamiento ni descargas", "input_sha256": inputs,
                  "output_sha256": {name: digest(OUT / name) for name in outputs}, "panel_rows": len(data),
                  "diagnostic_model": "random_forest/hybrid; selección descriptiva tras observar la ablación",
                  "diagnostic_scores": diagnostic, "new_learning_curves": False}
    (OUT / "procedencia.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(outputs)} figures with verified inputs and recorded provenance.")


if __name__ == "__main__":
    main()
