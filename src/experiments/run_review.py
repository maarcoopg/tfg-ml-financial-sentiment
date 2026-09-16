from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from src.data.audit_news_coverage import audit_coverage
from src.data.point_in_time import build_dataset
from src.experiments.ablations import feature_variants
from src.experiments.artifacts import artifact_dir, create_run, finish_run, sha256, write_json
from src.models.evaluation import evaluate_predictions
from src.models.temporal_validation import expanding_date_splits, paired_auc_interval, purged_train
from src.models.train_models import build_model


ROOT = Path(__file__).resolve().parents[2]
MODELS = ["logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm"]
GRIDS = {
    "logistic_regression": [{"model__C": c} for c in [0.1, 10.0]],
    "random_forest": [{"model__n_estimators": 100, "model__max_depth": d,
                       "model__min_samples_leaf": 20} for d in [4, 8]],
    "hist_gradient_boosting": [{"model__max_iter": 100, "model__max_leaf_nodes": n,
                                "model__early_stopping": False} for n in [7, 15]],
    "xgboost": [{"model__n_estimators": 100, "model__max_depth": d} for d in [2, 3]],
    "lightgbm": [{"model__n_estimators": 100, "model__num_leaves": n,
                  "model__subsample_freq": 1} for n in [7, 15]],
}


def choose_params(train: pd.DataFrame, columns: list[str], model_name: str, inner_splits: int):
    rows = []
    splits = expanding_date_splits(train, inner_splits)
    for config_id, params in enumerate(GRIDS[model_name]):
        scores = []
        for fold, (_, valid_dates) in enumerate(splits, start=1):
            fit = purged_train(train, valid_dates.min())
            valid = train[train["Date"].isin(valid_dates)]
            estimator = build_model(model_name).set_params(**params)
            estimator.fit(fit[columns], fit["target"])
            score = roc_auc_score(valid["target"], estimator.predict_proba(valid[columns])[:, 1])
            scores.append(score)
            rows.append({"config_id": config_id, "inner_fold": fold, "roc_auc": score,
                         "train_end": fit["Date"].max(), "train_target_end": fit["target_end"].max(),
                         "valid_start": valid["Date"].min(), "valid_end": valid["Date"].max()})
        rows[-1]["mean_auc"] = float(np.mean(scores))
    means = pd.DataFrame(rows).groupby("config_id")["roc_auc"].mean()
    best = int(means.idxmax())
    return GRIDS[model_name][best], rows, float(means.loc[best])


def nested_predictions(dataset: pd.DataFrame, variants: dict, models: list[str], output: Path,
                       outer_start: str, outer_splits: int, inner_splits: int,
                       model_dir: Path | None = None):
    dates = sorted(dataset.loc[dataset["Date"] >= pd.Timestamp(outer_start), "Date"].unique())
    if outer_splits < 2 or len(dates) < outer_splits * 2:
        raise ValueError("Fechas insuficientes para los folds externos")
    predictions, search_rows, selection_rows = [], [], []
    model_dir = model_dir if model_dir is not None else artifact_dir(output, "models")
    model_dir.mkdir(parents=True)
    for kind in ["predictions", "tuning"]:
        (output / kind).mkdir(parents=True, exist_ok=True)
    for outer_fold, test_dates in enumerate(np.array_split(dates, outer_splits), start=1):
        test = dataset[dataset["Date"].isin(test_dates)]
        train = purged_train(dataset, pd.Timestamp(test_dates[0]))
        if train.empty or train["target_end"].max() >= test["Date"].min():
            raise ValueError("Frontera externa no purgada")
        combinations = [("base", "dummy")] + [(v, m) for v in variants for m in models]
        for variant, model_name in combinations:
            columns = variants[variant]
            if model_name == "dummy":
                params, inner_rows, mean_auc = {}, [], 0.5
            else:
                params, inner_rows, mean_auc = choose_params(train, columns, model_name, inner_splits)
            estimator = build_model(model_name).set_params(**params)
            estimator.fit(train[columns], train["target"])
            frame = test[["ticker", "Date", "target"]].copy()
            frame["dataset_type"], frame["model_name"], frame["outer_fold"] = variant, model_name, outer_fold
            frame["probability"] = estimator.predict_proba(test[columns])[:, 1]
            frame["prediction"] = (frame["probability"] >= 0.5).astype(int)
            predictions.append(frame)
            for row in inner_rows:
                search_rows.append({"dataset_type": variant, "model_name": model_name, "outer_fold": outer_fold, **row})
            selection_rows.append({"dataset_type": variant, "model_name": model_name, "outer_fold": outer_fold,
                                   "params": json.dumps(params, sort_keys=True), "inner_mean_auc": mean_auc,
                                   "train_rows": len(train), "test_rows": len(test),
                                   "train_end": train["Date"].max(), "train_target_end": train["target_end"].max(),
                                   "test_start": test["Date"].min(), "test_end": test["Date"].max()})
            joblib.dump(estimator, model_dir / f"fold_{outer_fold}_{variant}_{model_name}.joblib")
            print(f"Fold {outer_fold}: {variant}/{model_name}, AUC CV interna={mean_auc:.4f}", flush=True)
        pd.concat(predictions, ignore_index=True).to_csv(output / "predictions/predictions.csv", index=False)
        pd.DataFrame(search_rows).to_csv(output / "tuning/inner_cv.csv", index=False)
        pd.DataFrame(selection_rows).to_csv(output / "tuning/selected_params.csv", index=False)
    return pd.concat(predictions, ignore_index=True)


def summarize(predictions: pd.DataFrame, output: Path, repeats: int):
    metrics, intervals = [], []
    groups = dict(tuple(predictions.groupby(["dataset_type", "model_name"])))
    for (variant, model_name), frame in groups.items():
        frame = frame.assign(year=pd.to_datetime(frame["Date"]).dt.year)
        table = evaluate_predictions(frame, ["ticker", "year", "outer_fold"])
        fold_auc = table.loc[table["scope"] == "outer_fold", "roc_auc"]
        table.loc[table["scope"] == "global", "mean_outer_auc"] = fold_auc.mean()
        table.loc[table["scope"] == "global", "std_outer_auc"] = fold_auc.std()
        table.insert(0, "model_name", model_name)
        table.insert(0, "dataset_type", variant)
        metrics.append(table)
        if model_name == "dummy":
            continue
        reference = ("base", model_name) if variant in ["news_volume", "hybrid", "financial_sentiment"] else ("hybrid", model_name)
        if variant in ["base", "sentiment_only"]:
            reference = ("base", "dummy")
        if reference not in groups:
            continue
        blocks = [5, 20, 60] if (variant, model_name) == ("lag_1", "lightgbm") else [20]
        for block in blocks:
            if frame["Date"].nunique() <= block:
                continue
            interval = paired_auc_interval(frame, groups[reference], block=block, repeats=repeats)
            intervals.append({"dataset_type": variant, "model_name": model_name,
                              "reference_dataset": reference[0], "reference_model": reference[1], **interval})
    metrics = pd.concat(metrics, ignore_index=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics/metrics.csv", index=False)
    metrics[metrics["scope"] == "global"].to_csv(output / "metrics/global_metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "metrics/auc_intervals.csv", index=False)


def run(args):
    config = {**vars(args), "parameter_grids": GRIDS, "threshold": 0.5,
              "prediction_time": "session close; publication time used as availability proxy",
              "primary_comparison": "lag_1/lightgbm minus hybrid/lightgbm",
              "uncertainty": "paired moving blocks; percentile 95%; no multiple-testing correction"}
    output, manifest = create_run(ROOT, args.run_dir, config)
    try:
        dataset, news, inputs = build_dataset(ROOT, artifact_dir(output, "data"), args.source_timezone)
        input_hashes = {str(p): sha256(p) for p in inputs}
        manifest["input_sha256"] = input_hashes
        manifest["coverage"] = audit_coverage(dataset, news, output / "coverage")
        dataset, variants = feature_variants(dataset)
        if args.variants:
            unknown = set(args.variants) - set(variants)
            if unknown:
                raise ValueError(f"Variantes desconocidas: {sorted(unknown)}")
            variants = {v: variants[v] for v in args.variants}
        if "base" not in variants:
            raise ValueError("Debe incluirse base para la referencia trivial")
        manifest["feature_sets"] = variants
        write_json(output / "manifest.json", manifest)
        with threadpool_limits(limits=1):
            predictions = nested_predictions(dataset, variants, args.models, output,
                                             args.outer_start, args.outer_splits, args.inner_splits)
        summarize(predictions, output, args.bootstrap_repeats)
        from src.visualization.plot_experiment import plot_experiment
        plot_experiment(output)
        if any(sha256(p) != input_hashes[str(p)] for p in inputs):
            raise RuntimeError("Los datos de entrada cambiaron durante la ejecucion")
        finish_run(output, manifest, inputs, ROOT)
        print(f"Ejecucion completada: {output}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(output / "manifest.json", manifest)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecuta la revision temporal con resultados aislados.")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--outer-start", default="2023-10-16")
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--source-timezone", default="UTC")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
