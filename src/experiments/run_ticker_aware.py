from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.audit_news_coverage import audit_coverage
from src.data.point_in_time import build_dataset
from src.data.split_train_test import MODELING_TICKERS, SENTIMENT_FEATURES
from src.experiments.ablations import feature_variants
from src.experiments.artifacts import artifact_dir, create_run, finish_run, sha256, write_json
from src.experiments.panel_uncertainty import panel_intervals
from src.experiments.run_per_company import verify_pairing
from src.experiments.run_review import GRIDS, MODELS, ROOT, nested_predictions
from src.models.evaluation import evaluate_predictions


VARIANTS = ["base", "hybrid", "lag_1"]
APPROACHES = ["raw_ticker", "relative_pooled", "relative_ticker", "raw_interactions", "relative_interactions"]
CONTRASTS = [("raw_ticker", "joint"), ("relative_pooled", "joint"),
             ("relative_ticker", "relative_pooled"), ("relative_ticker", "joint"),
             ("raw_interactions", "raw_ticker"), ("relative_interactions", "relative_ticker"),
             ("relative_ticker", "individual")]


def ticker_features(dataset):
    if not dataset.ticker.isin(MODELING_TICKERS).all():
        raise ValueError("Empresa fuera del universo de entrenamiento")
    df, previous = feature_variants(dataset)
    raw = {v: previous[v] for v in VARIANTS}
    # The existing relative hybrid includes relative news volume. Keep news
    # strictly out of the financial-only base and lag each company's own series.
    relative_base = [c for c in previous["relative_features"]
                     if c not in SENTIMENT_FEATURES and c != "news_count_relative"]
    df["news_count_relative_lag_1"] = df.groupby("ticker").news_count_relative.shift(1).fillna(0)
    relative = {"base": relative_base, "hybrid": previous["relative_features"],
                "lag_1": previous["relative_features"] + ["sentiment_mean_lag_1", "news_count_relative_lag_1"]}
    identity = []
    for ticker in MODELING_TICKERS[1:]:
        name = "company_" + ticker
        df[name] = (df.ticker == ticker).astype(float)
        identity.append(name)
    schemas = {"raw_ticker": {v: cols + identity for v, cols in raw.items()},
               "relative_pooled": relative,
               "relative_ticker": {v: cols + identity for v, cols in relative.items()}}
    additions = {}
    for name, features in [("raw_interactions", raw), ("relative_interactions", relative)]:
        schemas[name] = {}
        for variant, columns in features.items():
            products = []
            for company in identity:
                for column in columns:
                    product = f"{company}__{column}"
                    additions[product] = df[company] * df[column]
                    products.append(product)
            schemas[name][variant] = columns + identity + products
    df = pd.concat([df, pd.DataFrame(additions, index=df.index)], axis=1)
    used = sorted({c for variants in schemas.values() for cols in variants.values() for c in cols})
    if not np.isfinite(df[used].to_numpy(dtype=float)).all():
        raise ValueError("Variables no finitas")
    return df, schemas


def checked_reference(reference, args, manifest, inputs, dataset_path):
    path = reference / "manifest.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous["status"] != "complete":
        raise ValueError("Referencia incompleta")
    for key in ["outer_start", "outer_splits", "inner_splits", "source_timezone"]:
        if previous["config"][key] != getattr(args, key):
            raise ValueError(f"Protocolo de referencia incompatible: {key}")
    if (previous["config"]["parameter_grids"] != GRIDS or previous["config"]["threshold"] != 0.5
            or previous["versions"] != manifest["versions"]):
        raise ValueError("Rejilla, umbral o versiones incompatibles")
    normalized = lambda mapping: {k.replace("\\", "/"): value for k, value in mapping.items()}
    old_inputs = normalized(previous["input_sha256"])
    for p in inputs:
        if old_inputs.get(p.relative_to(ROOT).as_posix()) != sha256(p):
            raise ValueError(f"Datos distintos: {p.name}")
    old_sources = normalized(previous["source_sha256"])
    for p in ["src/data/point_in_time.py", "src/data/aggregate_daily_sentiment.py",
              "src/data/handle_days_without_news.py", "src/data/split_train_test.py",
              "src/experiments/ablations.py", "src/experiments/run_review.py",
              "src/models/train_models.py", "src/models/temporal_validation.py"]:
        if old_sources[p] != sha256(ROOT / p):
            raise ValueError(f"Codigo de referencia distinto: {p}")
    old_outputs = normalized(previous["output_sha256"])
    old_dataset = previous["artifact_paths"]["data"] + "/dataset.csv"
    if sha256(dataset_path) != old_outputs[old_dataset]:
        raise ValueError("El panel reconstruido no reproduce el panel anterior")
    paths = [path]
    frames = []
    for approach, filename in [("joint", "joint_reference.csv"), ("individual", "individual.csv")]:
        p = reference / "predictions" / filename
        if sha256(p) != old_outputs[p.relative_to(ROOT).as_posix()]:
            raise ValueError("Predicciones de referencia alteradas")
        frames.append(pd.read_csv(p, parse_dates=["Date"]).assign(approach=approach))
        paths.append(p)
    p = reference / "joint/tuning/selected_params.csv"
    if sha256(p) != old_outputs[p.relative_to(ROOT).as_posix()]:
        raise ValueError("Fronteras de referencia alteradas")
    paths.append(p)
    return pd.concat(frames, ignore_index=True), pd.read_csv(p), paths


def summarize_panel(predictions, output, repeats):
    metrics = []
    for (approach, variant, model, ticker), frame in predictions.groupby(
            ["approach", "dataset_type", "model_name", "ticker"]):
        table = evaluate_predictions(frame, ["outer_fold"])
        table.loc[table.scope == "global", "mean_outer_auc"] = table.loc[table.scope == "outer_fold", "roc_auc"].mean()
        metrics.append(table.assign(approach=approach, dataset_type=variant, model_name=model, ticker=ticker))
    metrics = pd.concat(metrics, ignore_index=True)
    directory = output / "metrics"
    directory.mkdir()
    metrics.to_csv(directory / "all_metrics.csv", index=False)
    companies = metrics[metrics.scope == "global"]
    companies.to_csv(directory / "company_metrics.csv", index=False)
    columns = ["roc_auc", "mean_outer_auc", "accuracy", "balanced_accuracy", "f1_macro", "mcc"]
    companies.groupby(["approach", "dataset_type", "model_name"])[columns].mean().reset_index().to_csv(
        directory / "macro_metrics.csv", index=False)
    panel_intervals(predictions, CONTRASTS, repeats).to_csv(directory / "paired_intervals.csv", index=False)


def run(args):
    output, manifest = create_run(ROOT, args.run_dir, {
        **vars(args), "parameter_grids": GRIDS, "threshold": 0.5,
        "approaches": APPROACHES, "contrasts": CONTRASTS,
        "primary_comparison": "relative_ticker minus joint; macro AUC, each algorithm and variant",
        "interaction_models": ["logistic_regression"], "reference_company": MODELING_TICKERS[0],
        "uncertainty": "paired blocks of 20 sessions shared across companies; percentile intervals; unadjusted",
        "selection_metric": "pooled inner ROC AUC, unchanged from joint reference",
    })
    try:
        data_dir = artifact_dir(output, "data")
        dataset, news, inputs = build_dataset(ROOT, data_dir, args.source_timezone)
        reference = args.reference.resolve()
        previous, old_boundaries, ref_inputs = checked_reference(reference, args, manifest, inputs, data_dir / "dataset.csv")
        inputs += ref_inputs
        hashes = {str(p): sha256(p) for p in inputs}
        manifest["reference_verification"] = "identical inputs, dataset bytes, core source, versions, grids and protocol"
        manifest["coverage"] = audit_coverage(dataset, news, output / "coverage")
        dataset, schemas = ticker_features(dataset)
        manifest["feature_sets"] = schemas
        write_json(output / "manifest.json", manifest)
        dataset.to_csv(data_dir / "features.csv", index=False)
        predictions, selections = [previous], []
        joint = previous[previous.approach == "joint"]
        for approach in APPROACHES:
            models = ["logistic_regression"] if approach.endswith("interactions") else MODELS
            destination = output / "approaches" / approach
            print(f"Enfoque: {approach}; algoritmos={len(models)}", flush=True)
            with threadpool_limits(limits=1):
                frame = nested_predictions(dataset, schemas[approach], models, destination,
                    args.outer_start, args.outer_splits, args.inner_splits,
                    model_dir=artifact_dir(output, "models") / approach)
            matched = joint[joint.model_name.isin(models + ["dummy"])]
            verify_pairing(frame, matched)
            selected = pd.read_csv(destination / "tuning/selected_params.csv")
            boundaries = selected.merge(old_boundaries, on=["dataset_type", "model_name", "outer_fold"],
                suffixes=("", "_old"), validate="one_to_one")
            for key in ["train_rows", "test_rows", "train_end", "train_target_end", "test_start", "test_end"]:
                if not boundaries[key].equals(boundaries[key + "_old"]):
                    raise ValueError(f"Fronteras distintas: {key}")
            predictions.append(frame.assign(approach=approach))
            selections.append(selected.assign(approach=approach))
        predictions = pd.concat(predictions, ignore_index=True)
        (output / "predictions").mkdir()
        (output / "tuning").mkdir()
        predictions.to_csv(output / "predictions/all_predictions.csv", index=False)
        pd.concat(selections, ignore_index=True).to_csv(output / "tuning/selected_params.csv", index=False)
        summarize_panel(predictions, output, args.bootstrap_repeats)
        if any(sha256(p) != hashes[str(p)] for p in inputs):
            raise RuntimeError("Las entradas cambiaron durante la ejecucion")
        finish_run(output, manifest, inputs, ROOT)
        print(f"Experimento completado: {output}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(output / "manifest.json", manifest)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Modelos conjuntos que identifican la empresa.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "reports/experiments/per-company-20260916")
    parser.add_argument("--outer-start", default="2023-10-16")
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--source-timezone", default="UTC")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
