from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.point_in_time import build_dataset
from src.experiments.artifacts import artifact_dir, create_run, finish_run, sha256, write_json
from src.experiments.news_quality import deduplicate_news, quality_audit, rebuild_sentiment
from src.experiments.panel_uncertainty import panel_intervals
from src.experiments.round_selection import ROUND_MODELS, candidates, choose_candidate
from src.experiments.run_per_company import verify_pairing
from src.experiments.run_review import ROOT
from src.experiments.sentiment_representation import enriched_panel, relative_panel
from src.experiments.training_windows import training_bounds, windowed_train
from src.models.evaluation import evaluate_predictions
from src.models.train_models import build_model


STAGES = {
    "reference": ("original", None, "pooled_auc", False),
    "deduplicated": ("clean", None, "pooled_auc", False),
    "window_3y": ("clean", 3, "pooled_auc", False),
    "window_5y": ("clean", 5, "pooled_auc", False),
    "enhanced": ("enhanced", None, "pooled_auc", False),
    "macro": ("enhanced", None, "macro_auc", False),
    "tuned": ("enhanced", None, "macro_auc", True),
}
CONTRASTS = [("deduplicated", "reference"), ("window_3y", "deduplicated"), ("window_5y", "deduplicated"),
             ("enhanced", "deduplicated"), ("macro", "enhanced"), ("tuned", "macro"), ("tuned", "reference")]


def summarize(predictions, output, repeats):
    tables = []
    for (approach, variant, model, ticker), frame in predictions.groupby(
            ["approach", "dataset_type", "model_name", "ticker"]):
        metrics = evaluate_predictions(frame, ["outer_fold"])
        metrics.loc[metrics.scope == "global", "mean_outer_auc"] = metrics.loc[metrics.scope == "outer_fold", "roc_auc"].mean()
        tables.append(metrics.assign(approach=approach, dataset_type=variant, model_name=model, ticker=ticker))
    metrics = pd.concat(tables, ignore_index=True)
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
    planned = {name: {m: candidates(m, years, expanded) for m in args.models}
               for name, (_, years, metric, expanded) in STAGES.items()}
    output, manifest = create_run(ROOT, args.run_dir, {**vars(args), "stages": STAGES,
        "candidates": planned, "threshold": 0.5, "deduplication_hours": 24,
        "primary_comparison": "tuned minus reference, hybrid, each prespecified algorithm; macro AUC",
        "contrasts": CONTRASTS, "uncertainty": "paired moving blocks of 20 dates; unadjusted; no refit"})
    try:
        data_dir = artifact_dir(output, "data")
        dataset, news, inputs = build_dataset(ROOT, data_dir, args.source_timezone)
        previous_dir = ROOT / "reports/experiments/ticker-aware-full-20260916"
        previous_path = previous_dir / "manifest.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous["status"] != "complete" or previous["versions"] != manifest["versions"]:
            raise ValueError("Referencia incompleta o versiones diferentes")
        old_inputs = {key.replace("\\", "/"): value for key, value in previous["input_sha256"].items()}
        if any(old_inputs.get(p.relative_to(ROOT).as_posix()) != sha256(p) for p in inputs):
            raise ValueError("Entradas distintas a la referencia")
        for key in ["outer_start", "outer_splits", "inner_splits", "source_timezone"]:
            if previous["config"][key] != getattr(args, key):
                raise ValueError(f"Protocolo de referencia diferente: {key}")
        old_dataset = previous["artifact_paths"]["data"] + "/dataset.csv"
        if sha256(data_dir / "dataset.csv") != previous["output_sha256"][old_dataset]:
            raise ValueError("El panel no reproduce la referencia")
        old_file = previous_dir / "predictions/all_predictions.csv"
        if sha256(old_file) != previous["output_sha256"][old_file.relative_to(ROOT).as_posix()]:
            raise ValueError("Predicciones de referencia modificadas")
        inputs += [old_file, previous_path]
        hashes = {str(p): sha256(p) for p in inputs}
        cleaned, links = deduplicate_news(news)
        links.to_csv(data_dir / "removed_news.csv", index=False)
        original = rebuild_sentiment(dataset, news)
        pd.testing.assert_frame_equal(dataset[original.columns].sort_values(["Date", "ticker"]).reset_index(drop=True),
                                      original, check_dtype=False)
        deduplicated = rebuild_sentiment(dataset, cleaned)
        manifest["quality_audit"] = quality_audit(dataset, news, cleaned, links, output / "quality")
        datasets = {"original": relative_panel(dataset), "clean": relative_panel(deduplicated),
                    "enhanced": enriched_panel(deduplicated, cleaned)}
        manifest["feature_sets"] = {name: schema for name, (_, schema) in datasets.items()}
        if datasets["original"][1] != previous["feature_sets"]["relative_pooled"]:
            raise ValueError("Variables de referencia diferentes")
        for name, (frame, _) in datasets.items():
            frame.to_csv(data_dir / f"{name}_features.csv", index=False)
        write_json(output / "manifest.json", manifest)
        predictions, traces, selections = [], [], []
        (output / "predictions").mkdir()
        (output / "tuning").mkdir()
        model_dir = artifact_dir(output, "models")
        old = pd.read_csv(old_file, parse_dates=["Date"], float_precision="round_trip")
        old = old[(old.approach == "relative_pooled") & old.model_name.isin(args.models + ["dummy"])]
        for stage, (name, years, metric, expanded) in STAGES.items():
            frame, schemas = datasets[name]
            dates = sorted(frame.loc[frame.Date >= args.outer_start, "Date"].unique())
            if len(dates) < args.outer_splits * 2 or args.outer_splits < 2:
                raise ValueError("Bloques externos insuficientes")
            (model_dir / stage).mkdir(parents=True)
            for fold, valid_dates in enumerate(np.array_split(dates, args.outer_splits), start=1):
                train = windowed_train(frame, valid_dates[0])
                valid = frame[frame.Date.isin(valid_dates)]
                pairs = [("base", "dummy")] + [(v, m) for v in schemas for m in args.models]
                for variant, model in pairs:
                    columns = schemas[variant]
                    with threadpool_limits(limits=1):
                        if model == "dummy":
                            chosen, rows, score = {"years": None, "params": {}}, [], 0.5
                        else:
                            chosen, rows, score = choose_candidate(train, columns, model, planned[stage][model], metric, args.inner_splits)
                        fit = windowed_train(frame, valid_dates[0], chosen["years"])
                        estimator = build_model(model).set_params(**chosen["params"])
                        estimator.fit(fit[columns], fit.target)
                        probability = estimator.predict_proba(valid[columns])[:, 1]
                    prediction = valid[["ticker", "Date", "target"]].copy()
                    prediction = prediction.assign(approach=stage, dataset_type=variant, model_name=model,
                        outer_fold=fold, probability=probability, prediction=(probability >= 0.5).astype(int))
                    predictions.append(prediction)
                    metadata = dict(approach=stage, dataset_type=variant, model_name=model, outer_fold=fold)
                    traces.extend([{**row, **metadata} for row in rows])
                    selections.append({**metadata, **training_bounds(fit, valid, chosen["years"]),
                        "selection_metric": metric, "inner_score": score, "features": len(columns),
                        "params": json.dumps(chosen["params"], sort_keys=True)})
                    joblib.dump(estimator, model_dir / stage / f"fold_{fold}_{variant}_{model}.joblib")
                    print(f"{stage}: bloque {fold}, {variant}/{model}, CV={score:.4f}, ventana={chosen['years']}", flush=True)
                pd.DataFrame(selections).to_csv(output / "tuning/selected_params.csv", index=False)
                pd.DataFrame(traces).to_csv(output / "tuning/inner_cv.csv", index=False)
                pd.concat(predictions, ignore_index=True).to_csv(output / "predictions/all_predictions.csv", index=False)
            if stage == "reference":
                reference = pd.concat(predictions, ignore_index=True)
                verify_pairing(reference, old)
                keys = ["ticker", "Date", "dataset_type", "model_name", "outer_fold"]
                paired = reference.merge(old, on=keys, validate="one_to_one", suffixes=("", "_old"))
                np.testing.assert_allclose(paired.probability, paired.probability_old, rtol=0, atol=1e-12)
                manifest["reference_predictions_verified"] = len(paired)
                write_json(output / "manifest.json", manifest)
                print("Referencia anterior reproducida", flush=True)
        predictions = pd.concat(predictions, ignore_index=True)
        reference = predictions[predictions.approach == "reference"]
        for stage in STAGES:
            verify_pairing(predictions[predictions.approach == stage], reference)
        summarize(predictions, output, args.bootstrap_repeats)
        if any(sha256(p) != hashes[str(p)] for p in inputs if str(p) in hashes):
            raise RuntimeError("Entradas modificadas durante la ejecucion")
        finish_run(output, manifest, inputs, ROOT)
        print(f"Ronda completada: {output}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(output / "manifest.json", manifest)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Primera ronda controlada de mejoras.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=ROUND_MODELS, default=ROUND_MODELS)
    parser.add_argument("--outer-start", default="2023-10-16")
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--source-timezone", default="UTC")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
