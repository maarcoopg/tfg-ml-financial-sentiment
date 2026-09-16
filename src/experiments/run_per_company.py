from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.audit_news_coverage import audit_coverage
from src.data.point_in_time import build_dataset
from src.data.split_train_test import MODELING_TICKERS
from src.experiments.ablations import feature_variants
from src.experiments.artifacts import artifact_dir, create_run, finish_run, sha256, write_json
from src.experiments.run_review import GRIDS, MODELS, ROOT, nested_predictions, summarize
from src.models.evaluation import evaluate_predictions
from src.models.temporal_validation import paired_auc_interval, validate_panel


VARIANTS = ["base", "hybrid", "lag_1"]
PAIR_KEYS = ["ticker", "Date", "target", "dataset_type", "model_name", "outer_fold"]


def company_data(dataset, news, ticker):
    """Filter before deriving lags or fitting any preprocessing/model component."""
    if ticker not in MODELING_TICKERS:
        raise ValueError(f"Empresa fuera del universo: {ticker}")
    frame = dataset.loc[dataset.ticker == ticker].copy()
    validate_panel(frame)
    frame, variants = feature_variants(frame)
    return frame, news.loc[news.ticker == ticker].copy(), {v: variants[v] for v in VARIANTS}


def verify_pairing(individual, joint):
    order = ["ticker", "dataset_type", "model_name", "Date"]
    a = individual.sort_values(order).reset_index(drop=True)
    b = joint.sort_values(order).reset_index(drop=True)
    if a.duplicated(order).any() or b.duplicated(order).any():
        raise ValueError("Predicciones duplicadas")
    if not a[PAIR_KEYS].equals(b[PAIR_KEYS]):
        raise ValueError("La comparacion requiere las mismas filas, objetivos y bloques")
    for frame in [a, b]:
        if not frame.probability.between(0, 1).all():
            raise ValueError("Probabilidades invalidas")
        if not (frame.prediction == (frame.probability >= 0.5).astype(int)).all():
            raise ValueError("Predicciones incompatibles con el umbral")


def compare_approaches(individual, joint, output, repeats):
    verify_pairing(individual, joint)
    metrics, intervals = [], []
    keys = ["ticker", "dataset_type", "model_name"]
    references = dict(tuple(joint.groupby(keys)))
    for key, frame in individual.groupby(keys):
        ticker, variant, model = key
        other = references[key]
        for approach, predictions in [("individual", frame), ("joint", other)]:
            table = evaluate_predictions(predictions, ["outer_fold"])
            folds = table.loc[table.scope == "outer_fold", "roc_auc"]
            table.loc[table.scope == "global", "mean_outer_auc"] = folds.mean()
            table.loc[table.scope == "global", "std_outer_auc"] = folds.std()
            metrics.append(table.assign(ticker=ticker, dataset_type=variant,
                                        model_name=model, approach=approach))
        if model != "dummy":
            interval = paired_auc_interval(frame, other, block=20, repeats=repeats)
            intervals.append(dict(ticker=ticker, dataset_type=variant, model_name=model,
                                  reference_approach="joint", **interval))
        print(f"Comparacion: {ticker}/{variant}/{model}", flush=True)
    metrics = pd.concat(metrics, ignore_index=True)
    directory = output / "metrics"
    directory.mkdir(exist_ok=True)
    metrics.to_csv(directory / "comparison_metrics.csv", index=False)
    globals_ = metrics[metrics.scope == "global"]
    globals_.to_csv(directory / "company_metrics.csv", index=False)
    numeric = ["roc_auc", "mean_outer_auc", "accuracy", "balanced_accuracy", "f1_macro", "mcc"]
    globals_.groupby(["approach", "dataset_type", "model_name"])[numeric].mean().reset_index().to_csv(
        directory / "macro_metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(directory / "individual_vs_joint_intervals.csv", index=False)


def run(args):
    output, manifest = create_run(ROOT, args.run_dir, {
        **vars(args), "parameter_grids": GRIDS, "variants": VARIANTS, "threshold": 0.5,
        "approach": "independent training and tuning for each company",
        "uncertainty": "paired moving blocks of 20 sessions; percentile 95%; no multiple-testing correction",
        "reference_policy": "retrain joint models in this run; identical inputs, code, grids and observation keys",
    })
    try:
        data_dir = artifact_dir(output, "data")
        dataset, news, inputs = build_dataset(ROOT, data_dir, args.source_timezone)
        companies = {ticker: company_data(dataset, news, ticker) for ticker in MODELING_TICKERS}
        expected_dates = sorted(dataset.Date.unique())
        for frame, _, _ in companies.values():
            if sorted(frame.Date.unique()) != expected_dates:
                raise ValueError("Las empresas deben compartir el calendario completo")
        variants = next(iter(companies.values()))[2]
        hashes = {str(path): sha256(path) for path in inputs}
        manifest["feature_sets"] = variants
        manifest["coverage"] = audit_coverage(dataset, news, output / "coverage")
        write_json(output / "manifest.json", manifest)
        joint_dataset, joint_variants = feature_variants(dataset)
        for ticker, (frame, _, features) in companies.items():
            columns = ["ticker", "Date", "target", "target_end"] + features["lag_1"]
            pd.testing.assert_frame_equal(frame[columns].reset_index(drop=True),
                joint_dataset.loc[joint_dataset.ticker == ticker, columns].reset_index(drop=True))
        print("Referencia conjunta: nuevo entrenamiento con el mismo protocolo", flush=True)
        with threadpool_limits(limits=1):
            joint = nested_predictions(joint_dataset, {v: joint_variants[v] for v in VARIANTS},
                args.models, output / "joint", args.outer_start, args.outer_splits,
                args.inner_splits, model_dir=artifact_dir(output, "models") / "joint")
        frames, selections = [], []
        for ticker, (frame, company_news, features) in companies.items():
            print(f"Empresa: {ticker}; filas={len(frame)}; noticias={len(company_news)}", flush=True)
            company_dir = output / "companies" / ticker
            local_data = data_dir / ticker
            local_data.mkdir()
            frame.to_csv(local_data / "dataset.csv", index=False)
            company_news.to_csv(local_data / "news.csv", index=False)
            with threadpool_limits(limits=1):
                predictions = nested_predictions(
                    frame, features, args.models, company_dir, args.outer_start,
                    args.outer_splits, args.inner_splits,
                    model_dir=artifact_dir(output, "models") / ticker)
            summarize(predictions, company_dir, args.bootstrap_repeats)
            frames.append(predictions)
            selections.append(pd.read_csv(company_dir / "tuning/selected_params.csv").assign(ticker=ticker))
        individual = pd.concat(frames, ignore_index=True)
        selection = pd.concat(selections, ignore_index=True)
        old_selection = pd.read_csv(output / "joint/tuning/selected_params.csv")
        boundary_keys = ["dataset_type", "model_name", "outer_fold"]
        boundaries = selection.merge(old_selection, on=boundary_keys, suffixes=("", "_joint"), validate="many_to_one")
        for key in ["train_end", "train_target_end", "test_start", "test_end"]:
            if not boundaries[key].equals(boundaries[key + "_joint"]):
                raise ValueError(f"Fronteras distintas a la referencia: {key}")
        for key in ["train_rows", "test_rows"]:
            if not (boundaries[key] * len(companies) == boundaries[key + "_joint"]).all():
                raise ValueError(f"Numero de observaciones incompatible: {key}")
        (output / "predictions").mkdir()
        (output / "tuning").mkdir()
        individual.to_csv(output / "predictions/individual.csv", index=False)
        joint.to_csv(output / "predictions/joint_reference.csv", index=False)
        selection.to_csv(output / "tuning/selected_params.csv", index=False)
        compare_approaches(individual, joint, output, args.bootstrap_repeats)
        within = pd.concat([pd.read_csv(output / "companies" / t / "metrics/auc_intervals.csv").assign(ticker=t)
                            for t in companies], ignore_index=True)
        within.to_csv(output / "metrics/within_company_intervals.csv", index=False)
        if any(sha256(path) != hashes[str(path)] for path in inputs):
            raise RuntimeError("Los archivos de entrada cambiaron durante la ejecucion")
        finish_run(output, manifest, inputs, ROOT)
        print(f"Experimento completado: {output}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(output / "manifest.json", manifest)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Experimento temporal independiente por empresa.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--outer-start", default="2023-10-16")
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--source-timezone", default="UTC")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
