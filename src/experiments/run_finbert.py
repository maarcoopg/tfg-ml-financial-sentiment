"""Prespecified, paired financial evaluation of the frozen FinBERT checkpoint."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.point_in_time import build_dataset
from src.experiments.artifacts import artifact_dir, create_run, finish_run, sha256, write_json
from src.experiments.finbert_panel import START, build_paired_panels, news_eligibility
from src.experiments.news_quality import deduplicate_news, quality_audit
from src.experiments.panel_uncertainty import panel_intervals
from src.experiments.round_selection import ROUND_MODELS, candidates, choose_candidate
from src.experiments.run_review import ROOT
from src.experiments.training_windows import training_bounds, windowed_train
from src.models.evaluation import evaluate_predictions
from src.models.train_models import build_model
from src.nlp.news_score_cache import score_corpus


APPROACHES = ["alpha_all", "alpha_matched", "finbert_matched"]
CONTRASTS = [("finbert_matched", "alpha_matched"), ("alpha_matched", "alpha_all")]
CONTRASTS += [(name, "financial") for name in APPROACHES]


def fit_panels(panels, schemas, output, outer_start="2023-10-16", outer_splits=3, inner_splits=3):
    dates = sorted(panels["financial"].loc[lambda f: f.Date >= outer_start, "Date"].unique())
    if len(dates) < outer_splits * 2:
        raise ValueError("Insufficient outer dates")
    for name in ["predictions", "tuning"]:
        (output / name).mkdir()
    model_dir = artifact_dir(output, "models")
    predictions, traces, selections = [], [], []
    combinations = [("financial", "base", "dummy")]
    combinations += [("financial", "base", model) for model in ROUND_MODELS]
    combinations += [(approach, variant, model) for approach in APPROACHES
                     for variant in ["hybrid", "lag_1"] for model in ROUND_MODELS]
    for fold, valid_dates in enumerate(np.array_split(dates, outer_splits), start=1):
        for approach, variant, model in combinations:
            frame, columns = panels[approach], schemas[variant]
            train = windowed_train(frame, valid_dates[0])
            valid = frame[frame.Date.isin(valid_dates)]
            with threadpool_limits(limits=1):
                if model == "dummy":
                    chosen, rows, score = {"years": None, "params": {}}, [], 0.5
                else:
                    chosen, rows, score = choose_candidate(train, columns, model,
                        candidates(model), "macro_auc", inner_splits)
                estimator = build_model(model).set_params(**chosen["params"])
                estimator.fit(train[columns], train.target)
                probability = estimator.predict_proba(valid[columns])[:, 1]
            metadata = dict(approach=approach, dataset_type=variant, model_name=model, outer_fold=fold)
            predictions.append(valid[["ticker", "Date", "target"]].assign(**metadata,
                probability=probability, prediction=(probability >= 0.5).astype(int)))
            traces.extend([{**row, **metadata} for row in rows])
            selections.append({**metadata, **training_bounds(train, valid, None),
                "selection_metric": "macro_auc", "inner_score": score, "features": len(columns),
                "params": json.dumps(chosen["params"], sort_keys=True)})
            directory = model_dir / approach
            directory.mkdir(parents=True, exist_ok=True)
            joblib.dump(estimator, directory / f"fold_{fold}_{variant}_{model}.joblib")
            print(f"Bloque {fold}: {approach}/{variant}/{model}; CV={score:.4f}", flush=True)
        pd.DataFrame(selections).to_csv(output / "tuning/selected_params.csv", index=False)
        pd.DataFrame(traces).to_csv(output / "tuning/inner_cv.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(output / "predictions/all_predictions.csv", index=False)
    return pd.concat(predictions, ignore_index=True)


def paired_predictions(predictions):
    financial = predictions[(predictions.approach == "financial") & (predictions.model_name != "dummy")]
    return pd.concat([predictions, financial.assign(dataset_type="hybrid"),
                      financial.assign(dataset_type="lag_1")], ignore_index=True)


def summarize(predictions, output, repeats):
    tables = []
    for (approach, variant, model, ticker), frame in predictions.groupby(
            ["approach", "dataset_type", "model_name", "ticker"]):
        metrics = evaluate_predictions(frame, ["outer_fold"])
        metrics.loc[metrics.scope == "global", "mean_outer_auc"] = metrics.loc[metrics.scope == "outer_fold", "roc_auc"].mean()
        tables.append(metrics.assign(approach=approach, dataset_type=variant, model_name=model, ticker=ticker))
    directory = output / "metrics"
    directory.mkdir()
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(directory / "all_metrics.csv", index=False)
    companies = metrics[metrics.scope == "global"]
    companies.to_csv(directory / "company_metrics.csv", index=False)
    columns = ["roc_auc", "mean_outer_auc", "accuracy", "balanced_accuracy", "f1_macro", "mcc"]
    companies.groupby(["approach", "dataset_type", "model_name"])[columns].mean().reset_index().to_csv(
        directory / "macro_metrics.csv", index=False)
    panel_intervals(paired_predictions(predictions), CONTRASTS, repeats).to_csv(
        directory / "paired_intervals.csv", index=False)


def run(args):
    config = {**vars(args), "training_start": START, "outer_start": "2023-10-16", "outer_splits": 3,
        "inner_splits": 3, "models": ROUND_MODELS, "candidates": {m: candidates(m) for m in ROUND_MODELS},
        "threshold": 0.5, "selection_metric": "macro_auc", "deduplication_hours": 24,
        "contrasts": CONTRASTS, "uncertainty": "1000 shared moving blocks of 20 dates; no refit; no multiplicity correction"}
    output, manifest = create_run(ROOT, args.run_dir, config)
    try:
        data_dir = artifact_dir(output, "data")
        dataset, news, inputs = build_dataset(ROOT, data_dir, "UTC")
        inputs += [ROOT / "docs/finbert_prediccion_protocolo.md", ROOT / "requirements-finbert.txt"]
        hashes = {str(p): sha256(p) for p in inputs}
        audit = news_eligibility(news)
        cleaned, links = deduplicate_news(audit)
        all_news = cleaned[cleaned.post_checkpoint].copy()
        matched = all_news[all_news.eligible].copy()
        if matched.empty or matched.news_id.duplicated().any():
            raise ValueError("Empty or duplicate matched news")
        (output / "quality").mkdir()
        audit["retained_after_dedup"] = audit.news_id.isin(cleaned.news_id)
        audit[["news_id", "text_sha256", "ticker", "published_at", "trading_date", "explicit_company",
               "dynamic_page", "post_checkpoint", "eligible", "retained_after_dedup"]].to_csv(
                   output / "quality/news_eligibility.csv", index=False)
        links.to_csv(data_dir / "removed_news.csv", index=False)
        manifest["coverage_audit"] = quality_audit(dataset[dataset.Date >= START], all_news, matched,
            links, output / "quality/coverage")
        manifest["coverage_audit"]["policy"] = "all post-2020 deduplicated news versus fixed eligible subset; see protocol"
        (output / "sentiment").mkdir()
        print(f"Noticias: {len(news)} alineadas; {len(all_news)} deduplicadas desde 2021; {len(matched)} emparejadas", flush=True)
        scores, inference = score_corpus(matched, args.cache, output / "sentiment", args.device, args.batch_size)
        panels, schema, converted = build_paired_panels(dataset, all_news, matched, scores)
        manifest["feature_sets"], manifest["inference"] = schema, inference
        manifest["news_counts"] = {"aligned": len(news), "post_2020_deduplicated": len(all_news),
            "matched": len(matched), "unique_scored_texts": len(scores), "truncated_texts": int(scores.truncated.sum()),
            "dynamic_pages_flagged": int(audit.dynamic_page.sum())}
        trace = matched[["news_id", "text_sha256", "ticker", "published_at", "trading_date", "market_close",
                         "sentiment_score", "sentiment_label", "relevance_score"]].rename(
                             columns={"sentiment_score": "alpha_score", "sentiment_label": "alpha_label"})
        trace = trace.merge(scores, on="text_sha256", validate="many_to_one")
        trace.to_csv(output / "sentiment/news_scores.csv", index=False)
        agreement = []
        for ticker, part in trace.groupby("ticker"):
            agreement.append(dict(ticker=ticker, news=len(part),
                label_agreement=float(part.alpha_label.eq(part.label).mean()),
                pearson=part.alpha_score.corr(part.score), spearman=part.alpha_score.corr(part.score, method="spearman")))
        pd.DataFrame(agreement).to_csv(output / "sentiment/provider_agreement.csv", index=False)
        converted.to_csv(data_dir / "finbert_news.csv", index=False)
        for name, frame in panels.items():
            frame.to_csv(data_dir / f"{name}_features.csv", index=False)
        write_json(output / "manifest.json", manifest)
        predictions = fit_panels(panels, schema, output)
        summarize(predictions, output, 1000)
        if any(sha256(p) != hashes[str(p)] for p in inputs):
            raise RuntimeError("Inputs changed during execution")
        finish_run(output, manifest, inputs, ROOT)
        print(f"FinBERT completado: {output}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(output / "manifest.json", manifest)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Paired FinBERT financial evaluation, issue 51.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/experiments/finbert-cache/scores.sqlite")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
