from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.experiments.artifacts import ROOT, artifact_dir, create_run, finish_run, sha256, write_json
from src.experiments.panel_uncertainty import panel_intervals
from src.experiments.run_per_company import verify_pairing
from src.experiments.sentiment_representation import EXTRA
from src.experiments.training_windows import training_bounds, windowed_train
from src.models.evaluation import evaluate_predictions
from src.models.train_models import build_model


CONTRASTS = [(f"{mode}_{feature}", reference)
             for mode, reference in [("add", "deduplicated"), ("drop", "enhanced")]
             for feature in EXTRA]


def feature_subset(full, clean, variant, feature, mode):
    if feature not in EXTRA or mode not in {"add", "drop"} or variant not in {"hybrid", "lag_1"}:
        raise ValueError("Invalid ablation")
    family = {feature, feature + "_lag_1"} if variant == "lag_1" else {feature}
    if not family.issubset(full) or not set(clean).issubset(full):
        raise ValueError("Inconsistent feature schema")
    if mode == "add":
        return [c for c in full if c in clean or c in family]
    return [c for c in full if c not in family]


def summarize(predictions, output, repeats):
    tables = []
    for (approach, variant, model, ticker), frame in predictions.groupby(
            ["approach", "dataset_type", "model_name", "ticker"]):
        metrics = evaluate_predictions(frame, ["outer_fold"])
        tables.append(metrics.assign(approach=approach, dataset_type=variant, model_name=model, ticker=ticker))
    metrics = pd.concat(tables, ignore_index=True)
    directory = output / "metrics"
    directory.mkdir()
    metrics.to_csv(directory / "all_metrics.csv", index=False)
    companies = metrics[metrics.scope == "global"]
    companies.to_csv(directory / "company_metrics.csv", index=False)
    columns = ["roc_auc", "accuracy", "balanced_accuracy", "f1_macro", "mcc"]
    companies.groupby(["approach", "dataset_type", "model_name"])[columns].mean().reset_index().to_csv(
        directory / "macro_metrics.csv", index=False)
    panel_intervals(predictions, CONTRASTS, repeats).to_csv(directory / "paired_intervals.csv", index=False)


def run(args):
    previous_dir = (ROOT / args.reference_dir).resolve()
    previous_path = previous_dir / "manifest.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    output, manifest = create_run(ROOT, args.run_dir, {**vars(args), "contrasts": CONTRASTS,
        "threshold": 0.5, "primary_comparison": "add-one hybrid, each of five features and three algorithms",
        "hyperparameters": "fixed from each corresponding reference, per fold and variant",
        "lag_family": "current feature and its lag removed or added together",
        "uncertainty": "paired moving blocks of 20 dates; unadjusted; no refit"})
    manifest["evaluation_status"] = "Exploratory fixed-hyperparameter refits on previously inspected dates; not an untouched holdout."
    try:
        if previous["status"] != "complete" or previous["versions"] != manifest["versions"]:
            raise ValueError("Incomplete reference or changed dependencies")
        old_sources = {key.replace("\\", "/"): value for key, value in previous["source_sha256"].items()}
        for name in ["src/models/train_models.py", "src/models/temporal_validation.py",
                     "src/experiments/training_windows.py", "src/experiments/sentiment_representation.py"]:
            if sha256(ROOT / name) != old_sources[name]:
                raise ValueError(f"Reference implementation changed: {name}")
        inputs = [previous_path]

        def checked(relative):
            path = ROOT / relative
            if sha256(path) != previous["output_sha256"][path.relative_to(ROOT).as_posix()]:
                raise ValueError(f"Reference artifact changed: {relative}")
            inputs.append(path)
            return path

        frame = pd.read_csv(checked(previous["artifact_paths"]["data"] + "/enhanced_features.csv"),
                            parse_dates=["Date", "target_end"], float_precision="round_trip")
        old = pd.read_csv(checked(previous_dir.relative_to(ROOT).as_posix() + "/predictions/all_predictions.csv"),
                          parse_dates=["Date"], float_precision="round_trip")
        selected = pd.read_csv(checked(previous_dir.relative_to(ROOT).as_posix() + "/tuning/selected_params.csv"))
        selected = selected[selected.approach.isin(["deduplicated", "enhanced"]) &
                            selected.dataset_type.isin(["hybrid", "lag_1"])]
        if len(selected) != 36:
            raise ValueError("Expected 36 reference models")
        schemas = previous["feature_sets"]
        manifest["feature_sets"] = {anchor: schemas[name] for anchor, name in
                                    [("deduplicated", "clean"), ("enhanced", "enhanced")]}
        for approach, reference in CONTRASTS:
            mode, feature = approach.split("_", 1)
            manifest["feature_sets"][approach] = {
                variant: feature_subset(schemas["enhanced"][variant], schemas["clean"][variant], variant, feature, mode)
                for variant in ["hybrid", "lag_1"]}
        old = old[old.approach.isin(["deduplicated", "enhanced"]) & old.dataset_type.isin(["hybrid", "lag_1"])]
        predictions, selections = [old], []
        (output / "predictions").mkdir()
        (output / "tuning").mkdir()
        keys = ["ticker", "Date", "target"]
        anchor_models = {}
        # Verify all anchors before interpreting any ablation against their saved forecasts.
        for row in selected.itertuples(index=False):
            valid_old = old[(old.approach == row.approach) & (old.dataset_type == row.dataset_type) &
                            (old.model_name == row.model_name) & (old.outer_fold == row.outer_fold)]
            valid = frame[frame.Date.isin(valid_old.Date.unique())]
            pd.testing.assert_frame_equal(valid[keys].reset_index(drop=True), valid_old[keys].reset_index(drop=True))
            path = checked(previous["artifact_paths"]["models"] +
                           f"/{row.approach}/fold_{row.outer_fold}_{row.dataset_type}_{row.model_name}.joblib")
            model = joblib.load(path)
            columns = manifest["feature_sets"][row.approach][row.dataset_type]
            if list(model.feature_names_in_) != columns:
                raise ValueError("Reference model feature mismatch")
            with threadpool_limits(limits=1):
                probability = model.predict_proba(valid[columns])[:, 1]
            np.testing.assert_allclose(probability, valid_old.probability, rtol=0, atol=1e-12)
            anchor_models[(row.approach, row.dataset_type, row.model_name, row.outer_fold)] = model
        manifest["verified_anchor_models"] = len(anchor_models)
        hashes = {p: sha256(p) for p in inputs}
        for approach, reference in CONTRASTS:
            for row in selected[selected.approach == reference].itertuples(index=False):
                columns = manifest["feature_sets"][approach][row.dataset_type]
                valid = frame[(frame.Date >= row.valid_start) & (frame.Date <= row.valid_end)]
                if str(row.window_years) != "all":
                    raise ValueError("Expected all-history reference")
                train = windowed_train(frame, valid.Date.min())
                bounds = training_bounds(train, valid, None)
                if bounds["train_rows"] != row.train_rows or bounds["valid_rows"] != row.valid_rows:
                    raise ValueError("Reference training or validation rows changed")
                for name in ["train_start", "train_end", "train_target_end", "valid_start", "valid_end"]:
                    if pd.Timestamp(bounds[name]) != pd.Timestamp(getattr(row, name)):
                        raise ValueError(f"Reference boundary changed: {name}")
                params = json.loads(row.params)
                model = build_model(row.model_name).set_params(**params)
                anchor = anchor_models[(reference, row.dataset_type, row.model_name, row.outer_fold)]
                for key, value in model.get_params().items():
                    if "__" in key and repr(value) != repr(anchor.get_params()[key]):
                        raise ValueError(f"Reference parameter changed: {key}")
                with threadpool_limits(limits=1):
                    model.fit(train[columns], train.target)
                    probability = model.predict_proba(valid[columns])[:, 1]
                prediction = valid[keys].copy().assign(approach=approach, dataset_type=row.dataset_type,
                    model_name=row.model_name, outer_fold=row.outer_fold, probability=probability,
                    prediction=(probability >= 0.5).astype(int))
                predictions.append(prediction)
                selections.append(dict(approach=approach, reference=reference, dataset_type=row.dataset_type,
                    model_name=row.model_name, outer_fold=row.outer_fold, **bounds,
                    features=len(columns), params=json.dumps(params, sort_keys=True)))
                path = artifact_dir(output, "models") / approach
                path.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path / f"fold_{row.outer_fold}_{row.dataset_type}_{row.model_name}.joblib")
                print(f"{len(selections)}/180: {approach}, {row.dataset_type}/{row.model_name}, fold {row.outer_fold}", flush=True)
            pd.DataFrame(selections).to_csv(output / "tuning/inherited_params.csv", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(output / "predictions/all_predictions.csv", index=False)
        predictions = pd.concat(predictions, ignore_index=True)
        for approach, reference in CONTRASTS:
            verify_pairing(predictions[predictions.approach == approach], predictions[predictions.approach == reference])
        summarize(predictions, output, args.bootstrap_repeats)
        if any(sha256(path) != digest for path, digest in hashes.items()):
            raise ValueError("Reference inputs changed during execution")
        manifest["trained_models"] = len(selections)
        finish_run(output, manifest, inputs, ROOT)
        print(f"Completed: {output}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled sentiment feature ablations")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=Path("reports/experiments/first-round-full-20260916"))
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    run(parser.parse_args())
