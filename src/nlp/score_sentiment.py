"""Evaluate a reserved panel only after explicit, validated human annotation."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.experiments.artifacts import artifact_dir, create_run, finish_run, sha256, write_json
from src.nlp.evaluate_sentiment import predict_variants
from src.nlp.finbert import ROOT, MODEL_ID, REVISION, load_finbert
from src.nlp.sentiment_data import validated_annotations
from src.nlp.sentiment_metrics import human_metrics, annotator_agreement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-run", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--adjudicated", type=Path)
    parser.add_argument("--allow-single-annotator", action="store_true")
    parser.add_argument("--partition", choices=["development", "evaluation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = ROOT / args.sample_run
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    sample_path = artifact_dir(run, "data") / "sample.csv"
    key = sample_path.relative_to(ROOT).as_posix()
    if sha256(sample_path) != manifest["output_sha256"][key]:
        raise ValueError("Sample no longer matches its frozen manifest")
    sample = pd.read_csv(sample_path, keep_default_na=False)
    primary = pd.read_csv(ROOT / args.primary, keep_default_na=False)
    validated_annotations(sample, primary)
    gold = primary
    inputs = [sample_path, ROOT / args.primary]
    agreement = {"status": "single_annotator_explicitly_accepted"}
    if args.secondary:
        secondary = pd.read_csv(ROOT / args.secondary, keep_default_na=False)
        expected = pd.read_csv(run / "annotation_secondary.csv", keep_default_na=False)
        if set(secondary.news_id) != set(expected.news_id):
            raise ValueError("Secondary annotations must match the predefined blind subset")
        agreement = annotator_agreement(sample, primary, secondary)
        inputs.append(ROOT / args.secondary)
        if agreement["adjudication_required"] and not args.adjudicated:
            raise ValueError("Resolve disagreements in a separate adjudicated file before evaluation")
    elif not args.allow_single_annotator:
        raise ValueError("Supply independent secondary annotations or explicitly accept the limitation")
    if args.adjudicated:
        if not args.secondary:
            raise ValueError("Adjudication requires the original independent annotations")
        gold = pd.read_csv(ROOT / args.adjudicated, keep_default_na=False)
        validated_annotations(sample, gold)
        before = primary.set_index("news_id")[["label", "language"]]
        after = gold.set_index("news_id")[["label", "language"]].reindex(before.index)
        changed = set(before.index[(before != after).any(axis=1)])
        if not changed <= set(agreement["disagreement_ids"]):
            raise ValueError("Adjudication changed labels outside the recorded disagreements")
        inputs.append(ROOT / args.adjudicated)
    subset = sample[sample.partition == args.partition]
    subset_gold = gold[gold.news_id.isin(subset.news_id)]
    # No model loading or inference until all human-input gates have passed.
    torch.set_num_threads(4)
    tokenizer, model = load_finbert()
    predictions = predict_variants(subset, tokenizer, model)
    metrics = human_metrics(subset, subset_gold, predictions)
    output, output_manifest = create_run(ROOT, args.output, {
        "model_id": MODEL_ID, "revision": REVISION, "partition": args.partition,
        "sample_run": str(args.sample_run), "annotation_files_sha256": {str(p): sha256(p) for p in inputs}})
    output_manifest["evaluation_status"] = "Human sentiment classification, not financial evaluation"
    predictions.to_csv(output / "predictions.csv", index=False)
    write_json(output / "quality.json", metrics)
    write_json(output / "annotator_agreement.json", agreement)
    finish_run(output, output_manifest, inputs, ROOT)
    print(output)


if __name__ == "__main__":
    main()
