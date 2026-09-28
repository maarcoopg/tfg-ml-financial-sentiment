"""Compare FinBERT with a frozen AI reference, never with claimed human gold."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.experiments.artifacts import ROOT, artifact_dir, create_run, finish_run, sha256, write_json
from src.nlp.sentiment_data import LABELS, validated_annotations
from src.nlp.sentiment_metrics import reference_metrics


def load_ai_review(review_dir, root=ROOT):
    metadata = json.loads((review_dir / "provenance.json").read_text(encoding="utf-8"))
    if metadata["annotation_kind"] != "ai":
        raise ValueError("This evaluator requires an explicit AI reference")
    source = root / metadata["source"]
    if sha256(source) != metadata["source_sha256"]:
        raise ValueError("Annotation source differs from the frozen row mapping")
    original = pd.read_csv(source, keep_default_na=False)
    review = pd.read_csv(review_dir / "review.tsv", sep="\t", keep_default_na=False)
    if len(original) != metadata["rows"] or review.row.tolist() != list(range(1, len(original) + 1)):
        raise ValueError("Review must cover every source row in its frozen order")
    if review.reason.str.strip().eq("").any():
        raise ValueError("Every AI judgment needs a reason")
    annotations = original.assign(label=review.label, reason=review.reason,
        language=[metadata.get("language_overrides", {}).get(str(i), "en") for i in review.row],
        annotation_kind="ai", annotator_id=metadata["annotator_id"], reviewed_at=metadata["reviewed_at"])
    validated_annotations(original, annotations, reference_kind="ai")
    return annotations, metadata


def compare_systems(sample, annotations, predictions):
    validated_annotations(sample, annotations, reference_kind="ai")
    # Reuse the strict variant/coverage gates before joining either system.
    reference_metrics(sample, annotations, predictions, reference_kind="ai")
    wide = predictions.pivot(index="news_id", columns="variant", values="label")
    wide["alpha_vantage"] = sample.set_index("news_id").sentiment_label
    if not wide.alpha_vantage.isin(LABELS).all():
        raise ValueError("Unknown Alpha Vantage labels; no silent conversion")
    wide["constant_positive"] = "positive"
    wide["constant_neutral"] = "neutral"
    joined = annotations.set_index("news_id").join(wide, validate="one_to_one")
    eligible = joined[(joined.language == "en") & joined.label.isin(LABELS)]
    results = []
    for partition, partition_rows in eligible.groupby("partition"):
        for ticker in ["all", *sorted(sample.ticker.unique())]:
            company = partition_rows if ticker == "all" else partition_rows[partition_rows.ticker == ticker]
            for panel, panel_rows in [("all_eligible", company),
                    ("paired_context", company[company.company_context != "abstain"])]:
                for system in wide.columns:
                    usable = panel_rows[panel_rows[system] != "abstain"]
                    row = dict(partition=partition, ticker=ticker, panel=panel, system=system,
                        eligible=len(panel_rows), scored=len(usable),
                        coverage=len(usable) / len(panel_rows) if len(panel_rows) else None)
                    if not usable.empty:
                        report = classification_report(usable.label, usable[system], labels=list(LABELS),
                                                       output_dict=True, zero_division=0)
                        row.update(agreement=float(accuracy_score(usable.label, usable[system])),
                            macro_f1=report["macro avg"]["f1-score"], report=report,
                            confusion=confusion_matrix(usable.label, usable[system], labels=list(LABELS)).tolist())
                    results.append(row)
    return {"reference_kind": "ai", "not_human_accuracy": True,
            "class_order": list(LABELS), "results": results}


def markdown_review(sample, annotations, predictions):
    wide = predictions.pivot(index="news_id", columns="variant", values="label")
    wide["alpha_vantage"] = sample.set_index("news_id").sentiment_label
    rows = annotations.join(wide, on="news_id", validate="one_to_one")
    lines = ["# Valoración de las 400 noticias", "",
        "Referencia elaborada por el asistente de IA, no por una persona. No constituye verdad de referencia.",
        "Las etiquetas se fijaron antes de consultar estas predicciones. Se conserva la exposición previa declarada en el protocolo.",
        "`insufficient` significa que el texto no permite atribuir sentimiento a la empresa; no equivale a neutral.", ""]
    for i, row in enumerate(rows.itertuples(), 1):
        lines += [f"## {i}. {row.ticker}: {row.title}", "", row.summary or "Sin resumen.", "",
            f"**Valoración IA:** `{row.label}`. {row.reason}", "",
            f"**FinBERT, titular y resumen:** `{row.full_text}`. **Alpha Vantage:** `{row.alpha_vantage}`.",
            f"Partición: `{row.partition}`. Idioma: `{row.language}`. ID: `{row.news_id}`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-run", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, review = ROOT / args.sample_run, ROOT / args.review
    annotations, metadata = load_ai_review(review)
    sample_path = artifact_dir(run, "data") / "sample.csv"
    previous = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if sha256(sample_path) != previous["output_sha256"][sample_path.relative_to(ROOT).as_posix()]:
        raise ValueError("Sample no longer matches its original manifest")
    sample = pd.read_csv(sample_path, keep_default_na=False)
    validated_annotations(sample, annotations, reference_kind="ai")
    output, manifest = create_run(ROOT, args.output, {
        "reference_kind": "ai", "human_validation": False, "sample_run": str(args.sample_run),
        "review": str(args.review), "annotation_provenance": metadata,
        "partitions": ["development", "evaluation"], "threads": 4, "seed": 50})
    manifest["evaluation_status"] = "Agreement against one AI reference; not human quality or financial prediction"
    manifest["study_status"] = "ai_reference_comparison_human_validation_unavailable"
    # Load optional model dependencies only after annotation and snapshot checks.
    import torch
    from huggingface_hub import hf_hub_download
    from src.nlp.evaluate_sentiment import predict_variants
    from src.nlp.finbert import CACHE, MODEL_ID, REVISION, load_finbert
    torch.set_num_threads(4)
    torch.manual_seed(50)
    tokenizer, model = load_finbert()
    manifest["config"].update(model_id=MODEL_ID, revision=REVISION)
    manifest["versions"].update({name: importlib.metadata.version(name) for name in
        ["torch", "transformers", "captum", "tokenizers", "huggingface-hub"]})
    manifest["model_weights_sha256"] = sha256(Path(hf_hub_download(MODEL_ID, "pytorch_model.bin",
        revision=REVISION, cache_dir=str(CACHE), local_files_only=True)))
    shutil.copy2(ROOT / "requirements-finbert.txt", artifact_dir(output, "source") / "requirements-finbert.txt")
    manifest["finbert_requirements_sha256"] = sha256(ROOT / "requirements-finbert.txt")
    print("Frozen AI annotations validated; running inference on both partitions", flush=True)
    predictions = predict_variants(sample, tokenizer, model)
    predictions.to_csv(output / "predictions.csv", index=False)
    write_json(output / "annotations.json", annotations.to_dict(orient="records"))
    write_json(output / "comparison.json", compare_systems(sample, annotations, predictions))
    write_json(output / "reference_metrics.json", {partition: reference_metrics(
        rows, annotations[annotations.news_id.isin(rows.news_id)],
        predictions[predictions.news_id.isin(rows.news_id)], reference_kind="ai")
        for partition, rows in sample.groupby("partition")})
    (output / "valoraciones.md").write_text(markdown_review(sample, annotations, predictions), encoding="utf-8")
    summary = {"annotation_kind": "ai", "human_validation": False,
        "labels": annotations.label.value_counts().to_dict(),
        "language": annotations.language.value_counts().to_dict(),
        "by_partition_company": annotations.groupby(["partition", "ticker", "label"]).size().rename("n").reset_index().to_dict(orient="records"),
        "review_input_sha256": sha256(review / "review.tsv"),
        "reserved_panel_predictions_now_computed": True}
    write_json(output / "summary.json", summary)
    finish_run(output, manifest, [sample_path, ROOT / metadata["source"], review / "review.tsv", review / "provenance.json"], ROOT)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
