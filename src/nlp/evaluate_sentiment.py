"""Run development-only sentiment diagnostics; leave human evaluation pending."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.experiments.artifacts import artifact_dir, create_run, finish_run, sha256, write_json
from src.nlp.finbert import CACHE, MODEL_ID, REVISION, ROOT, load_finbert
from src.nlp.sentiment_behavior import PROBES, infer_texts, integrated_attribution
from src.nlp.sentiment_data import CUTOFF, annotation_templates, prepare_news, sample_news


def predict_variants(sample, tokenizer, model):
    requests = []
    for row in sample.itertuples():
        for variant, text in [("headline", row.title), ("full_text", row.text), ("company_context", row.context)]:
            requests.append({"news_id": row.news_id, "variant": variant, "input_text": text})
    unique = list(dict.fromkeys(r["input_text"] for r in requests if r["input_text"].strip()))
    cache = dict(zip(unique, infer_texts(tokenizer, model, unique)))
    return pd.DataFrame([{**r, **cache.get(r["input_text"], {"label": "abstain"})} for r in requests])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--news", type=Path, default=Path("data/processed/news/financial_news_with_sentiment.csv"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(50)
    news_path = ROOT / args.news
    original = pd.read_csv(news_path, keep_default_na=False)
    data = prepare_news(original)
    sample = sample_news(data)
    tokenizer, model = load_finbert()
    output, manifest = create_run(ROOT, args.output, {
        "model_id": MODEL_ID, "revision": REVISION, "cutoff_utc": CUTOFF,
        "sample_per_ticker_partition": 50, "seed": 50, "inference_partition": "development",
        "device": "cpu", "threads": args.threads, "human_evaluation": "pending_annotations"})
    manifest["evaluation_status"] = "Development diagnostics only; human sentiment evaluation pending; no market training"
    source = artifact_dir(output, "source")
    shutil.copy2(ROOT / "requirements-finbert.txt", source / "requirements-finbert.txt")
    manifest["finbert_requirements_sha256"] = sha256(ROOT / "requirements-finbert.txt")
    manifest["versions"].update({name: importlib.metadata.version(name) for name in
                               ["torch", "transformers", "captum", "tokenizers", "huggingface-hub"]})
    from huggingface_hub import hf_hub_download
    manifest["model_weights_sha256"] = sha256(Path(hf_hub_download(MODEL_ID, "pytorch_model.bin",
                                   revision=REVISION, cache_dir=str(CACHE), local_files_only=True)))
    local = artifact_dir(output, "data")
    local.mkdir(parents=True)
    sample.to_csv(local / "sample.csv", index=False)
    first, second = annotation_templates(sample)
    first.to_csv(output / "annotation_primary.csv", index=False, encoding="utf-8-sig")
    second.to_csv(output / "annotation_secondary.csv", index=False, encoding="utf-8-sig")
    sample[["news_id", "group_id", "partition", "ticker", "published_at", "multi_company"]].to_csv(
        output / "sample_registry.csv", index=False)
    print("Sample and blind annotation templates saved", flush=True)

    unique_texts = list(dict.fromkeys(data.text.tolist()))
    lengths = {}
    for offset in range(0, len(unique_texts), 512):
        chunk = unique_texts[offset:offset + 512]
        ids = tokenizer(chunk, truncation=False, padding=False, verbose=False)["input_ids"]
        lengths.update({text: len(tokens) for text, tokens in zip(chunk, ids)})
    data["token_count"] = data.text.map(lengths)
    data["year"] = data.timestamp.dt.year
    data.groupby(["ticker", "year"]).agg(rows=("news_id", "size"),
        groups=("group_id", "nunique"), multi_company=("multi_company", "sum"),
        context_available=("context", lambda s: s.ne("").sum()),
        token_median=("token_count", "median"), token_max=("token_count", "max"),
        language_flags=("language_review_flag", "sum")).reset_index().to_csv(output / "coverage.csv", index=False)
    data.groupby(["ticker", "source_domain"]).size().rename("rows").reset_index().to_csv(output / "sources.csv", index=False)
    audit = {"rows": len(data), "unique_exact_texts": data.text.nunique(), "groups": data.group_id.nunique(),
             "duplicate_ids": int(data.news_id.duplicated().sum()), "cross_boundary_rows": int(data.cross_boundary.sum()),
             "cross_boundary_groups": int(data.loc[data.cross_boundary, "group_id"].nunique()),
             "empty_titles": int(data.title.str.strip().eq("").sum()),
             "empty_summaries": int(data.summary.str.strip().eq("").sum()),
             "empty_texts": int(data.text.str.strip().eq("").sum()),
             "rows_over_512_tokens": int(data.token_count.gt(512).sum()),
             "token_quantiles": data.token_count.quantile([0.5, 0.9, 0.99, 1]).to_dict(),
             "language_review_flags": int(data.language_review_flag.sum()),
             "language_verified": False, "sample_rows": len(sample), "double_annotation_rows": len(second),
             "sample_context_available": int(sample.context.ne("").sum()),
             "sample_multi_company": int(sample.multi_company.sum()),
             "evaluation_predictions_computed": False}
    write_json(output / "audit.json", audit)
    print("Corpus token audit complete; running development-only inference", flush=True)
    development = sample[sample.partition == "development"]
    predictions = predict_variants(development, tokenizer, model)
    assert set(predictions.news_id) == set(development.news_id)
    predictions.to_csv(output / "development_predictions.csv", index=False)
    paired = predictions.pivot(index="news_id", columns="variant", values="label")
    paired["alpha_vantage"] = development.set_index("news_id").sentiment_label
    agreements = []
    common = paired.company_context != "abstain"
    for a, b in [("headline", "full_text"), ("full_text", "company_context"), ("full_text", "alpha_vantage")]:
        subset = paired.loc[common] if "company_context" in (a, b) else paired
        agreements.append({"first": a, "second": b, "n": len(subset),
                           "agreement": float((subset[a] == subset[b]).mean()),
                           "interpretation": "agreement_between_systems_not_accuracy"})
    write_json(output / "development_agreement.json", agreements)
    behavior = []
    for name, first_text, second_text in PROBES:
        first_result, second_result = infer_texts(tokenizer, model, [first_text, second_text])
        behavior.append({"probe": name, "first_text": first_text, "second_text": second_text,
                         "first": first_result, "second": second_result,
                         "score_difference": first_result["score"] - second_result["score"],
                         "expected_first_higher": None if name == "mixed_entities" else True})
    write_json(output / "controlled_probes.json", behavior)
    attribution_texts = [PROBES[0][1], PROBES[1][2], PROBES[3][2],
                         development.sort_values(["timestamp", "news_id"]).iloc[0].title]
    explanations = []
    for index, text in enumerate(attribution_texts):
        for reference in ["pad", "mask"]:
            result = integrated_attribution(tokenizer, model, text, reference)
            result.update(example=index, kind="synthetic" if index < 3 else "real_development_headline")
            explanations.append(result)
            print(f"Attribution {index + 1}/4 {reference}: converged={result['converged']}, delta={result['delta']:.6f}", flush=True)
    write_json(output / "attributions.json", explanations)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    probes = [r for r in behavior if r["expected_first_higher"] is not None]
    ax.bar([r["probe"] for r in probes], [r["score_difference"] for r in probes], color="#397A66")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="Diferencia de tono: primera frase menos segunda", title="Pares sintéticos: hipótesis de ordenación, no exactitud")
    fig.savefig(output / "controlled_probes.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    for index, ax in enumerate(axes.flat):
        for row in [r for r in explanations if r["example"] == index]:
            ax.plot(row["attributions"], marker=".", label=f"{row['reference']}: delta={row['delta']:.3g}")
        tokens = explanations[index * 2]["tokens"]
        ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
        ax.set_title(f"Ejemplo {index + 1}: logit {explanations[index * 2]['target']}")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.legend(fontsize=8)
    fig.suptitle("Atribuciones por token: dependen de la referencia, no prueban causalidad")
    fig.savefig(output / "attributions.png", dpi=130)
    plt.close(fig)
    write_json(output / "human_evaluation_status.json", {
        "status": "pending_human_annotation", "primary_labels": 0, "secondary_labels": 0,
        "quality_metrics_computed": False, "evaluation_predictions_computed": False,
        "next_step": "Independent human annotation and adjudication before quality evaluation"})
    manifest["study_status"] = "diagnostics_complete_human_evaluation_pending"
    finish_run(output, manifest, [news_path], ROOT)
    print(json.dumps({"output": str(output), "sample": len(sample), "human_evaluation": "pending"}))


if __name__ == "__main__":
    main()
