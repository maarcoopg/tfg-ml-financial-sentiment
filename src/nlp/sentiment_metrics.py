"""Quality metrics require explicit human labels, not provider sentiment."""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from src.nlp.sentiment_data import LABELS, validated_annotations


def human_metrics(sample, annotations, predictions):
    gold = validated_annotations(sample, annotations)
    if predictions.duplicated(["news_id", "variant"]).any():
        raise ValueError("Duplicate predictions")
    if not set(predictions.news_id) <= set(gold.news_id):
        raise ValueError("Predictions contain unknown IDs")
    if not predictions.label.isin([*LABELS, "abstain"]).all():
        raise ValueError("Unknown model labels")
    if not set(predictions.variant) == {"headline", "full_text", "company_context"}:
        raise ValueError("All predefined variants are required")
    for _, rows in predictions.groupby("variant"):
        if set(rows.news_id) != set(gold.news_id):
            raise ValueError("Every variant must cover the requested panel, including abstentions")
    eligible = gold[(gold.language == "en") & gold.label.isin(LABELS)]
    if eligible.empty:
        raise ValueError("No English examples with a human three-class label")
    joined = predictions.merge(eligible[["news_id", "ticker", "label"]], on="news_id",
                               validate="many_to_one", suffixes=("", "_human"))
    context_ids = set(joined[(joined.variant == "company_context") & (joined.label != "abstain")].news_id)
    rows = []
    for variant, variant_rows in joined.groupby("variant"):
        for panel, panel_rows in [("all_available", variant_rows),
                                  ("paired_context", variant_rows[variant_rows.news_id.isin(context_ids)])]:
            for ticker in ["all", *sorted(gold.ticker.unique())]:
                subset = panel_rows if ticker == "all" else panel_rows[panel_rows.ticker == ticker]
                usable = subset[subset.label != "abstain"]
                item = {"variant": variant, "panel": panel, "ticker": ticker,
                        "eligible": len(subset), "scored": len(usable),
                        "coverage": len(usable) / len(subset) if len(subset) else None}
                if not usable.empty:
                    item["report"] = classification_report(usable.label_human, usable.label,
                        labels=list(LABELS), output_dict=True, zero_division=0)
                    item["confusion"] = confusion_matrix(usable.label_human, usable.label,
                                                         labels=list(LABELS)).tolist()
                rows.append(item)
    return {"status": "human_labels_supplied", "requested": len(gold), "eligible": len(eligible),
            "excluded": len(gold) - len(eligible), "class_order": list(LABELS), "metrics": rows}


def annotator_agreement(sample, primary, secondary):
    first = validated_annotations(sample, primary)
    if secondary.empty or not set(secondary.news_id) <= set(sample.news_id):
        raise ValueError("Invalid secondary annotation IDs")
    subset = sample[sample.news_id.isin(secondary.news_id)]
    second = validated_annotations(subset, secondary)
    paired = first.merge(second[["news_id", "label", "language", "annotator_id"]], on="news_id",
                         validate="one_to_one", suffixes=("_first", "_second"))
    if (paired.annotator_id_first.str.casefold() == paired.annotator_id_second.str.casefold()).any():
        raise ValueError("Double annotation requires independent annotators")
    disagreements = paired.loc[(paired.label_first != paired.label_second) |
                               (paired.language_first != paired.language_second), "news_id"].tolist()
    categories = set(paired.label_first) | set(paired.label_second)
    return {"count": len(paired), "agreement": float((paired.label_first == paired.label_second).mean()),
            "language_agreement": float((paired.language_first == paired.language_second).mean()),
            "kappa": float(cohen_kappa_score(paired.label_first, paired.label_second)) if len(categories) > 1 else None,
            "disagreement_ids": disagreements, "adjudication_required": bool(disagreements)}
