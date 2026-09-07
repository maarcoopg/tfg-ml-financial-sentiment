from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_DIR = Path("models/trained")
METADATA_DIR = Path("reports/model_metadata")
OUTPUT_DIR = Path("reports/feature_importance")
FEATURE_CONFIG_FILE = Path("data/processed/model/feature_sets.json")


def load_feature_groups() -> dict[str, set[str]]:
    with FEATURE_CONFIG_FILE.open(encoding="utf-8") as file:
        config = json.load(file)
    return {
        "financial": set(config["financial_features"]),
        "sentiment": set(config["sentiment_features"]),
    }


def feature_group(feature: str, feature_groups: dict[str, set[str]]) -> str:
    if feature in feature_groups["sentiment"]:
        return "sentiment"
    if feature in feature_groups["financial"]:
        return "financial"
    return "other"


def extract_importance(model_file: Path, metadata_file: Path) -> pd.DataFrame | None:
    with metadata_file.open(encoding="utf-8") as file:
        metadata = json.load(file)

    model = joblib.load(model_file)
    estimator = model.named_steps["model"]
    feature_columns = metadata["feature_columns"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "feature_importances"
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
        importance_type = "absolute_coefficient"
    else:
        return None

    feature_groups = load_feature_groups()
    importance = pd.DataFrame(
        {
            "dataset_type": metadata["dataset_type"],
            "model_name": metadata["model_name"],
            "feature": feature_columns,
            "importance": values,
            "importance_type": importance_type,
        }
    )
    importance["feature_group"] = importance["feature"].apply(
        lambda value: feature_group(value, feature_groups)
    )
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1

    return importance


def build_feature_importance() -> pd.DataFrame:
    frames = []
    for metadata_file in sorted(METADATA_DIR.glob("*.json")):
        model_file = MODEL_DIR / f"{metadata_file.stem}.joblib"
        if not model_file.exists():
            continue
        importance = extract_importance(model_file, metadata_file)
        if importance is not None:
            frames.append(importance)

    if not frames:
        raise RuntimeError("No se encontraron modelos con importancia de variables disponible")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance = build_feature_importance()

    all_file = OUTPUT_DIR / "feature_importance.csv"
    top_file = OUTPUT_DIR / "top_10_feature_importance.csv"
    group_file = OUTPUT_DIR / "feature_group_importance.csv"

    top_10 = (
        importance.sort_values(["dataset_type", "model_name", "rank"])
        .groupby(["dataset_type", "model_name"])
        .head(10)
        .reset_index(drop=True)
    )
    group_importance = (
        importance.groupby(["dataset_type", "model_name", "feature_group"], as_index=False)[
            "importance"
        ]
        .sum()
        .sort_values(["dataset_type", "model_name", "feature_group"])
    )

    importance.to_csv(all_file, index=False)
    top_10.to_csv(top_file, index=False)
    group_importance.to_csv(group_file, index=False)

    print(f"Guardado: {all_file} - {len(importance)} filas")
    print(f"Guardado: {top_file} - {len(top_10)} filas")
    print(f"Guardado: {group_file} - {len(group_importance)} filas")
    print(group_importance)


if __name__ == "__main__":
    main()
