import numpy as np
import pandas as pd

from src.data.split_train_test import FINANCIAL_FEATURES, SENTIMENT_FEATURES


def feature_variants(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = dataset.sort_values(["ticker", "Date"]).copy()
    base = list(FINANCIAL_FEATURES)
    hybrid = base + list(SENTIMENT_FEATURES)
    tone = [c for c in SENTIMENT_FEATURES if "count" not in c]
    variants = {
        "base": base,
        "news_volume": base + ["news_count", "has_news"],
        "sentiment_only": tone,
        "financial_sentiment": base + tone,
        "hybrid": hybrid,
    }
    for lag in [1, 2, 3]:
        columns = []
        for source in ["sentiment_mean", "news_count"]:
            name = f"{source}_lag_{lag}"
            df[name] = df.groupby("ticker")[source].shift(lag).fillna(0)
            columns.append(name)
        variants[f"lag_{lag}"] = hybrid + columns
    for window in [3, 5]:
        columns = []
        for source in ["sentiment_mean", "news_count"]:
            name = f"{source}_rolling_{window}"
            df[name] = df.groupby("ticker")[source].transform(lambda s: s.rolling(window, min_periods=1).mean())
            columns.append(name)
        variants[f"rolling_{window}"] = hybrid + columns
    variants["weighted_sentiment"] = [c for c in hybrid if c != "sentiment_mean"] + ["sentiment_weighted"]
    relative = []
    for window in [5, 20, 50]:
        name = f"distance_sma_{window}"
        df[name] = df["Adj Close"] / df[f"sma_{window}"] - 1
        relative.append(name)
    for source in ["MACD", "MACD_signal"]:
        name = source + "_relative"
        df[name] = df[source] / df["Adj Close"]
        relative.append(name)
    for source in ["Volume", "news_count"]:
        history = df.groupby("ticker")[source].transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
        name = source + "_relative"
        df[name] = (df[source] / history.replace(0, np.nan)).fillna(0)
        relative.append(name)
    variants["relative_features"] = ["daily_return", "RSI", "volatility_20"] + relative + [c for c in SENTIMENT_FEATURES if c != "news_count"]
    all_features = sorted(set(c for columns in variants.values() for c in columns))
    if not np.isfinite(df[all_features].to_numpy(dtype=float)).all():
        raise ValueError("Variables experimentales no finitas")
    return df.sort_values(["Date", "ticker"]).reset_index(drop=True), variants
