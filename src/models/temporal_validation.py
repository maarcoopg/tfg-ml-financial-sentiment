from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def validate_panel(df: pd.DataFrame) -> None:
    if df.empty or df[["ticker", "Date", "target"]].isna().any().any():
        raise ValueError("Panel vacio o identificadores/objetivos nulos")
    if df.duplicated(["ticker", "Date"]).any():
        raise ValueError("Duplicados ticker-fecha")
    if not df["target"].isin([0, 1]).all():
        raise ValueError("El objetivo debe ser binario")


def label_end_dates(df: pd.DataFrame) -> pd.Series:
    if "target_end" in df:
        ends = pd.to_datetime(df["target_end"], errors="raise")
        if ends.isna().any() or (ends <= pd.to_datetime(df["Date"])).any():
            raise ValueError("Horizonte de etiqueta no valido")
        return ends
    # Legacy one-session datasets omit the horizon: omit their last date too.
    dates = pd.Index(pd.to_datetime(df["Date"]).unique()).sort_values()
    mapping = pd.Series(dates[1:], index=dates[:-1])
    return pd.to_datetime(df["Date"]).map(mapping)


def purged_train(df: pd.DataFrame, validation_start: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(validation_start)
    return df[(pd.to_datetime(df["Date"]) < start) & (label_end_dates(df) < start)].copy()


def expanding_date_splits(df: pd.DataFrame, n_splits: int = 3):
    if n_splits < 1:
        raise ValueError("n_splits debe ser positivo")
    validate_panel(df)
    dates = pd.Series(sorted(pd.to_datetime(df["Date"]).unique()))
    fold_size = len(dates) // (n_splits + 1)
    if fold_size < 2:
        raise ValueError("Fechas insuficientes para validacion con purga")
    folds = []
    for i in range(n_splits):
        start = fold_size * (i + 1)
        end = fold_size * (i + 2) if i < n_splits - 1 else len(dates)
        valid = dates.iloc[start:end]
        train = purged_train(df, valid.min())
        if train.empty:
            raise ValueError("Entrenamiento vacio despues de purgar")
        folds.append((pd.Series(sorted(pd.to_datetime(train["Date"]).unique())), valid))
    return folds


def paired_auc_interval(left: pd.DataFrame, right: pd.DataFrame, block: int = 20,
                        repeats: int = 1000, seed: int = 42) -> dict:
    keys = ["Date", "ticker"]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    validate_panel(left)
    validate_panel(right)
    if not left[keys + ["target"]].equals(right[keys + ["target"]]):
        raise ValueError("Las predicciones pareadas deben tener las mismas filas y objetivos")
    if repeats < 100 or block < 1:
        raise ValueError("Bootstrap requiere al menos 100 replicas y bloque positivo")
    groups = list(left.groupby("Date", sort=True).indices.values())
    n = len(groups)
    if block >= n:
        raise ValueError("El bloque debe ser menor que el numero de fechas")
    y = left["target"].to_numpy()
    a, b = left["probability"].to_numpy(), right["probability"].to_numpy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Probabilidades no finitas")
    rng = np.random.default_rng(seed)
    scores, deltas = [], []
    for _ in range(repeats):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        dates = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = np.concatenate([groups[i] for i in dates])
        if len(np.unique(y[idx])) != 2:
            continue
        score = roc_auc_score(y[idx], a[idx])
        scores.append(score)
        deltas.append(score - roc_auc_score(y[idx], b[idx]))
    if len(scores) < repeats * 0.9:
        raise ValueError("Demasiadas replicas sin ambas clases")
    low, high = np.quantile(scores, [0.025, 0.975])
    delta_low, delta_high = np.quantile(deltas, [0.025, 0.975])
    return {"block_sessions": block, "replicates": len(scores),
            "roc_auc": roc_auc_score(y, a), "auc_low": low, "auc_high": high,
            "auc_delta": roc_auc_score(y, a) - roc_auc_score(y, b),
            "delta_low": delta_low, "delta_high": delta_high}
