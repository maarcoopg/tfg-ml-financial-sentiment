from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.temporal_validation import validate_panel


def block_weights(n_dates: int, repeats: int = 1000, block: int = 20, seed: int = 42):
    if repeats < 100 or not 0 < block < n_dates:
        raise ValueError("Bootstrap requiere 100 replicas y un bloque menor que el calendario")
    rng = np.random.default_rng(seed)
    weights = np.zeros((repeats, n_dates), dtype=np.int64)
    for i in range(repeats):
        starts = rng.integers(0, n_dates - block + 1, size=int(np.ceil(n_dates / block)))
        selected = np.concatenate([np.arange(start, start + block) for start in starts])[:n_dates]
        weights[i] = np.bincount(selected, minlength=n_dates)
    return weights


def weighted_auc_samples(y, probability, weights):
    """Evaluate bootstrap AUCs by weighted positive/negative pairs, including ties.

    Repeated dates become integer sample weights. Sorting scores once avoids
    sorting the same predictions in every replicate. Tests compare this formula
    with sklearn's ROC AUC and the existing paired bootstrap implementation.
    """
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights)
    if (y.ndim != 1 or probability.shape != y.shape or weights.ndim != 2
            or weights.shape[1] != len(y) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(probability).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("Entradas invalidas para el AUC ponderado")
    order = np.argsort(probability, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(probability[order])) + 1]
    positive = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    wins = (positive * (np.cumsum(negative, axis=1) - 0.5 * negative)).sum(axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    return np.divide(wins, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def panel_intervals(predictions: pd.DataFrame, contrasts: list[tuple[str, str]], repeats=1000):
    keys = ["ticker", "Date", "target", "outer_fold"]
    groups = dict(tuple(predictions.groupby(["approach", "dataset_type", "model_name"])))
    reference = next(iter(groups.values())).sort_values(["ticker", "Date"]).reset_index(drop=True)
    validate_panel(reference)
    dates = pd.Index(sorted(pd.to_datetime(reference.Date).unique()))
    tickers = sorted(reference.ticker.unique())
    weights = block_weights(len(dates), repeats)
    weights_with_original = np.vstack([np.ones((1, len(dates)), dtype=np.int64), weights])
    scores = {}
    for key, frame in groups.items():
        frame = frame.sort_values(["ticker", "Date"]).reset_index(drop=True)
        validate_panel(frame)
        if not frame[keys].equals(reference[keys]):
            raise ValueError("Todas las comparaciones requieren las mismas filas y bloques")
        samples = []
        for ticker in tickers:
            company = frame[frame.ticker == ticker]
            if not pd.Index(pd.to_datetime(company.Date)).equals(dates):
                raise ValueError("Panel incompleto por empresa")
            sample = weighted_auc_samples(company.target, company.probability, weights_with_original)
            scores[key + (ticker,)] = sample
            samples.append(sample)
        scores[key + ("macro",)] = np.mean(samples, axis=0)
    rows = []
    for candidate, baseline in contrasts:
        for approach, variant, model in groups:
            if approach != candidate or model == "dummy":
                continue
            if (baseline, variant, model) not in groups:
                raise ValueError(f"Falta la referencia {baseline}/{variant}/{model}")
            for ticker in tickers + ["macro"]:
                left = scores[(candidate, variant, model, ticker)]
                right = scores[(baseline, variant, model, ticker)]
                valid = np.isfinite(left[1:]) & np.isfinite(right[1:])
                if not np.isfinite(left[0] + right[0]) or valid.sum() < repeats * 0.9:
                    raise ValueError("Demasiadas replicas sin ambas clases")
                delta = left - right
                low, high = np.quantile(left[1:][valid], [0.025, 0.975])
                dl, dh = np.quantile(delta[1:][valid], [0.025, 0.975])
                rows.append(dict(approach=candidate, reference=baseline, dataset_type=variant,
                    model_name=model, ticker=ticker, roc_auc=left[0], reference_auc=right[0],
                    auc_low=low, auc_high=high, auc_delta=delta[0], delta_low=dl, delta_high=dh,
                    block_sessions=20, replicates=int(valid.sum())))
    return pd.DataFrame(rows)
