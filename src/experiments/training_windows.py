from __future__ import annotations

import pandas as pd

from src.models.temporal_validation import purged_train


def windowed_train(dataset, validation_start, years=None):
    if years not in (None, 3, 5):
        raise ValueError("Ventana permitida: todo el pasado, tres o cinco anos")
    start = pd.Timestamp(validation_start)
    train = purged_train(dataset, start)
    if years is not None:
        train = train.loc[pd.to_datetime(train.Date) >= start - pd.DateOffset(years=years)].copy()
    if train.empty:
        raise ValueError("Entrenamiento vacio tras ventana y purga")
    return train


def training_bounds(train, validation, years):
    return {"window_years": "all" if years is None else years,
            "train_start": train.Date.min(), "train_end": train.Date.max(),
            "train_target_end": train.target_end.max(), "train_rows": len(train),
            "valid_start": validation.Date.min(), "valid_end": validation.Date.max(),
            "valid_rows": len(validation)}
