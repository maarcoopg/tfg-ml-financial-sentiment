from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments.run_review import GRIDS
from src.experiments.training_windows import training_bounds, windowed_train
from src.models.temporal_validation import expanding_date_splits
from src.models.train_models import build_model


ROUND_MODELS = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


def candidates(model, years=None, expanded=False):
    if model not in ROUND_MODELS:
        raise ValueError("Algoritmo no incluido en esta ronda")
    initial = [{"years": years, "params": dict(params)} for params in GRIDS[model]]
    if not expanded:
        return initial
    initial = [{"years": None, "params": dict(params)} for params in GRIDS[model]]
    if model == "logistic_regression":
        return [{"years": y, "params": dict(p)} for y in [None, 3, 5] for p in GRIDS[model]]
    spaces = {
        "random_forest": {"model__n_estimators": [100, 200], "model__max_depth": [3, 4, 6, 8],
                          "model__min_samples_leaf": [5, 10, 20, 40], "model__max_features": ["sqrt", 0.5, 1.0]},
        "hist_gradient_boosting": {"model__max_iter": [100, 200], "model__max_leaf_nodes": [3, 7, 15, 31],
            "model__learning_rate": [0.025, 0.05, 0.1], "model__min_samples_leaf": [10, 20, 40, 80],
            "model__l2_regularization": [0.0, 1.0, 10.0], "model__early_stopping": [False]},
    }
    rng = np.random.default_rng(42)
    seen = {json.dumps(item, sort_keys=True) for item in initial}
    while len(initial) < 20:
        params = {key: values[int(rng.integers(len(values)))] for key, values in spaces[model].items()}
        item = {"years": [None, 3, 5][(len(initial) - 2) % 3], "params": params}
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            initial.append(item)
    return initial


def auc_scores(validation, probability):
    values = validation[["ticker", "target"]].assign(probability=np.asarray(probability))
    if not values.probability.between(0, 1).all() or values.target.nunique() != 2:
        raise ValueError("Probabilidades o clases invalidas")
    if not values.groupby("ticker").target.nunique().eq(2).all():
        raise ValueError("AUC macro requiere ambas clases en cada empresa")
    per_company = [roc_auc_score(frame.target, frame.probability) for _, frame in values.groupby("ticker")]
    return {"pooled_auc": roc_auc_score(values.target, values.probability), "macro_auc": float(np.mean(per_company))}


def choose_candidate(train, columns, model, configurations, metric="pooled_auc", splits=3):
    if metric not in ["pooled_auc", "macro_auc"] or not configurations:
        raise ValueError("Criterio o candidatos invalidos")
    folds = expanding_date_splits(train, splits)
    rows, means = [], []
    for config_id, config in enumerate(configurations):
        scores = []
        for fold, (_, dates) in enumerate(folds, start=1):
            fit = windowed_train(train, dates.min(), config["years"])
            valid = train[train.Date.isin(dates)]
            estimator = build_model(model).set_params(**config["params"])
            estimator.fit(fit[columns], fit.target)
            values = auc_scores(valid, estimator.predict_proba(valid[columns])[:, 1])
            scores.append(values[metric])
            rows.append({"config_id": config_id, "inner_fold": fold, "selection_metric": metric,
                "selection_score": values[metric], "params": json.dumps(config["params"], sort_keys=True),
                **values, **training_bounds(fit, valid, config["years"])})
        means.append(float(np.mean(scores)))
    best = int(np.argmax(means))
    return configurations[best], rows, means[best]
