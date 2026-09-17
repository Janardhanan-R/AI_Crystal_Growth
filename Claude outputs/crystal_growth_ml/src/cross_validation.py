"""
cross_validation.py
====================
5-fold cross-validation, run identically (same folds, same scoring) for
every model x target combination, for a fair comparison.

CV strategy justification: KFold(n_splits=5, shuffle=True, random_state=42).
GroupKFold / time-aware splitting is not used because the dataset has no
repeated experiments, batches, or time ordering to leak across folds (see
preprocessing.py docstring) -- every row is a distinct (T1..T5) design
point. Preprocessing (StandardScaler) is fit only on each training fold
because it lives inside the sklearn Pipeline handed to cross_validate,
which refits the whole pipeline per fold.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

CV_SPLITS = 5
RANDOM_STATE = 42

SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}


def cross_validate_model(pipeline, X, y_series) -> dict:
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        clone(pipeline), X, y_series, cv=kf, scoring=SCORING,
        return_train_score=False, n_jobs=-1,
    )
    out = {
        "MAE_mean": -cv_results["test_MAE"].mean(),
        "MAE_std": cv_results["test_MAE"].std(),
        "RMSE_mean": -cv_results["test_RMSE"].mean(),
        "RMSE_std": cv_results["test_RMSE"].std(),
        "R2_mean": cv_results["test_R2"].mean(),
        "R2_std": cv_results["test_R2"].std(),
        "fit_time_mean_s": cv_results["fit_time"].mean(),
        "fit_time_std_s": cv_results["fit_time"].std(),
    }
    return out


def run_full_cv(models: dict, X: pd.DataFrame, y: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """models: {model_name: unfitted Pipeline}. Returns a tidy DataFrame,
    one row per (model, target), with CV mean +/- std for MAE/RMSE/R2 and
    mean fit time."""
    rows = []
    for target in targets:
        y_t = y[target]
        for model_name, pipeline in models.items():
            res = cross_validate_model(pipeline, X, y_t)
            rows.append({"model": model_name, "target": target, **res})
    return pd.DataFrame(rows)


def run_multioutput_cv(models: dict, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Same 5-fold CV, but each model predicts all targets in y jointly
    (sklearn's MAE/RMSE/R2 scorers average uniformly across outputs for a
    multi-output y). Used to compare a joint multi-output model against the
    average of the independent per-target models on the SAME folds."""
    rows = []
    for model_name, pipeline in models.items():
        res = cross_validate_model(pipeline, X, y)
        rows.append({"model": model_name, **res})
    return pd.DataFrame(rows)
