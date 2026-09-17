"""
preprocessing.py
=================
Data-quality checks, leakage checks, and the train/test split + scaling
pipeline used by every model in this project.

Design-of-experiment note
--------------------------
The dataset is a small factorial-style design-of-experiment table (96 rows,
all unique (T1..T5) combinations, no repeated trials, no batches, no time
index). Two consequences for preprocessing:

1. IQR-based outlier REMOVAL is not applied. A handful of rows sit outside
   the 1.5*IQR whiskers for Lc/Le/P/D, but these are the corner points of
   the temperature grid (the most extreme T1-T5 combinations), not
   measurement errors. Removing them would silently delete the most
   informative, most extreme conditions in a deliberately small designed
   experiment -- the opposite of good practice here. They are flagged and
   reported, not dropped.

2. Cross-validation does not need GroupKFold or a time-aware split: there
   are no repeated experiments, batches, or timestamps to leak across
   folds. Plain shuffled KFold is appropriate (implemented in
   cross_validation.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def data_quality_report(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    outliers = {}
    for col in ["Lc", "Le", "P", "D"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        outliers[col] = {
            "n_outliers_iqr_rule": int(mask.sum()),
            "bounds": [float(lo), float(hi)],
            "action": "NOT removed - corner points of the designed "
                      "experiment, not measurement error (see module docstring).",
        }
    report["outlier_screen"] = outliers
    return report


def leakage_check(feature_cols: list[str], target_cols: list[str]) -> None:
    """Refuse to proceed if any target column (or a column deterministically
    derived from a target, i.e. 'P') has ended up in the feature list."""
    banned = set(target_cols) | {"P"}
    leaked = banned & set(feature_cols)
    if leaked:
        raise ValueError(
            f"Data leakage: target-derived column(s) {leaked} found in the "
            "feature set. Features must be built only from T1..T5."
        )


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Hold out a test set BEFORE any cross-validation or hyperparameter
    comparison. With n=96 this leaves ~19 test rows -- reported explicitly
    as a limitation (see README / results, Section: Limitations)."""
    leakage_check(feature_cols, target_cols)
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    return X_train, X_test, y_train, y_test


def make_scaling_pipeline(model) -> Pipeline:
    """Every model is wrapped in the same Pipeline(StandardScaler -> model).
    Scaling is required for Linear/Ridge/SVR/GPR to behave numerically well
    and is harmless (scale-invariant) for the tree ensembles, so applying it
    uniformly keeps every model's pipeline identical and prevents any risk
    of fitting a scaler on anything but the training fold."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])
