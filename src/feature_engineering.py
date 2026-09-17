"""
feature_engineering.py
=======================
Physically-justified derived features built ONLY from the five furnace
zone temperatures T1..T5. No target column is ever used to build a
feature (see preprocessing.leakage_check).

Each engineered feature has an explicit physical justification -- per the
project instructions, feature engineering that cannot be justified this way
is avoided, to limit overfitting risk on a 96-row dataset.

    dT_12 = T1 - T2   Local axial gradient between furnace zones 1-2
    dT_23 = T2 - T3   Local axial gradient between furnace zones 2-3
    dT_34 = T3 - T4   Local axial gradient between furnace zones 3-4
    dT_45 = T4 - T5   Local axial gradient between furnace zones 4-5
    dT_15 = T1 - T5   Total axial temperature drop across the whole
                      furnace (hot zone to cold zone) -- the single most
                      commonly reported control parameter in Bridgman
                      furnace studies, since it governs the overall
                      thermal driving force for growth.
    T_mean = mean(T1..T5)  Overall thermal level of the furnace, distinct
                      from the gradients above (a furnace can have the same
                      total drop at different absolute temperature levels).

These 6 engineered features are added ALONGSIDE the 5 raw temperatures
(11 features total), not in place of them, so tree-based models retain
access to the raw values while linear/kernel models are given the
physically-relevant combinations directly.

A modest feature set is used deliberately: with only 96 rows, adding many
more derived/interaction features would raise strong overfitting and
multicollinearity risk for limited additional benefit.
"""
from __future__ import annotations

import pandas as pd

ENGINEERED_FEATURES = ["dT_12", "dT_23", "dT_34", "dT_45", "dT_15", "T_mean"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dT_12"] = out["T1"] - out["T2"]
    out["dT_23"] = out["T2"] - out["T3"]
    out["dT_34"] = out["T3"] - out["T4"]
    out["dT_45"] = out["T4"] - out["T5"]
    out["dT_15"] = out["T1"] - out["T5"]
    out["T_mean"] = out[["T1", "T2", "T3", "T4", "T5"]].mean(axis=1)
    return out


def multicollinearity_report(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Pairwise correlation among the final feature set, so multicollinearity
    introduced by the engineered features (which are linear combinations of
    T1..T5) is visible and documented rather than hidden."""
    return df[feature_cols].corr().round(3)
