"""
02_residual_diagnostic.py
===========================
Why doesn't D improve? Pulls the actual 5-fold out-of-fold predictions for
the current best model (GaussianProcess) on target D and shows which rows
drive the error. Run from the project root:
    python3 experiments/02_residual_diagnostic.py
See IMPROVE_D_REPORT.md in this folder for the interpretation.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Table.csv")

import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone

from src.data_loader import load_raw, verify_identities, INPUT_FEATURES, TARGETS
from src.feature_engineering import ENGINEERED_FEATURES, add_engineered_features
from src.preprocessing import split_data
from src.models import build_models

df = load_raw(DATA_PATH)
verify_identities(df)
df = add_engineered_features(df)
ALL_FEATURES = INPUT_FEATURES + ENGINEERED_FEATURES
X_train, X_test, y_train, y_test = split_data(df, ALL_FEATURES, TARGETS, test_size=0.20, random_state=42)
y_D = y_train["D"]

pipe = build_models()["GaussianProcess"]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(clone(pipe), X_train, y_D, cv=kf, n_jobs=-1)
resid = (y_D.values - pred)
order = np.argsort(-np.abs(resid))

print(f"n train rows: {len(y_D)}  |  D range: [{y_D.min():.5f}, {y_D.max():.5f}]  std: {y_D.std():.5f}")
print("\nWorst 8 residuals (out of {} CV predictions):".format(len(y_D)))
print(f"{'D_true':>10} {'D_pred':>10} {'resid':>10}  T1..T5")
for i in order[:8]:
    row = X_train.iloc[i]
    print(f"{y_D.values[i]:>10.5f} {pred[i]:>10.5f} {resid[i]:>10.5f}  {row[INPUT_FEATURES].values}")

print(f"\nMean |resid| worst-8 vs rest: {np.abs(resid[order[:8]]).mean():.6f} vs {np.abs(resid[order[8:]]).mean():.6f}")
print(f"Worst-8 rows' share of total squared error: {(resid[order[:8]]**2).sum() / (resid**2).sum():.1%}")
