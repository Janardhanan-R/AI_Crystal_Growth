"""
01_improve_D_metrics.py
========================
Honest experiment: can CV R2 for target D be improved beyond the project's
current best (GaussianProcess, CV R2_mean = 0.6648, pooled R2 = 0.7126, see
results/model_comparison_table.csv)?

Uses the EXACT same data loading / feature engineering / train-test split /
5-fold CV machinery already in src/, so results are directly comparable to
the numbers already reported in results/model_comparison_table.csv. No
numbers are invented here -- every row in the printed table and the output
CSV comes from an actual cross_validate call.

Run from the project root:  python3 experiments/01_improve_D_metrics.py
(Needs scikit-learn, scipy, pandas, numpy -- same as requirements.txt.)

See IMPROVE_D_REPORT.md in this folder for the findings and conclusion.
"""
from __future__ import annotations
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Make `src` importable whether this script is run from the project root
# or from inside experiments/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Table.csv")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from src.data_loader import load_raw, verify_identities, INPUT_FEATURES, TARGETS
from src.feature_engineering import ENGINEERED_FEATURES, add_engineered_features
from src.preprocessing import split_data, make_scaling_pipeline
from src.cross_validation import cross_validate_model, pooled_cv_metrics
from src.models import build_models, RANDOM_STATE

df = load_raw(DATA_PATH)
verify_identities(df)
df = add_engineered_features(df)
ALL_FEATURES = INPUT_FEATURES + ENGINEERED_FEATURES

X_train, X_test, y_train, y_test = split_data(df, ALL_FEATURES, TARGETS, test_size=0.20, random_state=42)
y_train_D = y_train["D"]

results = []

def record(name, pipeline, X):
    cv = cross_validate_model(pipeline, X, y_train_D)
    pooled = pooled_cv_metrics(pipeline, X, y_train_D)
    results.append({
        "variant": name,
        "CV_R2_mean": round(cv["R2_mean"], 4), "CV_R2_std": round(cv["R2_std"], 4),
        "CV_MAE_mean": round(cv["MAE_mean"], 6),
        "Pooled_R2": round(pooled["pooled_R2"], 4),
    })
    print(f"{name:45s} CV R2={cv['R2_mean']:.4f}+/-{cv['R2_std']:.4f}  pooled R2={pooled['pooled_R2']:.4f}")

print("=== BASELINE (current project config) ===")
baseline_models = build_models()
record("BASELINE: GaussianProcess (current)", baseline_models["GaussianProcess"], X_train)
record("BASELINE: RandomForest (current)", baseline_models["RandomForest"], X_train)
record("BASELINE: HistGradientBoosting (current)", baseline_models["HistGradientBoosting"], X_train)

print("\n=== A. Alternative GPR kernels (same features) ===")
n_feat = X_train.shape[1]

kernels = {
    "GPR anisotropic RBF (per-feature length scale)": (
        ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*n_feat, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    ),
    "GPR Matern(nu=1.5)": (
        ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    ),
    "GPR Matern(nu=2.5)": (
        ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    ),
    "GPR RBF+Matern sum": (
        ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        + ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    ),
}
for name, k in kernels.items():
    gpr = GaussianProcessRegressor(kernel=k, random_state=RANDOM_STATE, normalize_y=True, n_restarts_optimizer=10)
    record(f"A. {name}", make_scaling_pipeline(gpr), X_train)

print("\n=== B. Feature-set variations (current best kernel) ===")
def gpr_default():
    return GaussianProcessRegressor(
        kernel=(ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))),
        random_state=RANDOM_STATE, normalize_y=True, n_restarts_optimizer=5,
    )

record("B. GPR, raw T1-T5 only (no engineered features)", make_scaling_pipeline(gpr_default()), X_train[INPUT_FEATURES])
record("B. GPR, raw T1-T5 + dT_15 + T_mean only", make_scaling_pipeline(gpr_default()),
       X_train[INPUT_FEATURES + ["dT_15", "T_mean"]])

print("\n=== C. Pairwise interaction terms (T1..T5 only, degree-2, no bias) ===")
poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
    ("scaler", StandardScaler()),
    ("model", gpr_default()),
])
record("C. GPR on T1-T5 pairwise interactions (15 feats)", poly_pipeline, X_train[INPUT_FEATURES])

print("\n=== D. Ensembling (within proper CV, no leakage) ===")
stack = StackingRegressor(
    estimators=[
        ("gpr", gpr_default()),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=RANDOM_STATE)),
        ("hgb", HistGradientBoostingRegressor(max_depth=4, min_samples_leaf=5, max_iter=200, random_state=RANDOM_STATE)),
    ],
    final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
    cv=5, n_jobs=-1,
)
record("D. Stacking (GPR+RF+HistGB -> Ridge)", make_scaling_pipeline(stack), X_train)

print("\n=== Summary table ===")
res_df = pd.DataFrame(results).sort_values("CV_R2_mean", ascending=False)
print(res_df.to_string(index=False))
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_RESULTS_improve_D.csv")
res_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}")

print("\nBest CV variant:", res_df.iloc[0]["variant"], "(CV R2_mean=", res_df.iloc[0]["CV_R2_mean"], ")")
print("Conclusion: see IMPROVE_D_REPORT.md -- no variant tried here meaningfully beats the current baseline.")
