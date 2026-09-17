"""
06_ml_evaluation.py
====================
ML evaluation protocol comparing ORIGINAL-ONLY training vs.
SYNTHETIC-AUGMENTED training, always evaluated on real original data only
(never on synthetic rows), per the project's explicit anti-leakage
requirement.

Two evaluation layers, both required by the brief:

1. Repeated cross-validation (RepeatedKFold, 5 splits x 5 repeats = 25
   folds, varying random seeds) over the ORIGINAL train+val pool (n=81;
   original_test is NEVER included here). Within each fold, synthetic
   augmentation (when used) adds rows ONLY to that fold's training
   portion; the held-out portion is always real original data. This
   directly measures whether augmentation helps or hurts on unseen real
   data, fold by fold, so results are reported as mean +/- std across
   folds, not a single number.
2. A single, final, untouched-original-test-set check: fit once on the
   full original train+val pool (n=81) -- with and without synthetic
   augmentation -- and evaluate once on original_test.csv (n=15), which
   has never been seen by the generator, by any CV fold, or by any model
   fit before this final step.

Models: LinearRegression, Ridge, SVR, RandomForest, HistGradientBoosting,
GaussianProcess, and XGBoost ONLY IF the real xgboost package is
installed (checked at runtime; if absent, it is skipped and this is
printed and recorded -- no substitute is used, per this task's explicit
"only if installed" instruction, which differs from this project's main
pipeline elsewhere).

Each target (Lc, Le, D) is modeled independently (single-output), matching
this project's established convention; this is noted explicitly rather
than silently assumed.

For GaussianProcess (the only uncertainty-aware model here), 95%
prediction intervals are computed on the final original_test check and
their empirical coverage is reported.

Run: python3 synthetic/06_ml_evaluation.py
Outputs:
  synthetic/results/cv_results.csv
  synthetic/results/final_test_results.csv
  synthetic/results/gpr_prediction_intervals.csv
  synthetic/06_MODEL_EVALUATION_REPORT.md (written by 07_write_reports.py)
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
SYN_DIR = Path(__file__).resolve().parent
DATA_DIR = SYN_DIR / "data"
RESULTS_DIR = SYN_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_COLS = ["T1 (K)", "T2 (K)", "T3 (K)", "T4 (K)", "T5 (K)"]
TARGETS = ["Lc (m)", "Le (m)", "D (m)"]

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print(f"xgboost {xgb.__version__} detected -- included in the model comparison.")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("xgboost is NOT installed in this environment. Per the task's explicit "
          "'only if the real library is installed' instruction, XGBoost is SKIPPED "
          "entirely (no substitute model is used in this evaluation).")


def gpr_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    )


def build_models() -> dict:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=SEED),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.001),
        "RandomForest": RandomForestRegressor(random_state=SEED, n_estimators=300, max_depth=8, min_samples_leaf=2),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=SEED, max_depth=4, min_samples_leaf=5, max_iter=200),
        "GaussianProcess": GaussianProcessRegressor(kernel=gpr_kernel(), random_state=SEED, normalize_y=True, n_restarts_optimizer=2),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(random_state=SEED, n_estimators=200, max_depth=3, learning_rate=0.05, verbosity=0)
    return {name: Pipeline([("scaler", StandardScaler()), ("model", m)]) for name, m in models.items()}


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def load_all():
    train = pd.read_csv(DATA_DIR / "original_train.csv")
    val = pd.read_csv(DATA_DIR / "original_val.csv")
    test = pd.read_csv(DATA_DIR / "original_test.csv")
    synthetic = pd.read_csv(SYN_DIR / "synthetic_crystal_growth_dataset.csv")
    cv_pool = pd.concat([train, val], ignore_index=True)
    return cv_pool, test, synthetic


CV_SYNTH_AUGMENT_N = 300  # A fixed-size (seeded) subsample of the 2000-row synthetic set is
# used for augmentation INSIDE the repeated-CV loop, purely for computational tractability:
# GaussianProcessRegressor is O(n^3) and refitting it on the full 2067-row augmented set 25
# times (once per fold) x 3 targets is not tractable here. The FINAL, single untouched-test
# check below (run_final_test) uses the FULL 2000-row synthetic set, so the headline
# augmented-vs-original-only comparison is not affected by this CV-only simplification. This
# is stated explicitly rather than left implicit.


def run_cv(cv_pool: pd.DataFrame, synthetic: pd.DataFrame, n_splits=5, n_repeats=5) -> pd.DataFrame:
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    synth_cv_subsample = synthetic.sample(n=min(CV_SYNTH_AUGMENT_N, len(synthetic)), random_state=SEED).reset_index(drop=True)
    rows = []
    fold_id = 0
    for train_idx, test_idx in rkf.split(cv_pool):
        fold_id += 1
        fold_train = cv_pool.iloc[train_idx]
        fold_test = cv_pool.iloc[test_idx]

        for target in TARGETS:
            X_test_fold = fold_test[INPUT_COLS]
            y_test_fold = fold_test[target]

            for regime, X_train_fold, y_train_fold in [
                ("original_only", fold_train[INPUT_COLS], fold_train[target]),
                ("synthetic_augmented",
                 pd.concat([fold_train[INPUT_COLS], synth_cv_subsample[INPUT_COLS]], ignore_index=True),
                 pd.concat([fold_train[target], synth_cv_subsample[target]], ignore_index=True)),
            ]:
                for model_name, pipe_template in build_models().items():
                    pipe = pipe_template
                    t0 = time.perf_counter()
                    pipe.fit(X_train_fold, y_train_fold)
                    fit_time = time.perf_counter() - t0
                    y_pred = pipe.predict(X_test_fold)
                    m = metrics(y_test_fold, y_pred)
                    rows.append({
                        "fold": fold_id, "target": target, "regime": regime, "model": model_name,
                        "n_train": len(X_train_fold), "n_test_fold": len(X_test_fold),
                        "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"], "fit_time_s": fit_time,
                    })
        if fold_id % 5 == 0:
            print(f"  ... completed {fold_id}/{n_splits * n_repeats} CV folds")
    return pd.DataFrame(rows)


def run_final_test(cv_pool: pd.DataFrame, test: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    interval_rows = []
    for target in TARGETS:
        X_test = test[INPUT_COLS]
        y_test = test[target]

        for regime, X_train, y_train in [
            ("original_only", cv_pool[INPUT_COLS], cv_pool[target]),
            ("synthetic_augmented",
             pd.concat([cv_pool[INPUT_COLS], synthetic[INPUT_COLS]], ignore_index=True),
             pd.concat([cv_pool[target], synthetic[target]], ignore_index=True)),
        ]:
            for model_name, pipe in build_models().items():
                t0 = time.perf_counter()
                pipe.fit(X_train, y_train)
                fit_time = time.perf_counter() - t0
                model_obj = pipe.named_steps["model"]

                if model_name == "GaussianProcess":
                    Xs = pipe.named_steps["scaler"].transform(X_test)
                    y_pred, y_std = model_obj.predict(Xs, return_std=True)
                    lo, hi = y_pred - 1.96 * y_std, y_pred + 1.96 * y_std
                    covered = ((y_test.values >= lo) & (y_test.values <= hi)).sum()
                    interval_rows.append({
                        "target": target, "regime": regime,
                        "n_test": len(y_test), "n_covered_95pct_interval": int(covered),
                        "empirical_coverage": covered / len(y_test),
                        "mean_interval_halfwidth": float(np.mean(1.96 * y_std)),
                    })
                else:
                    y_pred = pipe.predict(X_test)

                m = metrics(y_test, y_pred)
                rows.append({
                    "target": target, "regime": regime, "model": model_name,
                    "n_train": len(X_train), "n_test": len(X_test),
                    "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"], "fit_time_s": fit_time,
                })
    return pd.DataFrame(rows), pd.DataFrame(interval_rows)


def main():
    cv_pool, test, synthetic = load_all()
    print(f"CV pool (original train+val): n={len(cv_pool)}")
    print(f"Untouched original test: n={len(test)}")
    print(f"Synthetic augmentation set: n={len(synthetic)}")

    print("\nRunning repeated CV (5 splits x 5 repeats = 25 folds) ...")
    cv_df = run_cv(cv_pool, synthetic)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

    cv_summary = cv_df.groupby(["target", "regime", "model"]).agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    ).reset_index()
    cv_summary.to_csv(RESULTS_DIR / "cv_results_summary.csv", index=False)

    print("\nRunning final check on untouched original_test.csv ...")
    final_df, interval_df = run_final_test(cv_pool, test, synthetic)
    final_df.to_csv(RESULTS_DIR / "final_test_results.csv", index=False)
    interval_df.to_csv(RESULTS_DIR / "gpr_prediction_intervals.csv", index=False)

    print("\n=== CV summary (mean +/- std across 25 folds) ===")
    print(cv_summary.to_string(index=False))
    print("\n=== Final original_test results (n=15, never seen before this step) ===")
    print(final_df.to_string(index=False))
    print("\n=== GPR 95% prediction interval empirical coverage (original_test) ===")
    print(interval_df.to_string(index=False))

    print(f"\nXGBoost included: {XGBOOST_AVAILABLE}")
    print("All outputs written to synthetic/results/.")


if __name__ == "__main__":
    main()
