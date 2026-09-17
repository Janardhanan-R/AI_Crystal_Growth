"""
model_comparison.py
====================
Builds the final comparison table (CV results + held-out test results side
by side, clearly labeled which is which) and selects the best model per
target using ONLY the cross-validation results (never the test set, per
the "do not select a model using test-set information" requirement). The
held-out test metrics are reported afterward, purely as an independent
final check on the already-selected model.
"""
from __future__ import annotations

import pandas as pd


def build_comparison_table(cv_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    cv = cv_df.copy()
    cv["MAE (CV mean ± std)"] = cv["MAE_mean"].round(6).astype(str) + " ± " + cv["MAE_std"].round(6).astype(str)
    cv["RMSE (CV mean ± std)"] = cv["RMSE_mean"].round(6).astype(str) + " ± " + cv["RMSE_std"].round(6).astype(str)
    cv["R2 (CV mean ± std)"] = cv["R2_mean"].round(4).astype(str) + " ± " + cv["R2_std"].round(4).astype(str)
    cv["CV fit time (s, mean)"] = cv["fit_time_mean_s"].round(4)
    # Pooled-CV columns (see cross_validation.pooled_cv_metrics): a DIFFERENT statistic
    # from the mean-of-folds columns above, reported for comparability with papers (e.g.
    # Zhang & Cheng 2022) that pool out-of-fold predictions before scoring. Never used
    # for model selection -- see select_best_model_per_target, which uses mean-of-folds only.
    has_pooled = {"pooled_MAE", "pooled_RMSE", "pooled_R2"}.issubset(cv.columns)
    if has_pooled:
        cv["Pooled-CV MAE"] = cv["pooled_MAE"].round(6)
        cv["Pooled-CV RMSE"] = cv["pooled_RMSE"].round(6)
        cv["Pooled-CV R2"] = cv["pooled_R2"].round(4)

    test = test_df.copy()
    test["Test MAE"] = test["test_MAE"].round(6)
    test["Test RMSE"] = test["test_RMSE"].round(6)
    test["Test R2"] = test["test_R2"].round(4)
    test["Test train time (s)"] = test["train_time_s"].round(4)
    test["Inference time/sample (ms)"] = (test["inference_time_per_sample_s"] * 1000).round(4)

    merged = cv.merge(test, on=["model", "target"], how="outer")
    cols = [
        "target", "model",
        "MAE (CV mean ± std)", "RMSE (CV mean ± std)", "R2 (CV mean ± std)", "CV fit time (s, mean)",
    ]
    if has_pooled:
        cols += ["Pooled-CV MAE", "Pooled-CV RMSE", "Pooled-CV R2"]
    cols += ["Test MAE", "Test RMSE", "Test R2", "Test train time (s)", "Inference time/sample (ms)"]
    return merged[cols].sort_values(["target", "model"]).reset_index(drop=True)


def select_best_model_per_target(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Rank-based selection using ONLY cross-validation results: for each
    target, rank models by CV MAE, CV RMSE, and CV R2 (each equally
    weighted), and pick the lowest average rank. This avoids picking a
    'winner' from R2 alone, per the project's explicit requirement, and
    avoids any use of the held-out test set for selection."""
    rows = []
    for target, group in cv_df.groupby("target"):
        g = group.copy()
        g["rank_mae"] = g["MAE_mean"].rank(ascending=True)
        g["rank_rmse"] = g["RMSE_mean"].rank(ascending=True)
        g["rank_r2"] = g["R2_mean"].rank(ascending=False)
        g["avg_rank"] = g[["rank_mae", "rank_rmse", "rank_r2"]].mean(axis=1)
        best = g.sort_values(["avg_rank", "fit_time_mean_s"]).iloc[0]
        rows.append({
            "target": target,
            "selected_model": best["model"],
            "avg_rank": round(best["avg_rank"], 2),
            "CV_MAE_mean": round(best["MAE_mean"], 6),
            "CV_RMSE_mean": round(best["RMSE_mean"], 6),
            "CV_R2_mean": round(best["R2_mean"], 4),
            "CV_fit_time_mean_s": round(best["fit_time_mean_s"], 4),
        })
    return pd.DataFrame(rows)
