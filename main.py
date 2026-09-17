"""
main.py
=======
End-to-end pipeline for the Bridgman crystal-growth ML surrogate project,
run in the exact order specified for this project:

 1. Inspect the uploaded dataset
 2. Verify the target variables (Lc, Le, D)
 3. Analyze and preprocess the dataset
 4. Engineer valid features
 5. Split the data
 6. Build preprocessing pipelines
 7. Implement Random Forest, XGBoost, GPR
 8. Implement additional baselines
 9. 5-fold cross-validation
10. Calculate MAE, RMSE, R2
11. Compare all models
12. Select the best model per target (from CV results only)
13. Train the final surrogate model
14. Save the trained model and preprocessing pipeline
15. Fast prediction functionality (demoed)
16. Research-quality plots and tables (saved to plots/ and results/)
17. Workflow documented in README.md

Run with:  python3 main.py
Reproducibility: a single RANDOM_STATE (42) is used everywhere -- data
split, KFold shuffling, and every stochastic model.
"""
from __future__ import annotations

import json
import warnings

import pandas as pd

from src.data_loader import (
    DATASET_CITATION,
    INPUT_FEATURES,
    TARGET_DESCRIPTIONS,
    TARGETS,
    load_raw,
    verify_identities,
)
from src.feature_engineering import ENGINEERED_FEATURES, add_engineered_features, multicollinearity_report
from src.preprocessing import data_quality_report, split_data
from src.models import build_models, build_multioutput_models, tune_tree_models, XGBOOST_AVAILABLE
from src.cross_validation import run_full_cv, run_multioutput_cv
from src.evaluation import (
    fit_and_evaluate_test,
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_feature_importance,
    plot_gpr_uncertainty,
    plot_residuals,
)
from src.model_comparison import build_comparison_table, select_best_model_per_target
from src.surrogate import FastSurrogate

RANDOM_STATE = 42
ALL_FEATURES = INPUT_FEATURES + ENGINEERED_FEATURES


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    # ---------------------------------------------------------------- 1, 2
    section("STEP 1-2: Load raw dataset and verify targets")
    df_raw = load_raw("data/Table.csv")
    print(f"Loaded data/Table.csv: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    print(f"Source (cite in any paper/report using this data): {DATASET_CITATION}")
    print("SIMULATION data (CrysMAS finite-volume, 2D axis-symmetric, BaBrCl in a "
          "5-heater vertical Bridgman furnace) -- not physical experimental measurements.")
    identity_check = verify_identities(df_raw)
    print("Target verification (see src/data_loader.py docstring for full detail):")
    for t in TARGETS:
        print(f"  - {t}: {TARGET_DESCRIPTIONS[t]}")
    print(f"  - P (m) excluded from modeling (per user confirmation): "
          f"P = (Lc+Le)/2 exactly, max abs error {identity_check['max_abs_error_P_identity']:.2e}")
    print(f"  - D relabeled from 'Crystal Diameter' to 'interface deflection (Lc-Le)': "
          f"max abs error vs Lc-Le {identity_check['max_abs_error_D_identity']:.2e}")

    # ------------------------------------------------------------------- 3
    section("STEP 3: Data quality / preprocessing analysis")
    quality = data_quality_report(df_raw)
    print(json.dumps({k: v for k, v in quality.items() if k != "outlier_screen"}, indent=2, default=str))
    print("\nOutlier screen (IQR rule, NOT removed -- see preprocessing.py docstring):")
    print(json.dumps(quality["outlier_screen"], indent=2))

    # ------------------------------------------------------------------- 4
    section("STEP 4: Feature engineering")
    df = add_engineered_features(df_raw)
    print(f"Engineered features added: {ENGINEERED_FEATURES}")
    print("Justification for each is documented in src/feature_engineering.py")
    print("\nFeature multicollinearity (raw + engineered):")
    print(multicollinearity_report(df, ALL_FEATURES))

    # ------------------------------------------------------------------- 5
    section("STEP 5: Train / test split")
    X_train, X_test, y_train, y_test = split_data(df, ALL_FEATURES, TARGETS, test_size=0.20, random_state=RANDOM_STATE)
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows "
          f"(held out from ALL CV and hyperparameter tuning below)")
    print("LIMITATION: with only 96 total rows, the 20% test split is ~19 rows -- "
          "test-set metrics below have high variance and should be read as indicative, "
          "not as a precise generalization estimate. This is stated explicitly in the README.")

    # ------------------------------------------------------------------- 6, 7, 8
    section("STEP 6-8: Build preprocessing pipelines + models (RF, XGBoost/GPR + baselines)")
    if not XGBOOST_AVAILABLE:
        print("NOTE: xgboost is not installed in this sandbox (no PyPI network access). "
              "Using sklearn.GradientBoostingRegressor as an explicitly documented substitute "
              "for the 'XGBoost' model everywhere in this run. See src/models.py docstring.")
    print("Tuning RandomForest and the XGBoost-family model per target (GridSearchCV, cv=3, on TRAIN only)...")
    tuned_params_by_target = {}
    for target in TARGETS:
        tuned_params_by_target[target] = tune_tree_models(X_train, y_train[target], random_state=RANDOM_STATE)
        print(f"  {target}: {tuned_params_by_target[target]}")
    # use the union tuning result from the first target's search as the single fixed
    # configuration for the fair, same-folds 5-fold comparison across ALL targets
    # (documented simplification -- see models.py docstring on tuning approach)
    fixed_tuned_params = tuned_params_by_target[TARGETS[0]]
    models = build_models(fixed_tuned_params)
    print(f"\nModels built: {list(models.keys())}")

    # ------------------------------------------------------------------- 9, 10
    section("STEP 9-10: 5-fold cross-validation (MAE, RMSE, R2) -- TRAIN split only")
    cv_df = run_full_cv(models, X_train, y_train, TARGETS)
    pd.set_option("display.width", 140)
    print(cv_df.round(6).to_string(index=False))
    cv_df.to_csv("results/cv_results_independent.csv", index=False)

    section("STEP 9-10b: Multi-output vs. independent comparison")
    multi_models = build_multioutput_models(fixed_tuned_params)
    multi_cv_df = run_multioutput_cv(multi_models, X_train, y_train[TARGETS])
    print("Joint multi-output model CV (average across all 3 targets simultaneously):")
    print(multi_cv_df.round(6).to_string(index=False))

    indep_avg = cv_df.groupby("model")[["MAE_mean", "RMSE_mean", "R2_mean"]].mean().reset_index()
    print("\nAverage of the INDEPENDENT per-target models' CV metrics, for comparison:")
    print(indep_avg.round(6).to_string(index=False))
    multi_cv_df.to_csv("results/cv_results_multioutput.csv", index=False)
    indep_avg.to_csv("results/cv_results_independent_avg_across_targets.csv", index=False)

    # ------------------------------------------------------------------- held-out test evaluation
    section("Held-out TEST SET evaluation (fit on train, evaluate once on test -- NOT used for model selection)")
    test_df, fitted = fit_and_evaluate_test(models, X_train, y_train, X_test, y_test, TARGETS)
    print(test_df.round(6).to_string(index=False))
    test_df.to_csv("results/test_set_results.csv", index=False)

    # ------------------------------------------------------------------- 11
    section("STEP 11: Full model comparison table")
    comparison = build_comparison_table(cv_df, test_df)
    print(comparison.to_string(index=False))
    comparison.to_csv("results/model_comparison_table.csv", index=False)

    # ------------------------------------------------------------------- 12
    section("STEP 12: Best model selection (from CV results ONLY, not test set)")
    best_df = select_best_model_per_target(cv_df)
    print(best_df.to_string(index=False))
    best_df.to_csv("results/best_model_per_target.csv", index=False)

    # ------------------------------------------------------------------- 16 (plots, before final refit so the
    #                                                                       plots reflect the same train/test fit)
    section("STEP 16: Generating research-quality plots")
    plot_actual_vs_predicted(fitted, X_test, y_test, TARGETS, "plots")
    plot_residuals(fitted, X_test, y_test, TARGETS, "plots")
    plot_error_distribution(fitted, X_test, y_test, TARGETS, "plots")
    plot_feature_importance(fitted, ALL_FEATURES, TARGETS, "plots", X_test=X_test, y_test=y_test)
    plot_gpr_uncertainty(fitted, X_test, y_test, TARGETS, "plots")
    print("Plots written to plots/")

    # ------------------------------------------------------------------- 13, 14
    section("STEP 13-14: Train final surrogate (selected model per target, refit on ALL data) and persist")
    best_model_names = dict(zip(best_df["target"], best_df["selected_model"]))
    print(f"Selected model per target (by CV rank): {best_model_names}")
    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)
    # rebuild fresh (unfitted) copies so the surrogate isn't sharing fitted state with the CV/test run.
    # NOTE: FastSurrogate.fit() also clones defensively, since two targets can select the
    # same model type (as happened here -- GaussianProcess won for all three targets) and
    # would otherwise share, and silently overwrite, the same fitted pipeline object.
    fresh_models = build_models(fixed_tuned_params)
    per_target_pipelines = {t: fresh_models[best_model_names[t]] for t in TARGETS}
    surrogate = FastSurrogate.fit(per_target_pipelines, X_all, y_all)
    surrogate.save("models/fast_surrogate.joblib")
    print("Surrogate trained on all 96 rows (train+test) and saved to models/fast_surrogate.joblib")

    # ------------------------------------------------------------------- 15
    section("STEP 15: Fast prediction functionality (demo)")
    demo_X = X_all.iloc[:5]
    result = surrogate.predict(demo_X)
    print("Predictions for 5 sample inputs:")
    print(result["predictions"].round(6).to_string())
    if result["uncertainty"] is not None:
        print("\nUncertainty (std) where available (GPR targets only):")
        print(result["uncertainty"].round(6).to_string())
    print(f"\nInference time: {result['inference_time_s']*1000:.4f} ms total, "
          f"{result['inference_time_per_sample_ms']:.4f} ms/sample")
    if result["warnings"]:
        print("Validation warnings:", result["warnings"])
    else:
        print("No input/output validation warnings for this demo batch.")

    section("Pipeline complete. See results/, plots/, and models/ for all outputs.")


if __name__ == "__main__":
    main()
