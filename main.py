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

Candidate exploration (new, optional step 18 -- see README "Candidate
Exploration" section): generates, evaluates, and ranks many VIRTUAL
heater-setpoint candidates with the already-trained surrogate. It never
retrains any model. Runs by default as the final pipeline step; skip it
with --skip-candidate-exploration, or run it standalone against the
already-persisted surrogate (no retraining, much faster) with
--candidates-only. See `python3 main.py --help`.
"""
from __future__ import annotations

import argparse
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
from src.explainability import plot_shap_summary, SHAP_AVAILABLE
from src.optimization import optimize_setpoints
from src.candidate_exploration import ExplorationConfig, run_candidate_exploration

RANDOM_STATE = 42
ALL_FEATURES = INPUT_FEATURES + ENGINEERED_FEATURES

# Zhang & Cheng (2022) reported CV results, for direct comparison (their Figure 2 /
# Section 3.1 text). They modeled P (average interface position) and D (deflection)
# only, with Linear Regression (LR) and Random Forest (RF), 5-fold CV, pooled over
# all 96 out-of-fold predictions -- the same statistic pooled_cv_metrics computes here.
ZHANG_CHENG_2022_BASELINE = pd.DataFrame([
    {"paper_model": "Model 1 (LR)", "paper_target": "P (avg. position)", "paper_R2": 0.95, "paper_MAE_mm": 1.04},
    {"paper_model": "Model 2 (RF)", "paper_target": "P (avg. position)", "paper_R2": 0.95, "paper_MAE_mm": 0.95},
    {"paper_model": "Model 3 (LR)", "paper_target": "D (deflection)", "paper_R2": 0.32, "paper_MAE_mm": 0.43},
    {"paper_model": "Model 4 (RF)", "paper_target": "D (deflection)", "paper_R2": 0.78, "paper_MAE_mm": 0.28},
])


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bridgman crystal-growth ML surrogate pipeline. By default runs the "
                     "full training pipeline (steps 1-17) and then candidate exploration "
                     "(step 18) as an optional final step."
    )
    p.add_argument("--skip-candidate-exploration", action="store_true",
                    help="Run the full training pipeline but skip candidate exploration at the end.")
    p.add_argument("--candidates-only", action="store_true",
                    help="Skip the full training pipeline entirely and run ONLY candidate "
                         "exploration against the already-persisted surrogate at "
                         "models/fast_surrogate.joblib (fast -- no retraining). Fails with a "
                         "clear error if that file does not exist yet.")
    p.add_argument("--n-candidates", type=int, default=1000,
                    help="Number of virtual candidates to generate (default: 1000).")
    p.add_argument("--candidate-seed", type=int, default=42,
                    help="Random seed for candidate generation (default: 42).")
    p.add_argument("--candidate-sampling", choices=["uniform", "lhs"], default="uniform",
                    help="Sampling strategy for candidate T1..T5 (default: uniform).")
    p.add_argument("--candidate-objective", choices=["flat_interface", "low_deflection", "multi_objective"],
                    default="multi_objective",
                    help="Which objective framing to record for exported candidates (default: multi_objective).")
    return p.parse_args(argv)


def run_candidates_only(args: argparse.Namespace):
    """Fast path: load the already-persisted surrogate and run candidate
    exploration only, without re-running the full training pipeline."""
    import os
    section("Candidates-only mode: loading persisted surrogate (no retraining)")
    surrogate_path = "models/fast_surrogate.joblib"
    if not os.path.exists(surrogate_path):
        raise FileNotFoundError(
            f"{surrogate_path} not found. --candidates-only requires a previously-trained "
            "surrogate -- run `python3 main.py` (without --candidates-only) at least once first."
        )
    surrogate = FastSurrogate.load(surrogate_path)
    print(f"Loaded surrogate: targets={surrogate.targets}, features={surrogate.feature_names}")
    _run_candidate_exploration_step(surrogate, args)


def _run_candidate_exploration_step(surrogate, args: argparse.Namespace):
    section("STEP 18: Candidate exploration (screening tool, complements src/optimization.py)")
    print("Generating, evaluating, and ranking virtual heater-setpoint candidates with the\n"
          "ALREADY-TRAINED surrogate -- no model is retrained in this step. This complements,\n"
          "and does not replace, the single-best-point inverse-design search in STEP 13's demo\n"
          "above (src/optimization.py). See README 'Candidate Exploration' section for the full\n"
          "methodology, limitations, and the difference between the two tools.")
    config = ExplorationConfig(
        n_samples=args.n_candidates,
        random_state=args.candidate_seed,
        sampling_strategy=args.candidate_sampling,
        objective=args.candidate_objective,
    )
    candidates = run_candidate_exploration(surrogate, config, output_dir="results",
                                            plots_dir="plots/candidate_exploration")
    counts = candidates["support_category"].value_counts().to_dict()
    print(f"Support category breakdown: {counts}")
    print("REMINDER: no candidate above is a confirmed CrysMAS result, a confirmed optimum, "
          "or a validated furnace instruction -- see results/candidate_exploration_summary.json "
          "and README 'Candidate Exploration' for the required caveats before acting on any of them.")
    return candidates


def main(args: argparse.Namespace | None = None):
    args = args or parse_args([])
    if args.candidates_only:
        run_candidates_only(args)
        return
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

    # ------------------------------------------------------------------- pooled-CV vs Zhang & Cheng (2022)
    section("Pooled-CV comparison against Zhang & Cheng (2022) reported results")
    print("Zhang & Cheng (2022) appear to report a POOLED-CV statistic (all out-of-fold\n"
          "predictions from 5-fold CV combined into one R2/MAE), not a mean-of-folds\n"
          "statistic (their Figure 2 plots all 96 out-of-fold predictions in one scatter\n"
          "with a single R2/MAE annotation). cv_df already includes pooled_MAE/pooled_RMSE/\n"
          "pooled_R2, computed the same way, on the same data, for direct comparison:")
    print(ZHANG_CHENG_2022_BASELINE.to_string(index=False))
    rf_d = cv_df[(cv_df["model"] == "RandomForest") & (cv_df["target"] == "D")]
    comparison_rows = [
        {
            "source": "Zhang & Cheng (2022)", "model": prow["paper_model"],
            "target": prow["paper_target"], "pooled_R2": prow["paper_R2"],
            "pooled_MAE_mm": prow["paper_MAE_mm"],
        }
        for _, prow in ZHANG_CHENG_2022_BASELINE.iterrows()
    ]
    if not rf_d.empty:
        row = rf_d.iloc[0]
        print("\nThis project, RandomForest, D (deflection), same pooled-CV statistic:")
        print(f"  pooled_R2  = {row['pooled_R2']:.3f}   (paper's RF/D: R2=0.78)")
        print(f"  pooled_MAE = {row['pooled_MAE']*1000:.3f} mm   (paper's RF/D: MAE=0.28 mm)")
        print(f"  [for reference] mean-of-folds R2 = {row['R2_mean']:.3f} +/- {row['R2_std']:.3f}")
        print("This project does not model P directly (P is the algebraic average of Lc\n"
              "and Le and was excluded from modeling per user confirmation -- see Section 0\n"
              "of the README), so there is no direct P-row comparison; Lc and Le are each\n"
              "individually harder to predict than their average P, which is expected since\n"
              "averaging two noisy predictions reduces variance.")
        comparison_rows.append({
            "source": "This project", "model": "RandomForest", "target": "D (deflection)",
            "pooled_R2": round(float(row["pooled_R2"]), 3),
            "pooled_MAE_mm": round(float(row["pooled_MAE"]) * 1000, 3),
        })
    pd.DataFrame(comparison_rows).to_csv("results/zhang_cheng_2022_comparison.csv", index=False)
    print("\nSaved to results/zhang_cheng_2022_comparison.csv")

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

    section("STEP 16b: SHAP explainability (RandomForest, cf. Zhang & Cheng 2022 Fig. 3)")
    plot_shap_summary(fitted, X_test, ALL_FEATURES, TARGETS, "plots")

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

    # ------------------------------------------------------------------- inverse design demo
    section("Inverse-design demo: optimize furnace set-points for a target outcome\n"
            "(new contribution vs. Zhang & Cheng 2022's manual, non-systematic heuristic)")
    target_outcome = {"D": 0.0}
    print(f"Target: {target_outcome}  (a flat/undeflected interface, D = Lc-Le = 0, "
          "i.e. minimal radial thermal stress at the growth interface)")
    opt_result = optimize_setpoints(surrogate, target_outcome)
    print(f"Recommended set-points (K): "
          f"{ {k: round(v, 2) for k, v in opt_result.setpoints.items()} }")
    print(f"Predicted outcome at these set-points: "
          f"{ {k: round(v, 6) for k, v in opt_result.predictions.items()} }")
    if opt_result.uncertainty:
        print(f"Predictive uncertainty (std) at optimum: "
              f"{ {k: round(v, 6) for k, v in opt_result.uncertainty.items()} }")
    print(f"Converged: {opt_result.converged}  |  Objective value: {opt_result.objective_value:.6e}")
    if opt_result.warnings:
        print("Warnings:", opt_result.warnings)
    pd.DataFrame([{
        "target_D_m": 0.0,
        **{f"setpoint_{k}_K": v for k, v in opt_result.setpoints.items()},
        **{f"predicted_{k}_m": v for k, v in opt_result.predictions.items()},
        "converged": opt_result.converged,
    }]).to_csv("results/optimization_demo.csv", index=False)
    print("Saved to results/optimization_demo.csv")
    print("NOTE: this is a surrogate-driven recommendation for exploring the CrysMAS\n"
          "response surface, not a validated furnace control instruction -- re-running\n"
          "CrysMAS (or a physical trial) at these set-points is needed before acting on\n"
          "it. See src/surrogate.py and src/optimization.py docstrings for the full\n"
          "'what this is / is not' framing.")

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

    # ------------------------------------------------------------------- 18 (optional)
    if args.skip_candidate_exploration:
        section("STEP 18: Candidate exploration SKIPPED (--skip-candidate-exploration)")
    else:
        _run_candidate_exploration_step(surrogate, args)

    section("Pipeline complete. See results/, plots/, and models/ for all outputs.")


if __name__ == "__main__":
    main(parse_args())
