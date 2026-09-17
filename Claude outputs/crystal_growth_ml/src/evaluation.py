"""
evaluation.py
==============
Held-out TEST SET evaluation (separate from the 5-fold CV used for model
selection -- CV never touches this test split, per the "do not use test
data for model selection" requirement), plus the research-quality plots:
actual-vs-predicted, residuals, error distribution, tree feature
importance, and GPR predictive uncertainty.

Color palette: fixed categorical hues from the project's dataviz
reference palette (colorblind-checked ordering), assigned to models in a
FIXED order so a model's color never changes across figures.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_COLORS = {
    "LinearRegression": "#2a78d6",
    "Ridge": "#eb6834",
    "SVR": "#1baf7a",
    "RandomForest": "#eda100",
    "XGBoost": "#e87ba4",
    "HistGradientBoosting": "#008300",
    "GaussianProcess": "#4a3aa7",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.size": 9,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fit_and_evaluate_test(models: dict, X_train, y_train, X_test, y_test, targets: list[str]) -> pd.DataFrame:
    """Fit each model on the training split (only) and evaluate once on the
    held-out test split. Also measures per-sample inference time."""
    rows = []
    fitted = {}
    for target in targets:
        y_tr, y_te = y_train[target], y_test[target]
        for name, pipeline in models.items():
            m = clone(pipeline)
            t0 = time.perf_counter()
            m.fit(X_train, y_tr)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            preds = m.predict(X_test)
            n_infer = max(1, len(X_test))
            infer_time_per_sample = (time.perf_counter() - t0) / n_infer

            rows.append({
                "model": name,
                "target": target,
                "test_MAE": mean_absolute_error(y_te, preds),
                "test_RMSE": mean_squared_error(y_te, preds) ** 0.5,
                "test_R2": r2_score(y_te, preds),
                "train_time_s": train_time,
                "inference_time_per_sample_s": infer_time_per_sample,
            })
            fitted[(name, target)] = m
    return pd.DataFrame(rows), fitted


def plot_actual_vs_predicted(fitted: dict, X_test, y_test, targets: list[str], out_dir: str):
    for target in targets:
        models_for_target = sorted({name for (name, t) in fitted if t == target})
        n = len(models_for_target)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False)
        y_true = y_test[target].values
        lims = [min(y_true.min(), y_true.min()), max(y_true.max(), y_true.max())]
        for i, name in enumerate(models_for_target):
            ax = axes[i // ncols][i % ncols]
            preds = fitted[(name, target)].predict(X_test)
            all_vals = np.concatenate([y_true, preds])
            lo, hi = all_vals.min(), all_vals.max()
            pad = (hi - lo) * 0.08 or 1e-4
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, linewidth=1, linestyle="--")
            ax.scatter(y_true, preds, color=MODEL_COLORS.get(name, INK), s=28, alpha=0.85, edgecolor="white", linewidth=0.4)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"Actual vs. Predicted -- {target} (held-out test set)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{out_dir}/actual_vs_predicted_{target}.png", dpi=150)
        plt.close(fig)


def plot_residuals(fitted: dict, X_test, y_test, targets: list[str], out_dir: str):
    for target in targets:
        models_for_target = sorted({name for (name, t) in fitted if t == target})
        n = len(models_for_target)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), squeeze=False)
        y_true = y_test[target].values
        for i, name in enumerate(models_for_target):
            ax = axes[i // ncols][i % ncols]
            preds = fitted[(name, target)].predict(X_test)
            resid = y_true - preds
            ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
            ax.scatter(preds, resid, color=MODEL_COLORS.get(name, INK), s=28, alpha=0.85, edgecolor="white", linewidth=0.4)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual")
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"Residuals vs. Predicted -- {target} (held-out test set)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{out_dir}/residuals_{target}.png", dpi=150)
        plt.close(fig)


def plot_error_distribution(fitted: dict, X_test, y_test, targets: list[str], out_dir: str):
    for target in targets:
        models_for_target = sorted({name for (name, t) in fitted if t == target})
        fig, ax = plt.subplots(figsize=(7, 4.2))
        y_true = y_test[target].values
        for name in models_for_target:
            preds = fitted[(name, target)].predict(X_test)
            resid = y_true - preds
            ax.hist(resid, bins=10, alpha=0.45, label=name, color=MODEL_COLORS.get(name, INK), edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")
        ax.set_title(f"Error distribution -- {target} (held-out test set)", fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/error_distribution_{target}.png", dpi=150)
        plt.close(fig)


def plot_feature_importance(fitted: dict, feature_names: list[str], targets: list[str], out_dir: str,
                             X_test=None, y_test=None,
                             tree_models=("RandomForest", "XGBoost", "HistGradientBoosting")):
    """Native feature_importances_ where the model exposes it (RandomForest,
    and XGBoost/GradientBoosting); permutation importance on the held-out
    test set otherwise (HistGradientBoostingRegressor has no native
    feature_importances_ attribute in scikit-learn) -- labeled accordingly
    so the two are never confused."""
    for target in targets:
        available = [m for m in tree_models if (m, target) in fitted]
        if not available:
            continue
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), squeeze=False)
        for i, name in enumerate(available):
            ax = axes[0][i]
            pipeline = fitted[(name, target)]
            model = pipeline.named_steps["model"]
            importances = getattr(model, "feature_importances_", None)
            label = "Importance (native)"
            if importances is None and X_test is not None:
                perm = permutation_importance(
                    pipeline, X_test, y_test[target], n_repeats=30, random_state=42, scoring="neg_mean_absolute_error"
                )
                importances = perm.importances_mean
                label = "Importance (permutation, test set)"
            if importances is None:
                ax.axis("off")
                continue
            order = np.argsort(importances)
            ax.barh(np.array(feature_names)[order], np.array(importances)[order], color=MODEL_COLORS.get(name, INK))
            ax.set_title(name, fontsize=9)
            ax.set_xlabel(label, fontsize=8)
        fig.suptitle(f"Feature importance -- {target}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"{out_dir}/feature_importance_{target}.png", dpi=150)
        plt.close(fig)


def plot_gpr_uncertainty(fitted: dict, X_test, y_test, targets: list[str], out_dir: str):
    for target in targets:
        key = ("GaussianProcess", target)
        if key not in fitted:
            continue
        pipeline = fitted[key]
        scaler = pipeline.named_steps["scaler"]
        gpr = pipeline.named_steps["model"]
        X_scaled = scaler.transform(X_test)
        mean, std = gpr.predict(X_scaled, return_std=True)
        y_true = y_test[target].values
        order = np.argsort(y_true)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        x_axis = np.arange(len(y_true))
        ax.errorbar(x_axis, mean[order], yerr=1.96 * std[order], fmt="o", color=MODEL_COLORS["GaussianProcess"],
                    ecolor=MODEL_COLORS["GaussianProcess"], alpha=0.7, capsize=3, markersize=5, label="GPR prediction ± 95% CI")
        ax.scatter(x_axis, y_true[order], color=INK, marker="x", s=45, label="Actual", zorder=5)
        ax.set_xlabel("Test sample (sorted by actual value)")
        ax.set_ylabel(target)
        ax.set_title(f"GPR predictive uncertainty -- {target} (held-out test set)", fontsize=11)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/gpr_uncertainty_{target}.png", dpi=150)
        plt.close(fig)
