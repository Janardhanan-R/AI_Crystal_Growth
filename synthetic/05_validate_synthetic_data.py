"""
05_validate_synthetic_data.py
==============================
Statistical quality validation of synthetic_crystal_growth_dataset.csv
against the ORIGINAL data. Produces distribution comparison plots,
correlation comparison plots, a pair plot, KS-test and Wasserstein-distance
tables, duplicate/near-duplicate detection, outlier analysis, and a
predictive-relationship comparison (does a model trained on synthetic data
reproduce the same input-output relationship as the original?).

IMPORTANT (per the project's explicit requirement): statistical similarity
here is NOT treated as proof of physical validity. This script and its
report say so explicitly and repeatedly -- it checks whether the synthetic
data LOOKS statistically like the original sample, not whether it is
correct physics.

Run: python3 synthetic/05_validate_synthetic_data.py
Outputs:
  synthetic/results/ks_wasserstein_tests.csv
  synthetic/results/correlation_comparison.csv
  synthetic/plots/distribution_comparison_<col>.png
  synthetic/plots/correlation_comparison.png
  synthetic/plots/pairplot_comparison.png
  synthetic/05_VALIDATION_REPORT.md
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
SYN_DIR = Path(__file__).resolve().parent
DATA_DIR = SYN_DIR / "data"
PLOTS_DIR = SYN_DIR / "plots"
RESULTS_DIR = SYN_DIR / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_COLS = ["T1 (K)", "T2 (K)", "T3 (K)", "T4 (K)", "T5 (K)"]
OUTPUT_COLS = ["Lc (m)", "Le (m)", "P (m)", "D (m)"]
ALL_COLS = INPUT_COLS + OUTPUT_COLS


def load_data():
    original = pd.read_csv(ROOT / "data" / "Table.csv")
    synthetic = pd.read_csv(SYN_DIR / "synthetic_crystal_growth_dataset.csv")
    return original, synthetic


def ks_and_wasserstein(original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ALL_COLS:
        a, b = original[col].values, synthetic[col].values
        ks_stat, ks_p = stats.ks_2samp(a, b)
        wass = stats.wasserstein_distance(a, b)
        wass_norm = wass / (a.std() if a.std() > 0 else 1.0)  # normalized by original std, for comparability across columns
        rows.append({
            "column": col,
            "KS_statistic": round(ks_stat, 4),
            "KS_pvalue": round(ks_p, 4),
            "KS_reject_H0_similar_dist_at_0.05": bool(ks_p < 0.05),
            "wasserstein_distance": round(wass, 6),
            "wasserstein_distance_normalized_by_orig_std": round(wass_norm, 4),
        })
    return pd.DataFrame(rows)


def distribution_plots(original: pd.DataFrame, synthetic: pd.DataFrame):
    for col in ALL_COLS:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(original[col], bins=15, density=True, alpha=0.5, label=f"Original (n={len(original)})", color="C0")
        ax.hist(synthetic[col], bins=40, density=True, alpha=0.4, label=f"Synthetic (n={len(synthetic)})", color="C1")
        try:
            kde_o = stats.gaussian_kde(original[col])
            kde_s = stats.gaussian_kde(synthetic[col])
            xs = np.linspace(min(original[col].min(), synthetic[col].min()),
                              max(original[col].max(), synthetic[col].max()), 300)
            ax.plot(xs, kde_o(xs), color="C0", lw=2)
            ax.plot(xs, kde_s(xs), color="C1", lw=2)
        except np.linalg.LinAlgError:
            pass  # degenerate KDE (e.g. near-constant column) -- histogram alone is fine
        ax.set_title(f"Distribution comparison: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        safe_name = col.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(PLOTS_DIR / f"distribution_comparison_{safe_name}.png", dpi=150)
        plt.close(fig)


def correlation_comparison_plot(original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    corr_o = original[ALL_COLS].corr()
    corr_s = synthetic[ALL_COLS].corr()
    diff = (corr_o - corr_s).abs()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], corr_o, "Original correlation", "coolwarm", -1, 1),
        (axes[1], corr_s, "Synthetic correlation", "coolwarm", -1, 1),
        (axes[2], diff, "Absolute difference", "viridis", 0, diff.values.max()),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(ALL_COLS)))
        ax.set_yticks(range(len(ALL_COLS)))
        ax.set_xticklabels(ALL_COLS, rotation=90, fontsize=8)
        ax.set_yticklabels(ALL_COLS, fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_comparison.png", dpi=150)
    plt.close(fig)

    out = diff.reset_index().melt(id_vars="index", var_name="col2", value_name="abs_corr_diff")
    out = out.rename(columns={"index": "col1"})
    out = out[out["col1"] != out["col2"]].sort_values("abs_corr_diff", ascending=False)
    return out


def pairplot_comparison(original: pd.DataFrame, synthetic: pd.DataFrame, sample_n: int = 300):
    cols = ["Lc (m)", "Le (m)", "D (m)"]
    rng = np.random.default_rng(SEED)
    syn_sample = synthetic.sample(n=min(sample_n, len(synthetic)), random_state=SEED)

    fig, axes = plt.subplots(len(cols), len(cols), figsize=(10, 10))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(original[ci], bins=12, density=True, alpha=0.6, color="C0", label="Original")
                ax.hist(syn_sample[ci], bins=25, density=True, alpha=0.4, color="C1", label="Synthetic")
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
            else:
                ax.scatter(syn_sample[cj], syn_sample[ci], s=6, alpha=0.25, color="C1", label="Synthetic")
                ax.scatter(original[cj], original[ci], s=14, alpha=0.9, color="C0", edgecolor="k", linewidth=0.3, label="Original")
            if i == len(cols) - 1:
                ax.set_xlabel(cj, fontsize=8)
            if j == 0:
                ax.set_ylabel(ci, fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Pair plot: original (n={len(original)}) vs. synthetic sample (n={len(syn_sample)})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pairplot_comparison.png", dpi=150)
    plt.close(fig)


def duplicate_and_outlier_analysis(original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    exact_dup_inputs = synthetic[INPUT_COLS].duplicated().sum()
    ref = original[INPUT_COLS].values
    ref_std = ref.std(axis=0)
    ref_std[ref_std == 0] = 1.0
    syn_scaled = synthetic[INPUT_COLS].values / ref_std
    ref_scaled = ref / ref_std
    min_dists = np.array([np.sqrt(((ref_scaled - row) ** 2).sum(axis=1)).min() for row in syn_scaled])
    near_dup_005 = int((min_dists < 0.05).sum())
    near_dup_02 = int((min_dists < 0.2).sum())

    outlier_counts = {}
    for col in OUTPUT_COLS:
        q1, q3 = original[col].quantile(0.25), original[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = int(((synthetic[col] < lo) | (synthetic[col] > hi)).sum())

    return {
        "exact_duplicate_input_rows_within_synthetic": int(exact_dup_inputs),
        "synthetic_rows_within_0.05_normalized_distance_of_an_original_row": near_dup_005,
        "synthetic_rows_within_0.2_normalized_distance_of_an_original_row": near_dup_02,
        "n_synthetic": len(synthetic),
        "outliers_vs_original_iqr_fences": outlier_counts,
    }


def predictive_relationship_comparison(original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Fit a simple, identical linear model (T1..T5 -> Lc/Le/D) separately
    on original data and on synthetic data, and compare fitted
    coefficients. Similar coefficient SIGNS and RELATIVE MAGNITUDES suggest
    the synthetic data preserved the direction/relative strength of the
    real input-output relationship; this is a coarse, transparent check
    (deliberately not a black-box model), reported honestly as directional
    evidence only, not proof of fidelity."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    rows = []
    for target in ["Lc (m)", "Le (m)", "D (m)"]:
        lr_o = LinearRegression().fit(StandardScaler().fit_transform(original[INPUT_COLS]), original[target])
        lr_s = LinearRegression().fit(StandardScaler().fit_transform(synthetic[INPUT_COLS]), synthetic[target])
        for i, feat in enumerate(INPUT_COLS):
            rows.append({
                "target": target, "feature": feat,
                "coef_original": round(lr_o.coef_[i], 5),
                "coef_synthetic": round(lr_s.coef_[i], 5),
                "same_sign": bool(np.sign(lr_o.coef_[i]) == np.sign(lr_s.coef_[i])),
            })
    return pd.DataFrame(rows)


def main():
    original, synthetic = load_data()

    ks_df = ks_and_wasserstein(original, synthetic)
    ks_df.to_csv(RESULTS_DIR / "ks_wasserstein_tests.csv", index=False)

    distribution_plots(original, synthetic)
    corr_diff_df = correlation_comparison_plot(original, synthetic)
    corr_diff_df.to_csv(RESULTS_DIR / "correlation_comparison.csv", index=False)
    pairplot_comparison(original, synthetic)

    dup_report = duplicate_and_outlier_analysis(original, synthetic)
    pred_df = predictive_relationship_comparison(original, synthetic)
    pred_df.to_csv(RESULTS_DIR / "predictive_relationship_comparison.csv", index=False)

    print("KS / Wasserstein results:")
    print(ks_df.to_string(index=False))
    print("\nDuplicate/near-duplicate + outlier report:")
    for k, v in dup_report.items():
        print(f"  {k}: {v}")
    print(f"\nSign agreement (original vs. synthetic linear coefficients): "
          f"{pred_df['same_sign'].sum()}/{len(pred_df)}")
    print("\nOutputs written to synthetic/results/ and synthetic/plots/.")


if __name__ == "__main__":
    main()
