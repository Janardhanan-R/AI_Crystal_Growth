"""
01_dataset_analysis.py
=======================
Full, honest statistical analysis of the ORIGINAL Table.csv (n=96), before
any synthetic data is generated. No physical interpretation is invented
here: column meanings are imported from src/data_loader.py, which documents
them against the dataset's original Mendeley source record and the
Zhang & Cheng (2022) reference paper (both already verified in this
project). Everything else below (dtypes, ranges, distributions, outliers,
correlations, derived-variable identities) is computed directly from data.

Run: python3 synthetic/01_dataset_analysis.py
Outputs: synthetic/results/dataset_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def main():
    df = pd.read_csv(ROOT / "data" / "Table.csv")
    report: dict = {}

    report["shape"] = {"n_rows": df.shape[0], "n_cols": df.shape[1]}
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}
    report["duplicate_rows"] = int(df.duplicated().sum())

    inputs = ["T1 (K)", "T2 (K)", "T3 (K)", "T4 (K)", "T5 (K)"]
    outputs = ["Lc (m)", "Le (m)", "P (m)", "D (m)"]

    # Ranges / central tendency / spread
    desc = {}
    for c in df.columns:
        col = df[c]
        desc[c] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "median": float(col.median()),
            "std": float(col.std()),
            "skew": float(stats.skew(col)),
            "kurtosis": float(stats.kurtosis(col)),
            "n_unique": int(col.nunique()),
        }
    report["descriptive_stats"] = desc

    # Input factorial-design levels
    report["input_design_levels"] = {c: sorted(df[c].unique().tolist()) for c in inputs}

    # Physical ordering constraint T1>=T2>=T3>=T4>=T5
    order_ok = (df["T1 (K)"] >= df["T2 (K)"]) & (df["T2 (K)"] >= df["T3 (K)"]) & \
               (df["T3 (K)"] >= df["T4 (K)"]) & (df["T4 (K)"] >= df["T5 (K)"])
    report["ordering_constraint_T1_to_T5_monotone_nonincreasing"] = {
        "n_violations": int((~order_ok).sum()),
        "n_satisfied": int(order_ok.sum()),
    }

    # Uniqueness of design points
    n_unique_designs = df[inputs].drop_duplicates().shape[0]
    report["unique_input_combinations"] = {"n_unique": int(n_unique_designs), "n_total": int(df.shape[0])}

    # Derived-variable identities: P = (Lc+Le)/2, D = Lc - Le
    p_pred = (df["Lc (m)"] + df["Le (m)"]) / 2.0
    d_pred = df["Lc (m)"] - df["Le (m)"]
    report["derived_variable_identities"] = {
        "P_equals_mean_Lc_Le": {
            "max_abs_error": float((df["P (m)"] - p_pred).abs().max()),
            "verified": bool((df["P (m)"] - p_pred).abs().max() < 1e-9),
        },
        "D_equals_Lc_minus_Le": {
            "max_abs_error": float((df["D (m)"] - d_pred).abs().max()),
            "verified": bool((df["D (m)"] - d_pred).abs().max() < 1e-9),
        },
    }

    # Correlations
    report["correlation_matrix_pearson"] = df.corr(method="pearson").round(4).to_dict()
    report["correlation_matrix_spearman"] = df.corr(method="spearman").round(4).to_dict()

    # Input-output correlation summary
    io_corr = df.corr().loc[inputs, outputs].round(4)
    report["input_output_correlation"] = io_corr.to_dict()

    # Outliers via IQR rule (1.5x IQR)
    outlier_report = {}
    for c in df.columns:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[c] < lo) | (df[c] > hi)
        outlier_report[c] = {
            "n_outliers_iqr": int(mask.sum()),
            "outlier_indices": df.index[mask].tolist(),
            "lower_fence": float(lo),
            "upper_fence": float(hi),
        }
    report["outliers_iqr_1_5x"] = outlier_report

    # Physically-impossible-value checks (data-driven, not assumption-driven)
    checks = {}
    checks["Lc_negative"] = int((df["Lc (m)"] < 0).sum())
    checks["Le_negative"] = int((df["Le (m)"] < 0).sum())
    checks["D_negative"] = int((df["D (m)"] < 0).sum())  # NOT impossible: D is a signed deflection
    checks["T_any_nonpositive_K"] = int((df[inputs] <= 0).any(axis=1).sum())
    report["physically_flagged_values"] = checks
    report["physical_flag_notes"] = (
        "D (m) = Lc - Le is a SIGNED interface deflection (curvature direction), not a "
        "diameter or distance; negative D values are physically valid and are NOT treated "
        "as outliers or errors, and are NOT removed. Lc/Le negative values would indicate "
        "the melt-crystal interface sits below the reference plane, which is physically "
        "plausible depending on furnace geometry and is not flagged as impossible here "
        "unless it actually occurs (see counts above)."
    )

    # Input feature (in)dependence -- confirms the DOE is a constrained factorial,
    # not i.i.d. sampled inputs, which matters for how synthetic inputs should be drawn.
    input_corr = df[inputs].corr().round(4)
    report["input_feature_correlation"] = input_corr.to_dict()

    out_path = RESULTS_DIR / "dataset_analysis.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps({k: report[k] for k in ["shape", "duplicate_rows",
          "ordering_constraint_T1_to_T5_monotone_nonincreasing",
          "unique_input_combinations", "derived_variable_identities",
          "physically_flagged_values"]}, indent=2))


if __name__ == "__main__":
    main()
