"""
Statistical analysis for the CdTe scenario dataset.

IMPORTANT: the "original" dataset is a single record (n=1). Standard deviation,
variance, quartiles, and correlations are mathematically undefined for n=1 and are
reported as "Not computable (n=1)" rather than invented. Only the anchor's own value
is reported as its "mean" (trivially, since mean of one number is that number).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOT_AVAILABLE = "Not available in source dataset"
NOT_COMPUTABLE = "Not computable (n=1)"


def sigfig(x, n=4):
    """Round to n significant figures instead of a fixed decimal count, so very
    small-magnitude columns (e.g. Vacuum_Level_Torr ~ 1e-6) don't collapse to 0.0."""
    if x is None or pd.isna(x):
        return None
    if x == 0:
        return 0.0
    from math import log10, floor
    d = n - int(floor(log10(abs(x)))) - 1
    return round(float(x), d)

df = pd.read_csv("synthetic_CdTe_crystal_growth_5000.csv")
anchor = df[df["Data_Source"] == "Experimental"].iloc[0]
synth = df[df["Data_Source"] != "Experimental"].copy()

# numeric columns that actually have real synthetic values (exclude the 5 all-NOT_AVAILABLE columns)
candidate_numeric = [
    "Furnace_Set_Temperature_C", "Actual_Furnace_Temperature_C", "Temperature_Error_C",
    "Temperature_Gradient_C_per_cm", "Cooling_Rate_C_per_min", "Translation_Speed_mm_per_hr",
    "Growth_Time_hr", "Stage_Position_mm", "Crystal_Length_Grown_mm", "Solidification_Percentage",
    "Chamber_Pressure_Pa", "Vacuum_Level_Torr", "Initial_Charge_Mass_g",
    "Temperature_Variance_Last_10min", "Translation_Speed_Variance", "Average_Temperature_C",
    "dT_dt_C_per_min", "Stability_Index", "Crystal_Quality_Score", "Crack_Probability",
    "Dislocation_Density_cm2", "Oxygen_Concentration_ppm", "Crystal_Diameter_Uniformity_Percent",
]
always_na = ["Heating_Power_W", "Ambient_Temperature_C", "Dopant_Concentration_ppm"]
anchor_only_na = ["Temperature_Variance_Last_10min", "Translation_Speed_Variance", "dT_dt_C_per_min",
                   "Stability_Index", "Crystal_Quality_Score"]

# ---------------------------------------------------------------------------
# original_vs_synthetic_statistics.csv
# ---------------------------------------------------------------------------
rows = []
for col in candidate_numeric:
    orig_val = anchor[col]
    orig_display = NOT_AVAILABLE if orig_val == NOT_AVAILABLE else orig_val
    s = pd.to_numeric(synth[col], errors="coerce").dropna()
    rows.append({
        "column": col,
        "original_n": 1,
        "original_value_single_record": orig_display,
        "original_std": NOT_COMPUTABLE,
        "original_min": orig_display, "original_q1": NOT_COMPUTABLE,
        "original_median": orig_display, "original_q3": NOT_COMPUTABLE,
        "original_max": orig_display,
        "synthetic_n": int(s.shape[0]),
        "synthetic_mean": sigfig(s.mean()) if len(s) else None,
        "synthetic_std": sigfig(s.std()) if len(s) else None,
        "synthetic_min": sigfig(s.min()) if len(s) else None,
        "synthetic_q1": sigfig(s.quantile(0.25)) if len(s) else None,
        "synthetic_median": sigfig(s.median()) if len(s) else None,
        "synthetic_q3": sigfig(s.quantile(0.75)) if len(s) else None,
        "synthetic_max": sigfig(s.max()) if len(s) else None,
    })
for col in always_na:
    rows.append({
        "column": col, "original_n": 1, "original_value_single_record": NOT_AVAILABLE,
        "original_std": NOT_COMPUTABLE, "original_min": NOT_AVAILABLE, "original_q1": NOT_COMPUTABLE,
        "original_median": NOT_AVAILABLE, "original_q3": NOT_COMPUTABLE, "original_max": NOT_AVAILABLE,
        "synthetic_n": 0, "synthetic_mean": NOT_AVAILABLE, "synthetic_std": NOT_AVAILABLE,
        "synthetic_min": NOT_AVAILABLE, "synthetic_q1": NOT_AVAILABLE, "synthetic_median": NOT_AVAILABLE,
        "synthetic_q3": NOT_AVAILABLE, "synthetic_max": NOT_AVAILABLE,
    })
stats_df = pd.DataFrame(rows)
stats_df.to_csv("original_vs_synthetic_statistics.csv", index=False)

# ---------------------------------------------------------------------------
# correlation_comparison.csv (long format: original is not computable for every pair)
# ---------------------------------------------------------------------------
corr_cols = [c for c in candidate_numeric if c not in anchor_only_na]  # columns with real values in BOTH... but anchor has no numeric value for most targets anyway; original corr is n=1 regardless
synth_numeric = synth[candidate_numeric].apply(pd.to_numeric, errors="coerce")
synth_corr = synth_numeric.corr()
synth_corr.to_csv("synthetic_correlation_matrix.csv")

pairs = []
cols_all = candidate_numeric
for i, c1 in enumerate(cols_all):
    for c2 in cols_all[i + 1:]:
        val = synth_corr.loc[c1, c2] if c1 in synth_corr.columns and c2 in synth_corr.columns else None
        pairs.append({
            "variable_1": c1, "variable_2": c2,
            "original_correlation": NOT_COMPUTABLE,
            "synthetic_correlation": round(val, 4) if val is not None and not pd.isna(val) else None,
        })
pd.DataFrame(pairs).to_csv("correlation_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# Correlation heatmap (synthetic only - explicit caption that original doesn't exist)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(synth_corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(synth_corr.columns)))
ax.set_xticklabels(synth_corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(synth_corr.columns)))
ax.set_yticklabels(synth_corr.columns, fontsize=7)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Pearson correlation (synthetic-scenario data only)")
ax.set_title(
    "Synthetic-scenario correlation matrix\n"
    "NOTE: no original correlation matrix exists to compare against (n=1 real record)",
    fontsize=11, pad=14,
)
fig.tight_layout()
fig.savefig("synthetic_correlation_heatmap.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Distribution comparison plots: synthetic histogram + anchor marked as vertical line
# (cannot compare distributions since original has one point)
# ---------------------------------------------------------------------------
plot_cols = [
    "Furnace_Set_Temperature_C", "Temperature_Error_C", "Translation_Speed_mm_per_hr",
    "Growth_Time_hr", "Crack_Probability", "Dislocation_Density_cm2",
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, plot_cols):
    vals = pd.to_numeric(synth[col], errors="coerce").dropna()
    ax.hist(vals, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    anchor_val = anchor[col]
    if anchor_val != NOT_AVAILABLE:
        ax.axvline(float(anchor_val), color="#C44E52", linewidth=2, linestyle="--", label="Anchor (real, n=1)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.95, "no anchor value", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#C44E52")
    ax.set_title(col, fontsize=9)
    ax.tick_params(labelsize=7)
fig.suptitle("Synthetic-scenario distributions with the single real anchor value marked\n(no original distribution exists to overlay - n=1)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("distribution_comparison_plots.png", dpi=150)
plt.close(fig)

print("Analysis complete.")
print(stats_df.head(8).to_string())
