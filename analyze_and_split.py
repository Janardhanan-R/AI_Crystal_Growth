"""
Post-processing for the synthetic crystal-growth dataset:
  - summary statistics (describe)
  - numeric correlation matrix + heatmap PNG
  - a couple of relationship scatter plots for the methodology report
  - ML-ready 70/15/15 stratified train/val/test split (by Crystal_Quality_Category)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "crystal_growth_synthetic_dataset.csv"
df = pd.read_csv(DATA_PATH)

ID_COLS = ["Record_ID", "Data_Source"]
CATEGORICAL_FEATURES = ["Dopant_Type", "Crucible_Material"]
TARGET_COLS = [
    "Crystal_Quality_Score", "Crystal_Quality_Category", "Crack_Probability",
    "Dislocation_Density_cm2", "Oxygen_Concentration_ppm", "Crystal_Diameter_Uniformity_Percent",
]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS + TARGET_COLS]

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number])
summary = numeric_df.describe().T
summary["missing"] = df[numeric_df.columns].isna().sum()
summary.to_csv("summary_statistics.csv")

cat_summary_rows = []
for c in CATEGORICAL_FEATURES + ["Crystal_Quality_Category", "Data_Source"]:
    vc = df[c].value_counts()
    for level, count in vc.items():
        cat_summary_rows.append({"column": c, "level": level, "count": int(count), "pct": round(count / len(df) * 100, 2)})
pd.DataFrame(cat_summary_rows).to_csv("categorical_summary.csv", index=False)

# ---------------------------------------------------------------------------
# Correlation matrix + heatmap
# ---------------------------------------------------------------------------
corr = numeric_df.corr(numeric_only=True)
corr.to_csv("correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(15, 13))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=7)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Pearson correlation")
ax.set_title("Correlation matrix - crystal growth synthetic dataset", fontsize=13, pad=14)
fig.tight_layout()
fig.savefig("correlation_heatmap.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Relationship illustration scatter plots (for the methodology report)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
pairs = [
    ("Stability_Index", "Crystal_Quality_Score", "Stability Index", "Quality Score"),
    ("Temperature_Error_C", "Crack_Probability", "Temperature Error (deg C)", "Crack Probability"),
    ("Cooling_Rate_C_per_min", "Dislocation_Density_cm2", "Cooling Rate (deg C/min)", "Dislocation Density (cm^-2)"),
    ("Translation_Speed_mm_per_hr", "Crystal_Length_Grown_mm", "Translation Speed (mm/hr)", "Crystal Length Grown (mm)"),
]
colors = {"Excellent": "#2E7D32", "Good": "#66BB6A", "Average": "#FBC02D", "Poor": "#C62828"}
for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flat, pairs):
    sub = df.sample(1500, random_state=0)
    c = sub["Crystal_Quality_Category"].map(colors)
    ax.scatter(sub[xcol], sub[ycol], c=c, s=8, alpha=0.5, linewidths=0)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{ylabel} vs {xlabel}", fontsize=10)
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=v, label=k, markersize=7) for k, v in colors.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig("relationship_scatterplots.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Train / validation / test split (stratified by quality category)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(123)
anchor_row = df[df["Data_Source"] == "Experimental"]
synth = df[df["Data_Source"] == "Synthetic"].copy()

train_parts, val_parts, test_parts = [], [], []
for cat, group in synth.groupby("Crystal_Quality_Category"):
    idx = group.sample(frac=1.0, random_state=42).index
    n = len(idx)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    train_parts.append(synth.loc[train_idx])
    val_parts.append(synth.loc[val_idx])
    test_parts.append(synth.loc[test_idx])

train_df = pd.concat([anchor_row] + train_parts, ignore_index=True)  # anchor always in training
val_df = pd.concat(val_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

train_df.to_csv("crystal_growth_train.csv", index=False)
val_df.to_csv("crystal_growth_val.csv", index=False)
test_df.to_csv("crystal_growth_test.csv", index=False)

split_summary = pd.DataFrame({
    "split": ["train", "val", "test"],
    "n_rows": [len(train_df), len(val_df), len(test_df)],
    "pct_of_total": [round(len(train_df) / len(df) * 100, 2), round(len(val_df) / len(df) * 100, 2), round(len(test_df) / len(df) * 100, 2)],
})
split_summary.to_csv("split_summary.csv", index=False)

split_by_cat = pd.concat([
    train_df["Crystal_Quality_Category"].value_counts().rename("train"),
    val_df["Crystal_Quality_Category"].value_counts().rename("val"),
    test_df["Crystal_Quality_Category"].value_counts().rename("test"),
], axis=1).fillna(0).astype(int)
split_by_cat.to_csv("split_by_category.csv")

print("FEATURE_COLS:", FEATURE_COLS)
print("TARGET_COLS:", TARGET_COLS)
print(split_summary)
print(split_by_cat)
print("\nDone.")
