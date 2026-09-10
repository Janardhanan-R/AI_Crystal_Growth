import pandas as pd
import numpy as np

df = pd.read_csv("synthetic_CdTe_crystal_growth_5000.csv")
anchor_row = df[df["Data_Source"] == "Experimental"]
synth = df[df["Data_Source"] != "Experimental"].copy()

train_parts, val_parts, test_parts = [], [], []
for cat, group in synth.groupby("Crystal_Quality_Category"):
    idx = group.sample(frac=1.0, random_state=42).index
    n = len(idx)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_parts.append(synth.loc[idx[:n_train]])
    val_parts.append(synth.loc[idx[n_train:n_train + n_val]])
    test_parts.append(synth.loc[idx[n_train + n_val:]])

train_df = pd.concat([anchor_row] + train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

# near-duplicate check: since all synthetic rows come from independent continuous
# perturbation draws, exact or near-exact duplicates across splits are not expected -
# verify directly rather than assume.
def sig(d):
    numeric_cols = [c for c in d.columns if c not in
                    ("Record_ID", "Data_Source", "Original_Record_ID", "Extrapolated", "Synthetic_Generation_Method")]
    return d[numeric_cols].astype(str).agg("|".join, axis=1)

train_sig, val_sig, test_sig = sig(train_df), sig(val_df), sig(test_df)
overlap_tv = set(train_sig) & set(val_sig)
overlap_tt = set(train_sig) & set(test_sig)
overlap_vt = set(val_sig) & set(test_sig)
print("Exact/near-duplicate overlaps -> train/val:", len(overlap_tv), "train/test:", len(overlap_tt), "val/test:", len(overlap_vt))

train_df.to_csv("cdte_train.csv", index=False)
val_df.to_csv("cdte_val.csv", index=False)
test_df.to_csv("cdte_test.csv", index=False)

split_summary = pd.DataFrame({
    "split": ["train", "val", "test"],
    "n_rows": [len(train_df), len(val_df), len(test_df)],
    "pct_of_total": [round(len(train_df) / len(df) * 100, 2), round(len(val_df) / len(df) * 100, 2), round(len(test_df) / len(df) * 100, 2)],
})
split_by_cat = pd.concat([
    train_df["Crystal_Quality_Category"].value_counts().rename("train"),
    val_df["Crystal_Quality_Category"].value_counts().rename("val"),
    test_df["Crystal_Quality_Category"].value_counts().rename("test"),
], axis=1).fillna(0).astype(int)

split_summary.to_csv("split_summary.csv", index=False)
split_by_cat.to_csv("split_by_category.csv")
print(split_summary)
print(split_by_cat)
