"""
03_split_original_data.py
==========================
Splits the ORIGINAL 96-row dataset into train/validation/test BEFORE any
synthetic generation happens. This is the leakage boundary for the whole
project: the generator (04_generate_synthetic_data.py) is fit ONLY on
original_train.csv. original_val.csv is available for generator/model
tuning if needed. original_test.csv is never touched until final ML
evaluation (06_ml_evaluation.py) and is never used to fit or select the
generator or any model hyperparameter.

Split: 70% train / 15% val / 15% test, seeded, stratified loosely by
T2 (the strongest single driver of the outputs) via quantile binning so
each split spans the output range reasonably rather than by chance
clustering into one region of the small dataset.

Run: python3 synthetic/03_split_original_data.py
Outputs: synthetic/data/original_train.csv, original_val.csv, original_test.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(ROOT / "data" / "Table.csv").reset_index(drop=True)
    df["_orig_index"] = df.index  # provenance: trace every row back to Table.csv

    # Loose stratification bin on T2 (strongest single input-output correlate,
    # see 01_dataset_analysis.py) so train/val/test each span the output range.
    strata = pd.qcut(df["T2 (K)"], q=3, labels=False, duplicates="drop")

    train_df, temp_df, strata_train, strata_temp = train_test_split(
        df, strata, test_size=0.30, random_state=SEED, stratify=strata
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=strata_temp
    )

    for name, split in [("original_train", train_df), ("original_val", val_df), ("original_test", test_df)]:
        split = split.sort_values("_orig_index").reset_index(drop=True)
        split.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: n={len(split)}  orig_index range=[{split['_orig_index'].min()},{split['_orig_index'].max()}]")

    # Sanity: no overlap, full coverage
    idx_train = set(train_df["_orig_index"])
    idx_val = set(val_df["_orig_index"])
    idx_test = set(test_df["_orig_index"])
    assert not (idx_train & idx_val)
    assert not (idx_train & idx_test)
    assert not (idx_val & idx_test)
    assert idx_train | idx_val | idx_test == set(df["_orig_index"])
    print(f"\nTotal={len(df)}  Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
    print("No overlap between splits. Full coverage confirmed.")
    print("\nLEAKAGE BOUNDARY: only original_train.csv (and, for tuning only, "
          "original_val.csv) may be used to fit the synthetic-data generator. "
          "original_test.csv is reserved exclusively for final ML evaluation "
          "(06_ml_evaluation.py) and must never be seen by the generator.")


if __name__ == "__main__":
    main()
