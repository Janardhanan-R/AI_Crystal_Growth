"""
data_loader.py
===============
Loads the raw Bridgman crystal-growth dataset and documents exactly what the
columns mean, based on direct inspection of the data (not assumption).

Dataset provenance
-------------------
Source file: data/Table.csv (uploaded by the user).
96 rows, 9 columns, no missing values, no duplicate rows. All 96 input
combinations of (T1..T5) are unique -> this is a small designed-experiment
(DOE) table, not repeated/batched/time-series measurements.

Column meaning (verified programmatically, see notebooks/ or README for the
verification code):

    T1 (K) .. T5 (K)  : five furnace axial-zone temperatures (Kelvin),
                        the process/control INPUTS.
    Lc (m)            : axial position of the crystal-side growth interface
                        (metres). A genuine, independent modeled quantity.
    Le (m)            : axial position of the encapsulant/liquid-side
                        interface (metres). A genuine, independent modeled
                        quantity.
    P  (m)            : EXACTLY (Lc + Le) / 2 for every row (verified to
                        float precision, max abs error ~2.7e-17). This is
                        the midpoint of the two interface positions. It is
                        NOT modeled in this project (confirmed with the
                        user) because it carries no information beyond Lc
                        and Le.
    D  (m)            : EXACTLY Lc - Le for every row (verified to float
                        precision, max abs error ~2.7e-17). This is the
                        interface SEPARATION / deflection between the two
                        interfaces, not a crystal diameter. The label
                        "Crystal Diameter" used in early project notes is
                        not supported by the data (D takes near-zero and
                        even slightly NEGATIVE values, which is physically
                        impossible for a diameter) and has been corrected
                        here, per user confirmation, to
                        "interface separation (Lc - Le)".

TARGETS modeled in this project: Lc, Le, D (D modeled directly from T1..T5,
per user's explicit choice, and honestly documented as interface
separation rather than diameter).
"""
from __future__ import annotations

import pandas as pd

RAW_COLUMNS = {
    "T1 (K)": "T1",
    "T2 (K)": "T2",
    "T3 (K)": "T3",
    "T4 (K)": "T4",
    "T5 (K)": "T5",
    "Lc (m)": "Lc",
    "Le (m)": "Le",
    "P (m)": "P",
    "D (m)": "D",
}

INPUT_FEATURES = ["T1", "T2", "T3", "T4", "T5"]
TARGETS = ["Lc", "Le", "D"]
NOT_MODELED = ["P"]  # present in source data, deliberately excluded (see docstring)

TARGET_DESCRIPTIONS = {
    "Lc": "Axial position of the crystal-side growth interface (m)",
    "Le": "Axial position of the encapsulant/liquid-side interface (m)",
    "D": "Interface separation / deflection, D = Lc - Le (m). "
         "NOT a crystal diameter -- see data_loader module docstring.",
}


def load_raw(path: str = "data/Table.csv") -> pd.DataFrame:
    """Load the CSV exactly as provided and rename columns to safe,
    unit-documented short names. No values are modified."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns not found in {path}: {sorted(missing)}. "
            "Refusing to proceed with assumed/fabricated column mappings."
        )
    df = df.rename(columns=RAW_COLUMNS)
    return df[list(RAW_COLUMNS.values())]


def verify_identities(df: pd.DataFrame, tol: float = 1e-9) -> dict:
    """Confirm the exact algebraic relationships P=(Lc+Le)/2 and D=Lc-Le
    that were used to justify excluding P and relabeling D. Raises if the
    data no longer supports these identities (defensive check)."""
    p_err = (df["P"] - (df["Lc"] + df["Le"]) / 2).abs().max()
    d_err = (df["D"] - (df["Lc"] - df["Le"])).abs().max()
    if p_err > tol or d_err > tol:
        raise AssertionError(
            f"P/D identity check failed (max errors P={p_err}, D={d_err}). "
            "Do not trust downstream documentation without re-verifying "
            "column meanings against the raw data."
        )
    return {"max_abs_error_P_identity": float(p_err), "max_abs_error_D_identity": float(d_err)}


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(verify_identities(df))
