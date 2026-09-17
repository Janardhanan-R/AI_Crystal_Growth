"""
data_loader.py
===============
Loads the raw Bridgman crystal-growth dataset and documents exactly what the
columns mean, based on direct inspection of the data plus the dataset's
original source record (not assumption).

Dataset provenance
-------------------
Source file: data/Table.csv, originally distributed as:

    "Numerical simulation data of melt-crystal interface position and
    deflection of BaBrCl in a 5-heater vertical Bridgman furnace"
    Chang Zhang, University of Minnesota
    Mendeley Data, V1, 7 February 2022
    DOI: 10.17632/4hr6cx2p8m.1
    https://data.mendeley.com/datasets/4hr6cx2p8m/1

**This is SIMULATION-GENERATED data, not physical experimental
measurements.** It was produced with CrysMAS (finite-volume method, 2D
axis-symmetric, steady-state heat conduction + radiation) for a BaBrCl
crystal grown in a 5-heater vertical Bridgman furnace. Every row is one
CrysMAS run at a different combination of the five heater set-point
temperatures. Consequently, this project is best described as a
DATA-DRIVEN SURROGATE FOR A NUMERICAL SIMULATION (a standard, legitimate
ML pattern: a fast statistical model that approximates a slower physics
solver), not as a model trained on furnace measurements. Any paper or
report built on this project MUST cite the Mendeley dataset above and
state plainly that the underlying data is simulated, not experimental --
check the dataset's license terms on the Mendeley page before republishing
it or its derivatives.

96 rows, 9 columns, no missing values, no duplicate rows. All 96 (T1..T5)
combinations are unique -> a small simulation design-of-experiment table,
not repeated/batched/time-series data.

Column meaning (per the dataset's original documentation, cross-checked
programmatically against the algebraic identities below):

    T1 (K) .. T5 (K)  : target set-point temperatures for the furnace's
                        five heaters, the simulation INPUTS.
    Lc (m)            : vertical melt-crystal interface position AT THE
                        CENTERLINE (radius r = 0 m). A genuine,
                        independent simulated quantity.
    Le (m)            : vertical melt-crystal interface position AT THE
                        EDGE (radius r = 0.0065 m, i.e. at the ampoule
                        wall). A genuine, independent simulated quantity.
    P  (m)            : the AVERAGE interface position, EXACTLY
                        (Lc + Le) / 2 for every row (verified to float
                        precision, max abs error ~2.7e-17). NOT modeled
                        in this project (confirmed with the user) because
                        it carries no information beyond Lc and Le.
    D  (m)            : the INTERFACE DEFLECTION, EXACTLY Lc - Le for
                        every row (verified to float precision, max abs
                        error ~2.7e-17) -- i.e. how much the growth
                        interface bows between the centerline and the
                        edge. This is a standard, physically meaningful
                        quantity in Bridgman crystal growth (interface
                        curvature relates to radial thermal gradients and
                        thermal stress), NOT a crystal diameter. An
                        earlier project brief called this column "Crystal
                        Diameter"; that label is not supported by the data
                        (D takes near-zero and slightly NEGATIVE values,
                        physically impossible for a diameter) and was
                        corrected, per user confirmation, before any
                        modeling began.

TARGETS modeled in this project: Lc, Le, D (D modeled directly from
T1..T5, per the user's explicit choice, and documented honestly as
interface deflection rather than diameter).
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
    "Lc": "Melt-crystal interface position at the centerline, r=0m (m)",
    "Le": "Melt-crystal interface position at the edge, r=0.0065m (m)",
    "D": "Interface deflection, D = Lc - Le (m) -- how much the growth "
         "interface bows between centerline and edge. NOT a crystal "
         "diameter -- see data_loader module docstring.",
}

DATASET_CITATION = (
    "Zhang, Chang (2022), \"Numerical simulation data of melt-crystal "
    "interface position and deflection of BaBrCl in a 5-heater vertical "
    "Bridgman furnace\", Mendeley Data, V1, doi: 10.17632/4hr6cx2p8m.1"
)


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
