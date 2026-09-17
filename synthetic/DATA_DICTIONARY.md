# Data Dictionary — Bridgman Crystal Growth Dataset (`data/Table.csv`)

## Provenance (sourced, not assumed)

- **Source**: "Numerical simulation data of melt-crystal interface position and deflection of
  BaBrCl in a 5-heater vertical Bridgman furnace," Chang Zhang, University of Minnesota,
  Mendeley Data, V1, 7 February 2022. DOI: [10.17632/4hr6cx2p8m.1](https://data.mendeley.com/datasets/4hr6cx2p8m/1)
- **Companion paper**: Zhang, C. & Cheng, X. (2022), "Predicting Melt-Crystal Interface
  Position and Shape during the Manufacturing Process of Single Crystal via Explainable
  Machine Learning Models," *IOP Conf. Ser.: Mater. Sci. Eng.* 1258 012029. Read in full
  during this project; column meanings, the ordering constraint, and the CrysMAS simulation
  setup below are drawn directly from it.
- **Nature of the data**: CrysMAS (finite-volume, 2D axis-symmetric, steady-state heat
  conduction + radiation) simulation output for a BaBrCl scintillator crystal grown in a
  5-heater vertical Bridgman (VB) furnace. Every row is one simulation run at one combination
  of five heater set-point temperatures. **This is simulated data, not physical experimental
  measurement**, and every deliverable in this project states that explicitly.
- **Shape**: 96 rows × 9 columns. No missing values. No duplicate rows. All 96 `(T1..T5)`
  input combinations are unique — a constrained factorial design-of-experiments (DOE) table,
  not repeated or time-series data.

## Columns

| Column | Role | Units | Type | Observed range | Meaning |
|---|---|---|---|---|---|
| `T1 (K)` | input | Kelvin | int64 | 1255 – 1275 (2 levels) | Heater 1 set-point temperature (hottest heater, top of furnace) |
| `T2 (K)` | input | Kelvin | int64 | 1215 – 1255 (3 levels) | Heater 2 set-point temperature |
| `T3 (K)` | input | Kelvin | int64 | 1195 – 1235 (3 levels) | Heater 3 set-point temperature |
| `T4 (K)` | input | Kelvin | int64 | 1055 – 1095 (3 levels) | Heater 4 set-point temperature |
| `T5 (K)` | input | Kelvin | int64 | 1035 – 1055 (2 levels) | Heater 5 set-point temperature (coldest heater, bottom of furnace) |
| `Lc (m)` | output (simulated, independent) | metres | float64 | −0.2279 – −0.1946 | Melt–crystal interface position at the **centerline** (radius r = 0 m) |
| `Le (m)` | output (simulated, independent) | metres | float64 | −0.2273 – −0.1936 | Melt–crystal interface position at the **edge** (radius r = 0.0065 m, ampoule wall) |
| `P (m)` | output (**derived, not modeled**) | metres | float64 | −0.2276 – −0.1941 | Average interface position, exactly `(Lc + Le) / 2` for every row (verified to float precision, max abs error ≈ 2.7×10⁻¹⁷). Not used as a modeling target because it carries no information beyond `Lc`/`Le`. |
| `D (m)` | output (**derived**, modeled directly) | metres | float64 | −0.0010 – 0.0041 | Interface **deflection**, exactly `Lc − Le` for every row (verified to float precision, max abs error ≈ 2.7×10⁻¹⁷) — how much the growth interface bows between centerline and edge. A signed quantity: positive = interface bows toward the melt at the centerline relative to the edge, negative = the opposite. **This is not a crystal diameter.** |

## Physical constraint honored by the design

`T1 ≥ T2 ≥ T3 ≥ T4 ≥ T5` for every row (0 of 96 violations), consistent with the
directional-solidification requirement described in Zhang & Cheng (2022) §2.2.2 — the
furnace must run hotter at the top than the bottom for the melt-crystal interface to advance
correctly. Any synthetic input generation must preserve this ordering exactly; it is a
physical requirement of the VB process, not a statistical artifact.

## Sign conventions — stated honestly, not invented

`Lc` and `Le` are **negative in all 96 rows** (range ≈ −0.228 to −0.194 m), with a tight
spread (std ≈ 0.007 m for both). This is consistent across the entire dataset (uniform sign,
narrow band), so it reads as a fixed coordinate-reference convention in the CrysMAS setup
(interface position measured relative to a datum above the interface — e.g., ampoule/furnace
top) rather than a data error. The dataset's public documentation does not spell out the
exact zero-point of this axis, so **the exact physical zero-reference for `Lc`/`Le` is marked
uncertain** — what is *not* uncertain is that the sign and narrow range are a real,
reproducible feature of the simulation output, and synthetic data must preserve both rather
than clipping to positive values or rescaling the axis. `D`, by contrast, straddles zero
(39 of 96 rows negative, 57 non-negative) — expected for a signed deflection — and must not be
forced non-negative.

## Sample size and feature dependence

- n = 96, a small simulation DOE table (2×3×3×3×2 = 108 possible combinations, minus 12
  combinations that violate the `T1≥T2≥T3≥T4≥T5` constraint = 96 actual rows). This is stated
  explicitly wherever methodology or model performance is discussed.
- Inputs `T1..T5` are **not mutually independent by construction**: pairwise correlations are
  ≈0 except `T2`–`T3` (Pearson r = 0.231), reflecting the factorial design rather than i.i.d.
  sampling. A synthetic-input generator that draws `T1..T5` independently from their marginal
  distributions would not reproduce this — see `02_methodology_selection.md`.
- Input–output correlations (Pearson, original data): `T2` and `T3` are the strongest single
  drivers of `Lc`/`Le` (r ≈ −0.63 to −0.67), `T4` is next (r ≈ −0.50 to −0.51), and `T1`/`T5`
  are weak (|r| ≈ 0.07–0.12). `D` correlates most with `T2` (r = −0.654) and, notably,
  asymmetrically with `Lc` (r = 0.692) vs. `Le` (r = 0.595) — consistent with `D` being a
  difference of two highly collinear quantities (`Lc`–`Le` Pearson r = 0.992) and therefore
  much noisier to predict than either alone.

## Derived variables (identified programmatically, not assumed)

- `P (m) = (Lc + Le) / 2` — exact identity, confirmed to float precision on all 96 rows.
- `D (m) = Lc − Le` — exact identity, confirmed to float precision on all 96 rows.

Because both are exact algebraic functions of `Lc`/`Le`, they carry **zero independent
information** beyond `Lc` and `Le`. This has two consequences enforced throughout this
project: (1) `P` is excluded from modeling entirely; (2) `D` is modeled directly from
`T1..T5` (never computed from a *predicted* `Lc`/`Le`, and never used as an input feature to
predict `Lc`/`Le`, since that would leak the target into the feature set).

## Full statistics

See `synthetic/results/dataset_analysis.json` for the complete machine-readable output
(per-column dtype, min/max/mean/median/std/skew/kurtosis, IQR-based outlier indices, full
Pearson and Spearman correlation matrices, and the physically-flagged-value counts) that this
dictionary summarizes.
