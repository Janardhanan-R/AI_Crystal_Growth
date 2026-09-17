# Synthetic Data Generation Workflow — `synthetic/`

Complete, reproducible workflow for the LLM-Based Intelligent Decision Support System project's
synthetic-data-generation task, built on top of the existing 96-row `data/Table.csv`. Every
deliverable requested is below, with the honesty/leakage/realism constraints from the task
brief enforced in code, not just asserted in prose. Start with `07_ASSUMPTIONS_AND_LIMITATIONS.md`
if you only read one file — the single most important requirement of this task was scientific
honesty over dataset size or accuracy.

## The 10 deliverables

| # | Deliverable | File(s) |
|---|---|---|
| 1 | Synthetic dataset (CSV) | `synthetic_crystal_growth_dataset.csv` (n=2000) |
| 2 | Reproducible generation code | `04_generate_synthetic_data.py` (+ `01_*.py`, `03_*.py` it depends on) |
| 3 | Data validation/comparison notebook | `10_data_validation_comparison.ipynb` |
| 4 | Distribution comparison plots | `plots/distribution_comparison_*.png` |
| 5 | Correlation comparison plots | `plots/correlation_comparison.png`, `plots/pairplot_comparison.png` |
| 6 | Model evaluation report | `06_MODEL_EVALUATION_REPORT.md` (+ `06_ml_evaluation.py`, `results/*.csv`) |
| 7 | Data dictionary | `DATA_DICTIONARY.md` |
| 8 | Assumptions and limitations | `07_ASSUMPTIONS_AND_LIMITATIONS.md` |
| 9 | Reproducibility instructions | `08_REPRODUCIBILITY_INSTRUCTIONS.md` |
| 10 | Research-paper-ready methodology section | `09_METHODOLOGY_SECTION.md` |

## Supporting files

- `01_DATASET_ANALYSIS_REPORT.md` / `01_dataset_analysis.py` — full original-data analysis
  (dtypes, ranges, distributions, outliers, correlations, derived-variable identities,
  physically-flagged values).
- `02_METHODOLOGY_SELECTION.md` — comparison of all 5 candidate generation methods and
  justification for the one selected (GP-based).
- `03_split_original_data.py` — the leakage-safe train/val/test split of the *original* 96
  rows, performed before any generation.
- `05_VALIDATION_REPORT.md` / `05_validate_synthetic_data.py` — KS tests, Wasserstein
  distance, correlation/duplicate/outlier analysis, predictive-relationship comparison.
- `data/original_{train,val,test}.csv` — the leakage-safe splits themselves.
- `results/` — every machine-readable output (JSON/CSV) referenced by the reports above.
- `build_notebook.py` — the (deterministic) script that generated the `.ipynb` in #3; re-run
  it if the notebook's cell structure needs regenerating.

## One-line honesty summary

The synthetic data is a Gaussian-Process statistical interpolant of the 96-row CrysMAS
simulation sample — not new simulation output, not experimental data, and not a validated
physical model. It is statistically similar to the original sample for `Lc`/`Le`/`P` and less
so for `D` (the weakest-modeled target throughout this project). On the untouched original
test set, synthetic augmentation did **not** improve the best model (Gaussian Process
Regression) for any target; it helped some weaker models and substantially hurt one (SVR). See
`07_ASSUMPTIONS_AND_LIMITATIONS.md` and `06_MODEL_EVALUATION_REPORT.md` for the full picture.

## Run order

See `08_REPRODUCIBILITY_INSTRUCTIONS.md` for exact commands. Summary:
`01 → 03 → 04 → 05 → 06`, then open `10_data_validation_comparison.ipynb`.
