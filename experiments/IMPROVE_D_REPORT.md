# Can we improve the D (interface deflection) metric? — Findings

Generated 2026-10-03. Prompted by a direct question during mentor-meeting prep:
"Can you improve that metric?" (R2/MAE for D, which lags Zhang & Cheng (2022)'s
reported Random Forest R2=0.78 vs. our GaussianProcess pooled-CV R2=0.713).

**Short answer: no, not meaningfully — and the reason why is itself a useful
finding.** This doc is the honest record of what was tried, exact numbers
(re-run of the project's own CV code on the real data, not estimated), and
the diagnostic that explains why tuning didn't help.

## What was tried

Nine variants, all evaluated with the exact same 5-fold CV setup already in
`src/cross_validation.py` (same KFold, same random_state=42, same train
split), against the current baseline (GaussianProcess, CV R2_mean=0.6648,
pooled R2=0.7126 — see `results/model_comparison_table.csv`). Full numbers
in `01_RESULTS_improve_D.csv`; reproduce with `01_improve_D_metrics.py`.

| Variant | CV R2 mean | vs. baseline |
|---|---|---|
| GPR, RBF+Matern sum kernel | 0.6655 | +0.0007 — within noise (fold std is ~0.10) |
| GPR, raw T1-T5 only (no engineered features) | 0.6394 | worse |
| GPR, anisotropic RBF (per-feature length scale) | 0.6078 | worse |
| GPR, Matern(nu=2.5) kernel | 0.5864 | worse |
| GPR on T1-T5 pairwise interaction terms | 0.5799 | worse |
| GPR, reduced feature set (dT_15, T_mean only) | 0.5711 | worse |
| GPR, Matern(nu=1.5) kernel | 0.5452 | worse |
| Stacking (GPR+RF+HistGB -> Ridge) | **-0.1218** | much worse |

The best variant found (RBF+Matern sum) beats the current baseline by
0.0007 — smaller than the fold-to-fold standard deviation (+/-0.10), so this
is not a real improvement, just noise. Every other change tested made
things worse, in some cases (stacking) catastrophically.

## Why — residual diagnostic

Reproduce with `02_residual_diagnostic.py`. Pulled the actual 5-fold
out-of-fold predictions for the baseline GaussianProcess model on D and
ranked rows by absolute residual:

- **8 of 76 training rows (~10.5%) account for 74.7% of the total squared
  error.**
- Those 8 rows are specifically the most extreme corners of the heater grid
  — T1 near its maximum (~1275K) paired with T5 near its minimum
  (~1035-1055K), i.e. the steepest axial thermal gradient conditions.
- The model fits the "normal" operating range well; it specifically cannot
  generalize to these few, sparsely-sampled extreme conditions.

## Conclusion

This is a **data-density problem at the extremes of the operating range,
not a modeling problem** — which is also why kernel changes, feature
changes, and stacking all failed to help: there is no algorithmic fix for
having only ~8 examples near the hardest region of the input space.

This is consistent with, and adds a fourth independent data point to, three
things already documented elsewhere in this project:
1. D has the weakest CV R2 of the three targets (Section 16 of the main README).
2. D had the highest outlier rate and largest correlation-structure shift in
   the synthetic-data validation (`synthetic/05_VALIDATION_REPORT.md`).
3. Synthetic-data augmentation did not improve D for the best model either
   (`synthetic/06_MODEL_EVALUATION_REPORT.md`).

**What would actually move this number**: more CrysMAS simulation runs
specifically at those extreme heater combinations (more data where it's
sparse), not further ML tuning. This has not been done as part of this
project.

## Honesty notes

- All numbers above are from actually re-running the project's own CV code
  against `data/Table.csv` just now, not estimated or recalled.
- No change from this experiment was adopted into the main pipeline
  (`src/models.py` is unchanged) — nothing here beat the baseline by a
  margin worth calling an improvement.
- Run `python3 experiments/01_improve_D_metrics.py` and
  `python3 experiments/02_residual_diagnostic.py` (from the project root,
  with `requirements.txt` installed) to reproduce every number in this
  document independently.
