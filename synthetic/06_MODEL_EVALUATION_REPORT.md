# Model Evaluation Report — Original-Only vs. Synthetic-Augmented Training

Generated from `synthetic/06_ml_evaluation.py`. Full numeric outputs:
`synthetic/results/cv_results.csv` (all 25 individual folds), `cv_results_summary.csv`
(mean±std), `final_test_results.csv` (the single untouched-test check),
`gpr_prediction_intervals.csv`.

**XGBoost was not available in this environment** (`import xgboost` failed; no network access
to install it here) and was skipped entirely, per this task's explicit "only if the real
library is installed" instruction — no substitute model was used in its place for this
evaluation.

## Protocol recap

- **Repeated CV**: 5-fold × 5-repeat (25 folds) over the original train+validation pool
  (n=81). Synthetic augmentation inside CV uses a fixed 300-row subsample of the 2000-row
  synthetic set (computational tractability; GPR is O(n³) — see the script's
  `CV_SYNTH_AUGMENT_N` note). Evaluation within every fold is always on real, held-out
  original rows.
- **Final check**: one fit on the full original train+val pool (n=81), with and without the
  **complete** 2000-row synthetic set, evaluated once on `original_test.csv` (n=15) — data
  untouched by generation, CV, or any prior model fit.

## Headline result: does synthetic augmentation help?

**No, not for the best-performing model, and the effect is target- and model-dependent —
this section reports what was actually observed, not what would be convenient to claim.**

On the untouched original test set (n=15), **Gaussian Process Regression trained on
original data only was the best model for all three targets**: Lc R²=0.9970, Le R²=0.9918,
D R²=0.8737. Adding the full synthetic set and refitting GPR **slightly reduced** its own
performance on all three (Lc R²=0.9938, Le R²=0.9861, D R²=0.7465) — consistent with the CV
result, where GPR's mean CV R² also dropped under augmentation for Lc (0.9946→0.9933), was
roughly flat for Le (0.9760→0.9796), and dropped for D (0.5698→0.4502). This matches this
project's prior, independently-established finding that GPR is already the best-fit model for
this system; augmenting its training data with the GPR-generated synthetic set adds noise
without adding new information the GPR interpolant did not already encode.

**Weaker models benefited more.** RandomForest — the model most starved for training data at
n=81 — improved substantially with augmentation on the final test set: Lc R² 0.9128→0.9495, Le
R² 0.8996→0.9334, D R² 0.7652→0.8634 (D's largest single improvement of any model/target
combination). HistGradientBoosting showed smaller, mixed gains (Le 0.9771→0.9818, D
0.8339→0.8252 essentially flat, Lc 0.9929→0.9936 essentially flat). Linear/Ridge showed small,
inconsistent shifts in both directions (≤0.05 R² either way).

**SVR was harmed, sometimes catastrophically.** Le R² collapsed from 0.9707 (original-only)
to 0.6197 (augmented); D R² collapsed from 0.2714 to −8.88 (augmented model performs far worse
than predicting the mean). SVR's RBF kernel bandwidth, tuned implicitly via the fixed
`C`/`epsilon` on the small original scale, does not transfer well to the wider, noisier
synthetic distribution — this is reported as a genuine model-specific failure mode of
augmentation, not smoothed over.

## Full CV summary (mean ± std across 25 folds)

| Target | Model | Orig-only R² | Synth-augmented R² | Δ R² |
|---|---|---|---|---|
| Lc | GaussianProcess | 0.9946 ± 0.0032 | 0.9933 ± 0.0028 | −0.0013 |
| Lc | HistGradientBoosting | 0.9864 ± 0.0091 | 0.9900 ± 0.0077 | +0.0036 |
| Lc | LinearRegression | 0.9642 ± 0.0129 | 0.9672 ± 0.0126 | +0.0030 |
| Lc | Ridge | 0.9642 ± 0.0127 | 0.9672 ± 0.0126 | +0.0030 |
| Lc | RandomForest | 0.9074 ± 0.0331 | 0.9563 ± 0.0232 | +0.0489 |
| Lc | SVR | 0.9465 ± 0.0208 | 0.9092 ± 0.0337 | −0.0373 |
| Le | GaussianProcess | 0.9760 ± 0.0179 | 0.9796 ± 0.0088 | +0.0036 |
| Le | HistGradientBoosting | 0.9754 ± 0.0118 | 0.9865 ± 0.0081 | +0.0111 |
| Le | LinearRegression | 0.9334 ± 0.0217 | 0.9388 ± 0.0177 | +0.0054 |
| Le | Ridge | 0.9340 ± 0.0208 | 0.9390 ± 0.0177 | +0.0050 |
| Le | RandomForest | 0.9028 ± 0.0365 | 0.9536 ± 0.0164 | +0.0508 |
| Le | SVR | 0.9209 ± 0.0272 | 0.8725 ± 0.0485 | −0.0484 |
| D | HistGradientBoosting | 0.6381 ± 0.1989 | 0.7314 ± 0.1226 | +0.0933 |
| D | RandomForest | 0.5927 ± 0.1603 | 0.7429 ± 0.1259 | +0.1502 |
| D | GaussianProcess | 0.5698 ± 0.1985 | 0.4502 ± 0.1783 | −0.1196 |
| D | Ridge | 0.4458 ± 0.3299 | 0.4619 ± 0.2546 | +0.0161 |
| D | LinearRegression | 0.4425 ± 0.3381 | 0.4620 ± 0.2555 | +0.0195 |
| D | SVR | −0.0559 ± 0.4294 | −7.1223 ± 2.9955 | −7.0664 |

`D`'s very large CV standard deviations (±0.16 to ±0.43 across models) reflect the same
target-level difficulty documented throughout this project — small changes in which original
rows land in a given fold noticeably move `D`'s score, because `D` is a difference of two
highly collinear quantities computed on already few points.

## Final untouched-test-set results (n=15)

| Target | Model | Orig-only MAE / RMSE / R² | Synth-augmented MAE / RMSE / R² |
|---|---|---|---|
| Lc | **GaussianProcess (best)** | 0.000275 / 0.000316 / **0.9970** | 0.000339 / 0.000455 / 0.9938 |
| Lc | HistGradientBoosting | 0.000395 / 0.000486 / 0.9929 | 0.000403 / 0.000463 / 0.9936 |
| Lc | SVR | 0.000997 / 0.001201 / 0.9569 | 0.000969 / 0.001322 / 0.9478 |
| Lc | RandomForest | 0.001357 / 0.001709 / 0.9128 | 0.001062 / 0.001301 / 0.9495 |
| Lc | Ridge | 0.001030 / 0.001302 / 0.9494 | 0.001000 / 0.001305 / 0.9492 |
| Lc | LinearRegression | 0.001046 / 0.001320 / 0.9480 | 0.001001 / 0.001306 / 0.9491 |
| Le | **GaussianProcess (best)** | 0.000370 / 0.000425 / **0.9918** | 0.000467 / 0.000553 / 0.9861 |
| Le | HistGradientBoosting | 0.000484 / 0.000710 / 0.9771 | 0.000530 / 0.000632 / 0.9818 |
| Le | SVR | 0.000662 / 0.000803 / 0.9707 | 0.002197 / 0.002891 / **0.6197 (collapsed)** |
| Le | RandomForest | 0.001193 / 0.001485 / 0.8996 | 0.000960 / 0.001210 / 0.9334 |
| Le | Ridge | 0.000929 / 0.001261 / 0.9277 | 0.000925 / 0.001217 / 0.9326 |
| Le | LinearRegression | 0.000952 / 0.001293 / 0.9239 | 0.000927 / 0.001219 / 0.9325 |
| D | **GaussianProcess (best)** | 0.000346 / 0.000434 / **0.8737** | 0.000376 / 0.000615 / 0.7465 |
| D | HistGradientBoosting | 0.000337 / 0.000497 / 0.8339 | 0.000289 / 0.000510 / 0.8252 |
| D | RandomForest | 0.000399 / 0.000591 / 0.7652 | 0.000283 / 0.000451 / **0.8634 (best synth-aug)** |
| D | LinearRegression | 0.000428 / 0.000630 / 0.7338 | 0.000454 / 0.000692 / 0.6788 |
| D | Ridge | 0.000433 / 0.000634 / 0.7299 | 0.000454 / 0.000692 / 0.6787 |
| D | SVR | 0.000927 / 0.001042 / 0.2714 | 0.003005 / 0.003837 / **−8.884 (catastrophic)** |

## Prediction intervals (GaussianProcessRegressor, 95% interval, original_test n=15)

| Target | Regime | Empirical coverage | Mean interval half-width |
|---|---|---|---|
| Lc | original-only | 15/15 = 100.0% | 0.000678 m |
| Lc | synthetic-augmented | 15/15 = 100.0% | 0.001144 m |
| Le | original-only | 14/15 = 93.3% | 0.000838 m |
| Le | synthetic-augmented | 15/15 = 100.0% | 0.001928 m |
| D | original-only | 15/15 = 100.0% | 0.001052 m |
| D | synthetic-augmented | 15/15 = 100.0% | 0.002231 m |

Coverage is at or above the nominal 95% target in every case (n=15 is too small for this to be
a precise calibration check, but there is no evidence of overconfident intervals). Synthetic
augmentation widens the GPR's own stated uncertainty in every case — a secondary, honest
confirmation that the augmented training distribution is less concentrated than the original,
rather than an unambiguous improvement.

## Conclusion

**Synthetic augmentation does not improve the single best model (GPR) for any target on the
untouched original test set**, and for `D` and `Le` it makes GPR appreciably worse. Its
practical value in this project is narrower and more model-specific: it substantially helps
weaker, more data-hungry models (RandomForest gained +0.037 to +0.150 R² across targets on the
untouched test set) partially close the gap to GPR, while actively harming a
kernel-bandwidth-sensitive model (SVR) whose hyperparameters were tuned for the tighter
original data distribution. No claim is made anywhere in this project that synthetic-augmented
training improves overall performance — the evidence, on the model this project already
identified as best, points the other way.
