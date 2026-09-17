# Bridgman Crystal Growth ML Surrogate

A data-driven regression pipeline that predicts three melt-crystal
interface quantities (**Lc**, **Le**, **D**) for a BaBrCl crystal grown in
a 5-heater vertical Bridgman furnace, from the five heater set-point
temperatures (**T1..T5**), compares seven regression models under 5-fold
cross-validation, and packages the best model per target as a fast
surrogate for exploring the simulation's response surface -- with SHAP
explainability and a surrogate-driven inverse-design (optimization) module
on top.

## 0. Data source and a critical correction to the original project brief -- read this first

**Data source.** `data/Table.csv` is a numerical-simulation dataset, not
physical experimental measurements:

> Zhang, Chang (2022), "Numerical simulation data of melt-crystal
> interface position and deflection of BaBrCl in a 5-heater vertical
> Bridgman furnace", Mendeley Data, V1, doi: 10.17632/4hr6cx2p8m.1
> <https://data.mendeley.com/datasets/4hr6cx2p8m/1>

It was generated with **CrysMAS** (finite-volume method, 2D
axis-symmetric, steady-state heat conduction + radiation) -- each of the
96 rows is one CrysMAS run at a different combination of the five heater
set points. **Cite this dataset in any paper or report built on this
project, and state plainly that the underlying data is simulated, not
experimental.** Check the license on the Mendeley page before
republishing the data or derivatives of it.

**Column correction.** The original project brief asked for a **"Crystal
Diameter" target called D**. The dataset's own documentation (and direct
inspection of `data/Table.csv`) shows this is not correct:

| Column | What it actually is | Verified relationship |
|---|---|---|
| `Lc (m)` | Melt-crystal interface position **at the centerline** (r = 0 m) | genuine, independent |
| `Le (m)` | Melt-crystal interface position **at the edge** (r = 0.0065 m, the ampoule wall) | genuine, independent |
| `P (m)` | Average interface position | **exactly** `(Lc + Le) / 2` for all 96 rows (max abs error 2.8e-17) |
| `D (m)` | Interface **deflection** -- how much the interface bows between centerline and edge, not a diameter | **exactly** `Lc - Le` for all 96 rows (max abs error 2.7e-17) |

`D` ranges from about -0.001 m to +0.004 m and is frequently negative --
physically impossible for a diameter. There is no diameter in the
dataset; interface deflection is, however, a standard and physically
meaningful quantity in Bridgman crystal growth (it relates to radial
thermal gradients and thermal stress in the growing crystal).

**This mismatch was flagged before any modeling began, per the project's
own "do not fabricate target values or column mappings" requirement.**
The user confirmed:
* `D` should be **modeled directly** as an independent regression target
  from T1..T5 (not derived algebraically from Lc/Le), but must be
  **documented honestly as interface deflection**, never as "Crystal
  Diameter", anywhere in this project's code, plots, or tables.
* `P` is **not** modeled (it would add no information beyond Lc and Le).

Every module, plot, and table in this repository uses the corrected,
sourced labeling. **Targets modeled: Lc, Le, D** (D = interface deflection).

## 1. Relationship to Zhang & Cheng (2022) -- the paper this dataset comes from

The Mendeley dataset above is the same dataset used in:

> Zhang, C. & Cheng, Q. (2022). "Predicting Melt-Crystal Interface Position
> and Shape during the Manufacturing Process of Single Crystal via
> Explainable Machine Learning Models," *IOP Conf. Series: Materials
> Science and Engineering*, 1258, 012029.
> doi: 10.1088/1757-899X/1258/1/012029

That paper trained **2 models** (Linear Regression, Random Forest) on
**2 targets** (`P` and `D` only -- not `Lc`/`Le` individually), used SHAP to
explain the Random Forest, and closed with a **manual** application: reading
the SHAP summary plot by eye, proposing 4 candidate heater adjustments by
hand, and checking them against 12 scenarios already present in the same
96-point dataset.

This project is built as a direct methodological extension, not a
re-statement, of that paper:

| | Zhang & Cheng (2022) | This project |
|---|---|---|
| Targets | `P`, `D` | `Lc`, `Le`, `D` individually (`P` excluded as redundant -- see Section 0) |
| Models compared | 2 (LR, RF) | 7 (LR, Ridge, SVR, RandomForest, XGBoost/substitute, HistGradientBoosting, Gaussian Process) |
| Uncertainty quantification | none | Gaussian Process predictive std on every prediction |
| Model selection | not described as a systematic comparison | rank-averaged over CV MAE/RMSE/R2, CV-only, documented in Section 7 |
| Multi-output modeling | not explored | independent vs. joint multi-output compared (Section 5) |
| Explainability | SHAP on RF | SHAP on RF (Section 11), for direct comparability |
| "Application to practice" | manual: read SHAP plot, propose 4 heater changes by hand, check against existing dataset rows | **automated**: `src/optimization.py` runs a bounded, constrained numerical search (`scipy.optimize.differential_evolution`) over the trained surrogate to recommend heater set-points for a *specified target outcome*, penalizing candidate set-points where the surrogate's own predictive uncertainty is high (Section 13) |
| Evaluation statistic reported | pooled 5-fold CV (all out-of-fold predictions combined into one R2/MAE -- inferred from their Figure 2) | **both** pooled-CV (for direct comparability, Section 9) **and** mean-of-folds CV mean +/- std (used for model selection, since it also gives a stability measure the pooled statistic cannot) |

Section 9 below reports the direct numeric comparison against their
published RF/D result.

## 2. Dataset

* `data/Table.csv`: 96 rows, 9 columns, **no missing values, no duplicate
  rows**. Every row is a unique (T1..T5) combination from the CrysMAS
  simulation design -- this is a small **simulation design-of-experiment**,
  not repeated trials, batches, or a time series, and not independent
  physical replication.
* Inputs: `T1 (K)` .. `T5 (K)`, the five furnace heaters' set-point
  temperatures. T1 has 2 levels, T2/T3/T4 have 3 levels each, T5 has 2
  levels (96 of the possible combinations are present).
* A small number of rows sit outside the 1.5x-IQR whisker range for
  Lc/Le/P/D (1-3 rows per column) -- these are the most extreme corner
  points of the temperature grid, not simulation errors, and are **kept**,
  not removed (see `src/preprocessing.py`).
* **Limitation**: n=96 is small for a 7-model comparison with an 80/20
  test split (~19 test rows). Test-set metrics below should be read as
  indicative, not precise generalization estimates. 5-fold CV (run on the
  77-row training split only) is the primary basis for model comparison
  and selection. All numbers describe fit to this simulation's output, not
  validation against physical furnace measurements.

## 3. Feature engineering

Raw features: `T1, T2, T3, T4, T5`.

Engineered features (all physically justified, all derived only from
T1..T5 -- see `src/feature_engineering.py` for the full justification of
each):

* `dT_12, dT_23, dT_34, dT_45` -- local axial gradients between adjacent
  furnace zones
* `dT_15` -- total axial temperature drop, hot zone to cold zone
* `T_mean` -- overall thermal level of the furnace

11 features total. Kept deliberately modest given n=96, to limit
overfitting and multicollinearity risk.

## 4. Models compared

| Model | Role |
|---|---|
| Linear Regression | Interpretable baseline |
| Ridge Regression | Regularized linear baseline |
| SVR (RBF kernel) | Nonlinear kernel baseline |
| Random Forest | Nonlinear tree ensemble, feature importance |
| **XGBoost*** | Nonlinear tree ensemble with regularization |
| HistGradientBoosting | Second, independent gradient-boosting implementation |
| Gaussian Process Regression | Smooth nonlinear surface + calibrated predictive uncertainty |

**\*XGBoost availability**: the development sandbox has no outbound PyPI
network access, so the real `xgboost` package could not be installed here.
`src/models.py` tries to import it first and automatically uses the real
`XGBRegressor` if available; in this environment it fell back to
`sklearn.ensemble.GradientBoostingRegressor`, a classic gradient-boosted
tree model, as an explicitly logged and documented substitute -- never
silently. Every plot/table that says "XGBoost" in this repository's
generated output was produced with that substitute. Installing xgboost
(`pip install xgboost`, or uncomment it in `requirements.txt`) and rerunning
`main.py` will use the genuine implementation with no code changes.

All 7 models use the identical `Pipeline(StandardScaler -> model)` and the
identical 5 CV folds, for a fair comparison.

RandomForest and the XGBoost-family model were each tuned once via
`GridSearchCV(cv=3)` on the training split only (see `src/models.py`); a
full nested search inside every outer CV fold was judged too unstable at
n~77 training rows and was not attempted.

## 5. Multi-output vs. independent models

RandomForest (native multi-output) and the XGBoost-family model (wrapped in
`MultiOutputRegressor`) were each also trained to predict all three targets
jointly, and compared against the average of their independent per-target
CV performance on the same folds:

| Model | CV MAE (joint) | CV RMSE (joint) | CV R2 (joint) |
|---|---|---|---|
| RandomForest (multi-output) | 0.001385 | 0.001756 | 0.764 |
| RandomForest (avg of independent fits) | 0.001370 | 0.001763 | 0.716 |
| XGBoost-family (multi-output) | 0.000716 | 0.000996 | 0.758 |
| XGBoost-family (avg of independent fits) | 0.000716 | 0.000996 | 0.758 |

The joint multi-output RandomForest is essentially tied with (marginally
better R2 than) its independent counterpart; the boosting model's joint
and independent results are effectively identical. **Independent
per-target models were used for the final surrogate** since the
per-target model-selection step (Section 7) is more transparent and
allows a different model per target, which mattered here (GPR won on all
three targets, but this is only knowable by comparing independently).

## 6. Cross-validation

`KFold(n_splits=5, shuffle=True, random_state=42)` on the **training split
only** (the test split, ~20% of the data, is never touched by CV or
hyperparameter tuning). GroupKFold / time-aware splitting was considered
and rejected: there are no repeated experiments, batches, or timestamps in
this dataset to leak across folds (every row is a distinct design point).

Full results: `results/cv_results_independent.csv`. Best CV result per
target (mean-of-folds statistic, used for selection):

| Target | Selected model | CV MAE (mean) | CV RMSE (mean) | CV R2 (mean) |
|---|---|---|---|---|
| Lc | Gaussian Process | 0.000395 | 0.000517 | **0.994** |
| Le | Gaussian Process | 0.000667 | 0.000914 | **0.977** |
| D  | Gaussian Process | 0.000334 | 0.000536 | 0.665 |

Note the much lower R2 for D across every model (best case 0.665) compared
to Lc/Le (>0.97). This is expected: D (deflection) is the small difference
of two close, similarly-scaled quantities (Lc and Le), so it inherits the
noise of both while its signal is an order of magnitude smaller -- a
harder regression problem by construction, not a modeling failure. This is
reported plainly rather than glossed over. Section 9 shows that a
meaningful part (not all) of this apparent gap versus Zhang & Cheng (2022)
is a difference in *evaluation statistic*, not purely a difference in model
quality.

`cross_validation.py` also computes, for every model x target, a **second,
pooled** statistic (`pooled_MAE`, `pooled_RMSE`, `pooled_R2` columns in
`results/cv_results_independent.csv`) -- see Section 9.

## 7. Model selection (from CV results only)

Selection used a rank-average of CV MAE, CV RMSE, and CV R2 (equal
weight -- not R2 alone, per the project requirement), computed **only**
from the mean-of-folds 5-fold CV on the training split (never the pooled
statistic from Section 9, and never the test set). The held-out test set
was never used for selection, only for an independent check afterward.

**Gaussian Process Regression won on all three targets** -- by every
metric, not narrowly. It also trains in well under 0.1s at this dataset
size (GPR is O(n^3), trivial at n<=96, so no sparse/inducing-point
approximation was needed) and predicts in well under 0.1ms/sample, so its
usual "does not scale" caveat does not apply here. See
`results/best_model_per_target.csv`.

## 8. Held-out test set (independent check, not used for selection)

| Target | Test MAE | Test RMSE | Test R2 | Inference time/sample |
|---|---|---|---|---|
| Lc | 0.000335 | 0.000442 | 0.996 | 0.068 ms |
| Le | 0.000458 | 0.000680 | 0.987 | 0.066 ms |
| D  | 0.000465 | 0.000822 | 0.617 | 0.066 ms |

Consistent with the CV results (including D's much lower R2). Full table
with all 7 models x 3 targets, including the pooled-CV columns from
Section 9: `results/model_comparison_table.csv`.

## 9. Pooled-CV comparison against Zhang & Cheng (2022)

Zhang & Cheng (2022) report a single R2/MAE per model, computed (as best
can be determined from their Figure 2, which plots all 96 out-of-fold
predictions in one scatter with one annotation) by **pooling** every
fold's out-of-fold predictions into one array before scoring -- not by
averaging the 5 per-fold scores, which is what Section 6/7 use for model
selection. These are genuinely different statistics on a small, noisy
target, so `pooled_cv_metrics()` (`src/cross_validation.py`) computes the
paper's apparent statistic on this project's own models, on the same
96-row dataset, for a fair, like-for-like comparison. Full table:
`results/zhang_cheng_2022_comparison.csv`.

For **D (deflection)**, their headline result (their "Model 4", Random
Forest):

| Model | Statistic | R2 | MAE |
|---|---|---|---|
| Zhang & Cheng (2022), Random Forest | pooled | 0.78 | 0.28 mm |
| This project, Random Forest | pooled | 0.508 | 0.446 mm |
| This project, Random Forest | mean-of-folds | 0.407 +/- 0.266 | 0.446 mm |
| **This project, Gaussian Process (selected model)** | **pooled** | **0.713** | **0.334 mm** |
| This project, Gaussian Process (selected model) | mean-of-folds | 0.665 | 0.334 mm |

Two findings, reported honestly rather than picking the flattering one:

1. **The pooled statistic does close part of the gap.** Even holding the
   model fixed at Random Forest, switching from mean-of-folds R2 (0.407) to
   the paper's apparent pooled statistic (0.508) moves substantially toward
   their 0.78 -- confirming that some of the originally-observed gap was a
   statistic mismatch, not a model-quality difference.
2. **It does not close all of it.** This project's own selected model
   (Gaussian Process, chosen by CV on Lc/Le/D independently, per Section 7)
   reaches pooled R2 = 0.713 for D -- closer to their 0.78 than our own
   Random Forest gets, but still below it. Plausible remaining causes,
   *not* verified further here: their RF may use different
   hyperparameters or a different random seed/fold assignment (not fully
   specified in the paper); they may predict `D` using a feature set that
   includes derived quantities not reconstructed here; or 5-fold CV at
   n=96 simply has enough fold-to-fold variance (std of 0.266 in the
   mean-of-folds R2 above) that a single-seed run cannot be expected to
   land on their exact number. This is reported as an open, honestly
   unresolved discrepancy -- not claimed as either "we matched the paper"
   or "the paper's numbers don't replicate."

## 10. Plots (`plots/`)

* `actual_vs_predicted_{Lc,Le,D}.png` -- all 7 models, held-out test set
* `residuals_{Lc,Le,D}.png`
* `error_distribution_{Lc,Le,D}.png`
* `feature_importance_{Lc,Le,D}.png` -- native importance for
  RandomForest/XGBoost-family; **permutation importance** (test set) for
  HistGradientBoosting, which has no native `feature_importances_` in
  scikit-learn -- labeled accordingly on each panel so the two are never
  confused
* `gpr_uncertainty_{Lc,Le,D}.png` -- GPR mean prediction +/- 95% CI vs.
  actual, held-out test set
* `shap_summary_{Lc,Le,D}.png` -- see Section 11

`T_mean` (overall furnace thermal level) and `dT_12` (zone 1-2 gradient)
are consistently the top-ranked features for Lc across models -- see
`plots/feature_importance_Lc.png`.

## 11. SHAP explainability (`src/explainability.py`)

`plot_shap_summary()` runs `shap.TreeExplainer` on the fitted Random
Forest for each target (the same model Zhang & Cheng explain with SHAP in
their Figure 3), on the standardized feature space the model was actually
trained on, and saves `plots/shap_summary_{Lc,Le,D}.png`.

**Not available in the development sandbox**: `shap` could not be
installed here (same no-PyPI-network-access limitation as xgboost -- see
Section 4). The module is written and tested for graceful degradation: it
detects `shap`'s absence at import time, prints a clear message, and skips
without affecting the rest of the pipeline (confirmed by an end-to-end run
of `main.py` in this sandbox). It will produce real SHAP plots as soon as
it is run with `shap` installed (`pip install shap`, or uncomment it in
`requirements.txt`) -- no code changes needed. (It has since been run
successfully outside the sandbox, with real SHAP plots produced -- see
the comparison below.)

**Comparison to Zhang & Cheng's own SHAP result (their Figure 3(d), Model
4).** They report, in their own words, that for interface deflection D:
*"T2 has the strongest impact ... followed by T4, T3, T1, and T5."*
Restricting this project's SHAP output to the same raw T1..T5 features
(excluding the engineered features, which their model did not have) gives
**T2 > T4 > T5 > T3 > T1**. The top two drivers match exactly between the
two independently-trained models on two different data splits -- both
find T2 (upper heater) and T4 (lower heater) dominate interface
deflection by a wide margin over the rest. The bottom three (T3, T1, T5)
reshuffle between the two studies, but both agree they are minor
contributors; this is consistent with fold-to-fold / seed-to-seed noise
on genuinely small effects, not a disagreement about the physics. For
interface position, their Figure 3(b) similarly finds T2 strongest,
"followed by T3 and T4," with T1/T5 much weaker -- this project's raw-T
ranking for Lc (T2 > T3 > T4 > T1 > T5) and Le (T3 > T2 > T1 > T5 > T4)
again puts the same trio (T2, T3, T4) ahead of the same pair (T1, T5),
consistent with their finding.

**What the engineered features add that raw T1..T5 alone could not
show**: Zhang & Cheng's models take only the five raw set-points as
input, so their SHAP analysis is structurally limited to ranking T1..T5
against each other. This project's engineered features rank *above* the
raw temperatures for Lc and Le: `T_mean` and `dT_12` are the top two SHAP
features for both targets, ahead of every individual raw heater
temperature, and `dT_12` ranks third for D, ahead of three of the five
raw temperatures. This indicates that the overall thermal level and the
zone 1-2 gradient carry explanatory signal beyond what the raw set-points
convey individually -- a finding their feature set could not have
surfaced, and a concrete point of methodological differentiation for a
paper (see Section 1).

## 12. Fast surrogate model (`src/surrogate.py`)

The `FastSurrogate` class wraps the selected model per target (GPR for
all three here), refit on the **full 96-row dataset** (train+test
combined -- CV/test were used only to select and validate the approach).

What it is / is not, explicitly:

* **A data-driven surrogate for the CrysMAS simulation**, not a
  physics-based or hybrid digital twin in its own right. The training data
  is itself CrysMAS's output, so this surrogate is a fast statistical
  approximation of that slower finite-volume solver -- a standard,
  well-established ML pattern -- not a model built from furnace
  measurements. No heat-transfer, fluid-flow, or interface-thermodynamics
  equations are encoded in *this* project's code (CrysMAS has its own, but
  they were not supplied here, so none were invented).
* Fast: well under 1ms per sample for all models, including GPR.
* Provides calibrated predictive uncertainty (GPR standard deviation) for
  every prediction.
* Input validation warns (does not block) when a query point falls outside
  the training data's observed T1..T5 range -- i.e. extrapolation.
* Output validation warns if a prediction falls far outside the observed
  training range for that target.
* Persisted with `joblib` to `models/fast_surrogate.joblib`, including the
  fitted `StandardScaler` for every target (the whole pipeline is saved,
  not just the model).

Example:

```python
from src.surrogate import FastSurrogate
import pandas as pd

surrogate = FastSurrogate.load("models/fast_surrogate.joblib")
X_new = pd.DataFrame([{"T1": 1265, "T2": 1240, "T3": 1210, "T4": 1080, "T5": 1045,
                        "dT_12": 25, "dT_23": 30, "dT_34": 130, "dT_45": 35,
                        "dT_15": 220, "T_mean": 1168}])
result = surrogate.predict(X_new)
print(result["predictions"])      # Lc, Le, D
print(result["uncertainty"])      # GPR std per target
print(result["warnings"])         # extrapolation / out-of-range flags, if any
```

## 13. Inverse-design optimization (`src/optimization.py`) -- new contribution

Zhang & Cheng's own "application to crystal growth practice" (their
Section 3.3) is manual: read a SHAP plot, propose 4 heater changes by
hand, check them against 12 scenarios already present in the dataset. It
is not a systematic search and does not use predictive uncertainty.

`optimize_setpoints(surrogate, target, ...)` instead runs a bounded,
constrained numerical search directly against the trained surrogate:

* **Optimizer**: `scipy.optimize.differential_evolution` -- gradient-free,
  appropriate since the tree-ensemble/GPR surrogates are not
  differentiable in closed form.
* **Physical constraint enforced**: `T1 >= T2 >= T3 >= T4 >= T5`. This is
  not invented here -- it is the same constraint Zhang & Cheng (2022,
  Section 2.2.2) impose in their own design of experiment, because a
  positive axial temperature gradient (hot at top, cold at bottom) is
  physically necessary to drive directional solidification. Candidate
  set-points that violate it are heavily penalized, not silently allowed.
* **Uncertainty-aware**: for targets whose selected model is the Gaussian
  Process, candidate set-points where the model's own predictive
  uncertainty is high are penalized -- the optimizer is actively
  discouraged from recommending furnace settings the surrogate itself is
  unsure about, rather than extrapolating with false confidence.
* **Search bounds default to the observed training range** for each
  heater -- i.e. it searches inside the simulated regime by default, since
  outside it the surrogate's accuracy is unknown.

Demo in `main.py` (target: `D = 0`, a flat/undeflected interface, i.e.
minimal radial thermal stress at the growth interface):

```
Recommended set-points (K): {'T1': 1255.0, 'T2': 1235.1, 'T3': 1215.1, 'T4': 1075.1, 'T5': 1035.0}
Predicted outcome:           {'Lc': -0.2142, 'Le': -0.2145, 'D': 0.00018}
```

Saved to `results/optimization_demo.csv`. **This is a surrogate-driven
recommendation for exploring the CrysMAS response surface, not a
validated furnace control instruction** -- re-running CrysMAS (or a
physical trial) at these set-points is needed before acting on it. See
the "what this is / is not" framing in Section 12 and in
`src/optimization.py`'s module docstring.

## 14. Project structure

```
crystal_growth_ml/
├── data/Table.csv                  # uploaded dataset, unmodified
├── src/
│   ├── data_loader.py              # load + target/column verification
│   ├── preprocessing.py            # quality checks, leakage guard, split
│   ├── feature_engineering.py      # physically-justified derived features
│   ├── models.py                   # 7 models + hyperparameter tuning
│   ├── cross_validation.py         # 5-fold CV (independent + multi-output + pooled)
│   ├── evaluation.py               # test-set metrics + all plots
│   ├── model_comparison.py         # comparison table + CV-only selection
│   ├── surrogate.py                # FastSurrogate: predict/validate/persist
│   ├── explainability.py           # SHAP summary plots (optional dependency)
│   └── optimization.py             # surrogate-driven inverse design
├── models/fast_surrogate.joblib    # trained, persisted surrogate
├── results/*.csv                   # every numeric table referenced above
├── plots/*.png                     # every plot referenced above
├── requirements.txt
├── README.md
└── main.py                         # runs the full pipeline, in order
```

Run everything with:

```bash
pip install -r requirements.txt
python3 main.py
```

Reproducibility: `RANDOM_STATE = 42` is used for the train/test split,
every KFold shuffle, and every stochastic model.

## 15. Limitations and future work

* **The training data is simulated (CrysMAS), not physically measured.**
  Every number in this repository describes how well these models fit
  that simulation's output at 96 sampled operating points -- not how well
  they predict a real furnace. Presenting this work (e.g. in a paper)
  should say so explicitly; see Section 0 for the required citation.
* **n = 96** is small. All CV and test-set numbers above should be
  understood as measured on this specific 96-point simulation design, not
  validated against independent new simulation runs or physical
  experiments.
* The test split (~19 rows) gives high-variance test metrics; 5-fold CV
  on the 77-row training split is the more stable comparison basis and is
  what model selection is based on.
* `D` (interface deflection) is intrinsically harder to predict than Lc/Le
  (R2 ~0.6-0.7 vs. >0.97) because it is a small difference of two larger,
  correlated quantities -- reported plainly, not hidden.
* The pooled-vs-mean-of-folds CV discrepancy with Zhang & Cheng (2022) for
  the `D` target (Section 9) is only partially resolved -- our best model
  (GPR, pooled R2=0.713) does not fully reach their reported RF R2=0.78,
  and the remaining gap's exact cause is not identified here.
* No governing physical equations were supplied directly to this project
  (only CrysMAS's simulated output was), so the surrogate is purely
  data-driven. If Bridgman interface-position equations become available,
  they could be incorporated as (a) a physics-informed mean function in
  the GPR prior, or (b) monotonicity constraints in the tree ensembles --
  either would upgrade this from a data-driven surrogate toward a hybrid
  digital twin, but neither is implemented here since inventing such
  equations was explicitly out of scope.
* xgboost and shap could not be installed in the development sandbox (see
  Sections 4 and 11); results labeled "XGBoost" throughout this repository
  were produced with the documented `GradientBoostingRegressor`
  substitute, and SHAP plots were not generated here. Re-running with
  genuine xgboost/shap installed is expected to give similar, not
  necessarily identical, results.
* Extrapolation beyond the training T1..T5 grid (1195K-1275K across the
  five zones) is not validated and is flagged by `FastSurrogate`'s input
  validation, not silently extrapolated with confidence. The optimizer in
  Section 13 defaults to searching only inside this same range.
* The optimization demo (Section 13) is a single example (target D=0);
  it has not been validated by re-running CrysMAS at the recommended
  set-points, and should be treated as a screening suggestion, not a
  furnace instruction.

## 16. Citation

If you use this dataset (directly or via this pipeline) in a paper,
report, or presentation, cite the original data source and the paper it
was published with:

> Zhang, Chang (2022), "Numerical simulation data of melt-crystal
> interface position and deflection of BaBrCl in a 5-heater vertical
> Bridgman furnace", Mendeley Data, V1, doi: 10.17632/4hr6cx2p8m.1

> Zhang, C. & Cheng, Q. (2022). "Predicting Melt-Crystal Interface Position
> and Shape during the Manufacturing Process of Single Crystal via
> Explainable Machine Learning Models," *IOP Conf. Series: Materials
> Science and Engineering*, 1258, 012029. doi: 10.1088/1757-899X/1258/1/012029

Check the license terms on the [Mendeley dataset
page](https://data.mendeley.com/datasets/4hr6cx2p8m/1) before
republishing the data itself or any large derivative of it.
