# Bridgman Crystal Growth ML Surrogate

A data-driven regression pipeline that predicts three Bridgman crystal-growth
interface-position quantities (**Lc**, **Le**, **D**) from five furnace
zone temperatures (**T1..T5**), compares seven regression models under
5-fold cross-validation, and packages the best model per target as a fast
surrogate for process exploration.

## 0. Critical correction to the original project brief -- read this first

The original brief asked for a **"Crystal Diameter" target called D**.
Direct inspection of the uploaded dataset (`data/Table.csv`, 96 rows) shows
this is not correct:

| Column | What it actually is | Verified relationship |
|---|---|---|
| `Lc (m)` | Axial position of the crystal-side growth interface | genuine, independent |
| `Le (m)` | Axial position of the encapsulant/liquid-side interface | genuine, independent |
| `P (m)` | Midpoint of the two interfaces | **exactly** `(Lc + Le) / 2` for all 96 rows (max abs error 2.8e-17) |
| `D (m)` | Interface **separation/deflection**, not a diameter | **exactly** `Lc - Le` for all 96 rows (max abs error 2.7e-17) |

`D` ranges from about -0.001 m to +0.004 m and is frequently negative --
physically impossible for a diameter. There is no diameter measurement
anywhere in the dataset.

**This was flagged to the user before any modeling began, per the project's
own "do not fabricate target values or column mappings" requirement.** The
user confirmed:
* `D` should be **modeled directly** as an independent regression target
  from T1..T5 (not derived algebraically from Lc/Le), but must be
  **documented honestly as "interface separation (Lc - Le)"**, never as
  "Crystal Diameter", anywhere in this project's code, plots, or tables.
* `P` is **not** modeled (it would add no information beyond Lc and Le).

Every module, plot, and table in this repository uses the corrected,
honest labeling. **Targets modeled: Lc, Le, D** (D = interface separation).

## 1. Dataset

* `data/Table.csv`: 96 rows, 9 columns, **no missing values, no duplicate
  rows**. Every row is a unique combination of (T1..T5) -- this is a small
  **designed experiment / factorial grid**, not repeated trials, batches,
  or a time series.
* Inputs: `T1 (K)` .. `T5 (K)`, five furnace axial-zone temperatures.
  T1 has 2 levels, T2/T3/T4 have 3 levels each, T5 has 2 levels (96 of the
  possible combinations are present).
* A small number of rows sit outside the 1.5x-IQR whisker range for
  Lc/Le/P/D (1-3 rows per column) -- these are the most extreme corner
  points of the temperature grid, not measurement errors, and are **kept**,
  not removed (see `src/preprocessing.py`).
* **Limitation**: n=96 is small for a 7-model comparison with an 80/20
  test split (~19 test rows). Test-set metrics below should be read as
  indicative, not precise generalization estimates. 5-fold CV (run on the
  77-row training split only) is the primary basis for model comparison
  and selection.

## 2. Feature engineering

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

## 3. Models compared

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

## 4. Multi-output vs. independent models

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
per-target model-selection step (Section 6) is more transparent and
allows a different model per target, which mattered here (GPR won on all
three targets, but this is only knowable by comparing independently).

## 5. Cross-validation

`KFold(n_splits=5, shuffle=True, random_state=42)` on the **training split
only** (the test split, ~20% of the data, is never touched by CV or
hyperparameter tuning). GroupKFold / time-aware splitting was considered
and rejected: there are no repeated experiments, batches, or timestamps in
this dataset to leak across folds (every row is a distinct design point).

Full results: `results/cv_results_independent.csv`. Best CV result per
target:

| Target | Selected model | CV MAE (mean) | CV RMSE (mean) | CV R2 (mean) |
|---|---|---|---|---|
| Lc | Gaussian Process | 0.000395 | 0.000517 | **0.994** |
| Le | Gaussian Process | 0.000667 | 0.000914 | **0.977** |
| D  | Gaussian Process | 0.000334 | 0.000536 | 0.665 |

Note the much lower R2 for D across every model (best case 0.665) compared
to Lc/Le (>0.97). This is expected: D is the small difference of two
close, similarly-scaled quantities (Lc and Le), so it inherits the noise
of both while its signal is an order of magnitude smaller -- a harder
regression problem by construction, not a modeling failure. This is
reported plainly rather than glossed over.

## 6. Model selection (from CV results only)

Selection used a rank-average of CV MAE, CV RMSE, and CV R2 (equal
weight -- not R2 alone, per the project requirement), computed **only**
from the 5-fold CV on the training split. The held-out test set was never
used for selection, only for an independent check afterward.

**Gaussian Process Regression won on all three targets** -- by every
metric, not narrowly. It also trains in well under 0.1s at this dataset
size (GPR is O(n^3), trivial at n<=96, so no sparse/inducing-point
approximation was needed) and predicts in well under 0.1ms/sample, so its
usual "does not scale" caveat does not apply here. See
`results/best_model_per_target.csv`.

## 7. Held-out test set (independent check, not used for selection)

| Target | Test MAE | Test RMSE | Test R2 | Inference time/sample |
|---|---|---|---|---|
| Lc | 0.000335 | 0.000442 | 0.996 | 0.068 ms |
| Le | 0.000458 | 0.000680 | 0.987 | 0.066 ms |
| D  | 0.000465 | 0.000822 | 0.617 | 0.066 ms |

Consistent with the CV results (including D's much lower R2). Full table
with all 7 models x 3 targets: `results/model_comparison_table.csv`.

## 8. Plots (`plots/`)

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

`T_mean` (overall furnace thermal level) and `dT_12` (zone 1-2 gradient)
are consistently the top-ranked features for Lc across models -- see
`plots/feature_importance_Lc.png`.

## 9. Fast surrogate model (`src/surrogate.py`)

The `FastSurrogate` class wraps the selected model per target (GPR for
all three here), refit on the **full 96-row dataset** (train+test
combined -- CV/test were used only to select and validate the approach).

What it is / is not, explicitly:

* **Data-driven surrogate**, not a physics-based or hybrid digital twin.
  No heat-transfer, fluid-flow, or interface-thermodynamics equations are
  encoded. No governing equations were supplied for this project, so none
  were invented.
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

## 10. Project structure

```
crystal_growth_ml/
├── data/Table.csv                  # uploaded dataset, unmodified
├── src/
│   ├── data_loader.py              # load + target/column verification
│   ├── preprocessing.py            # quality checks, leakage guard, split
│   ├── feature_engineering.py      # physically-justified derived features
│   ├── models.py                   # 7 models + hyperparameter tuning
│   ├── cross_validation.py         # 5-fold CV (independent + multi-output)
│   ├── evaluation.py               # test-set metrics + all plots
│   ├── model_comparison.py         # comparison table + CV-only selection
│   └── surrogate.py                # FastSurrogate: predict/validate/persist
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

## 11. Limitations and future work

* **n = 96** is small. All CV and test-set numbers above should be
  understood as measured on this specific 96-point designed experiment,
  not validated against independent new experiments or real furnace runs.
* The test split (~19 rows) gives high-variance test metrics; 5-fold CV
  on the 77-row training split is the more stable comparison basis and is
  what model selection is based on.
* `D` (interface separation) is intrinsically harder to predict than Lc/Le
  (R2 ~0.6-0.7 vs. >0.97) because it is a small difference of two larger,
  correlated quantities -- reported plainly, not hidden.
* No governing physical equations were available for this project, so the
  surrogate is purely data-driven. If Bridgman interface-position
  equations become available, they could be incorporated as (a) a
  physics-informed mean function in the GPR prior, or (b) monotonicity
  constraints in the tree ensembles -- either would upgrade this from a
  data-driven surrogate toward a hybrid digital twin, but neither is
  implemented here since no such equations were supplied (inventing them
  was explicitly out of scope).
* xgboost could not be installed in the development sandbox (see Section
  3); results labeled "XGBoost" throughout this repository were produced
  with the documented `GradientBoostingRegressor` substitute. Re-running
  with genuine xgboost installed is expected to give similar, not
  necessarily identical, results.
* Extrapolation beyond the training T1..T5 grid (1195K-1275K across the
  five zones) is not validated and is flagged by `FastSurrogate`'s input
  validation, not silently extrapolated with confidence.
