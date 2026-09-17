# Methodology — Synthetic Data Generation (research-paper-ready section)

*Drafted for direct inclusion in a manuscript methods section. Cross-references (figure/table
numbers) are left as placeholders (`[Fig. X]`, `[Table X]`) for insertion into the target
document; all numerical claims below are drawn directly from this project's own computed
outputs (`synthetic/results/*.csv`, `synthetic/results/*.json`) and are not restated from
memory.*

## Dataset and leakage-safe partitioning

The original dataset (Zhang, 2022, Mendeley Data, DOI: 10.17632/4hr6cx2p8m.1) comprises 96
CrysMAS finite-volume steady-state simulation runs of a five-heater vertical Bridgman furnace
growing a BaBrCl crystal, each parameterized by five heater set-point temperatures
(`T1`–`T5`, K) and producing two independently simulated melt–crystal interface positions
(`Lc` at the centerline, `Le` at the ampoule wall, both in metres) from which the average
position `P = (Lc + Le)/2` and interface deflection `D = Lc − Le` are derived exactly. All 96
rows satisfy the directional-solidification ordering constraint `T1 ≥ T2 ≥ T3 ≥ T4 ≥ T5`
required for this process (Zhang & Cheng, 2022). Prior to any synthetic data generation, the
96 rows were partitioned into original-train (n=67), original-validation (n=14), and
original-test (n=15) subsets using a stratified split on `T2` (the strongest single
input–output correlate, r ≈ −0.65 to −0.67). The original-test partition was held out from
every subsequent step — generation, generator/model tuning, and cross-validation — and used
exactly once, at the end, as an independent final check.

## Methodology selection

Five candidate synthetic-generation approaches were evaluated against the constraints of this
dataset: (i) statistical marginal-distribution sampling, (ii) multivariate Gaussian/copula
generation, (iii) Gaussian Process (GP)-based generation, (iv) physics-informed generation, and
(v) conditional deep generative models. Given the small sample size (n=67 available to the
generator after the leakage-safe split), methods (iv) and (v) were rejected outright: (iv)
would require an independent reduced-order physical model not available to this project, and
fabricating one from 67 points would constitute an unsupported physical claim; (v) requires
sample sizes far beyond 67 to avoid memorization/mode collapse. Method (ii) was rejected
because a 9-dimensional covariance/copula fit at this sample size is unstable and, more
importantly, does not represent the directional, nonlinear input→output relationship already
established for this system via SHAP analysis in prior work on this dataset. **Method (iii),
GP-based generation, was selected**: it is well-suited to small-sample regression, natively
returns a predictive standard deviation usable as a principled per-point noise model, and
this project already possesses an independently validated GP surrogate for this exact
system (cross-validated R² ≈ 0.977–0.994 for `Lc`/`Le`; ≈ 0.665–0.713 for `D`). Statistical
sampling (i) was retained, but only for the five *input* variables, combined with rejection
sampling to enforce the ordering constraint — an appropriate use of method (i), since the
inputs are design variables with no antecedent relationship to model.

## Generation procedure

For each synthetic row, `T1`–`T5` were sampled independently and uniformly within the ranges
observed in original-train, then accepted only if `T1 ≥ T2 ≥ T3 ≥ T4 ≥ T5` (rejection
sampling; 0 constraint violations by construction, matching the 96/96 compliance rate of the
original data). `Lc` and `Le` were then predicted from two independently fitted Gaussian
Process regressors (kernel: constant × RBF + white noise, `scikit-learn`
`GaussianProcessRegressor`, identical architecture to the project's previously validated
surrogate), trained exclusively on original-train (n=67). Gaussian noise with standard
deviation equal to each GP's own predictive standard deviation at the sampled input point was
added to each prediction — i.e., injected noise magnitude is data-driven per point, not a
fixed or arbitrarily tuned constant. `D` and `P` were computed algebraically from the noised
`Lc`/`Le` (`D = Lc − Le`, `P = (Lc + Le)/2`), preserving the exact algebraic identities present
in the original data; an initial design that instead fit an independent GP for `D` directly
was rejected during development because it broke this identity by amounts comparable to `D`'s
entire observed range. `D`'s per-row uncertainty is instead reported as the propagated
quantity `sqrt(σ²_Lc + σ²_Le)` under an explicit (conservative) independence assumption.
Candidate rows within a normalized Euclidean distance of 0.05 (in per-column-standard-deviation
units) of any original input combination were discarded and resampled to prevent
near-duplication of the source design points; candidate outputs falling more than one training
span outside the observed original-train output range were likewise discarded as
implausible extrapolations. 2,000 synthetic rows were generated under a fixed seed
(`SEED=42`), a moderate (~20×) multiple of the original sample size chosen because a
small-sample GP interpolant does not gain additional real information from being sampled more
densely — see Limitations.

## Quality validation

The synthetic dataset was compared to the full original 96-row sample using two-sample
Kolmogorov–Smirnov tests and Wasserstein distance per column, pairwise Pearson correlation
comparison across all nine columns, exact- and near-duplicate detection on the input space,
IQR-based outlier screening against the original data's own fences, and a coarse linear
input–output coefficient-sign comparison. `Lc`, `Le`, and `P` distributions were statistically
indistinguishable from the original sample by the KS test (p > 0.1 for all three); `D`'s
distribution likewise passed (p = 0.115) but showed the largest correlation-structure
divergence of any column (|Δr| up to 0.42, vs. < 0.05 for all non-`D` pairs) and the highest
outlier rate (1.85% of synthetic rows vs. ≤0.1% for `Lc`/`Le`/`P`), consistent with `D`'s known
status as the weakest-modeled target. `T1`–`T5` distributions did not match the original by
design (the original inputs are a discrete factorial design; the synthetic inputs are sampled
continuously) — this divergence is mechanical, expected, and does not indicate a generation
defect. No exact or near-duplicate synthetic input row was found relative to the original 96.

## ML evaluation protocol

Model performance was compared under two training regimes — original-only and
synthetic-augmented — across seven candidate regressors (Linear Regression, Ridge, SVR,
Random Forest, HistGradientBoosting, Gaussian Process Regression, and XGBoost when available)
for each of the three modeled targets (`Lc`, `Le`, `D`) independently. Evaluation used
repeated 5-fold cross-validation (5 repeats, 25 folds total, varying random partitions) over
the original train+validation pool (n=81), with synthetic augmentation applied only to each
fold's training partition and evaluation always performed on the held-out real (original)
fold — never on synthetic rows. A single final check fit each model on the full original
train+validation pool, with and without the complete synthetic set, and evaluated once on the
original-test partition (n=15), untouched by every prior step. This design directly targets
the risk that synthetic augmentation could otherwise appear to improve performance only
because it makes the evaluation set easier, rather than because it improves generalization to
real data — see `synthetic/06_MODEL_EVALUATION_REPORT.md` for the resulting numbers and
whether an augmentation benefit was or was not observed on the untouched original test set.

## Limitations

n = 96 (67 for the generator) is a small sample for any generative method. The selected GP
approach is a statistical interpolant of the existing 96-row CrysMAS simulation output, not an
independent physical model, and is not used or validated outside the observed input ranges. No
claim of physical validity, experimental confirmation, or new CrysMAS simulation is made for
any synthetic value produced by this pipeline.
