# Methodology Selection for Synthetic Data Generation

## Constraint that governs this decision

n = 96, a constrained-factorial DOE with strongly non-independent inputs (see
`01_DATASET_ANALYSIS_REPORT.md`). Any method that needs to *learn* a distribution or a
relationship is learning it from at most ~67 rows once a leakage-safe train/val/test split is
taken (see `03_split_original_data.py`). That is stated plainly wherever it is relevant below;
no method here is described as more capable than the data supports.

## The five candidate methods, compared on this specific dataset

**1. Statistical distribution-based sampling** (fit a marginal distribution per column, sample
independently). *Pro*: simplest, most transparent, lowest overfitting risk at small n.
*Con*: **cannot** reproduce the `T1..T5` factorial structure or the `T1≥T2≥...≥T5` ordering
constraint if sampled independently per column, and captures no input→output relationship at
all — it would need `Lc/Le/D` sampled from their own marginals, completely decoupled from
which heater settings produced them. This fails the user's "model nonlinear relationships"
requirement outright and was rejected for the output columns, though it *is* used (with the
ordering constraint enforced by rejection sampling) for the input columns — see §"Selected
approach."

**2. Multivariate Gaussian / copula-based generation.** *Pro*: captures linear/rank
correlation structure between all 9 columns at once, more expressive than #1. *Con*: a
copula or multivariate-Gaussian fit at n≈67 (post-split) over 9 dimensions is fitting on the
order of one sample per 7–8 correlation-matrix parameters — a known small-sample regime where
fitted correlations are unstable. More importantly, it treats `T1..T5→outputs` as symmetric
correlations rather than a directional input→output relationship, so it cannot honor "held-out
test inputs must never be seen" in a principled way (the whole joint distribution is fit on
whatever data it's given, inputs and outputs together) and cannot express the genuinely
nonlinear response the dataset's own SHAP analysis (earlier in this project) already showed
(engineered features like `T_mean`, `dT_12` outrank raw temperatures for `Lc`/`Le` — a
nonlinear/interaction effect a Gaussian copula does not represent). Rejected as the primary
output-generation mechanism.

**3. Gaussian Process (GP)-based generation.** *Pro*: models `outputs = f(T1..T5)` directly
(the correct causal direction), is well-suited to small-n regression (GPs are a standard
small-data method precisely because they do not require large training sets to fit, unlike
deep generative models), and — critically for this project — **a GPR surrogate for this exact
dataset already exists and was already validated** (`src/surrogate.py`, `FastSurrogate`,
selected as the best of 7 candidate models for all three targets in this project's main
pipeline: CV R² ≈0.994 for Lc, ≈0.977 for Le, ≈0.665–0.713 for D). GPs also natively provide a
predictive standard deviation per prediction, which gives a principled, data-driven noise
model for synthetic outputs (see §"Selected approach") rather than an arbitrary noise
injection. *Con*: still a model fit on <100 rows, so it will not extrapolate reliably outside
the observed T1–T5 ranges, and it inherits the same weak-target problem for `D` that the
original pipeline already found.

**4. Physics-informed synthetic data generation** (e.g., fit or assume an analytic VB
heat-transfer relationship between heater set points and interface position, then sample
around it). *Pro*: would be the most defensible approach *if* a validated analytic or reduced-
order physical model were available. *Con*: **no such model is available in this project.**
Deriving one would require either the actual CrysMAS finite-volume solver (not available here)
or fitting an empirical physical law to 96 points and asserting it as physics — which is
exactly the fabrication the user's brief explicitly forbids ("do not fabricate... physical
laws"). This method is honestly rejected for that reason: the dataset is too small and this
project has no independent physical model to inform it with. (The GP-based approach in #3 is
"physics-*aware*" only in the weak sense that it respects the ordering constraint and known
sign conventions — it is not a physics-informed *model* in the reduced-order-physics sense.)

**5. Conditional generative models** (GANs, VAEs, diffusion, conditional normalizing flows).
*Pro*: state of the art for large tabular datasets. *Con*: these methods need hundreds to
thousands of training rows to fit stably and are well documented to mode-collapse or overfit
catastrophically below that; at n≈67 training rows they would almost certainly memorize/
near-duplicate the training set rather than learn a generalizable distribution — precisely the
"blindly duplicate/randomly perturb" failure mode the user's brief explicitly forbids.
Rejected outright as inappropriate for this sample size.

## Selected approach: GP-based generation (method 3), with rejection-sampled factorial inputs

1. **Inputs (`T1..T5`)**: generated by sampling *within* the observed per-column ranges (not
   the discrete DOE levels themselves, since treating the output as continuous is more honest
   about what "synthetic" should mean here — see `04_generate_synthetic_data.py`), then
   rejecting any draw that violates `T1≥T2≥T3≥T4≥T5`. This is method **#1** applied correctly
   to inputs only (statistical sampling is appropriate here because the inputs have no
   antecedent to model — they are the design variables), combined with a hard physical
   constraint, not a statistical one.
2. **Outputs (`Lc`, `Le`)**: predicted from a `GaussianProcessRegressor` (matching the
   project's own already-validated `FastSurrogate` architecture) **refit only on the
   leakage-safe original-train split** (never on validation or test rows — see
   `03_split_original_data.py`), plus noise drawn from that GP's own predictive standard
   deviation at each generated input point — i.e., the injected noise is exactly the model's
   own stated uncertainty, not an arbitrary or tuned number.
3. **`D`**: predicted the same way, but explicitly flagged in every deliverable as
   substantially higher-uncertainty (train-split GPR R² for `D` is expected to be well below
   `Lc`/`Le`'s, consistent with the ≈0.665–0.713 already found on the full dataset). No
   confidence is claimed for `D` that the underlying model does not have.
4. **`P`**: computed algebraically as `(Lc + Le)/2` from the generated `Lc`/`Le`, exactly as
   in the original data — never generated independently.

## Explicit limitation statement (per the user's requirement)

This dataset is too small (n≈67 after a leakage-safe train split) to establish a reliable
independent *physical* law relating heater temperatures to interface position — the GP-based
generator above is a **statistical interpolant of the existing simulation output**, not a new
physical model, and every synthetic row is labeled as such. Extrapolation outside the observed
T1–T5 ranges is not attempted and is called out as unsupported. The generator will reproduce
the *statistical* relationships visible in the 96-row CrysMAS sample; it cannot validate
whether those relationships hold outside that sample, and no deliverable in this project
claims that it does.
