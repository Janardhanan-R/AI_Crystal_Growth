"""
candidate_exploration.py
=========================
Candidate exploration: generate a large batch of VIRTUAL heater-temperature
candidates, evaluate them with the already-trained FastSurrogate, flag how
well-supported each one is by the original 96-row dataset, and rank/export
them -- WITHOUT training any new model and WITHOUT retrofitting ground-truth
CrysMAS outputs into the ranking.

Relationship to src/optimization.py
-------------------------------------
`optimization.py` (existing) runs a single, bounded numerical SEARCH
(differential evolution) for the one best set-point combination that hits a
specific target outcome. This module is the complementary, broader-coverage
tool: it SAMPLES many candidates across the explored space, scores all of
them, and lets you inspect the whole population -- useful for understanding
the shape of the response surface, sanity-checking the optimizer's output
against nearby alternatives, or screening a shortlist by hand. It does not
replace the optimizer and does not re-run or modify it.

What this module is NOT
-------------------------
* NOT a new model. Every prediction here comes from the same persisted
  `FastSurrogate` used elsewhere in this project (GPR for Lc, Le, D per the
  current README) -- no retraining happens in this module.
* NOT a physical validation. A candidate surviving every check here (in
  range, ordering-satisfied, inside the training data's convex hull) is
  merely well-SUPPORTED by the existing simulation sample -- it is not a
  confirmed CrysMAS result and never a "confirmed optimum". Re-running
  CrysMAS (or a physical trial) is required before acting on any candidate.
* NOT a diameter model. `D` is interface deflection (`Lc - Le`), never a
  crystal diameter -- see src/data_loader.py.

Uncertainty caveat
-------------------
GPR predictive standard deviation is a MODEL-INTERNAL uncertainty estimate
(how far a point is, in the GPR's kernel-induced metric, from the training
data), not a calibrated physical confidence interval -- no calibration study
(e.g. checking empirical coverage of GPR intervals against independent
CrysMAS runs) has been performed in this project. It is used here only as a
relative SCREENING signal between candidates, never presented as a proven
probability.

Convex-hull screening
-----------------------
Containment in the 5-D convex hull of the training (T1..T5) points is
tested with `scipy.spatial.Delaunay.find_simplex` (verified numerically
stable on this dataset: 96 points, 42 hull vertices in 5-D, see the
`_build_hull` docstring for the fallback if it ever fails on a re-run with
different data). Being inside the hull means the candidate is a convex
combination of already-simulated points -- it does NOT guarantee accurate
interpolation (the GPR could still be locally uncertain inside the hull),
and being outside the hull does not guarantee a bad prediction, only that
it is a genuine extrapolation in the strict geometric sense. Both checks
are reported; neither is treated as sufficient on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from scipy.stats import qmc

from .data_loader import INPUT_FEATURES, TARGETS, load_raw
from .feature_engineering import add_engineered_features

RAW_ORDER = INPUT_FEATURES  # ["T1", "T2", "T3", "T4", "T5"]

SUPPORT_IN_RANGE = "in_range"
SUPPORT_OUTSIDE_RANGE = "outside_range"
SUPPORT_OUTSIDE_HULL = "outside_convex_hull"
SUPPORT_INVALID_CONSTRAINT = "invalid_constraint"


# --------------------------------------------------------------------------- config

@dataclass
class ExplorationConfig:
    """All tunable knobs for one exploration run, in one place, so a run is
    fully reproducible from this object alone plus the surrogate/dataset."""
    n_samples: int = 1000
    random_state: int = 42
    sampling_strategy: str = "uniform"          # "uniform" | "lhs"
    dedupe_decimals: int = 4                    # round T1..T5 to this many decimals
                                                 # (K) before checking for duplicates
    max_generation_attempts_multiplier: int = 50  # rejection-sampling safety cap
    uncertainty_flag_percentile: float = 90.0    # candidates at/above this percentile
                                                 # of THIS candidate set's own aggregate
                                                 # uncertainty are flagged high_uncertainty
    objective: str = "flat_interface"            # "flat_interface" | "low_deflection" | "multi_objective"
    multi_objective_weights: dict = field(default_factory=lambda: {
        "abs_D": 0.5, "uncertainty": 0.3, "support_penalty": 0.2,
    })


# --------------------------------------------------------------------------- 1. generation

def _sample_uniform(n: int, bounds: dict, rng: np.random.Generator) -> np.ndarray:
    lo = np.array([bounds[f][0] for f in RAW_ORDER])
    hi = np.array([bounds[f][1] for f in RAW_ORDER])
    return rng.uniform(lo, hi, size=(n, len(RAW_ORDER)))


def _sample_lhs(n: int, bounds: dict, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sampling over the 5-D box -- better space-filling
    than plain uniform sampling for a fixed candidate budget, at the cost
    of losing pure i.i.d. independence between samples. Optional; default
    strategy is "uniform" per the task's "keep the default simple and
    reliable" instruction."""
    sampler = qmc.LatinHypercube(d=len(RAW_ORDER), seed=rng)
    unit = sampler.random(n=n)
    lo = np.array([bounds[f][0] for f in RAW_ORDER])
    hi = np.array([bounds[f][1] for f in RAW_ORDER])
    return qmc.scale(unit, lo, hi)


def _get_bounds_from_surrogate(surrogate) -> dict:
    """Reuse the surrogate's own stored feature_ranges (computed from the
    real training data at fit time -- see src/surrogate.py FastSurrogate.fit)
    rather than hardcoding or re-deriving bounds, per the task instruction
    to use the existing dataset/project configuration."""
    missing = set(RAW_ORDER) - set(surrogate.feature_ranges)
    if missing:
        raise ValueError(f"Surrogate is missing feature_ranges for {missing}; "
                          "cannot determine sampling bounds from it.")
    return {f: surrogate.feature_ranges[f] for f in RAW_ORDER}


def generate_candidates(
    surrogate,
    config: ExplorationConfig | None = None,
    bounds: dict | None = None,
) -> pd.DataFrame:
    """Generate `config.n_samples` virtual (T1..T5) candidates.

    - Bounds default to the surrogate's own observed training range per
      heater (`surrogate.feature_ranges`), i.e. the same range the model
      was actually fit on -- override with `bounds={"T1": (lo, hi), ...}`
      only if you deliberately want to explore beyond it (not recommended
      without treating the result as extrapolation).
    - Rejection-samples until every candidate satisfies the physical
      ordering constraint T1 >= T2 >= T3 >= T4 >= T5 (same constraint
      enforced in src/optimization.py, sourced from Zhang & Cheng 2022,
      Section 2.2.2).
    - Drops duplicate candidates (T1..T5 rounded to `dedupe_decimals`)
      where practical; if generation cannot reach the requested count
      after `max_generation_attempts_multiplier`x oversampling, returns
      fewer rows and records this in the returned DataFrame's attrs
      (checked/reported by the caller, never silently padded with
      duplicates).

    Returns a DataFrame with columns T1..T5 and a stable `candidate_id`
    (0-indexed), NOT yet evaluated by the surrogate.
    """
    config = config or ExplorationConfig()
    if config.n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {config.n_samples}")
    if config.sampling_strategy not in ("uniform", "lhs"):
        raise ValueError(f"sampling_strategy must be 'uniform' or 'lhs', got {config.sampling_strategy!r}")

    resolved_bounds = bounds or _get_bounds_from_surrogate(surrogate)
    for f in RAW_ORDER:
        lo, hi = resolved_bounds[f]
        if lo > hi:
            raise ValueError(f"Invalid bounds for {f}: lo={lo} > hi={hi}")

    rng = np.random.default_rng(config.random_state)
    sampler_fn = _sample_lhs if config.sampling_strategy == "lhs" else _sample_uniform

    collected: list[np.ndarray] = []
    seen_keys: set[tuple] = set()
    attempts = 0
    max_attempts = config.n_samples * config.max_generation_attempts_multiplier

    while sum(len(c) for c in collected) < config.n_samples and attempts < max_attempts:
        batch_n = max(config.n_samples * 2, 200)
        batch = sampler_fn(batch_n, resolved_bounds, rng)
        attempts += batch_n

        ordered_ok = np.all(np.diff(batch, axis=1) <= 0, axis=1)  # T1>=T2>=...>=T5
        batch = batch[ordered_ok]

        kept_rows = []
        for row in batch:
            key = tuple(np.round(row, config.dedupe_decimals))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            kept_rows.append(row)
        if kept_rows:
            collected.append(np.array(kept_rows))

    all_rows = np.concatenate(collected, axis=0) if collected else np.empty((0, len(RAW_ORDER)))
    n_generated = len(all_rows)
    shortfall = config.n_samples - n_generated
    all_rows = all_rows[: config.n_samples]

    df = pd.DataFrame(all_rows, columns=RAW_ORDER)
    df.insert(0, "candidate_id", range(len(df)))
    df.attrs["requested_n_samples"] = config.n_samples
    df.attrs["n_generated"] = len(df)
    df.attrs["shortfall"] = max(0, shortfall)
    df.attrs["sampling_strategy"] = config.sampling_strategy
    df.attrs["random_state"] = config.random_state
    df.attrs["bounds_used"] = resolved_bounds
    df.attrs["is_physically_validated"] = False  # explicit, never implied otherwise
    return df


# --------------------------------------------------------------------------- 2. evaluation

def evaluate_candidates(candidates: pd.DataFrame, surrogate) -> pd.DataFrame:
    """Compute engineered features (via the SAME add_engineered_features
    used everywhere else in this project) and predict Lc/Le/D + uncertainty
    for every candidate with the already-trained surrogate. Does NOT train
    anything and does NOT consult any ground-truth CrysMAS value."""
    if candidates.empty:
        raise ValueError("evaluate_candidates: no candidates to evaluate (empty DataFrame).")

    raw = candidates[RAW_ORDER].copy()
    engineered = add_engineered_features(raw)
    missing_feats = set(surrogate.feature_names) - set(engineered.columns)
    if missing_feats:
        raise ValueError(f"Feature engineering did not produce required surrogate "
                          f"features: {missing_feats}")

    result = surrogate.predict(engineered[surrogate.feature_names], return_std=True)
    preds = result["predictions"]
    unc = result["uncertainty"]

    out = candidates.copy()
    for t in TARGETS:
        out[f"pred_{t}"] = preds[t].values if t in preds.columns else np.nan
        if unc is not None and t in unc.columns:
            out[f"unc_{t}"] = unc[t].values
        else:
            out[f"unc_{t}"] = np.nan
    for col in ["dT_12", "dT_23", "dT_34", "dT_45", "dT_15", "T_mean"]:
        out[col] = engineered[col].values

    out.attrs.update(candidates.attrs)
    out.attrs["surrogate_predict_warnings"] = result["warnings"]
    return out


# --------------------------------------------------------------------------- 3. support / extrapolation checks

def _build_hull_triangulation(training_points: np.ndarray) -> Delaunay | None:
    """Attempt a Delaunay triangulation of the training (T1..T5) points for
    convex-hull containment testing. Verified to succeed on the current
    96-row dataset (42 hull vertices in 5-D). If it ever fails on a
    different/future dataset (e.g. QhullError from near-degenerate,
    lower-rank point clouds), returns None and the caller records
    convex-hull screening as unavailable rather than fabricating an
    answer -- see check_candidate_support."""
    try:
        return Delaunay(training_points)
    except QhullError:
        return None


def check_candidate_support(
    candidates: pd.DataFrame,
    surrogate,
    training_data_path: str = "data/Table.csv",
) -> pd.DataFrame:
    """Add support / extrapolation-screening columns to an evaluated
    candidate set:

      in_range_T1..T5      bool, per-heater, against surrogate.feature_ranges
      all_in_range         bool, AND of the above
      ordering_satisfied   bool, T1>=T2>=T3>=T4>=T5
      inside_convex_hull   bool or NaN (NaN = hull screening unavailable)
      convex_hull_checked  bool, whether hull screening actually ran
      support_category     one of: in_range / outside_range /
                            outside_convex_hull / invalid_constraint
                            (priority: invalid_constraint > outside_range >
                            outside_convex_hull > in_range -- the first
                            applicable label wins, so a candidate with
                            multiple issues is not silently only flagged
                            for the least severe one)

    Being "in_range" and "inside_convex_hull" means the candidate is
    well-SUPPORTED by the training sample -- it is NOT a claim of physical
    validity or that the GPR prediction there is accurate; see this
    module's docstring.
    """
    out = candidates.copy()

    for f in RAW_ORDER:
        lo, hi = surrogate.feature_ranges[f]
        out[f"in_range_{f}"] = (out[f] >= lo) & (out[f] <= hi)
    out["all_in_range"] = out[[f"in_range_{f}" for f in RAW_ORDER]].all(axis=1)

    ordering_matrix = out[RAW_ORDER].values
    out["ordering_satisfied"] = np.all(np.diff(ordering_matrix, axis=1) <= 1e-9, axis=1)

    training_df = load_raw(training_data_path)
    training_points = training_df[RAW_ORDER].values.astype(float)
    tri = _build_hull_triangulation(training_points)
    if tri is not None:
        simplex_idx = tri.find_simplex(out[RAW_ORDER].values.astype(float))
        out["inside_convex_hull"] = simplex_idx >= 0
        out["convex_hull_checked"] = True
    else:
        out["inside_convex_hull"] = np.nan
        out["convex_hull_checked"] = False

    def _categorize(row) -> str:
        if not row["ordering_satisfied"]:
            return SUPPORT_INVALID_CONSTRAINT
        if not row["all_in_range"]:
            return SUPPORT_OUTSIDE_RANGE
        if row["convex_hull_checked"] and not row["inside_convex_hull"]:
            return SUPPORT_OUTSIDE_HULL
        return SUPPORT_IN_RANGE

    out["support_category"] = out.apply(_categorize, axis=1)
    out.attrs.update(candidates.attrs)
    out.attrs["convex_hull_screening_available"] = tri is not None
    return out


# --------------------------------------------------------------------------- 4. objectives

def compute_objectives(candidates: pd.DataFrame, config: ExplorationConfig | None = None) -> pd.DataFrame:
    """Compute objective/screening values. Documented explicitly, per the
    task's "avoid arbitrary scientific weights without documenting them"
    requirement:

      abs_D               |pred_D| -- a SURROGATE-DEFINED interface-shape
                           objective (flat interface), NOT a proxy for
                           thermal stress or crystal quality on its own.
      aggregate_uncertainty
                           mean of unc_Lc, unc_Le, unc_D, each first
                           min-max normalized to [0, 1] ACROSS THIS
                           CANDIDATE SET (so it is only meaningful as a
                           relative ranking within this one exploration
                           run, not an absolute or cross-run quantity).
                           Equal weight across the 3 targets -- documented
                           here as a simple, transparent choice, not a
                           scientifically derived weighting.
      support_penalty      0.0 if support_category == in_range, else 1.0.
                           Binary, transparent; used only inside the
                           multi-objective score below.
      screening_score       config.multi_objective_weights-weighted sum of
                           (normalized abs_D, normalized aggregate_uncertainty,
                           support_penalty). Lower is better. Weights default
                           to {abs_D: 0.5, uncertainty: 0.3, support: 0.2}
                           and are fully configurable via
                           ExplorationConfig.multi_objective_weights -- there
                           is no claim that this specific weighting is
                           physically optimal.
    """
    config = config or ExplorationConfig()
    out = candidates.copy()

    out["abs_D"] = out["pred_D"].abs()

    unc_cols = [c for c in ["unc_Lc", "unc_Le", "unc_D"] if c in out.columns]
    norm_unc = pd.DataFrame(index=out.index)
    for c in unc_cols:
        vals = out[c]
        span = vals.max() - vals.min()
        norm_unc[c] = (vals - vals.min()) / span if span > 0 else 0.0
    out["aggregate_uncertainty"] = norm_unc.mean(axis=1) if unc_cols else np.nan

    if "support_category" in out.columns:
        out["support_penalty"] = (out["support_category"] != SUPPORT_IN_RANGE).astype(float)
    else:
        out["support_penalty"] = np.nan

    d_span = out["abs_D"].max() - out["abs_D"].min()
    norm_abs_D = (out["abs_D"] - out["abs_D"].min()) / d_span if d_span > 0 else out["abs_D"] * 0.0

    if config.objective == "multi_objective" and unc_cols and "support_category" in out.columns:
        w = config.multi_objective_weights
        out["screening_score"] = (
            w.get("abs_D", 0.0) * norm_abs_D
            + w.get("uncertainty", 0.0) * out["aggregate_uncertainty"].fillna(0.0)
            + w.get("support_penalty", 0.0) * out["support_penalty"].fillna(1.0)
        )
    else:
        out["screening_score"] = np.nan

    # High-uncertainty screening flag: a configurable PERCENTILE threshold
    # derived from THIS candidate set's own aggregate_uncertainty -- an
    # explicit relative screening rule, not a calibrated absolute cutoff.
    if unc_cols:
        threshold = np.nanpercentile(out["aggregate_uncertainty"], config.uncertainty_flag_percentile)
        out["high_uncertainty_flag"] = out["aggregate_uncertainty"] >= threshold
        out.attrs["uncertainty_flag_threshold"] = float(threshold)
        out.attrs["uncertainty_flag_percentile"] = config.uncertainty_flag_percentile
    else:
        out["high_uncertainty_flag"] = np.nan

    out.attrs.update(candidates.attrs)
    return out


# --------------------------------------------------------------------------- 5. filtering

def filter_candidates(
    candidates: pd.DataFrame,
    supported_only: bool = False,
    exclude_high_uncertainty: bool = False,
    max_abs_D: float | None = None,
) -> pd.DataFrame:
    """Configurable filtering. Every filter is opt-in (defaults keep all
    rows) so callers can see the full unfiltered population if they want
    it -- filtering never happens silently inside generate/evaluate."""
    out = candidates
    if supported_only:
        out = out[out["support_category"] == SUPPORT_IN_RANGE]
    if exclude_high_uncertainty and "high_uncertainty_flag" in out.columns:
        out = out[~out["high_uncertainty_flag"].fillna(False)]
    if max_abs_D is not None:
        out = out[out["abs_D"] <= max_abs_D]
    return out.copy()


# --------------------------------------------------------------------------- 6. ranking

def rank_candidates(candidates: pd.DataFrame, by: str = "abs_D", supported_only: bool = False) -> pd.DataFrame:
    """Rank candidates by one of several transparent, single-purpose views
    (no single unexplained universal score is imposed):

      "abs_D"          ascending |predicted D| (flat-interface objective)
      "uncertainty"     ascending aggregate_uncertainty
      "screening_score" ascending combined multi-objective score (only
                        meaningful if compute_objectives ran with
                        config.objective == "multi_objective")

    `supported_only=True` restricts the ranking to support_category ==
    "in_range" candidates FIRST, so extrapolated/invalid candidates never
    rise to the top of a ranking unless the caller explicitly asks to
    include them (supported_only=False, the default, still leaves the
    support_category column visible so every consumer of the ranked table
    can see which rows are extrapolated).
    """
    valid_by = {"abs_D", "uncertainty", "screening_score"}
    if by not in valid_by:
        raise ValueError(f"`by` must be one of {valid_by}, got {by!r}")
    col = {"abs_D": "abs_D", "uncertainty": "aggregate_uncertainty", "screening_score": "screening_score"}[by]
    if col not in candidates.columns:
        raise ValueError(f"Column '{col}' not present -- did you run compute_objectives first?")

    pool = candidates[candidates["support_category"] == SUPPORT_IN_RANGE] if supported_only else candidates
    ranked = pool.sort_values(col, ascending=True, na_position="last").reset_index(drop=True)
    ranked.insert(0, f"rank_{by}", range(1, len(ranked) + 1))
    return ranked


def top_n_views(candidates: pd.DataFrame, n_values=(5, 10, 20)) -> dict:
    """Top-5/10/20 candidates under each of the standard ranking views,
    only including an (n, view) pair if enough valid candidates exist for
    it -- never pads a short ranking with unlabeled/invalid rows."""
    views = {}
    for by in ["abs_D", "uncertainty", "screening_score"]:
        if by == "screening_score" and candidates["screening_score"].isna().all():
            continue
        for supported_only, label in [(True, "supported_only"), (False, "all_candidates")]:
            ranked = rank_candidates(candidates, by=by, supported_only=supported_only)
            for n in n_values:
                key = f"{by}__{label}__top{n}"
                if len(ranked) >= 1:
                    views[key] = ranked.head(min(n, len(ranked)))
                else:
                    views[key] = ranked.head(0)
    return views


# --------------------------------------------------------------------------- 7. export

CORE_EXPORT_COLUMNS = (
    ["candidate_id"] + RAW_ORDER
    + ["dT_12", "dT_23", "dT_34", "dT_45", "dT_15", "T_mean"]
    + [f"pred_{t}" for t in TARGETS] + [f"unc_{t}" for t in TARGETS]
    + ["abs_D", "aggregate_uncertainty", "support_penalty", "screening_score", "high_uncertainty_flag"]
    + [f"in_range_{f}" for f in RAW_ORDER]
    + ["all_in_range", "ordering_satisfied", "inside_convex_hull", "convex_hull_checked", "support_category"]
)


def export_candidates(
    candidates: pd.DataFrame,
    output_dir: str = "results",
    all_filename: str = "candidate_exploration_all.csv",
    top10_filename: str = "candidate_exploration_top10.csv",
    summary_filename: str = "candidate_exploration_summary.json",
    top10_ranking: str = "abs_D",
) -> dict:
    """Writes the three required output files. Never overwrites existing
    optimization results (results/optimization_demo.csv) -- uses its own,
    distinctly-named files. Column meanings/units are documented in this
    module's docstrings and in README.md's Candidate Exploration section
    (T1..T5 in K, Lc/Le/D and their uncertainties in m, abs_D in m,
    aggregate_uncertainty/support_penalty/screening_score dimensionless)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_cols = [c for c in CORE_EXPORT_COLUMNS if c in candidates.columns]
    all_path = out_dir / all_filename
    candidates[export_cols].to_csv(all_path, index=False)

    top10 = rank_candidates(candidates, by=top10_ranking, supported_only=False).head(10)
    top10_path = out_dir / top10_filename
    top10[[f"rank_{top10_ranking}"] + export_cols].to_csv(top10_path, index=False)

    summary = {
        "n_candidates_requested": candidates.attrs.get("requested_n_samples"),
        "n_candidates_generated": candidates.attrs.get("n_generated", len(candidates)),
        "generation_shortfall": candidates.attrs.get("shortfall", 0),
        "sampling_strategy": candidates.attrs.get("sampling_strategy"),
        "random_state": candidates.attrs.get("random_state"),
        "bounds_used_K": candidates.attrs.get("bounds_used"),
        "convex_hull_screening_available": candidates.attrs.get("convex_hull_screening_available"),
        "uncertainty_flag_threshold": candidates.attrs.get("uncertainty_flag_threshold"),
        "uncertainty_flag_percentile": candidates.attrs.get("uncertainty_flag_percentile"),
        "n_supported_in_range": int((candidates["support_category"] == SUPPORT_IN_RANGE).sum())
            if "support_category" in candidates.columns else None,
        "support_category_counts": candidates["support_category"].value_counts().to_dict()
            if "support_category" in candidates.columns else None,
        "is_physically_validated": False,
        "notes": (
            "This exploration is a SCREENING tool over a data-driven surrogate of a "
            "CrysMAS simulation, not a validated furnace controller. No candidate here "
            "is a confirmed optimum or a confirmed CrysMAS/physical result. D is "
            "interface deflection (Lc - Le), never a crystal diameter."
        ),
    }
    summary_path = out_dir / summary_filename
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {"all_csv": str(all_path), "top10_csv": str(top10_path), "summary_json": str(summary_path)}


# --------------------------------------------------------------------------- orchestration

def run_candidate_exploration(
    surrogate,
    config: ExplorationConfig | None = None,
    output_dir: str = "results",
    plots_dir: str | None = "plots/candidate_exploration",
    training_data_path: str = "data/Table.csv",
) -> pd.DataFrame:
    """Runs the full pipeline: generate -> evaluate -> support-check ->
    objectives -> export (+ plots if plots_dir is not None). Returns the
    final, fully-annotated candidate DataFrame. Does not train any model."""
    config = config or ExplorationConfig()

    candidates = generate_candidates(surrogate, config)
    if candidates.attrs.get("shortfall", 0) > 0:
        print(f"WARNING: candidate generation produced only "
              f"{candidates.attrs['n_generated']}/{candidates.attrs['requested_n_samples']} "
              f"unique, ordering-valid candidates after "
              f"{config.max_generation_attempts_multiplier}x oversampling. "
              "Proceeding with the smaller set rather than padding with duplicates.")
    if candidates.empty:
        raise RuntimeError("No valid candidates could be generated -- check ExplorationConfig "
                            "bounds and max_generation_attempts_multiplier.")

    candidates = evaluate_candidates(candidates, surrogate)
    candidates = check_candidate_support(candidates, surrogate, training_data_path)
    candidates = compute_objectives(candidates, config)

    export_paths = export_candidates(candidates, output_dir=output_dir)
    print(f"Candidate exploration: {len(candidates)} candidates evaluated, "
          f"{int((candidates['support_category'] == SUPPORT_IN_RANGE).sum())} fully supported "
          f"(in_range + ordering_satisfied + inside_convex_hull).")
    print(f"Exported: {export_paths}")

    if plots_dir is not None:
        from .candidate_exploration_plots import make_all_plots
        make_all_plots(candidates, plots_dir)
        print(f"Plots written to {plots_dir}/")

    return candidates
