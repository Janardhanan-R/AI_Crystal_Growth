"""
optimization.py
================
Surrogate-driven INVERSE DESIGN: given a target interface outcome (some
subset of Lc, Le, D), search for furnace heater set-points (T1..T5) that
achieve it, using the trained FastSurrogate as a fast stand-in for the
CrysMAS simulation.

Why this is the main new contribution over Zhang & Cheng (2022)
------------------------------------------------------------------
Zhang & Cheng's own "application to crystal growth practice" (their
Section 3.3) is a MANUAL exercise: they read a SHAP summary plot, proposed
4 candidate heater actions by hand, and checked the outcome against 12
scenarios already present in their 96-point dataset. It is not a
systematic search, and it does not use predictive uncertainty at all.

This module instead runs a bounded, constrained numerical optimization
(scipy's differential evolution -- gradient-free, appropriate since the
tree-ensemble surrogates are not differentiable) directly over the
surrogate's predicted outputs, and -- when the target's selected model is
the Gaussian Process -- PENALIZES candidate set-points where the model's
own predictive uncertainty is high. That means the optimizer is actively
discouraged from recommending furnace settings the surrogate is unsure
about, rather than silently extrapolating with false confidence. This is
still only as good as the underlying simulation-trained surrogate (see
src/surrogate.py's "what this is/is not" docstring) -- it is a fast
screening tool for exploring the CrysMAS response surface, not a
substitute for re-running CrysMAS or a physical experiment to confirm a
recommended set-point.

Physical constraint enforced: T1 >= T2 >= T3 >= T4 >= T5. This is not
invented here -- it is the same constraint Zhang & Cheng (2022, Section
2.2.2) impose during their own design of experiment, because positive
axial temperature gradients (hot at top, cold at bottom) are physically
necessary to drive directional solidification in this furnace. Candidate
set-points that violate it are heavily penalized, not silently allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .feature_engineering import add_engineered_features

RAW_ORDER = ["T1", "T2", "T3", "T4", "T5"]


@dataclass
class OptimizationResult:
    setpoints: dict            # {"T1":.., ..., "T5":..}
    predictions: dict          # {"Lc":.., "Le":.., "D":..} at the optimum
    uncertainty: dict | None   # GPR std at the optimum, per target with a GPR model
    target: dict                # the requested target outcome
    objective_value: float
    converged: bool
    warnings: list


def _build_feature_row(T: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    raw = pd.DataFrame([dict(zip(RAW_ORDER, T))])
    full = add_engineered_features(raw)
    return full[feature_names]


def _ordering_violation(T: np.ndarray) -> float:
    """Sum of squared violations of T1>=T2>=T3>=T4>=T5 (Zhang & Cheng 2022,
    Section 2.2.2 -- required for directional solidification)."""
    diffs = [T[i] - T[i + 1] for i in range(len(T) - 1)]  # want every diff >= 0
    return sum(max(0.0, -d) ** 2 for d in diffs)


def optimize_setpoints(
    surrogate,
    target: dict,
    bounds: dict | None = None,
    uncertainty_weight: float = 200.0,
    ordering_penalty_weight: float = 1e5,
    seed: int = 42,
    maxiter: int = 200,
    popsize: int = 20,
) -> OptimizationResult:
    """
    surrogate: a fitted src.surrogate.FastSurrogate.
    target: dict subset of {"Lc": value, "Le": value, "D": value} (metres)
        -- the interface outcome you want the furnace to achieve.
    bounds: optional {"T1": (lo, hi), ...}; defaults to the observed
        training range for each heater (see surrogate.feature_ranges) --
        i.e. the optimizer searches only inside the simulated regime by
        default, since outside it the surrogate's accuracy is unknown.
    uncertainty_weight: how strongly to penalize predictive uncertainty at
        candidate set-points (only applies to targets whose selected model
        is the GaussianProcess, which is the only model here that reports
        uncertainty). 0 disables the uncertainty penalty.
    """
    unknown = set(target) - set(surrogate.targets)
    if unknown:
        raise ValueError(f"Unknown target(s) {unknown}; must be a subset of {surrogate.targets}")

    bounds = bounds or {}
    bounds_list = [bounds.get(f, surrogate.feature_ranges[f]) for f in RAW_ORDER]

    def objective(T):
        row = _build_feature_row(np.asarray(T), surrogate.feature_names)
        result = surrogate.predict(row, return_std=True)
        pred = result["predictions"].iloc[0]

        err = sum((pred[k] - v) ** 2 for k, v in target.items())

        unc_penalty = 0.0
        if uncertainty_weight and result["uncertainty"] is not None:
            unc = result["uncertainty"].iloc[0]
            relevant = [unc[k] for k in target if k in unc.index]
            if relevant:
                unc_penalty = uncertainty_weight * float(np.mean(np.square(relevant)))

        order_penalty = ordering_penalty_weight * _ordering_violation(np.asarray(T))
        return err + unc_penalty + order_penalty

    res = differential_evolution(
        objective, bounds_list, seed=seed, maxiter=maxiter, popsize=popsize,
        tol=1e-12, polish=True, updating="deferred", workers=1,
    )

    T_opt = res.x
    row = _build_feature_row(T_opt, surrogate.feature_names)
    final = surrogate.predict(row, return_std=True)
    pred = final["predictions"].iloc[0].to_dict()
    unc = final["uncertainty"].iloc[0].to_dict() if final["uncertainty"] is not None else None

    return OptimizationResult(
        setpoints=dict(zip(RAW_ORDER, T_opt)),
        predictions=pred,
        uncertainty=unc,
        target=target,
        objective_value=float(res.fun),
        converged=bool(res.success),
        warnings=final["warnings"],
    )
