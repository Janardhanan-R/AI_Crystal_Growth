"""
surrogate.py
============
A fast, data-driven surrogate model for Lc, Le, D as functions of the five
furnace zone temperatures.

What this IS and IS NOT
-------------------------
* This is a DATA-DRIVEN SURROGATE MODEL: a statistical function fit to 96
  simulated/measured (T1..T5) -> (Lc, Le, D) data points. It reproduces the
  input/output mapping present in that data, nothing more.
* This is NOT a physics-based digital twin: it contains no heat-transfer,
  fluid-flow, or interface-thermodynamics equations, and cannot extrapolate
  outside the physical regime represented by the training data with any
  guarantee.
* This is NOT a hybrid digital twin: no physical constraints (e.g. energy
  balance, known monotonicity of interface position with temperature) are
  embedded in the model structure. Such constraints could be added -- e.g.
  as monotonicity constraints in the tree ensembles, or as a physics-based
  mean function in the GPR prior -- but none are implemented here because no
  governing equations were supplied for this project; doing so would mean
  inventing physics, which was explicitly ruled out.
* Because the training data is a 96-point designed experiment (not a large
  historical process log), this surrogate should be treated as a
  RESEARCH-STAGE / offline optimization aid, not a validated real-time
  control component. Its inference latency (measured below, typically
  sub-millisecond per prediction for every model in this comparison) is
  fast enough for real-time use, but the STATISTICAL ACCURACY of that
  real-time use has only been checked against a small held-out test set
  and 5-fold CV on the same 96-row dataset -- not against independent new
  experiments. That distinction is deliberately kept explicit throughout
  this project's documentation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone


@dataclass
class FastSurrogate:
    models: dict                      # {target: fitted Pipeline}
    feature_names: list
    targets: list
    feature_ranges: dict = field(default_factory=dict)   # {feature: (min, max)} from training data
    target_ranges: dict = field(default_factory=dict)    # {target: (min, max)} from training data
    supports_uncertainty: dict = field(default_factory=dict)  # {target: bool}

    @classmethod
    def fit(cls, per_target_pipelines: dict, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """per_target_pipelines: {target: unfitted Pipeline}. Fits one
        pipeline per target on the FULL available training data (train +
        the previously-held-out test rows are combined at this final stage,
        per standard practice: CV/test were used to select and validate the
        approach; the deployed surrogate is then refit on all data)."""
        fitted = {}
        supports_uncertainty = {}
        for target, pipeline in per_target_pipelines.items():
            # clone defensively: two targets can select the same model type
            # (e.g. both use GaussianProcess), and without cloning they would
            # share -- and overwrite -- the same fitted object.
            pipeline = clone(pipeline)
            pipeline.fit(X_train, y_train[target])
            fitted[target] = pipeline
            supports_uncertainty[target] = hasattr(pipeline.named_steps["model"], "predict") and \
                pipeline.named_steps["model"].__class__.__name__ == "GaussianProcessRegressor"

        feature_ranges = {c: (float(X_train[c].min()), float(X_train[c].max())) for c in X_train.columns}
        target_ranges = {c: (float(y_train[c].min()), float(y_train[c].max())) for c in y_train.columns}
        return cls(
            models=fitted,
            feature_names=list(X_train.columns),
            targets=list(y_train.columns),
            feature_ranges=feature_ranges,
            target_ranges=target_ranges,
            supports_uncertainty=supports_uncertainty,
        )

    def _validate_input(self, X: pd.DataFrame) -> list:
        """Returns a list of warning strings for any row/feature outside the
        training data's observed range (extrapolation warning). Does not
        block prediction -- the caller decides how to react."""
        warnings = []
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required input features: {sorted(missing)}")
        for feat in self.feature_names:
            lo, hi = self.feature_ranges[feat]
            below = X[feat] < lo
            above = X[feat] > hi
            if below.any() or above.any():
                warnings.append(
                    f"Feature '{feat}' has {int(below.sum() + above.sum())} row(s) outside the "
                    f"training range [{lo:.3f}, {hi:.3f}] -- prediction there is EXTRAPOLATION."
                )
        return warnings

    def _validate_output(self, target: str, preds: np.ndarray) -> list:
        warnings = []
        lo, hi = self.target_ranges[target]
        span = hi - lo
        below = preds < (lo - 0.5 * span)
        above = preds > (hi + 0.5 * span)
        if below.any() or above.any():
            warnings.append(
                f"Prediction for '{target}' has {int(np.sum(below) + np.sum(above))} value(s) far "
                f"outside the observed training range [{lo:.4f}, {hi:.4f}] -- treat with caution."
            )
        return warnings

    def predict(self, X: pd.DataFrame, return_std: bool = True) -> dict:
        """Multi-target prediction with input/output validation and timing.
        Returns {'predictions': DataFrame, 'uncertainty': DataFrame or None,
        'warnings': [...], 'inference_time_s': float}."""
        X = X[self.feature_names]
        warnings = self._validate_input(X)

        preds = {}
        stds = {}
        t0 = time.perf_counter()
        for target, pipeline in self.models.items():
            model = pipeline.named_steps["model"]
            scaler = pipeline.named_steps["scaler"]
            X_scaled = scaler.transform(X)
            if return_std and model.__class__.__name__ == "GaussianProcessRegressor":
                mean, std = model.predict(X_scaled, return_std=True)
                preds[target] = mean
                stds[target] = std
            else:
                preds[target] = model.predict(X_scaled)
        inference_time_s = time.perf_counter() - t0

        preds_df = pd.DataFrame(preds, index=X.index)
        for target in self.models:
            warnings.extend(self._validate_output(target, preds_df[target].values))

        uncertainty_df = pd.DataFrame(stds, index=X.index) if stds else None
        return {
            "predictions": preds_df,
            "uncertainty": uncertainty_df,
            "warnings": warnings,
            "inference_time_s": inference_time_s,
            "inference_time_per_sample_ms": (inference_time_s / max(1, len(X))) * 1000,
        }

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FastSurrogate":
        return joblib.load(path)
