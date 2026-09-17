"""
models.py
=========
Defines every regression model used in the comparison, and a light
hyperparameter-tuning step for the two tree-ensemble models.

XGBoost availability
---------------------
This sandbox has no outbound access to PyPI, so `pip install xgboost`
cannot complete here. The code below tries to import the real xgboost
package first; if it is not installed (as in this environment) it falls
back to sklearn's GradientBoostingRegressor -- the closest available
substitute (classic gradient-boosted trees) -- and records that a
substitution happened. If this project is later run in an environment
with network access / xgboost installed, it will automatically use the
real XGBRegressor instead, with no code changes required.

The "additional baseline" HistGradientBoostingRegressor is *also* a
gradient-boosted tree model (histogram-based, the same algorithm family
XGBoost/LightGBM use), so between it and the GradientBoostingRegressor
fallback, the comparison still includes two distinct boosting
implementations even without real xgboost.

Hyperparameter tuning
-----------------------
With only ~77 training rows (96 rows x 80%), a full nested
hyperparameter search inside every outer CV fold would itself be
unstable and is not implemented. Instead, RandomForest and the
XGBoost-family model are each tuned ONCE per target with a small
GridSearchCV(cv=3) on the training split (evaluated by negative MAE),
and the resulting best hyperparameters are then held FIXED for the
outer, model-comparison 5-fold CV -- so every model is judged on
identical folds, per the project's fairness requirement.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from .preprocessing import make_scaling_pipeline

RANDOM_STATE = 42

try:
    from xgboost import XGBRegressor  # type: ignore
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def _boosting_model(**params):
    """Return a real XGBRegressor if available, else the documented
    GradientBoostingRegressor fallback with translated parameter names."""
    if XGBOOST_AVAILABLE:
        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
            **params,
        )
    warnings.warn(
        "xgboost is not installed in this environment (no PyPI network "
        "access) -- using sklearn.GradientBoostingRegressor as an honestly "
        "documented substitute for the 'XGBoost' model.",
        RuntimeWarning,
    )
    # translate the small param set we tune to GradientBoostingRegressor's names
    translated = dict(params)
    if "n_estimators" in translated:
        pass
    if "max_depth" in translated:
        pass
    if "learning_rate" in translated:
        pass
    translated.pop("subsample", None)
    return GradientBoostingRegressor(random_state=RANDOM_STATE, **translated)


def gpr_kernel():
    """Constant * RBF (smooth nonlinear response surface, appropriate for a
    physically smooth interface-position function of temperature) plus a
    WhiteKernel to absorb residual/process noise so the GPR does not
    interpolate noise as signal. Fully tractable exactly (no sparse/
    inducing-point approximation needed) at n<=96 -- GPR training cost is
    O(n^3) which is trivial at this scale (~10^6 flops), so no
    approximation is used; this is noted here rather than left unstated."""
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    )


def tune_tree_models(X_train, y_train_series, random_state: int = RANDOM_STATE) -> dict:
    """Small GridSearchCV(cv=3) for RandomForest and the boosting model,
    for a single target's training data. Returns best_params for each."""
    rf_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 4, 8],
        "model__min_samples_leaf": [1, 2, 4],
    }
    rf_pipe = make_scaling_pipeline(RandomForestRegressor(random_state=random_state))
    rf_search = GridSearchCV(rf_pipe, rf_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    rf_search.fit(X_train, y_train_series)

    if XGBOOST_AVAILABLE:
        b_grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.1],
        }
    else:
        b_grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.1],
        }
    b_pipe = make_scaling_pipeline(_boosting_model())
    b_search = GridSearchCV(b_pipe, b_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    b_search.fit(X_train, y_train_series)

    return {
        "RandomForest": {k.replace("model__", ""): v for k, v in rf_search.best_params_.items()},
        "XGBoost": {k.replace("model__", ""): v for k, v in b_search.best_params_.items()},
    }


def build_models(tuned_params: dict | None = None) -> dict:
    """Return {model_name: unfitted Pipeline(StandardScaler, model)} for
    every single-output model in the comparison."""
    tuned_params = tuned_params or {}
    rf_params = tuned_params.get("RandomForest", {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2})
    b_params = tuned_params.get("XGBoost", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05})

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.001),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, **rf_params),
        "XGBoost": _boosting_model(**b_params),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE, max_depth=4, min_samples_leaf=5, max_iter=200
        ),
        "GaussianProcess": GaussianProcessRegressor(
            kernel=gpr_kernel(), random_state=RANDOM_STATE, normalize_y=True, n_restarts_optimizer=5
        ),
    }
    return {name: make_scaling_pipeline(m) for name, m in models.items()}


def build_multioutput_models(tuned_params: dict | None = None) -> dict:
    """Multi-output variants for the RF-vs-independent and
    boosting-vs-independent comparisons. RandomForest supports multi-output
    natively (joint variance-reduction splits); the boosting model does not,
    so it is wrapped in MultiOutputRegressor (fits one model per target
    internally -- included for a like-for-like multi-output API comparison,
    not because it changes the underlying independence of the fits)."""
    tuned_params = tuned_params or {}
    rf_params = tuned_params.get("RandomForest", {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2})
    b_params = tuned_params.get("XGBoost", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05})
    return {
        "RandomForest_MultiOutput": make_scaling_pipeline(
            RandomForestRegressor(random_state=RANDOM_STATE, **rf_params)
        ),
        "XGBoost_MultiOutput": make_scaling_pipeline(
            MultiOutputRegressor(_boosting_model(**b_params))
        ),
    }
