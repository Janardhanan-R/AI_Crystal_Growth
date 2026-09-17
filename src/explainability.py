"""
explainability.py
==================
SHAP-based explanation for the tree-ensemble models, specifically to make
this project's explainability results directly comparable to Zhang & Cheng
(2022) -- the paper this dataset and problem come from -- which explains
its RandomForest model with SHAP summary plots (their Figure 3(b), 3(d)).

Not available in the development sandbox
------------------------------------------
`shap` could not be installed here (same no-PyPI-network-access sandbox
limitation as xgboost -- see src/models.py). This module is written and
documented so it works correctly once you `pip install shap` on your own
machine and rerun `main.py`; it degrades gracefully (skips with a clear
message, does not crash the rest of the pipeline) when shap is absent.

Method
-------
`shap.TreeExplainer` is used on the fitted RandomForest (Model 2/4 in
Zhang & Cheng's own labeling) for each target, computed on the SAME
STANDARDIZED feature space the model was actually trained on (the
StandardScaler step of the pipeline) -- consistent with how this
project's native `feature_importances_` plots are already computed (see
src/evaluation.py). A summary plot is saved per target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_shap_summary(fitted: dict, X_test: pd.DataFrame, feature_names: list[str],
                       targets: list[str], out_dir: str, model_name: str = "RandomForest") -> list[str]:
    """fitted: {(model_name, target): fitted Pipeline}, as produced by
    evaluation.fit_and_evaluate_test. Saves one SHAP summary plot per
    target for `model_name` (RandomForest by default, matching Zhang &
    Cheng's explained model). Returns the list of files written (empty
    list, with a printed message, if shap is not installed)."""
    if not SHAP_AVAILABLE:
        print("SHAP explainability SKIPPED: `shap` is not installed in this "
              "environment. Run `pip install shap` and rerun main.py to "
              "generate plots/shap_summary_<target>.png.")
        return []

    written = []
    for target in targets:
        key = (model_name, target)
        if key not in fitted:
            continue
        pipeline = fitted[key]
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["model"]
        X_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False)
        plt.title(f"SHAP summary -- {model_name} -- {target}\n"
                  f"(standardized feature space; cf. Zhang & Cheng 2022, Fig. 3)", fontsize=9)
        plt.tight_layout()
        path = f"{out_dir}/shap_summary_{target}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path}")
    return written
